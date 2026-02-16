"""Nonlinear cam-bar-motor hybrid model solved with torchsde."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np

from ..config import HybridNonlinearCamConfig
from .base import BaseModel
from .torchsde_utils import (
    ControlledPathMixin,
    inverse_softplus,
    optimize_with_adam,
    simulate_controlled_sde,
)


class _NonlinearCamSDEFunc(ControlledPathMixin):
    """SDE drift for nonlinear cam-bar-motor model with zero diffusion."""

    noise_type = "diagonal"
    sde_type = "ito"
    _POSITIVE_PARAMS = {"R", "r", "e", "L", "I", "J", "k", "k_t", "R_M", "L_M"}

    def __init__(self, dt, params, trainable_params, device,
                 min_m_eff=1e-3, max_acc=1e4):
        import torch, torch.nn as nn

        self._eps = 1e-8
        self._device = device
        self._trainable = set(trainable_params)
        self._min_m_eff = float(min_m_eff)
        self._max_acc = float(max_acc)

        self._raw_params: Dict[str, nn.Parameter] = {}
        self._const_params: Dict[str, torch.Tensor] = {}
        for name, value in params.items():
            if name in self._trainable:
                init_raw = inverse_softplus(value) if name in self._POSITIVE_PARAMS else float(value)
                self._raw_params[name] = nn.Parameter(
                    torch.tensor(init_raw, dtype=torch.float64, device=device))
            else:
                self._const_params[name] = torch.tensor(float(value), dtype=torch.float64, device=device)

        self._init_control_path(dt=dt, input_dim=1, device=device, dtype=torch.float64)

    def parameters(self):
        return list(self._raw_params.values())

    def train(self): return self
    def eval(self): return self

    def _decode_param(self, name):
        import torch.nn.functional as F
        if name in self._raw_params:
            raw = self._raw_params[name]
            return (F.softplus(raw) + self._eps) if name in self._POSITIVE_PARAMS else raw
        return self._const_params[name]

    def _decoded_params(self):
        all_names = set(self._raw_params) | set(self._const_params)
        return {n: self._decode_param(n) for n in all_names}

    @staticmethod
    def _safe_div(num, den, eps=1e-8):
        import torch
        den_safe = torch.where(torch.abs(den) < eps, torch.full_like(den, eps), den)
        return num / den_safe

    @staticmethod
    def _geometry_terms(theta, p):
        import torch
        Rp = p["R"] + p["r"]
        sin_t, cos_t = torch.sin(theta), torch.cos(theta)
        inside = Rp**2 - (p["e"]**2) * (sin_t**2)
        S = torch.sqrt(torch.clamp(inside, min=1e-10))
        sin_phi = torch.clamp((p["e"] * sin_t) / Rp, -1.0 + 1e-8, 1.0 - 1e-8)
        cos_phi = torch.sqrt(torch.clamp(1.0 - sin_phi**2, min=1e-8))
        y_geom = S - p["e"] * cos_t - (Rp - p["e"])
        A = -(p["e"]**2) * sin_t * cos_t / S + p["e"] * sin_t
        B = (p["e"] * cos_t - (p["e"]**2) * (cos_t**2 - sin_t**2) / S
             - (p["e"]**4) * (sin_t**2) * (cos_t**2) / (S**3))
        return y_geom, cos_phi, A, B

    def _theta_ddot(self, theta, omega, current, p):
        import torch
        y_geom, cos_phi, A, B = self._geometry_terms(theta, p)
        inertial_coeff = (4.0 * p["I"]) / (p["L"]**2 * cos_phi)
        spring_coeff = (2.0 * p["k"]) / (p["L"] * cos_phi)
        m_eff = p["J"] + inertial_coeff * A
        m_eff = torch.clamp(torch.abs(m_eff), min=self._min_m_eff) * torch.sign(m_eff + self._eps)
        rhs = p["k_t"] * current - spring_coeff * (y_geom + p["delta"]) - inertial_coeff * B * (omega**2)
        acc = self._safe_div(rhs, m_eff)
        return torch.clamp(acc, -self._max_acc, self._max_acc)

    def f(self, t, y):
        import torch
        p = self._decoded_params()
        theta, omega, current = y[:, 0], y[:, 1], y[:, 2]
        u_t = self._u_at(t, y.shape[0])[:, 0]

        omega_c = 500.0 * torch.tanh(omega / 500.0)
        current_c = 50.0 * torch.tanh(current / 50.0)

        acc = self._theta_ddot(theta, omega_c, current_c, p)
        i_dot = -(p["R_M"] / p["L_M"]) * current_c + (p["k_b"] / p["L_M"]) * omega_c + u_t / p["L_M"]
        i_dot = torch.clamp(i_dot, -1e4, 1e4)
        return torch.stack([omega_c, acc, i_dot], dim=-1)

    def g(self, t, y):
        import torch
        return torch.zeros_like(y)

    def decoded_parameter_dict(self):
        params = self._decoded_params()
        return {n: float(v.detach().cpu().item()) for n, v in params.items()}


class HybridNonlinearCam(BaseModel):
    """Nonlinear cam-bar-motor hybrid model."""

    def __init__(self, config: HybridNonlinearCamConfig | None = None, **kwargs):
        if config is None:
            config = HybridNonlinearCamConfig(**kwargs)
        super().__init__(config)
        self.sde_func_ = None
        self._device = None

    def _get_physics_dict(self):
        c = self.config
        return {"R": c.R, "r": c.r, "e": c.e, "L": c.L, "I": c.I,
                "J": c.J, "k": c.k, "delta": c.delta, "k_t": c.k_t,
                "k_b": c.k_b, "R_M": c.R_M, "L_M": c.L_M}

    def _simulate_state(self, u_t, theta0, omega0, current0):
        import torch
        c = self.config
        x0 = torch.stack([theta0, omega0, current0])
        return simulate_controlled_sde(
            sde_func=self.sde_func_, u_path=u_t, x0=x0,
            dt=c.dt, method="euler",
            integration_dt=c.dt / c.integration_substeps)

    # ── training ──────────────────────────────────────────────────────

    def _fit(self, u, y, *, val_data=None, logger=None):
        import torch
        c = self.config
        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()

        self._device = self._resolve_torch_device()

        self.sde_func_ = _NonlinearCamSDEFunc(
            dt=c.dt, params=self._get_physics_dict(),
            trainable_params=c.trainable_params, device=self._device)

        u_t = torch.tensor(u.reshape(-1, 1), dtype=torch.float64, device=self._device)
        y_t = torch.tensor(y, dtype=torch.float64, device=self._device)
        theta0 = y_t[0]
        omega0 = (y_t[1] - y_t[0]) / c.dt
        current0 = torch.tensor(c.initial_current, dtype=torch.float64, device=self._device)

        optim_vars = self.sde_func_.parameters()
        if not optim_vars:
            self.training_loss_ = []
            return

        def _loss_fn():
            path = self._simulate_state(u_t, theta0, omega0, current0)
            return torch.mean((path[:, 0] - y_t) ** 2)

        def _log_epoch(epoch, loss_value, grad_norm):
            if logger and c.wandb_log_every > 0 and epoch % c.wandb_log_every == 0:
                payload = {"train/epoch": epoch, "train/loss": loss_value, "train/grad_norm": grad_norm}
                payload.update({f"params/{k}": v for k, v in self.sde_func_.decoded_parameter_dict().items()})
                logger.log_metrics(payload, step=epoch)

        self.training_loss_ = optimize_with_adam(
            parameters=optim_vars, loss_fn=_loss_fn,
            epochs=c.epochs, learning_rate=c.learning_rate,
            on_epoch_end=_log_epoch, max_grad_norm=0.5,
            verbose=c.verbose, progress_desc="Training HybridNonlinearCam")

    # ── predict ───────────────────────────────────────────────────────

    def predict_osa(self, u, y):
        import torch
        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        dt = self.config.dt
        preds = np.copy(y)
        current = self.config.initial_current
        with torch.no_grad():
            for k in range(1, len(y) - 1):
                theta_k = torch.tensor(y[k], dtype=torch.float64, device=self._device)
                omega_k = torch.tensor((y[k] - y[k-1]) / dt, dtype=torch.float64, device=self._device)
                current_t = torch.tensor(current, dtype=torch.float64, device=self._device)
                u_step = torch.tensor([[u[k]], [u[k]]], dtype=torch.float64, device=self._device)
                path = self._simulate_state(u_step, theta_k, omega_k, current_t)
                preds[k+1] = float(path[-1, 0].cpu().item())
                current = float(path[-1, 2].cpu().item())
        return preds

    def predict_free_run(self, u, y_initial):
        import torch
        u = np.asarray(u, dtype=float).flatten()
        y0 = np.asarray(y_initial, dtype=float).flatten()
        dt = self.config.dt
        with torch.no_grad():
            u_t = torch.tensor(u.reshape(-1, 1), dtype=torch.float64, device=self._device)
            theta0 = torch.tensor(y0[0], dtype=torch.float64, device=self._device)
            omega0 = torch.tensor((y0[1] - y0[0]) / dt, dtype=torch.float64, device=self._device)
            current0 = torch.tensor(self.config.initial_current, dtype=torch.float64, device=self._device)
            path = self._simulate_state(u_t, theta0, omega0, current0)
        return path[:, 0].cpu().numpy()

    # ── save / load hooks ─────────────────────────────────────────────

    def _collect_state(self) -> Dict[str, Any]:
        if self.sde_func_ is None:
            return {}
        import torch
        state = {}
        for name, p in self.sde_func_._raw_params.items():
            state[f"raw_{name}"] = p.data.cpu()
        return state

    def _restore_state(self, state):
        for name, p in self.sde_func_._raw_params.items():
            key = f"raw_{name}"
            if key in state:
                p.data.copy_(state[key])

    def _build_for_load(self):
        c = self.config
        self._device = self._resolve_torch_device("cpu")
        self.sde_func_ = _NonlinearCamSDEFunc(
            dt=c.dt, params=self._get_physics_dict(),
            trainable_params=c.trainable_params, device=self._device)

    def get_physics_params(self) -> Dict[str, float]:
        if self.sde_func_ is None:
            raise RuntimeError("Model not fitted")
        return self.sde_func_.decoded_parameter_dict()

    def __repr__(self):
        c = self.config
        return f"HybridNonlinearCam(dt={c.dt}, trainable={c.trainable_params}, epochs={c.epochs})"
