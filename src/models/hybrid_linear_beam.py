"""Linear physics-guided hybrid model for beam dynamics using torchsde.

    J*theta_ddot + R*theta_dot + K*(theta + delta) = tau*V
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import BaseModel, resolve_device
from .torchsde_utils import (
    ControlledPathMixin,
    inverse_softplus,
    optimize_with_adam,
    simulate_controlled_sde,
)


class _LinearBeamSDEFunc(ControlledPathMixin):
    """SDE drift for beam dynamics with zero diffusion."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, dt, tau, estimate_delta, init_params, device):
        import torch, torch.nn as nn

        self.tau = float(tau)
        self.estimate_delta = bool(estimate_delta)
        self._eps = 1e-8
        self._device = device

        self.raw_J = nn.Parameter(torch.tensor(inverse_softplus(init_params["J"]), dtype=torch.float64, device=device))
        self.raw_R = nn.Parameter(torch.tensor(inverse_softplus(init_params["R"]), dtype=torch.float64, device=device))
        self.raw_K = nn.Parameter(torch.tensor(inverse_softplus(init_params["K"]), dtype=torch.float64, device=device))

        if self.estimate_delta:
            self.delta = nn.Parameter(torch.tensor(float(init_params["delta"]), dtype=torch.float64, device=device))
        else:
            self.delta = torch.tensor(0.0, dtype=torch.float64, device=device)

        self._init_control_path(dt=dt, input_dim=1, device=device, dtype=torch.float64)

    def parameters(self):
        params = [self.raw_J, self.raw_R, self.raw_K]
        if hasattr(self.delta, "requires_grad") and self.delta.requires_grad:
            params.append(self.delta)
        return params

    def train(self): return self
    def eval(self): return self

    def _decoded_params(self):
        import torch.nn.functional as F
        J = F.softplus(self.raw_J) + self._eps
        R = F.softplus(self.raw_R) + self._eps
        K = F.softplus(self.raw_K) + self._eps
        delta = self.delta if self.estimate_delta else self.delta.detach()
        return J, R, K, delta

    def f(self, t, y):
        import torch
        J, R, K, delta = self._decoded_params()
        theta, omega = y[:, 0], y[:, 1]
        voltage = self._u_at(t, y.shape[0])[:, 0]
        acc = (self.tau * voltage - R * omega - K * (theta + delta)) / J
        return torch.stack([omega, acc], dim=-1)

    def g(self, t, y):
        import torch
        return torch.zeros_like(y)

    def decoded_parameter_dict(self) -> Dict[str, float]:
        J, R, K, delta = self._decoded_params()
        return {"J": float(J.item()), "R": float(R.item()),
                "K": float(K.item()), "delta": float(delta.item())}


class HybridLinearBeam(BaseModel):
    """Linear beam physics model: J*θ̈ + R*θ̇ + K*(θ+δ) = τ*V."""

    def __init__(self, config=None):
        from ..config import HybridLinearBeamConfig
        if config is None:
            config = HybridLinearBeamConfig()
        super().__init__(config)
        self.sde_func_ = None
        self._device = None

    # ── helpers ───────────────────────────────────────────────────────

    def _initial_guess(self, u, y):
        c = self.config
        dt = c.dt
        y_dot = np.gradient(y, dt)
        y_ddot = np.gradient(y_dot, dt)
        if c.estimate_delta:
            phi = np.column_stack([-y_dot, -y, u, np.ones_like(y)])
        else:
            phi = np.column_stack([-y_dot, -y, u])
        reg = phi.T @ phi + c.ridge * np.eye(phi.shape[1])
        theta = np.linalg.solve(reg, phi.T @ y_ddot)
        a1, a0, b0 = float(theta[0]), float(theta[1]), float(theta[2])
        bias = float(theta[3]) if c.estimate_delta else 0.0
        b0_safe = b0 if not np.isclose(b0, 0.0) else 1e-6
        J0 = max(c.tau / b0_safe, 1e-6)
        R0 = max(a1 * J0, 1e-6)
        K0 = max(a0 * J0, 1e-6)
        delta0 = float(-bias / a0) if c.estimate_delta and not np.isclose(a0, 0.0) else 0.0
        return {"J": J0, "R": R0, "K": K0, "delta": delta0}

    def _simulate_theta(self, u_t, theta0, omega0):
        import torch
        c = self.config
        x0 = torch.stack([theta0, omega0])
        int_dt = c.dt / c.integration_substeps if c.integration_substeps > 1 else None
        path = simulate_controlled_sde(
            sde_func=self.sde_func_, u_path=u_t, x0=x0,
            dt=c.dt, method="euler", integration_dt=int_dt)
        return path[:, 0]

    # ── training ──────────────────────────────────────────────────────

    def _fit(self, u, y, *, val_data=None, logger=None):
        import torch
        c = self.config
        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()

        self._device = resolve_device(c.device)

        init_params = self._initial_guess(u, y)
        self.sde_func_ = _LinearBeamSDEFunc(
            dt=c.dt, tau=c.tau, estimate_delta=c.estimate_delta,
            init_params=init_params, device=self._device)

        u_t = torch.tensor(u.reshape(-1, 1), dtype=torch.float64, device=self._device)
        y_t = torch.tensor(y, dtype=torch.float64, device=self._device)
        theta0 = y_t[0]
        omega0 = (y_t[1] - y_t[0]) / c.dt

        def _loss_fn():
            hat = self._simulate_theta(u_t, theta0, omega0)
            return torch.mean((hat - y_t) ** 2)

        def _log_epoch(epoch, loss_value, grad_norm):
            if logger and c.wandb_log_every > 0 and epoch % c.wandb_log_every == 0:
                payload = {"train/epoch": epoch, "train/loss": loss_value, "train/grad_norm": grad_norm}
                payload.update({f"params/{k}": v for k, v in self.sde_func_.decoded_parameter_dict().items()})
                logger.log_metrics(payload, step=epoch)

        self.training_loss_ = optimize_with_adam(
            parameters=self.sde_func_.parameters(), loss_fn=_loss_fn,
            epochs=c.epochs, learning_rate=c.learning_rate,
            on_epoch_end=_log_epoch, verbose=c.verbose,
            progress_desc="Training HybridLinearBeam")

    # ── predict ───────────────────────────────────────────────────────

    def predict_osa(self, u, y):
        import torch
        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        dt = self.config.dt
        preds = np.copy(y)
        with torch.no_grad():
            for k in range(1, len(y) - 1):
                theta_k = torch.tensor(y[k], dtype=torch.float64, device=self._device)
                omega_k = torch.tensor((y[k] - y[k-1]) / dt, dtype=torch.float64, device=self._device)
                u_step = torch.tensor([[u[k]], [u[k]]], dtype=torch.float64, device=self._device)
                path = self._simulate_theta(u_step, theta_k, omega_k)
                preds[k+1] = float(path[-1].cpu().item())
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
            hat = self._simulate_theta(u_t, theta0, omega0)
        return hat.cpu().numpy()

    # ── save / load hooks ─────────────────────────────────────────────

    def _collect_state(self) -> Dict[str, Any]:
        if self.sde_func_ is None:
            return {}
        import torch
        return {name: p.data.cpu() for name, p in
                zip(["raw_J", "raw_R", "raw_K", "delta"],
                    [self.sde_func_.raw_J, self.sde_func_.raw_R,
                     self.sde_func_.raw_K, self.sde_func_.delta])}

    def _restore_state(self, state):
        for name, val in state.items():
            getattr(self.sde_func_, name).data.copy_(val)

    def _build_for_load(self):
        import torch
        c = self.config
        self._device = torch.device("cpu")
        init_params = {"J": 0.1, "R": 0.1, "K": 1.0, "delta": 0.0}
        self.sde_func_ = _LinearBeamSDEFunc(
            dt=c.dt, tau=c.tau, estimate_delta=c.estimate_delta,
            init_params=init_params, device=self._device)

    def get_physics_params(self) -> Dict[str, float]:
        if self.sde_func_ is None:
            raise RuntimeError("Model not fitted")
        return self.sde_func_.decoded_parameter_dict()

    def __repr__(self):
        c = self.config
        return f"HybridLinearBeam(dt={c.dt}, tau={c.tau}, epochs={c.epochs})"
