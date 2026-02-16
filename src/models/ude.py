"""Universal Differential Equation (UDE) model.

Combines linear beam physics with a small neural residual on acceleration:
    dθ/dt = ω
    dω/dt = (τV - Rω - K(θ+δ))/J + f_nn(ω)
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..config import UDEConfig
from .base import BaseModel, DEFAULT_GRAD_CLIP
from .torchsde_utils import (
    ControlledPathMixin,
    inverse_softplus,
    train_sequence_batches,
)


# ── internal SDE function ─────────────────────────────────────────────

class _UDEFunc(ControlledPathMixin):
    """Physics drift + neural residual on dω/dt, zero diffusion."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, dt, tau, init_params, hidden_layers, device, dtype):
        import torch, torch.nn as nn
        self._device, self._dtype = device, dtype
        self.tau = float(tau)
        self._eps = 1e-8

        self.raw_J = nn.Parameter(torch.tensor(inverse_softplus(init_params["J"]), dtype=dtype, device=device))
        self.raw_R = nn.Parameter(torch.tensor(inverse_softplus(init_params["R"]), dtype=dtype, device=device))
        self.raw_K = nn.Parameter(torch.tensor(inverse_softplus(init_params["K"]), dtype=dtype, device=device))
        self.delta = nn.Parameter(torch.tensor(float(init_params["delta"]), dtype=dtype, device=device))

        layers: list[nn.Module] = []
        prev = 1
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.residual_net = nn.Sequential(*layers).to(dtype).to(device)
        for m in self.residual_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self._init_control_path(dt, 1, device, dtype)

    def _decoded_physics(self):
        import torch.nn.functional as F
        J = F.softplus(self.raw_J) + self._eps
        R = F.softplus(self.raw_R) + self._eps
        K = F.softplus(self.raw_K) + self._eps
        return J, R, K, self.delta

    def parameters(self):
        return [self.raw_J, self.raw_R, self.raw_K, self.delta] + list(self.residual_net.parameters())

    def to(self, device):
        self.residual_net = self.residual_net.to(device)
        self._u_path = self._u_path.to(device)
        return self

    def train(self):
        self.residual_net.train()

    def eval(self):
        self.residual_net.eval()

    def f(self, t, y):
        import torch
        J, R, K, delta = self._decoded_physics()
        theta, omega = y[:, 0], y[:, 1]
        voltage = self._u_at(t, y.shape[0])[:, 0]
        acc_phys = (self.tau * voltage - R * omega - K * (theta + delta)) / J
        acc_res = self.residual_net(omega.unsqueeze(-1)).squeeze(-1)
        return torch.stack([omega, acc_phys + acc_res], dim=-1)

    def g(self, t, y):
        import torch
        return torch.zeros_like(y)

    def decoded_parameter_dict(self):
        J, R, K, delta = self._decoded_physics()
        return {"J": float(J.item()), "R": float(R.item()),
                "K": float(K.item()), "delta": float(delta.item())}


# ── public model ──────────────────────────────────────────────────────

class UDE(BaseModel):
    """Universal Differential Equation: physics + neural residual."""

    def __init__(self, config: UDEConfig | None = None, **kwargs):
        if config is None:
            config = UDEConfig(**kwargs)
        super().__init__(config)
        self.sde_func_ = None
        self._device = None
        self._dtype = None

    # ── helpers ───────────────────────────────────────────────────────

    def _initial_guess(self, u, y):
        dt = self.config.dt
        yd = np.gradient(y, dt)
        ydd = np.gradient(yd, dt)
        phi = np.column_stack([-yd, -y, u, np.ones_like(y)])
        ridge = 1e-8
        theta = np.linalg.solve(phi.T @ phi + ridge * np.eye(4), phi.T @ ydd)
        a1, a0, b0, bias = map(float, theta)
        b0_safe = b0 if not np.isclose(b0, 0.0) else 1e-6
        J0 = max(1.0 / b0_safe, 1e-6)
        R0, K0 = max(a1 * J0, 1e-6), max(a0 * J0, 1e-6)
        delta0 = -bias / a0 if not np.isclose(a0, 0.0) else 0.0
        return {"J": J0, "R": R0, "K": K0, "delta": delta0}

    def _simulate_trajectory(self, u_path, x0):
        from .torchsde_utils import simulate_controlled_sde
        return simulate_controlled_sde(
            self.sde_func_, u_path, x0, dt=self.config.dt, method="euler",
        )

    # ── training ──────────────────────────────────────────────────────

    def _fit(self, u, y, *, val_data=None, logger=None):
        import torch
        c = self.config
        self._device = self._resolve_torch_device()
        self._dtype = torch.float32

        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        init_params = self._initial_guess(u, y)

        self.sde_func_ = _UDEFunc(
            dt=c.dt, tau=1.0, init_params=init_params,
            hidden_layers=c.hidden_layers,
            device=self._device, dtype=self._dtype)

        if c.training_mode == "full":
            self.training_loss_ = self._train_full(u, y, logger)
        else:
            dt = c.dt
            omega = np.gradient(y, dt)
            y_state = np.column_stack([y, omega])
            self.training_loss_ = train_sequence_batches(
                sde_func=self.sde_func_,
                simulate_fn=lambda u_seq, x0: self._simulate_trajectory(u_seq, x0),
                u=u.reshape(-1, 1), y=y_state,
                input_dim=1, state_dim=2,
                sequence_length=c.sequence_length, epochs=c.epochs,
                learning_rate=c.learning_rate,
                device=self._device, dtype=self._dtype,
                verbose=c.verbose, progress_desc="Training UDE",
                logger=logger, log_every=c.wandb_log_every,
                sequences_per_epoch=getattr(c, 'sequences_per_epoch', 24),
                early_stopping_patience=c.early_stopping_patience,
            )

    def _train_full(self, u, y, logger):
        import torch, torch.optim as optim
        from tqdm.auto import tqdm
        c = self.config
        dt = c.dt

        u_t = torch.tensor(u.reshape(-1, 1), dtype=self._dtype, device=self._device)
        y_t = torch.tensor(y, dtype=self._dtype, device=self._device)
        omega_t = torch.tensor(np.gradient(y, dt), dtype=self._dtype, device=self._device)
        x0 = torch.tensor([y_t[0].item(), (y_t[1] - y_t[0]).item() / dt],
                           dtype=self._dtype, device=self._device)

        params = list(self.sde_func_.parameters())
        optimizer = optim.Adam(params, lr=c.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=200, min_lr=1e-6)
        history, best_loss, best_state = [], float("inf"), None

        it = range(c.epochs)
        if c.verbose:
            it = tqdm(it, desc="Training UDE (full)", unit="epoch")

        for epoch in it:
            self.sde_func_.train(); optimizer.zero_grad()
            pred = self._simulate_trajectory(u_t, x0)
            loss_th = torch.mean((pred[:, 0] - y_t) ** 2)
            loss_om = torch.mean((pred[:, 1] - omega_t) ** 2)
            loss = loss_th + 0.2 * loss_om
            if torch.isnan(loss) or torch.isinf(loss):
                history.append(float("nan")); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, DEFAULT_GRAD_CLIP)
            optimizer.step()
            loss_val = loss.item(); history.append(loss_val); scheduler.step(loss_val)
            if loss_val < best_loss:
                best_loss = loss_val
                best_state = {id(p): p.data.clone() for p in params}
            if c.verbose and hasattr(it, "set_postfix"):
                it.set_postfix(loss=loss_val)
            if logger and c.wandb_log_every > 0 and (epoch + 1) % c.wandb_log_every == 0:
                logger.log_metrics({"train/loss": loss_val, "train/epoch": epoch + 1}, step=epoch + 1)

        if best_state:
            for p in params:
                p.data.copy_(best_state[id(p)])
        return history

    # ── predict ───────────────────────────────────────────────────────

    def predict_osa(self, u, y):
        import torch
        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        omega = np.gradient(y, self.config.dt)
        preds = []
        self.sde_func_.eval()
        with torch.no_grad():
            for t in range(len(y) - 1):
                x_t = torch.tensor([y[t], omega[t]], dtype=self._dtype, device=self._device)
                u_t = torch.tensor([[u[t]], [u[t]]], dtype=self._dtype, device=self._device)
                x_next = self._simulate_trajectory(u_t, x_t)
                preds.append(float(x_next[-1, 0].cpu().item()))
        return np.asarray(preds)

    def predict_free_run(self, u, y_initial):
        import torch
        u = np.asarray(u, dtype=float).flatten()
        y0 = np.asarray(y_initial, dtype=float).flatten()
        dt = self.config.dt
        omega0 = (y0[1] - y0[0]) / dt if len(y0) >= 2 else 0.0
        self.sde_func_.eval()
        with torch.no_grad():
            u_t = torch.tensor(u.reshape(-1, 1), dtype=self._dtype, device=self._device)
            x0 = torch.tensor([y0[0], omega0], dtype=self._dtype, device=self._device)
            pred = self._simulate_trajectory(u_t, x0)
        return pred[:, 0].cpu().numpy()

    # ── save / load hooks ─────────────────────────────────────────────

    def _collect_state(self) -> Dict[str, Any]:
        if self.sde_func_ is None:
            return {}
        return {"params": {id_str: p.data.cpu() for id_str, p in
                           zip(["raw_J", "raw_R", "raw_K", "delta"], [
                               self.sde_func_.raw_J, self.sde_func_.raw_R,
                               self.sde_func_.raw_K, self.sde_func_.delta])},
                "residual_net": self.sde_func_.residual_net.state_dict()}

    def _restore_state(self, state):
        import torch
        for name, val in state["params"].items():
            getattr(self.sde_func_, name).data.copy_(val)
        self.sde_func_.residual_net.load_state_dict(state["residual_net"])

    def _collect_extra_state(self):
        if self.sde_func_ is None:
            return {}
        return {"tau": self.sde_func_.tau}

    def _restore_extra_state(self, extra):
        self._tau_for_load = extra.get("tau", 1.0)

    def _build_for_load(self):
        import torch
        self._device = self._resolve_torch_device("cpu")
        self._dtype = torch.float32
        c = self.config
        init_params = {"J": 0.1, "R": 0.1, "K": 1.0, "delta": 0.0}
        self.sde_func_ = _UDEFunc(
            dt=c.dt, tau=getattr(self, '_tau_for_load', 1.0),
            init_params=init_params, hidden_layers=c.hidden_layers,
            device=self._device, dtype=self._dtype)

    def get_physics_params(self) -> Dict[str, float]:
        if self.sde_func_ is None:
            raise RuntimeError("Model not fitted")
        return self.sde_func_.decoded_parameter_dict()

    def __repr__(self):
        c = self.config
        return f"UDE(dt={c.dt}, hidden={c.hidden_layers}, epochs={c.epochs})"
