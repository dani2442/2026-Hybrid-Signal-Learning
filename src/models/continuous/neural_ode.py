"""Neural ODE model for system identification.

Learns a drift function ``dy/dt = f(y, u; θ)`` and integrates it with
simple fixed-step solvers or ``torchdiffeq`` when available.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.config import NeuralODEConfig
from src.models.base import BaseModel, PickleStateMixin
from src.models.continuous.interpolation import make_u_func
from src.models.registry import register_model
from src.models.training import (
    EarlyStopper,
    make_optimizer,
    make_scheduler,
    train_loop,
)


# ─────────────────────────────────────────────────────────────────────
# ODE function (drift)
# ─────────────────────────────────────────────────────────────────────

_ACTIVATIONS = {"relu": nn.ReLU, "selu": nn.SELU, "tanh": nn.Tanh}


class _DriftNet(nn.Module):
    """``dy/dt = f(y, u; θ)``."""

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        hidden_layers: List[int],
        activation: str = "selu",
    ) -> None:
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation, nn.SELU)
        layers: list[nn.Module] = []
        prev = state_dim + input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([y, u], dim=-1))


# ─────────────────────────────────────────────────────────────────────
# Simple ODE integrators
# ─────────────────────────────────────────────────────────────────────

def _euler_integrate(drift_net, y0, u_func, t_span, device):
    """Fixed-step Euler integration."""
    ys = [y0]
    y = y0
    for i in range(len(t_span) - 1):
        dt = t_span[i + 1] - t_span[i]
        t_i = t_span[i]
        u_i = u_func(t_i).unsqueeze(0) if u_func(t_i).dim() == 0 else u_func(t_i).unsqueeze(-1) if u_func(t_i).dim() == 0 else u_func(t_i)
        if u_i.dim() == 0:
            u_i = u_i.unsqueeze(0)
        if u_i.dim() == 1 and y.dim() == 1:
            dydt = drift_net(y.unsqueeze(0), u_i.unsqueeze(0)).squeeze(0)
        else:
            dydt = drift_net(y, u_i)
        y = y + dt * dydt
        ys.append(y)
    return torch.stack(ys)


def _rk4_integrate(drift_net, y0, u_func, t_span, device):
    """Fixed-step RK4 integration."""
    ys = [y0]
    y = y0
    for i in range(len(t_span) - 1):
        dt = t_span[i + 1] - t_span[i]
        t_i = t_span[i]
        t_mid = t_i + dt / 2
        t_next = t_span[i + 1]

        def _eval(yy, tt):
            uu = u_func(tt)
            if uu.dim() == 0:
                uu = uu.unsqueeze(0)
            if yy.dim() == 1 and uu.dim() == 1:
                return drift_net(yy.unsqueeze(0), uu.unsqueeze(0)).squeeze(0)
            return drift_net(yy, uu)

        k1 = _eval(y, t_i)
        k2 = _eval(y + dt / 2 * k1, t_mid)
        k3 = _eval(y + dt / 2 * k2, t_mid)
        k4 = _eval(y + dt * k3, t_next)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys.append(y)
    return torch.stack(ys)


def _integrate(drift_net, y0, u_func, t_span, solver, device):
    solvers = {"euler": _euler_integrate, "rk4": _rk4_integrate}
    fn = solvers.get(solver)
    if fn is None:
        raise ValueError(f"Unknown solver '{solver}'. Choose from {list(solvers)}")
    return fn(drift_net, y0, u_func, t_span, device)


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────

@register_model("neural_ode", NeuralODEConfig)
class NeuralODEModel(PickleStateMixin, BaseModel):
    """Neural ODE for system identification."""

    def __init__(self, config: NeuralODEConfig | None = None) -> None:
        super().__init__(config or NeuralODEConfig())
        self.config: NeuralODEConfig
        self.drift_net: _DriftNet | None = None

    def _fit(self, u, y, *, val_data=None, logger=None) -> None:
        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        y_norm = self._normalize_y(y)

        self.drift_net = _DriftNet(
            cfg.state_dim, cfg.input_dim, cfg.hidden_layers, cfg.activation
        ).to(device)

        optimizer = make_optimizer(self.drift_net.parameters(), cfg.learning_rate)
        scheduler = make_scheduler(
            optimizer, cfg.scheduler_patience, cfg.scheduler_factor, cfg.scheduler_min_lr
        )
        stopper = EarlyStopper(cfg.early_stopping_patience)

        N = len(u_norm)
        dt = cfg.dt
        window = cfg.train_window_size
        seqs_per_epoch = cfg.sequences_per_epoch

        def step_fn(epoch: int) -> float:
            self.drift_net.train()
            total_loss = 0.0
            for _ in range(seqs_per_epoch):
                max_start = max(N - window, 1)
                start = np.random.randint(0, max_start)
                end = min(start + window, N)

                u_sub = u_norm[start:end]
                y_sub = y_norm[start:end]
                n_sub = len(u_sub)
                t_sub = torch.linspace(0, (n_sub - 1) * dt, n_sub, device=device)

                u_func = make_u_func(u_sub, dt=dt, device=device)
                y0 = torch.tensor(
                    [y_sub[0]], dtype=torch.float32, device=device
                ).unsqueeze(0)
                y_target = torch.tensor(
                    y_sub, dtype=torch.float32, device=device
                )

                optimizer.zero_grad()
                y_pred = _integrate(
                    self.drift_net, y0, u_func, t_sub, cfg.solver, device
                )
                y_pred_flat = y_pred.squeeze()[:n_sub]
                loss = nn.functional.mse_loss(y_pred_flat, y_target)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.drift_net.parameters(), cfg.grad_clip
                )
                optimizer.step()
                total_loss += loss.item()
            return total_loss / seqs_per_epoch

        train_loop(
            step_fn,
            epochs=cfg.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=stopper,
            logger=logger,
            verbose=cfg.verbose,
            desc="NeuralODE",
        )

    def _predict(self, u, *, y0=None) -> np.ndarray:
        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        N = len(u_norm)

        t_span = torch.linspace(
            0, (N - 1) * cfg.dt, N, device=device
        )
        u_func = make_u_func(u_norm, dt=cfg.dt, device=device)

        if y0 is not None:
            y0_val = self._normalize_y(np.atleast_1d(y0))[0]
        else:
            y0_val = 0.0
        y0_t = torch.tensor(
            [y0_val], dtype=torch.float32, device=device
        ).unsqueeze(0)

        self.drift_net.eval()
        with torch.no_grad():
            y_pred = _integrate(
                self.drift_net, y0_t, u_func, t_span, cfg.solver, device
            )

        y_pred_np = y_pred.squeeze().cpu().numpy()[:N]
        return self._denormalize_y(y_pred_np)
