"""Universal Differential Equation (UDE) model.

Combines a known linear ODE with a universal (neural-network) term.
``dy/dt = A·y + B·u + NN(y, u)``
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.config import UDEConfig
from src.models.base import BaseModel, PickleStateMixin
from src.models.continuous.interpolation import make_u_func
from src.models.registry import register_model
from src.models.training import (
    EarlyStopper,
    make_optimizer,
    make_scheduler,
    train_loop,
)


_ACTIVATIONS = {"relu": nn.ReLU, "selu": nn.SELU, "tanh": nn.Tanh}


class _UDEFunc(nn.Module):
    """``dy/dt = A·state + B·u + NN(state, u)``."""

    def __init__(
        self,
        state_dim: int = 1,
        input_dim: int = 1,
        hidden_layers: List[int] = None,
        activation: str = "selu",
    ) -> None:
        super().__init__()
        hidden_layers = hidden_layers or [64, 64]

        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.1)

        act_cls = _ACTIVATIONS.get(activation, nn.SELU)
        layers: list[nn.Module] = []
        prev = state_dim + input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, state_dim))
        self.nn = nn.Sequential(*layers)
        self._u_func = None

    def set_u_func(self, u_func):
        self._u_func = u_func

    def forward(self, t, y):
        u = self._u_func(t)
        if u.dim() == 0:
            u = u.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
            u = u.unsqueeze(0)

        linear_part = y @ self.A.T + u @ self.B.T
        nn_part = self.nn(torch.cat([y, u], dim=-1))
        return linear_part + nn_part


def _euler(func, y0, t_span):
    ys = [y0]
    y = y0
    for i in range(len(t_span) - 1):
        dt = t_span[i + 1] - t_span[i]
        y = y + dt * func(t_span[i], y)
        ys.append(y)
    return torch.stack(ys)


def _rk4(func, y0, t_span):
    ys = [y0]
    y = y0
    for i in range(len(t_span) - 1):
        dt = t_span[i + 1] - t_span[i]
        t = t_span[i]
        k1 = func(t, y)
        k2 = func(t + dt / 2, y + dt / 2 * k1)
        k3 = func(t + dt / 2, y + dt / 2 * k2)
        k4 = func(t + dt, y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys.append(y)
    return torch.stack(ys)


_SOLVERS = {"euler": _euler, "rk4": _rk4}


@register_model("ude", UDEConfig)
class UDEModel(PickleStateMixin, BaseModel):
    """Universal Differential Equation model."""

    def __init__(self, config: UDEConfig | None = None) -> None:
        super().__init__(config or UDEConfig())
        self.config: UDEConfig
        self.ude_func: _UDEFunc | None = None

    def _fit(self, u, y, *, val_data=None, logger=None) -> None:
        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        y_norm = self._normalize_y(y)

        self.ude_func = _UDEFunc(
            state_dim=1, input_dim=1,
            hidden_layers=cfg.hidden_layers,
            activation=cfg.activation,
        ).to(device)

        optimizer = make_optimizer(self.ude_func.parameters(), cfg.learning_rate)
        scheduler = make_scheduler(
            optimizer, cfg.scheduler_patience, cfg.scheduler_factor, cfg.scheduler_min_lr
        )
        stopper = EarlyStopper(cfg.early_stopping_patience)

        N = len(u_norm)
        dt = cfg.dt
        window = cfg.train_window_size
        solver_fn = _SOLVERS.get(cfg.solver, _euler)
        training_mode = cfg.training_mode

        def step_fn(epoch: int) -> float:
            self.ude_func.train()
            if training_mode == "full":
                t_span = torch.linspace(0, (N - 1) * dt, N, device=device)
                u_func = make_u_func(u_norm, dt=dt, device=device)
                self.ude_func.set_u_func(u_func)

                y0 = torch.tensor(
                    [[y_norm[0]]], dtype=torch.float32, device=device
                )
                y_target = torch.tensor(
                    y_norm, dtype=torch.float32, device=device
                )

                optimizer.zero_grad()
                ys = solver_fn(self.ude_func, y0, t_span)
                loss = nn.functional.mse_loss(ys.squeeze()[:N], y_target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ude_func.parameters(), cfg.grad_clip)
                optimizer.step()
                return loss.item()
            else:
                total = 0.0
                n_seqs = max(1, N // window)
                for _ in range(n_seqs):
                    start = np.random.randint(0, max(N - window, 1))
                    end = min(start + window, N)
                    u_sub = u_norm[start:end]
                    y_sub = y_norm[start:end]
                    n = len(u_sub)
                    t_sub = torch.linspace(0, (n - 1) * dt, n, device=device)

                    u_func = make_u_func(u_sub, dt=dt, device=device)
                    self.ude_func.set_u_func(u_func)

                    y0 = torch.tensor(
                        [[y_sub[0]]], dtype=torch.float32, device=device
                    )
                    y_target = torch.tensor(
                        y_sub, dtype=torch.float32, device=device
                    )

                    optimizer.zero_grad()
                    ys = solver_fn(self.ude_func, y0, t_sub)
                    loss = nn.functional.mse_loss(ys.squeeze()[:n], y_target)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.ude_func.parameters(), cfg.grad_clip
                    )
                    optimizer.step()
                    total += loss.item()
                return total / n_seqs

        train_loop(
            step_fn,
            epochs=cfg.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=stopper,
            logger=logger,
            verbose=cfg.verbose,
            desc="UDE",
        )

    def _predict(self, u, *, y0=None) -> np.ndarray:
        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        N = len(u_norm)

        t_span = torch.linspace(0, (N - 1) * cfg.dt, N, device=device)
        u_func = make_u_func(u_norm, dt=cfg.dt, device=device)
        self.ude_func.set_u_func(u_func)

        y0_val = self._normalize_y(np.atleast_1d(y0))[0] if y0 is not None else 0.0
        y0_t = torch.tensor(
            [[y0_val]], dtype=torch.float32, device=device
        )

        solver_fn = _SOLVERS.get(cfg.solver, _euler)
        self.ude_func.eval()
        with torch.no_grad():
            ys = solver_fn(self.ude_func, y0_t, t_span)

        return self._denormalize_y(ys.squeeze().cpu().numpy()[:N])
