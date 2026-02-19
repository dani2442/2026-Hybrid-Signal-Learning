"""Physics-informed ODE models.

* :class:`LinearPhysicsModel` — second-order linear spring-damper.
* :class:`StribeckPhysicsModel` — second-order system with Stribeck
  friction.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import LinearPhysicsConfig, StribeckPhysicsConfig
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
# Simple integrators
# ─────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────
# Linear 2nd-order: m*y'' + c*y' + k*y = u
# ─────────────────────────────────────────────────────────────────────

class _LinearODE(nn.Module):
    """State: ``[y, ẏ]``.  Learns ``(wn, zeta, gain)``."""

    def __init__(self) -> None:
        super().__init__()
        self.log_wn = nn.Parameter(torch.tensor(1.0))
        self.log_zeta = nn.Parameter(torch.tensor(0.0))
        self.log_gain = nn.Parameter(torch.tensor(0.0))
        self._u_func = None

    def set_u_func(self, u_func):
        self._u_func = u_func

    def forward(self, t, state):
        wn = F.softplus(self.log_wn)
        zeta = torch.sigmoid(self.log_zeta)
        gain = F.softplus(self.log_gain)

        u_val = self._u_func(t)
        if u_val.dim() == 0:
            u_val = u_val.unsqueeze(0)

        y = state[..., 0:1]
        yd = state[..., 1:2]

        ydd = -2 * zeta * wn * yd - wn ** 2 * y + gain * u_val
        return torch.cat([yd, ydd], dim=-1)


@register_model("linear_physics", LinearPhysicsConfig)
class LinearPhysicsModel(PickleStateMixin, BaseModel):
    """Linear second-order physical model (spring-damper)."""

    def __init__(self, config: LinearPhysicsConfig | None = None) -> None:
        super().__init__(config or LinearPhysicsConfig())
        self.config: LinearPhysicsConfig
        self.ode_func: _LinearODE | None = None

    def _fit(self, u, y, *, val_data=None, logger=None) -> None:
        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        y_norm = self._normalize_y(y)

        self.ode_func = _LinearODE().to(device)
        optimizer = make_optimizer(self.ode_func.parameters(), cfg.learning_rate)
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
            self.ode_func.train()
            if training_mode == "full":
                t_span = torch.linspace(0, (N - 1) * dt, N, device=device)
                u_func = make_u_func(u_norm, dt=dt, device=device)
                self.ode_func.set_u_func(u_func)

                y0_state = torch.tensor(
                    [y_norm[0], 0.0], dtype=torch.float32, device=device
                )
                y_target = torch.tensor(y_norm, dtype=torch.float32, device=device)

                optimizer.zero_grad()
                ys = solver_fn(self.ode_func, y0_state, t_span)
                loss = nn.functional.mse_loss(ys[:, 0], y_target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ode_func.parameters(), cfg.grad_clip)
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
                    self.ode_func.set_u_func(u_func)

                    y0_state = torch.tensor(
                        [y_sub[0], 0.0], dtype=torch.float32, device=device
                    )
                    y_target = torch.tensor(y_sub, dtype=torch.float32, device=device)

                    optimizer.zero_grad()
                    ys = solver_fn(self.ode_func, y0_state, t_sub)
                    loss = nn.functional.mse_loss(ys[:, 0], y_target)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ode_func.parameters(), cfg.grad_clip)
                    optimizer.step()
                    total += loss.item()
                return total / n_seqs

        # Build validation step (full val trajectory, no grad)
        val_sfn = None
        if val_data is not None:
            u_val_norm = self._normalize_u(val_data[0])
            y_val_norm = self._normalize_y(val_data[1])
            n_val = len(u_val_norm)
            t_val = torch.linspace(0, (n_val - 1) * dt, n_val, device=device)
            u_func_val = make_u_func(u_val_norm, dt=dt, device=device)
            y0_val_t = torch.tensor([y_val_norm[0], 0.0], dtype=torch.float32, device=device)
            y_target_val = torch.tensor(y_val_norm, dtype=torch.float32, device=device)

            def val_sfn() -> float:
                self.ode_func.set_u_func(u_func_val)
                self.ode_func.eval()
                with torch.no_grad():
                    ys_v = solver_fn(self.ode_func, y0_val_t, t_val)
                    return nn.functional.mse_loss(ys_v[:, 0], y_target_val).item()

        train_loop(
            step_fn,
            epochs=cfg.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=stopper,
            logger=logger,
            verbose=cfg.verbose,
            desc="LinearPhysics",
            val_step_fn=val_sfn,
            model_params=list(self.ode_func.parameters()),
        )

    def _predict(self, u, *, y0=None) -> np.ndarray:
        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        N = len(u_norm)

        t_span = torch.linspace(0, (N - 1) * cfg.dt, N, device=device)
        u_func = make_u_func(u_norm, dt=cfg.dt, device=device)
        self.ode_func.set_u_func(u_func)

        y0_val = self._normalize_y(np.atleast_1d(y0))[0] if y0 is not None else 0.0
        y0_state = torch.tensor(
            [y0_val, 0.0], dtype=torch.float32, device=device
        )

        solver_fn = _SOLVERS.get(cfg.solver, _euler)
        self.ode_func.eval()
        with torch.no_grad():
            ys = solver_fn(self.ode_func, y0_state, t_span)

        return self._denormalize_y(ys[:, 0].cpu().numpy()[:N])


# ─────────────────────────────────────────────────────────────────────
# Stribeck friction model
# ─────────────────────────────────────────────────────────────────────

class _StribeckODE(nn.Module):
    """Second-order system with Stribeck friction."""

    def __init__(self) -> None:
        super().__init__()
        self.log_wn = nn.Parameter(torch.tensor(1.0))
        self.log_zeta = nn.Parameter(torch.tensor(0.0))
        self.log_gain = nn.Parameter(torch.tensor(0.0))
        self.log_fc = nn.Parameter(torch.tensor(-1.0))  # Coulomb friction
        self.log_fs = nn.Parameter(torch.tensor(-0.5))  # Static friction
        self.log_vs = nn.Parameter(torch.tensor(-1.0))  # Stribeck velocity
        self._u_func = None

    def set_u_func(self, u_func):
        self._u_func = u_func

    def forward(self, t, state):
        wn = F.softplus(self.log_wn)
        zeta = torch.sigmoid(self.log_zeta)
        gain = F.softplus(self.log_gain)
        fc = F.softplus(self.log_fc)
        fs = F.softplus(self.log_fs)
        vs = F.softplus(self.log_vs) + 1e-6

        u_val = self._u_func(t)
        if u_val.dim() == 0:
            u_val = u_val.unsqueeze(0)

        y = state[..., 0:1]
        yd = state[..., 1:2]

        # Stribeck friction: F = (Fc + (Fs - Fc) * exp(-(v/vs)^2)) * sign(v)
        friction = (fc + (fs - fc) * torch.exp(-(yd / vs) ** 2)) * torch.sign(yd)

        ydd = -2 * zeta * wn * yd - wn ** 2 * y - friction + gain * u_val
        return torch.cat([yd, ydd], dim=-1)


@register_model("stribeck_physics", StribeckPhysicsConfig)
class StribeckPhysicsModel(PickleStateMixin, BaseModel):
    """Second-order system with Stribeck friction."""

    def __init__(self, config: StribeckPhysicsConfig | None = None) -> None:
        super().__init__(config or StribeckPhysicsConfig())
        self.config: StribeckPhysicsConfig
        self.ode_func: _StribeckODE | None = None

    def _fit(self, u, y, *, val_data=None, logger=None) -> None:
        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        y_norm = self._normalize_y(y)

        self.ode_func = _StribeckODE().to(device)
        optimizer = make_optimizer(self.ode_func.parameters(), cfg.learning_rate)
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
            self.ode_func.train()
            if training_mode == "full":
                t_span = torch.linspace(0, (N - 1) * dt, N, device=device)
                u_func = make_u_func(u_norm, dt=dt, device=device)
                self.ode_func.set_u_func(u_func)

                y0_state = torch.tensor(
                    [y_norm[0], 0.0], dtype=torch.float32, device=device
                )
                y_target = torch.tensor(y_norm, dtype=torch.float32, device=device)

                optimizer.zero_grad()
                ys = solver_fn(self.ode_func, y0_state, t_span)
                loss = nn.functional.mse_loss(ys[:, 0], y_target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ode_func.parameters(), cfg.grad_clip)
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
                    self.ode_func.set_u_func(u_func)

                    y0_state = torch.tensor(
                        [y_sub[0], 0.0], dtype=torch.float32, device=device
                    )
                    y_target = torch.tensor(y_sub, dtype=torch.float32, device=device)

                    optimizer.zero_grad()
                    ys = solver_fn(self.ode_func, y0_state, t_sub)
                    loss = nn.functional.mse_loss(ys[:, 0], y_target)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ode_func.parameters(), cfg.grad_clip)
                    optimizer.step()
                    total += loss.item()
                return total / n_seqs

        # Build validation step (full val trajectory, no grad)
        val_sfn = None
        if val_data is not None:
            u_val_norm = self._normalize_u(val_data[0])
            y_val_norm = self._normalize_y(val_data[1])
            n_val = len(u_val_norm)
            t_val = torch.linspace(0, (n_val - 1) * dt, n_val, device=device)
            u_func_val = make_u_func(u_val_norm, dt=dt, device=device)
            y0_val_t = torch.tensor([y_val_norm[0], 0.0], dtype=torch.float32, device=device)
            y_target_val = torch.tensor(y_val_norm, dtype=torch.float32, device=device)

            def val_sfn() -> float:
                self.ode_func.set_u_func(u_func_val)
                self.ode_func.eval()
                with torch.no_grad():
                    ys_v = solver_fn(self.ode_func, y0_val_t, t_val)
                    return nn.functional.mse_loss(ys_v[:, 0], y_target_val).item()

        train_loop(
            step_fn,
            epochs=cfg.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=stopper,
            logger=logger,
            verbose=cfg.verbose,
            desc="StribeckPhysics",
            val_step_fn=val_sfn,
            model_params=list(self.ode_func.parameters()),
        )

    def _predict(self, u, *, y0=None) -> np.ndarray:
        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        N = len(u_norm)

        t_span = torch.linspace(0, (N - 1) * cfg.dt, N, device=device)
        u_func = make_u_func(u_norm, dt=cfg.dt, device=device)
        self.ode_func.set_u_func(u_func)

        y0_val = self._normalize_y(np.atleast_1d(y0))[0] if y0 is not None else 0.0
        y0_state = torch.tensor(
            [y0_val, 0.0], dtype=torch.float32, device=device
        )

        solver_fn = _SOLVERS.get(cfg.solver, _euler)
        self.ode_func.eval()
        with torch.no_grad():
            ys = solver_fn(self.ode_func, y0_state, t_span)

        return self._denormalize_y(ys[:, 0].cpu().numpy()[:N])
