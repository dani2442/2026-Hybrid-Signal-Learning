"""Hybrid nonlinear cam-follower model.

Combines known multi-body dynamics of a DC-motor-driven cam-follower
mechanism with a neural network that learns unmeasured nonlinear terms.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import HybridNonlinearCamConfig
from src.models.base import BaseModel, PickleStateMixin
from src.models.continuous.interpolation import make_u_func
from src.models.registry import register_model
from src.models.training import (
    EarlyStopper,
    inverse_softplus,
    make_optimizer,
    make_scheduler,
    train_loop,
)


class _CamNNCorrection(nn.Module):
    """Neural correction for unknown cam dynamics."""

    def __init__(self, in_dim: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SELU(),
            nn.Linear(hidden, hidden), nn.SELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("hybrid_nonlinear_cam", HybridNonlinearCamConfig)
class HybridNonlinearCamModel(PickleStateMixin, BaseModel):
    """Hybrid: cam-follower physics + neural correction."""

    def __init__(self, config: HybridNonlinearCamConfig | None = None) -> None:
        super().__init__(config or HybridNonlinearCamConfig())
        self.config: HybridNonlinearCamConfig
        self.nn_correction: _CamNNCorrection | None = None
        self._trainable: nn.ParameterDict | None = None

    def _init_params(self, device):
        cfg = self.config
        params = {}
        defaults = {
            "J": cfg.J, "k": cfg.k, "delta": cfg.delta,
            "k_t": cfg.k_t, "k_b": cfg.k_b,
        }
        for name in cfg.trainable_params:
            val = defaults.get(name, 0.01)
            if val > 0:
                params[name] = nn.Parameter(
                    torch.tensor(inverse_softplus(val), device=device)
                )
            else:
                params[name] = nn.Parameter(torch.tensor(0.0, device=device))

        self._trainable = nn.ParameterDict(params)
        self.nn_correction = _CamNNCorrection(in_dim=4).to(device)

    def _get_param(self, name: str) -> torch.Tensor:
        if self._trainable is not None and name in self._trainable:
            return F.softplus(self._trainable[name])
        return torch.tensor(
            getattr(self.config, name), dtype=torch.float32, device=self.device
        )

    def _ode_rhs(self, t, state, u_func):
        cfg = self.config
        J = self._get_param("J")
        k = self._get_param("k")
        delta = self._get_param("delta")
        k_t = self._get_param("k_t")
        k_b = self._get_param("k_b")

        R_M = torch.tensor(cfg.R_M, device=self.device)
        L_M = torch.tensor(cfg.L_M, device=self.device)

        u_val = u_func(t)
        if u_val.dim() == 0:
            u_val = u_val.unsqueeze(0)

        # State: [theta, theta_dot, current]
        theta = state[..., 0:1]
        theta_dot = state[..., 1:2]
        current = state[..., 2:3] if state.shape[-1] > 2 else torch.zeros_like(theta)

        # Motor dynamics
        di_dt = (u_val - R_M * current - k_b * theta_dot) / (L_M + 1e-8)
        torque_motor = k_t * current

        # Mechanical dynamics
        spring_torque = k * theta + delta * theta ** 3

        # Neural correction for unmeasured terms
        nn_in = torch.cat([theta, theta_dot, current, u_val], dim=-1)
        nn_corr = self.nn_correction(nn_in)

        theta_ddot = (torque_motor - spring_torque + nn_corr) / (J + 1e-8)

        if state.shape[-1] > 2:
            return torch.cat([theta_dot, theta_ddot, di_dt], dim=-1)
        return torch.cat([theta_dot, theta_ddot], dim=-1)

    def _integrate(self, y0_state, u_func, t_span, substeps=1):
        ys = [y0_state]
        state = y0_state
        for i in range(len(t_span) - 1):
            dt = (t_span[i + 1] - t_span[i]) / substeps
            t = t_span[i]
            for _ in range(substeps):
                dydt = self._ode_rhs(t, state, u_func)
                state = state + dt * dydt
                t = t + dt
            ys.append(state)
        return torch.stack(ys)

    def _fit(self, u, y, *, val_data=None, logger=None) -> None:
        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        y_norm = self._normalize_y(y)

        self._init_params(device)

        all_params = list(self.nn_correction.parameters())
        if self._trainable is not None:
            all_params += list(self._trainable.parameters())

        optimizer = make_optimizer(all_params, cfg.learning_rate)
        scheduler = make_scheduler(
            optimizer, cfg.scheduler_patience, cfg.scheduler_factor, cfg.scheduler_min_lr
        )
        stopper = EarlyStopper(cfg.early_stopping_patience)

        N = len(u_norm)
        dt = cfg.dt
        t_span = torch.linspace(0, (N - 1) * dt, N, device=device)
        u_func = make_u_func(u_norm, dt=dt, device=device)
        y_target = torch.tensor(y_norm, dtype=torch.float32, device=device)

        n_state = 2
        y0_state = torch.zeros(n_state, dtype=torch.float32, device=device)
        y0_state[0] = y_norm[0]

        def step_fn(epoch: int) -> float:
            self.nn_correction.train()
            optimizer.zero_grad()
            ys = self._integrate(
                y0_state, u_func, t_span, cfg.integration_substeps
            )
            loss = nn.functional.mse_loss(ys[:, 0], y_target)
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
            optimizer.step()
            return loss.item()

        # Build validation step (full val trajectory, no grad)
        val_sfn = None
        if val_data is not None:
            u_val_norm = self._normalize_u(val_data[0])
            y_val_norm = self._normalize_y(val_data[1])
            n_val = len(u_val_norm)
            t_val = torch.linspace(0, (n_val - 1) * dt, n_val, device=device)
            u_func_val = make_u_func(u_val_norm, dt=dt, device=device)
            n_state_v = n_state
            y0_val_state = torch.zeros(n_state_v, dtype=torch.float32, device=device)
            y0_val_state[0] = y_val_norm[0]
            y_target_val = torch.tensor(y_val_norm, dtype=torch.float32, device=device)

            def val_sfn() -> float:
                self.nn_correction.eval()
                with torch.no_grad():
                    ys_v = self._integrate(
                        y0_val_state, u_func_val, t_val, cfg.integration_substeps
                    )
                    return nn.functional.mse_loss(ys_v[:, 0], y_target_val).item()

        train_loop(
            step_fn,
            epochs=cfg.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=stopper,
            logger=logger,
            verbose=cfg.verbose,
            desc="HybridNonlinearCam",
            val_step_fn=val_sfn,
            model_params=all_params,
        )

    def _predict(self, u, *, y0=None) -> np.ndarray:
        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        N = len(u_norm)

        t_span = torch.linspace(0, (N - 1) * cfg.dt, N, device=device)
        u_func = make_u_func(u_norm, dt=cfg.dt, device=device)

        n_state = 2
        y0_state = torch.zeros(n_state, dtype=torch.float32, device=device)
        if y0 is not None:
            y0_state[0] = self._normalize_y(np.atleast_1d(y0))[0]

        self.nn_correction.eval()
        with torch.no_grad():
            ys = self._integrate(
                y0_state, u_func, t_span, cfg.integration_substeps
            )

        return self._denormalize_y(ys[:, 0].cpu().numpy()[:N])
