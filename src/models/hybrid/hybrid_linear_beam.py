"""Hybrid linear beam model.

Combines known 2nd-order linear physics with a neural-network
correction.  Physical parameters are estimated initially via ridge
regression and then fine-tuned jointly with the network.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import HybridLinearBeamConfig
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


def _logit(x: float) -> float:
    x = max(min(x, 1 - 1e-6), 1e-6)
    return math.log(x / (1 - x))


class _NNCorrection(nn.Module):
    """Small MLP producing a scalar correction to the acceleration."""

    def __init__(self, in_dim: int = 3, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SELU(),
            nn.Linear(hidden, hidden), nn.SELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("hybrid_linear_beam", HybridLinearBeamConfig)
class HybridLinearBeamModel(PickleStateMixin, BaseModel):
    """Hybrid: linear beam physics + neural correction."""

    def __init__(self, config: HybridLinearBeamConfig | None = None) -> None:
        super().__init__(config or HybridLinearBeamConfig())
        self.config: HybridLinearBeamConfig
        self._raw_wn: nn.Parameter | None = None
        self._raw_zeta: nn.Parameter | None = None
        self._raw_gain: nn.Parameter | None = None
        self._raw_delta: nn.Parameter | None = None
        self.nn_correction: _NNCorrection | None = None

    def _init_params(self, u, y, dt, device):
        """Estimate initial physics parameters via least-squares."""
        # Simple heuristic: estimate wn from FFT dominant frequency
        Y = np.fft.fft(y - y.mean())
        freqs = np.fft.fftfreq(len(y), d=dt)
        pos = freqs > 0
        if np.any(pos):
            peak_idx = np.argmax(np.abs(Y[pos]))
            wn_est = max(2 * np.pi * freqs[pos][peak_idx], 0.5)
        else:
            wn_est = 1.0
        zeta_est = 0.1
        gain_est = max(abs(y.std() / (u.std() + 1e-8)), 0.01)

        self._raw_wn = nn.Parameter(
            torch.tensor(inverse_softplus(wn_est), device=device)
        )
        self._raw_zeta = nn.Parameter(
            torch.tensor(_logit(zeta_est), device=device)
        )
        self._raw_gain = nn.Parameter(
            torch.tensor(inverse_softplus(gain_est), device=device)
        )
        if self.config.estimate_delta:
            self._raw_delta = nn.Parameter(torch.tensor(0.0, device=device))

        self.nn_correction = _NNCorrection(in_dim=3).to(device)

    def _ode_rhs(self, t, state, u_func):
        wn = F.softplus(self._raw_wn)
        zeta = torch.sigmoid(self._raw_zeta)
        gain = F.softplus(self._raw_gain)

        u_val = u_func(t)
        if u_val.dim() == 0:
            u_val = u_val.unsqueeze(0)

        y_val = state[..., 0:1]
        yd = state[..., 1:2]

        ydd_phys = -2 * zeta * wn * yd - wn ** 2 * y_val + gain * u_val

        if self.config.estimate_delta and self._raw_delta is not None:
            delta = F.softplus(self._raw_delta)
            ydd_phys = ydd_phys - delta * y_val ** 3

        nn_in = torch.cat([y_val, yd, u_val], dim=-1)
        ydd_nn = self.nn_correction(nn_in)
        ydd = ydd_phys + ydd_nn

        return torch.cat([yd, ydd], dim=-1)

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

        self._init_params(u, y, cfg.dt, device)

        all_params = list(self.nn_correction.parameters()) + [
            self._raw_wn, self._raw_zeta, self._raw_gain,
        ]
        if self._raw_delta is not None:
            all_params.append(self._raw_delta)

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

        y0_state = torch.tensor(
            [y_norm[0], 0.0], dtype=torch.float32, device=device
        )

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

        train_loop(
            step_fn,
            epochs=cfg.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=stopper,
            logger=logger,
            verbose=cfg.verbose,
            desc="HybridLinearBeam",
        )

    def _predict(self, u, *, y0=None) -> np.ndarray:
        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        N = len(u_norm)

        t_span = torch.linspace(0, (N - 1) * cfg.dt, N, device=device)
        u_func = make_u_func(u_norm, dt=cfg.dt, device=device)

        y0_val = self._normalize_y(np.atleast_1d(y0))[0] if y0 is not None else 0.0
        y0_state = torch.tensor(
            [y0_val, 0.0], dtype=torch.float32, device=device
        )

        self.nn_correction.eval()
        with torch.no_grad():
            ys = self._integrate(
                y0_state, u_func, t_span, cfg.integration_substeps
            )

        return self._denormalize_y(ys[:, 0].cpu().numpy()[:N])
