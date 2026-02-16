"""Neural SDE model for system identification.

Learns drift ``f(y, u; θ)`` and diffusion ``g(y, u; θ)`` functions,
integrated using ``torchsde``.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.config import NeuralSDEConfig
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
# SDE function
# ─────────────────────────────────────────────────────────────────────

def _build_mlp(in_dim, hidden, out_dim, activation="selu"):
    _act = {"relu": nn.ReLU, "selu": nn.SELU, "tanh": nn.Tanh}
    act_cls = _act.get(activation, nn.SELU)
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class _SDEFunc(nn.Module):
    """SDE function compatible with ``torchsde``."""

    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        hidden_layers: List[int],
        diffusion_hidden_layers: List[int],
    ) -> None:
        super().__init__()
        self.drift_net = _build_mlp(state_dim + input_dim, hidden_layers, state_dim)
        self.diff_net = _build_mlp(state_dim + input_dim, diffusion_hidden_layers, state_dim)
        self._u_func = None  # set before integration

    def set_u_func(self, u_func):
        self._u_func = u_func

    def f(self, t, y):
        u = self._u_func(t)
        if u.dim() == 0:
            u = u.unsqueeze(0)
        u = u.expand(y.shape[0], -1) if u.dim() == 1 and y.dim() == 2 else u
        return self.drift_net(torch.cat([y, u], dim=-1))

    def g(self, t, y):
        u = self._u_func(t)
        if u.dim() == 0:
            u = u.unsqueeze(0)
        u = u.expand(y.shape[0], -1) if u.dim() == 1 and y.dim() == 2 else u
        return self.diff_net(torch.cat([y, u], dim=-1))


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────

@register_model("neural_sde", NeuralSDEConfig)
class NeuralSDEModel(PickleStateMixin, BaseModel):
    """Neural SDE for system identification (requires ``torchsde``)."""

    def __init__(self, config: NeuralSDEConfig | None = None) -> None:
        super().__init__(config or NeuralSDEConfig())
        self.config: NeuralSDEConfig
        self.sde_func: _SDEFunc | None = None

    def _fit(self, u, y, *, val_data=None, logger=None) -> None:
        try:
            import torchsde
        except ImportError as exc:
            raise ImportError(
                "torchsde required: pip install torchsde"
            ) from exc

        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        y_norm = self._normalize_y(y)

        self.sde_func = _SDEFunc(
            cfg.state_dim, cfg.input_dim,
            cfg.hidden_layers, cfg.diffusion_hidden_layers,
        ).to(device)

        optimizer = make_optimizer(self.sde_func.parameters(), cfg.learning_rate)
        scheduler = make_scheduler(
            optimizer, cfg.scheduler_patience, cfg.scheduler_factor, cfg.scheduler_min_lr
        )
        stopper = EarlyStopper(cfg.early_stopping_patience)

        N = len(u_norm)
        dt = cfg.dt
        window = cfg.train_window_size
        seqs_per_epoch = cfg.sequences_per_epoch

        def step_fn(epoch: int) -> float:
            self.sde_func.train()
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
                self.sde_func.set_u_func(u_func)

                y0 = torch.tensor(
                    [[y_sub[0]]], dtype=torch.float32, device=device
                )
                y_target = torch.tensor(
                    y_sub, dtype=torch.float32, device=device
                )

                optimizer.zero_grad()
                y_pred = torchsde.sdeint(
                    self.sde_func, y0, t_sub, method="euler", dt=dt
                )
                y_pred_flat = y_pred.squeeze()[:n_sub]
                loss = nn.functional.mse_loss(y_pred_flat, y_target)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.sde_func.parameters(), cfg.grad_clip
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
            desc="NeuralSDE",
        )

    def _predict(self, u, *, y0=None) -> np.ndarray:
        try:
            import torchsde
        except ImportError as exc:
            raise ImportError("torchsde required") from exc

        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        N = len(u_norm)

        t_span = torch.linspace(0, (N - 1) * cfg.dt, N, device=device)
        u_func = make_u_func(u_norm, dt=cfg.dt, device=device)
        self.sde_func.set_u_func(u_func)

        if y0 is not None:
            y0_val = self._normalize_y(np.atleast_1d(y0))[0]
        else:
            y0_val = 0.0
        y0_t = torch.tensor(
            [[y0_val]], dtype=torch.float32, device=device
        )

        self.sde_func.eval()
        with torch.no_grad():
            y_pred = torchsde.sdeint(
                self.sde_func, y0_t, t_span, method="euler", dt=cfg.dt
            )

        y_pred_np = y_pred.squeeze().cpu().numpy()[:N]
        return self._denormalize_y(y_pred_np)
