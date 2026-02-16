"""Blackbox 2-D SDE models: Vanilla, Structured, Adaptive.

Same three ODE-function variants but with added diagonal diffusion.
Integrates with ``torchsde``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.config import BlackboxSDE2DConfig
from src.models.base import BaseModel, PickleStateMixin
from src.models.blackbox.networks import (
    AdaptiveODEFunc,
    DiagonalDiffusion,
    StructuredODEFunc,
    VanillaODEFunc,
)
from src.models.continuous.interpolation import make_u_func
from src.models.registry import register_model
from src.models.training import (
    EarlyStopper,
    make_optimizer,
    make_scheduler,
    train_loop,
)


# ─────────────────────────────────────────────────────────────────────
# torchsde wrapper
# ─────────────────────────────────────────────────────────────────────

class _SDEWrapper(nn.Module):
    """Wraps (drift_func, diff_func) into a torchsde-compatible module."""

    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, drift_func, diff_func, u_func=None):
        super().__init__()
        self.drift_func = drift_func
        self.diff_func = diff_func
        self._u_func = u_func

    def set_u_func(self, u_func):
        self._u_func = u_func

    def f(self, t, y):
        u = self._u_func(t)
        if u.dim() == 0:
            u = u.unsqueeze(0)
        u = u.expand(y.shape[0], -1) if u.dim() == 1 and y.dim() == 2 else u
        return self.drift_func(y, u)

    def g(self, t, y):
        u = self._u_func(t)
        if u.dim() == 0:
            u = u.unsqueeze(0)
        u = u.expand(y.shape[0], -1) if u.dim() == 1 and y.dim() == 2 else u
        return self.diff_func(y, u)


class _BlackboxSDE2DBase(PickleStateMixin, BaseModel):
    """Shared training logic for blackbox 2-D SDE variants."""

    def __init__(self, config, drift_cls) -> None:
        super().__init__(config)
        self.config: BlackboxSDE2DConfig
        self._drift_cls = drift_cls
        self.sde_wrapper: _SDEWrapper | None = None

    def _fit(self, u, y, *, val_data=None, logger=None) -> None:
        try:
            import torchsde
        except ImportError as exc:
            raise ImportError("torchsde required") from exc

        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        y_norm = self._normalize_y(y)

        drift = self._drift_cls(cfg.state_dim, cfg.input_dim, cfg.hidden_dim).to(device)
        diff = DiagonalDiffusion(
            cfg.state_dim, cfg.input_dim, cfg.diffusion_hidden_dim
        ).to(device)
        self.sde_wrapper = _SDEWrapper(drift, diff)

        all_params = list(drift.parameters()) + list(diff.parameters())
        optimizer = make_optimizer(all_params, cfg.learning_rate)
        scheduler = make_scheduler(
            optimizer, cfg.scheduler_patience, cfg.scheduler_factor, cfg.scheduler_min_lr
        )
        stopper = EarlyStopper(cfg.early_stopping_patience)

        N = len(u_norm)
        k = cfg.k_steps
        dt = cfg.dt

        n_win = N - k
        if n_win <= 0:
            raise ValueError(f"Signal length ({N}) too short for k_steps={k}")

        u_arr = np.asarray(u_norm, dtype=np.float32).ravel()
        y_arr = np.asarray(y_norm, dtype=np.float32).ravel()

        y0_all = torch.tensor(
            np.column_stack([y_arr[:n_win], np.zeros(n_win)]),
            dtype=torch.float32, device=device,
        )
        y_target_all = torch.tensor(
            np.lib.stride_tricks.sliding_window_view(y_arr, k + 1)[:n_win],
            dtype=torch.float32, device=device,
        )

        dataset = torch.utils.data.TensorDataset(y0_all, y_target_all)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True
        )

        def step_fn(epoch: int) -> float:
            drift.train()
            diff.train()
            total = 0.0
            count = 0
            for y0_b, yt_b in loader:
                B = y0_b.shape[0]
                # Pick random start index and build u_func for this window
                start = np.random.randint(0, max(n_win, 1))
                u_sub = u_arr[start: start + k]
                u_func = make_u_func(u_sub, dt=dt, device=device)
                self.sde_wrapper.set_u_func(u_func)

                t_span = torch.linspace(0, k * dt, k + 1, device=device)

                optimizer.zero_grad()
                y_pred = torchsde.sdeint(
                    self.sde_wrapper, y0_b, t_span, method="euler", dt=dt
                )
                # y_pred: [k+1, B, state_dim]
                loss = nn.functional.mse_loss(
                    y_pred[:, :, 0].T, yt_b
                )
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
                optimizer.step()
                total += loss.item()
                count += 1
            return total / max(count, 1)

        train_loop(
            step_fn,
            epochs=cfg.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=stopper,
            logger=logger,
            verbose=cfg.verbose,
            desc=f"{self.name}",
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

        u_func = make_u_func(u_norm, dt=cfg.dt, device=device)
        self.sde_wrapper.set_u_func(u_func)

        y0_val = self._normalize_y(np.atleast_1d(y0))[0] if y0 is not None else 0.0
        y0_t = torch.tensor(
            [[y0_val, 0.0]], dtype=torch.float32, device=device
        )
        t_span = torch.linspace(0, (N - 1) * cfg.dt, N, device=device)

        self.sde_wrapper.drift_func.eval()
        self.sde_wrapper.diff_func.eval()
        with torch.no_grad():
            y_pred = torchsde.sdeint(
                self.sde_wrapper, y0_t, t_span, method="euler", dt=cfg.dt
            )

        return self._denormalize_y(y_pred[:, 0, 0].cpu().numpy()[:N])


@register_model("vanilla_nsde_2d", BlackboxSDE2DConfig)
class VanillaNSDE2D(_BlackboxSDE2DBase):
    def __init__(self, config=None):
        super().__init__(config or BlackboxSDE2DConfig(), VanillaODEFunc)


@register_model("structured_nsde", BlackboxSDE2DConfig)
class StructuredNSDE(_BlackboxSDE2DBase):
    def __init__(self, config=None):
        super().__init__(config or BlackboxSDE2DConfig(), StructuredODEFunc)


@register_model("adaptive_nsde", BlackboxSDE2DConfig)
class AdaptiveNSDE(_BlackboxSDE2DBase):
    def __init__(self, config=None):
        super().__init__(config or BlackboxSDE2DConfig(), AdaptiveODEFunc)
