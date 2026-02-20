"""Blackbox 2-D ODE models: Vanilla, Structured, Adaptive.

All three variants use k-step shooting for training and full-
trajectory Euler integration for prediction.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.config import BlackboxODE2DConfig
from src.data.torch_datasets import WindowedTrainDataset
from src.models.base import BaseModel, PickleStateMixin
from src.models.blackbox.networks import (
    AdaptiveODEFunc,
    StructuredODEFunc,
    VanillaODEFunc,
)
from src.models.registry import register_model
from src.models.training import (
    EarlyStopper,
    make_optimizer,
    make_scheduler,
    train_loop,
)


# ─────────────────────────────────────────────────────────────────────
# Shared training / prediction
# ─────────────────────────────────────────────────────────────────────

def _integrate_euler(ode_func, y0, u_data, t_data, dt):
    """Euler-integrate ``dy/dt = f(y, u)`` returning ``[T, batch, state]``."""
    ys = [y0]
    y = y0
    for i in range(len(t_data) - 1):
        # Interpolate u at current time
        u_i = u_data[:, i, :] if u_data.dim() == 3 else u_data[i:i + 1]
        dydt = ode_func(y, u_i)
        y = y + dt * dydt
        ys.append(y)
    return torch.stack(ys, dim=0)  # [T, batch, state]


class _BlackboxODE2DBase(PickleStateMixin, BaseModel):
    """Shared training logic for blackbox 2-D ODE variants."""

    def __init__(self, config, ode_func_cls) -> None:
        super().__init__(config)
        self.config: BlackboxODE2DConfig
        self._ode_func_cls = ode_func_cls
        self.ode_func: nn.Module | None = None

    def _fit(self, u, y, *, val_data=None, logger=None) -> None:
        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        y_norm = self._normalize_y(y)

        self.ode_func = self._ode_func_cls(
            cfg.state_dim, cfg.input_dim, cfg.hidden_dim
        ).to(device)

        optimizer = make_optimizer(self.ode_func.parameters(), cfg.learning_rate)
        scheduler = make_scheduler(
            optimizer, cfg.scheduler_patience, cfg.scheduler_factor, cfg.scheduler_min_lr
        )
        stopper = EarlyStopper(cfg.early_stopping_patience)

        N = len(u_norm)
        k = cfg.k_steps
        dt = cfg.dt

        if N <= k:
            raise ValueError(f"Signal length ({N}) too short for k_steps={k}")

        dataset = WindowedTrainDataset(
            u_norm, y_norm, window_size=k + 1,
            samples_per_epoch=max(N - k, cfg.batch_size),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True
        )

        def step_fn(epoch: int) -> float:
            self.ode_func.train()
            total = 0.0
            count = 0
            for u_sub, y_sub in loader:
                # u_sub: [B, k+1], y_sub: [B, k+1]
                u_sub = u_sub.to(device)
                y_sub = y_sub.to(device)
                B = u_sub.shape[0]

                y0 = torch.stack(
                    [y_sub[:, 0], torch.zeros(B, device=device)], dim=-1
                )  # [B, 2]

                optimizer.zero_grad()

                # Euler k-step integration
                y_pred = [y0]
                state = y0
                for step in range(k):
                    u_step = u_sub[:, step:step + 1]  # [B, 1]
                    dydt = self.ode_func(state, u_step)
                    state = state + dt * dydt
                    y_pred.append(state)
                y_pred = torch.stack(y_pred, dim=1)  # [B, k+1, 2]

                loss = nn.functional.mse_loss(y_pred[:, :, 0], y_sub)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.ode_func.parameters(), cfg.grad_clip
                )
                optimizer.step()
                total += loss.item()
                count += 1
            return total / max(count, 1)

        # Build validation step (full val trajectory Euler integration, no grad)
        val_sfn = None
        if val_data is not None:
            u_val_norm = self._normalize_u(val_data[0])
            y_val_norm = self._normalize_y(val_data[1])
            n_val = len(u_val_norm)
            u_val_t = torch.tensor(
                np.asarray(u_val_norm, dtype=np.float32).ravel(),
                dtype=torch.float32, device=device,
            )
            y_val_t = torch.tensor(
                np.asarray(y_val_norm, dtype=np.float32).ravel(),
                dtype=torch.float32, device=device,
            )

            def val_sfn() -> float:
                self.ode_func.eval()
                with torch.no_grad():
                    state_v = torch.tensor(
                        [[y_val_norm[0], 0.0]], dtype=torch.float32, device=device
                    )
                    preds_v = [state_v[:, 0]]
                    for i in range(n_val - 1):
                        u_i = u_val_t[i:i + 1].unsqueeze(-1)
                        dydt_v = self.ode_func(state_v, u_i)
                        state_v = state_v + dt * dydt_v
                        preds_v.append(state_v[:, 0])
                    y_pred_v = torch.cat(preds_v, dim=0)
                    return nn.functional.mse_loss(y_pred_v, y_val_t).item()

        train_loop(
            step_fn,
            epochs=cfg.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=stopper,
            logger=logger,
            verbose=cfg.verbose,
            desc=f"{self.name}",
            val_step_fn=val_sfn,
            model_params=list(self.ode_func.parameters()),
        )

    def _predict(self, u, *, y0=None) -> np.ndarray:
        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        N = len(u_norm)

        u_t = torch.tensor(
            u_norm, dtype=torch.float32, device=device
        ).unsqueeze(-1)

        y0_val = self._normalize_y(np.atleast_1d(y0))[0] if y0 is not None else 0.0
        state = torch.tensor(
            [y0_val, 0.0], dtype=torch.float32, device=device
        ).unsqueeze(0)

        self.ode_func.eval()
        preds = [state]
        with torch.no_grad():
            for i in range(N - 1):
                u_i = u_t[i].unsqueeze(0)
                dydt = self.ode_func(state, u_i)
                state = state + cfg.dt * dydt
                preds.append(state)

        y_pred = torch.cat(preds, dim=0)[:, 0].cpu().numpy()[:N]
        return self._denormalize_y(y_pred)


@register_model("vanilla_node_2d", BlackboxODE2DConfig)
class VanillaNODE2D(_BlackboxODE2DBase):
    def __init__(self, config=None):
        super().__init__(config or BlackboxODE2DConfig(), VanillaODEFunc)


@register_model("structured_node", BlackboxODE2DConfig)
class StructuredNODE(_BlackboxODE2DBase):
    def __init__(self, config=None):
        super().__init__(config or BlackboxODE2DConfig(), StructuredODEFunc)


@register_model("adaptive_node", BlackboxODE2DConfig)
class AdaptiveNODE(_BlackboxODE2DBase):
    def __init__(self, config=None):
        super().__init__(config or BlackboxODE2DConfig(), AdaptiveODEFunc)
