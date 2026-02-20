"""Blackbox 2-D CDE models: Vanilla, Structured, Adaptive.

Same three drift variants but driven by controlled paths via
``torchcde``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.config import BlackboxCDE2DConfig
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
# CDE wrapper
# ─────────────────────────────────────────────────────────────────────

class _CDEFuncWrapper(nn.Module):
    """Wraps a drift func into torchcde-compatible vector field.

    ``f(t, z) → [hidden_dim, input_channels]`` matrix.
    """

    def __init__(self, drift_func, state_dim: int, input_channels: int):
        super().__init__()
        self.drift_func = drift_func
        self.state_dim = state_dim
        self.input_channels = input_channels
        self.proj = nn.Linear(state_dim, state_dim * input_channels)

    def forward(self, t, z):
        # Use a dummy zero input for the drift to get a state derivative
        dummy_u = torch.zeros(
            *z.shape[:-1], 1, device=z.device, dtype=z.dtype
        )
        base = self.drift_func(z, dummy_u)
        w = self.proj(base)
        return w.view(*z.shape[:-1], self.state_dim, self.input_channels)


class _BlackboxCDE2DBase(PickleStateMixin, BaseModel):
    """Shared training logic for blackbox 2-D CDE variants."""

    def __init__(self, config, drift_cls) -> None:
        super().__init__(config)
        self.config: BlackboxCDE2DConfig
        self._drift_cls = drift_cls
        self.cde_func: _CDEFuncWrapper | None = None
        self.output_net: nn.Module | None = None

    def _make_path(self, u_seq, t_seq, device):
        import torchcde
        X = torch.stack([t_seq, u_seq], dim=-1)
        if X.dim() == 2:
            X = X.unsqueeze(0)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X)
        return torchcde.CubicSpline(coeffs)

    def _fit(self, u, y, *, val_data=None, logger=None) -> None:
        try:
            import torchcde
        except ImportError as exc:
            raise ImportError("torchcde required") from exc

        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        y_norm = self._normalize_y(y)

        drift = self._drift_cls(cfg.state_dim, cfg.input_dim, cfg.hidden_dim).to(device)
        input_channels = 2  # [time, u]
        self.cde_func = _CDEFuncWrapper(drift, cfg.state_dim, input_channels).to(device)
        self.output_net = nn.Linear(cfg.state_dim, 1).to(device)

        all_params = (
            list(self.cde_func.parameters())
            + list(self.output_net.parameters())
        )
        optimizer = make_optimizer(all_params, cfg.learning_rate)
        scheduler = make_scheduler(
            optimizer, cfg.scheduler_patience, cfg.scheduler_factor, cfg.scheduler_min_lr
        )
        stopper = EarlyStopper(cfg.early_stopping_patience)

        N = len(u_norm)
        k = cfg.k_steps
        dt = cfg.dt

        if N <= k:
            raise ValueError(f"Signal too short for k_steps={k}")

        dataset = WindowedTrainDataset(
            u_norm, y_norm, window_size=k + 1,
            samples_per_epoch=max(N - k, cfg.batch_size),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True
        )

        def step_fn(epoch: int) -> float:
            self.cde_func.train()
            self.output_net.train()
            total = 0.0
            count = 0

            for u_sub, y_sub in loader:
                # u_sub: [B, k+1], y_sub: [B, k+1]
                u_sub = u_sub.to(device)
                y_sub = y_sub.to(device)
                B = u_sub.shape[0]

                # Process each sample individually (CDE paths
                # are not directly batchable from random windows).
                batch_loss = torch.tensor(0.0, device=device)
                for i in range(B):
                    u_i = u_sub[i, :k]  # [k]
                    t_sub_i = torch.linspace(0, k * dt, k, device=device)
                    X_path = self._make_path(u_i, t_sub_i, device)

                    z0 = torch.stack(
                        [y_sub[i, 0:1], torch.zeros(1, device=device)], dim=-1
                    )  # [1, 2]
                    t_eval = torch.linspace(0, k * dt, k + 1, device=device)

                    z_pred = torchcde.cdeint(
                        X=X_path, z0=z0, func=self.cde_func,
                        t=t_eval, method="rk4",
                    )
                    y_pred = self.output_net(z_pred.squeeze(0)).squeeze(-1)
                    batch_loss = batch_loss + nn.functional.mse_loss(y_pred, y_sub[i])

                optimizer.zero_grad()
                avg_loss = batch_loss / B
                avg_loss.backward()
                nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
                optimizer.step()
                total += avg_loss.item()
                count += 1

            return total / max(count, 1)

        # Build validation step (full val trajectory, no grad)
        val_sfn = None
        if val_data is not None:
            u_val_norm = self._normalize_u(val_data[0])
            y_val_norm = self._normalize_y(val_data[1])
            n_val = len(u_val_norm)
            u_vt = torch.tensor(u_val_norm, dtype=torch.float32, device=device)
            t_vt = torch.linspace(0, (n_val - 1) * dt, n_val, device=device)
            z0_val = torch.tensor(
                [[y_val_norm[0], 0.0]], dtype=torch.float32, device=device
            )
            y_target_val = torch.tensor(y_val_norm, dtype=torch.float32, device=device)

            def val_sfn() -> float:
                self.cde_func.eval()
                self.output_net.eval()
                with torch.no_grad():
                    X_p = self._make_path(u_vt, t_vt, device)
                    z_p = torchcde.cdeint(
                        X=X_p, z0=z0_val, func=self.cde_func,
                        t=t_vt, method="rk4",
                    )
                    y_p = self.output_net(z_p.squeeze(0)).squeeze(-1)
                    return nn.functional.mse_loss(y_p, y_target_val).item()

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
            model_params=all_params,
        )

    def _predict(self, u, *, y0=None) -> np.ndarray:
        try:
            import torchcde
        except ImportError as exc:
            raise ImportError("torchcde required") from exc

        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        N = len(u_norm)
        dt = cfg.dt

        u_t = torch.tensor(u_norm, dtype=torch.float32, device=device)
        t_t = torch.linspace(0, (N - 1) * dt, N, device=device)
        X_path = self._make_path(u_t, t_t, device)

        y0_val = self._normalize_y(np.atleast_1d(y0))[0] if y0 is not None else 0.0
        z0 = torch.tensor(
            [[y0_val, 0.0]], dtype=torch.float32, device=device
        )

        self.cde_func.eval()
        self.output_net.eval()
        with torch.no_grad():
            z_pred = torchcde.cdeint(
                X=X_path, z0=z0, func=self.cde_func,
                t=t_t, method="rk4",
            )
            y_pred = self.output_net(z_pred.squeeze(0)).squeeze(-1)

        return self._denormalize_y(y_pred.cpu().numpy()[:N])


@register_model("vanilla_ncde_2d", BlackboxCDE2DConfig)
class VanillaNCDE2D(_BlackboxCDE2DBase):
    def __init__(self, config=None):
        super().__init__(config or BlackboxCDE2DConfig(), VanillaODEFunc)


@register_model("structured_ncde", BlackboxCDE2DConfig)
class StructuredNCDE(_BlackboxCDE2DBase):
    def __init__(self, config=None):
        super().__init__(config or BlackboxCDE2DConfig(), StructuredODEFunc)


@register_model("adaptive_ncde", BlackboxCDE2DConfig)
class AdaptiveNCDE(_BlackboxCDE2DBase):
    def __init__(self, config=None):
        super().__init__(config or BlackboxCDE2DConfig(), AdaptiveODEFunc)
