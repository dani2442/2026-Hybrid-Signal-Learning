"""Neural CDE (Controlled Differential Equation) model.

Uses ``torchcde`` for path interpolation and integration.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.config import NeuralCDEConfig
from src.data.torch_datasets import WindowedTrainDataset
from src.models.base import BaseModel, PickleStateMixin
from src.models.registry import register_model
from src.models.training import (
    EarlyStopper,
    make_optimizer,
    make_scheduler,
    train_loop,
)


# ─────────────────────────────────────────────────────────────────────
# CDE vector field
# ─────────────────────────────────────────────────────────────────────

class _CDEFunc(nn.Module):
    """CDE vector field: ``dz/dt = f(z) · dX/dt``."""

    def __init__(self, hidden_dim: int, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim * input_dim),
        )
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, t, z):
        out = self.net(z)
        return out.view(*z.shape[:-1], self.hidden_dim, self.input_dim)


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────

@register_model("neural_cde", NeuralCDEConfig)
class NeuralCDEModel(PickleStateMixin, BaseModel):
    """Neural CDE for system identification (requires ``torchcde``)."""

    def __init__(self, config: NeuralCDEConfig | None = None) -> None:
        super().__init__(config or NeuralCDEConfig())
        self.config: NeuralCDEConfig
        self.cde_func: _CDEFunc | None = None
        self.initial_net: nn.Module | None = None
        self.output_net: nn.Module | None = None

    def _make_path(self, u_seq, y_seq, t_seq, device):
        """Build an interpolated controlled path from data."""
        import torchcde

        # Stack [time, u, y] or [time, u] as path channels
        if y_seq is not None:
            X_raw = torch.stack([t_seq, u_seq, y_seq], dim=-1)
        else:
            X_raw = torch.stack([t_seq, u_seq], dim=-1)

        if X_raw.dim() == 2:
            X_raw = X_raw.unsqueeze(0)

        interp = self.config.interpolation
        if interp == "cubic":
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_raw)
            return torchcde.CubicSpline(coeffs)
        elif interp == "linear":
            coeffs = torchcde.linear_interpolation_coeffs(X_raw)
            return torchcde.LinearInterpolation(coeffs)
        else:
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_raw)
            return torchcde.CubicSpline(coeffs)

    def _fit(self, u, y, *, val_data=None, logger=None) -> None:
        try:
            import torchcde
        except ImportError as exc:
            raise ImportError(
                "torchcde required: pip install torchcde"
            ) from exc

        cfg = self.config
        device = self.device
        u_norm = self._normalize_u(u)
        y_norm = self._normalize_y(y)

        hidden_dim = cfg.hidden_dim

        # Path channels: [time, u] — must match prediction (no y)
        path_channels = cfg.input_dim
        self.cde_func = _CDEFunc(hidden_dim, path_channels).to(device)
        self.initial_net = nn.Linear(path_channels, hidden_dim).to(device)
        self.output_net = nn.Linear(hidden_dim, 1).to(device)

        all_params = (
            list(self.cde_func.parameters())
            + list(self.initial_net.parameters())
            + list(self.output_net.parameters())
        )
        optimizer = make_optimizer(all_params, cfg.learning_rate)
        scheduler = make_scheduler(
            optimizer, cfg.scheduler_patience, cfg.scheduler_factor, cfg.scheduler_min_lr
        )
        stopper = EarlyStopper(cfg.early_stopping_patience)

        N = len(u_norm)
        dt = getattr(cfg, "dt", 0.05)
        window = cfg.train_window_size
        seqs = cfg.sequences_per_epoch

        dataset = WindowedTrainDataset(
            u_norm, y_norm, window, samples_per_epoch=seqs
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        def step_fn(epoch: int) -> float:
            self.cde_func.train()
            self.initial_net.train()
            self.output_net.train()
            total_loss = 0.0
            count = 0

            for u_sub, y_sub in loader:
                u_sub = u_sub.squeeze(0).to(device)
                y_sub = y_sub.squeeze(0).to(device)
                n_sub = u_sub.shape[0]
                t_sub = torch.linspace(0, (n_sub - 1) * dt, n_sub, device=device)

                X_path = self._make_path(u_sub, None, t_sub, device)

                # Initial hidden state from first observation
                x0 = X_path.evaluate(t_sub[0]).squeeze(0)
                z0 = self.initial_net(x0)
                if z0.dim() == 1:
                    z0 = z0.unsqueeze(0)

                solver_name = cfg.solver if cfg.solver != "euler" else "euler"

                z_T = torchcde.cdeint(
                    X=X_path, z0=z0, func=self.cde_func,
                    t=t_sub,
                    method=solver_name,
                )
                y_pred = self.output_net(z_T.squeeze(0))
                y_target = y_sub

                optimizer.zero_grad()
                loss = nn.functional.mse_loss(y_pred.squeeze(), y_target)
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
                optimizer.step()
                total_loss += loss.item()
                count += 1

            return total_loss / max(count, 1)

        # Build validation step (full val trajectory, no grad)
        val_sfn = None
        if val_data is not None:
            u_val_norm = self._normalize_u(val_data[0])
            y_val_norm = self._normalize_y(val_data[1])
            n_val = len(u_val_norm)
            dt_v = dt
            u_vt = torch.tensor(u_val_norm, dtype=torch.float32, device=device)
            y_vt = torch.tensor(y_val_norm, dtype=torch.float32, device=device)
            t_vt = torch.linspace(0, (n_val - 1) * dt_v, n_val, device=device)
            solver_v = cfg.solver if cfg.solver != "euler" else "euler"

            def val_sfn() -> float:
                self.cde_func.eval()
                self.initial_net.eval()
                self.output_net.eval()
                with torch.no_grad():
                    X_p = self._make_path(u_vt, None, t_vt, device)
                    x0_v = X_p.evaluate(t_vt[0]).squeeze(0)
                    z0_v = self.initial_net(x0_v)
                    if z0_v.dim() == 1:
                        z0_v = z0_v.unsqueeze(0)
                    z_T_v = torchcde.cdeint(
                        X=X_p, z0=z0_v, func=self.cde_func,
                        t=t_vt, method=solver_v,
                    )
                    y_p = self.output_net(z_T_v.squeeze(0)).squeeze(-1)
                    return nn.functional.mse_loss(y_p, y_vt).item()

        train_loop(
            step_fn,
            epochs=cfg.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=stopper,
            logger=logger,
            verbose=cfg.verbose,
            desc="NeuralCDE",
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
        dt = getattr(cfg, "dt", 0.05)

        u_t = torch.tensor(u_norm, dtype=torch.float32, device=device)
        t_t = torch.linspace(0, (N - 1) * dt, N, device=device)

        X_path = self._make_path(u_t, None, t_t, device)

        x0 = X_path.evaluate(t_t[0]).squeeze(0)
        z0 = self.initial_net(x0)
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)

        self.cde_func.eval()
        self.output_net.eval()

        with torch.no_grad():
            z_T = torchcde.cdeint(
                X=X_path, z0=z0, func=self.cde_func,
                t=t_t,
                method=cfg.solver if cfg.solver != "euler" else "euler",
            )
            y_pred = self.output_net(z_T.squeeze(0))

        y_pred_np = y_pred.squeeze().cpu().numpy()[:N]
        return self._denormalize_y(y_pred_np)
