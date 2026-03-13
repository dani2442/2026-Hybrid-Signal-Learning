"""Physics-based ODE models for system identification.

LinearPhysics:  J·θ̈ + R·θ̇ + K·(θ + δ) = τ·V
StribeckPhysics: above + Stribeck friction

Parameters are log-parameterised (strictly positive) and trained
by back-propagating through torchsde (zero diffusion).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import BaseModel, resolve_device, DEFAULT_GRAD_CLIP
from .torchsde_utils import interp_u


# ── ODE right-hand sides (nn.Module with f/g SDE interface) ──────────

def _build_linear_ode():
    import torch, torch.nn as nn

    class LinearPhysODE(nn.Module):
        """Linear 2nd-order beam dynamics."""
        noise_type = "diagonal"; sde_type = "ito"

        def __init__(self):
            super().__init__()
            self.log_J = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
            self.log_R = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
            self.log_K = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))
            self.delta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.log_Tau = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))
            self.u_series = self.t_series = self.batch_start_times = None

        def get_params(self):
            return (torch.exp(self.log_J), torch.exp(self.log_R),
                    torch.exp(self.log_K), self.delta, torch.exp(self.log_Tau))

        def f(self, t, x):
            J, R, K, delta, Tau = self.get_params()
            u_t = interp_u(self, t, x)
            th, thd = x[:, 0:1], x[:, 1:2]
            thdd = (Tau * u_t - R * thd - K * (th + delta)) / J
            return torch.cat([thd, thdd], dim=1)

        def g(self, t, x):
            return torch.zeros_like(x)

    return LinearPhysODE()


def _build_stribeck_ode():
    import torch, torch.nn as nn

    class StribeckPhysODE(nn.Module):
        """Linear beam + Stribeck friction dynamics."""
        noise_type = "diagonal"; sde_type = "ito"

        def __init__(self):
            super().__init__()
            self.log_J = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
            self.log_R = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
            self.log_K = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))
            self.delta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.log_Tau = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))
            self.log_Fc = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
            self.log_Fs = nn.Parameter(torch.tensor(np.log(0.2), dtype=torch.float32))
            self.log_vs = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
            self.log_b = nn.Parameter(torch.tensor(np.log(0.01), dtype=torch.float32))
            self.u_series = self.t_series = self.batch_start_times = None

        def get_params(self):
            return (torch.exp(self.log_J), torch.exp(self.log_R),
                    torch.exp(self.log_K), self.delta, torch.exp(self.log_Tau),
                    torch.exp(self.log_Fc), torch.exp(self.log_Fs),
                    torch.exp(self.log_vs), torch.exp(self.log_b))

        def f(self, t, x):
            J, R, K, delta, Tau, Fc, Fs, vs, b = self.get_params()
            u_t = interp_u(self, t, x)
            th, thd = x[:, 0:1], x[:, 1:2]
            sgn = torch.tanh(thd / 1e-3)
            F_str = (Fc + (Fs - Fc) * torch.exp(-(thd / vs) ** 2)) * sgn + b * thd
            thdd = (Tau * u_t - R * thd - K * (th + delta) - F_str) / J
            return torch.cat([thd, thdd], dim=1)

        def g(self, t, x):
            return torch.zeros_like(x)

    return StribeckPhysODE()


# ── shared wrapper ────────────────────────────────────────────────────

class _PhysODEModel(BaseModel):
    """Shared base for physics ODE models. State = [θ, θ̇], observe θ."""

    def __init__(self, config, ode_factory):
        super().__init__(config)
        self.ode_factory = ode_factory
        self.dt = config.dt
        self.ode_func_ = None
        self._device = None
        self._dtype = None

    def _simulate(self, u_t, x0, t_grid=None):
        import torch, torchsde
        n = u_t.shape[0]
        if t_grid is None:
            t_grid = torch.arange(n, dtype=self._dtype, device=self._device) * self.dt
        self.ode_func_.t_series = t_grid
        self.ode_func_.u_series = u_t
        self.ode_func_.batch_start_times = None
        x0b = x0 if x0.ndim == 2 else x0.unsqueeze(0)
        return torchsde.sdeint(
            self.ode_func_, x0b, t_grid, method="euler", dt=self.dt,
        )[:, 0, :]

    # ── training ──────────────────────────────────────────────────────

    def _fit(self, u, y, *, val_data=None, logger=None):
        import torch
        c = self.config
        self._device = resolve_device(c.device)
        self._dtype = torch.float32
        self.ode_func_ = self.ode_factory().to(self._device)

        u = np.asarray(u, dtype=float).flatten()
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 2 and y_arr.shape[1] >= 2:
            y = y_arr[:, 0].flatten()
            y_dot = y_arr[:, 1].flatten()
        else:
            y = y_arr.flatten()
            y_dot = None

        if c.training_mode == "full":
            self.training_loss_ = self._train_full(u, y, logger, y_dot=y_dot)
        else:
            self.training_loss_ = self._train_subseq(u, y, logger, y_dot=y_dot)

    def _train_full(self, u, y, logger, *, y_dot=None):
        import torch, torch.optim as optim
        from tqdm.auto import tqdm
        c = self.config

        u_t = torch.tensor(u.reshape(-1, 1), dtype=self._dtype, device=self._device)
        y_t = torch.tensor(y, dtype=self._dtype, device=self._device)
        y_dot_t = (
            torch.tensor(y_dot, dtype=self._dtype, device=self._device)
            if y_dot is not None
            else None
        )
        t_grid = torch.arange(len(y_t), dtype=self._dtype, device=self._device) * self.dt
        if y_dot_t is not None and len(y_dot_t) > 0:
            omega0 = y_dot_t[0].item()
        else:
            omega0 = (y_t[1] - y_t[0]).item() / self.dt if len(y_t) > 1 else 0.0
        x0 = torch.tensor([[y_t[0].item(), omega0]], dtype=self._dtype, device=self._device)

        params = list(self.ode_func_.parameters())
        optimizer = optim.Adam(params, lr=c.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=200, min_lr=1e-6)
        history, best_loss, best_state = [], float("inf"), None

        it = range(c.epochs)
        if c.verbose:
            it = tqdm(it, desc=f"Training {type(self).__name__} (full)", unit="epoch")

        for epoch in it:
            self.ode_func_.train(); optimizer.zero_grad()
            pred = self._simulate(u_t, x0, t_grid)
            loss = torch.mean((pred[:, 0] - y_t) ** 2)
            if torch.isnan(loss) or torch.isinf(loss):
                history.append(float("nan")); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, DEFAULT_GRAD_CLIP)
            optimizer.step()
            loss_val = loss.item(); history.append(loss_val); scheduler.step(loss_val)
            if loss_val < best_loss:
                best_loss = loss_val
                best_state = {k: v.clone() for k, v in self.ode_func_.state_dict().items()}
            if c.verbose and hasattr(it, "set_postfix"):
                it.set_postfix(loss=loss_val)
            if logger and c.wandb_log_every > 0 and (epoch + 1) % c.wandb_log_every == 0:
                logger.log_metrics({"train/loss": loss_val, "train/epoch": epoch + 1}, step=epoch + 1)

        if best_state:
            self.ode_func_.load_state_dict(best_state)
        return history

    def _train_subseq(self, u, y, logger, *, y_dot=None):
        import torch, torch.optim as optim, torchsde
        from tqdm.auto import tqdm
        c = self.config
        seq_len = max(2, int(c.sequence_length))

        u_t = torch.tensor(u.reshape(-1, 1), dtype=self._dtype, device=self._device)
        y_t = torch.tensor(y, dtype=self._dtype, device=self._device)
        y_dot_t = (
            torch.tensor(y_dot, dtype=self._dtype, device=self._device)
            if y_dot is not None
            else None
        )
        if len(y_t) <= seq_len:
            # Fallback for short trajectories: subsequences are ill-defined.
            return self._train_full(u, y, logger, y_dot=y_dot)

        t_full = torch.arange(len(y_t), dtype=self._dtype, device=self._device) * self.dt
        t_local = torch.arange(seq_len, dtype=self._dtype, device=self._device) * self.dt
        valid_starts = np.arange(0, len(y_t) - seq_len, dtype=int)
        if valid_starts.size == 0:
            return self._train_full(u, y, logger, y_dot=y_dot)

        params = list(self.ode_func_.parameters())
        optimizer = optim.Adam(params, lr=c.learning_rate)
        history = []
        batch_size = max(1, int(c.batch_size))
        offsets = torch.arange(seq_len, device=self._device, dtype=torch.long)

        it = range(c.epochs)
        if c.verbose:
            it = tqdm(it, desc=f"Training {type(self).__name__} (subseq)", unit="epoch")

        for epoch in it:
            self.ode_func_.train()
            optimizer.zero_grad()

            starts_np = np.random.choice(valid_starts, size=batch_size, replace=True)
            starts = torch.tensor(starts_np, dtype=torch.long, device=self._device)

            theta0 = y_t[starts]
            if y_dot_t is not None:
                omega0 = y_dot_t[starts]
            else:
                next_starts = torch.clamp(starts + 1, max=len(y_t) - 1)
                omega0 = (y_t[next_starts] - y_t[starts]) / self.dt
            x0 = torch.stack([theta0, omega0], dim=1)  # (B, 2)

            # Match notebook-style batched subsequence rollout:
            # absolute time = batch_start + local rollout time.
            self.ode_func_.t_series = t_full
            self.ode_func_.u_series = u_t
            self.ode_func_.batch_start_times = t_full[starts].unsqueeze(1)
            pred = torchsde.sdeint(
                self.ode_func_,
                x0,
                t_local,
                method="euler",
                dt=self.dt,
            )  # (K, B, 2)
            pred_theta = pred[:, :, 0]

            target_idx = starts.unsqueeze(1) + offsets.unsqueeze(0)  # (B, K)
            y_target = y_t[target_idx].transpose(0, 1)  # (K, B)
            loss = torch.mean((pred_theta - y_target) ** 2)
            if torch.isnan(loss) or torch.isinf(loss):
                history.append(float("nan"))
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, DEFAULT_GRAD_CLIP)
            optimizer.step()
            loss_val = loss.item()
            history.append(loss_val)
            if c.verbose and hasattr(it, "set_postfix"):
                it.set_postfix(loss=loss_val)
            if logger and c.wandb_log_every > 0 and (epoch + 1) % c.wandb_log_every == 0:
                logger.log_metrics({"train/loss": loss_val, "train/epoch": epoch + 1}, step=epoch + 1)
        self.ode_func_.batch_start_times = None
        return history

    # ── predict ───────────────────────────────────────────────────────

    def predict_osa(self, u, y):
        import torch
        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        preds = []
        self.ode_func_.eval()
        with torch.no_grad():
            for k in range(len(y) - 1):
                omega = (y[k] - y[k-1]) / self.dt if k > 0 else (y[1] - y[0]) / self.dt
                x0 = torch.tensor([[y[k], omega]], dtype=self._dtype, device=self._device)
                u_seg = torch.tensor([[u[k]], [u[min(k+1, len(u)-1)]]],
                                     dtype=self._dtype, device=self._device)
                pred = self._simulate(u_seg, x0)
                preds.append(pred[-1, :].cpu().numpy())  # full [theta, theta_dot]
        return np.asarray(preds)  # shape (N, 2)

    def predict_free_run(self, u, y_initial, *, return_full_state=False):
        import torch
        u = np.asarray(u, dtype=float).flatten()
        y0 = np.asarray(y_initial, dtype=float).flatten()
        omega0 = (y0[1] - y0[0]) / self.dt if len(y0) > 1 else 0.0
        self.ode_func_.eval()
        with torch.no_grad():
            u_t = torch.tensor(u.reshape(-1, 1), dtype=self._dtype, device=self._device)
            x0 = torch.tensor([[y0[0], omega0]], dtype=self._dtype, device=self._device)
            pred = self._simulate(u_t, x0)
        if return_full_state:
            return pred.cpu().numpy()  # (N, 2) — [theta, theta_dot]
        return pred[:, 0].cpu().numpy()

    # ── save / load hooks ─────────────────────────────────────────────

    def _collect_state(self) -> Dict[str, Any]:
        if self.ode_func_ is None:
            return {}
        return {"ode_func": self.ode_func_.state_dict()}

    def _restore_state(self, state):
        self.ode_func_.load_state_dict(state["ode_func"])

    def _build_for_load(self):
        import torch
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self.ode_func_ = self.ode_factory().to(self._device)


# ── public classes ────────────────────────────────────────────────────

class LinearPhysics(_PhysODEModel):
    """Linear 2nd-order beam: J·θ̈ + R·θ̇ + K·(θ+δ) = τ·V"""

    def __init__(self, config=None):
        from ..config import LinearPhysicsConfig
        if config is None:
            config = LinearPhysicsConfig()
        super().__init__(config, ode_factory=_build_linear_ode)

    def __repr__(self):
        return f"LinearPhysics(dt={self.dt}, epochs={self.config.epochs})"


class StribeckPhysics(_PhysODEModel):
    """Linear beam + Stribeck friction."""

    def __init__(self, config=None):
        from ..config import StribeckPhysicsConfig
        if config is None:
            config = StribeckPhysicsConfig()
        super().__init__(config, ode_factory=_build_stribeck_ode)

    def __repr__(self):
        return f"StribeckPhysics(dt={self.dt}, epochs={self.config.epochs})"
