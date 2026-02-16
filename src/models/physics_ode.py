"""Physics-based ODE models for system identification.

LinearPhysics:  J·θ̈ + R·θ̇ + K·(θ + δ) = τ·V
StribeckPhysics: above + Stribeck friction

Parameters are log-parameterised (strictly positive) and trained
by back-propagating through torchsde (zero diffusion).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..config import LinearPhysicsConfig, StribeckPhysicsConfig
from .base import BaseModel, DEFAULT_GRAD_CLIP
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
        self._device = self._resolve_torch_device()
        self._dtype = torch.float32
        self.ode_func_ = self.ode_factory().to(self._device)

        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()

        if c.training_mode == "full":
            self.training_loss_ = self._train_full(u, y, logger)
        else:
            self.training_loss_ = self._train_subseq(u, y, logger)

    def _train_full(self, u, y, logger):
        import torch, torch.optim as optim
        from tqdm.auto import tqdm
        c = self.config

        u_t = torch.tensor(u.reshape(-1, 1), dtype=self._dtype, device=self._device)
        y_t = torch.tensor(y, dtype=self._dtype, device=self._device)
        t_grid = torch.arange(len(y_t), dtype=self._dtype, device=self._device) * self.dt
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

    def _train_subseq(self, u, y, logger):
        import torch, torch.optim as optim
        from tqdm.auto import tqdm
        c = self.config
        seq_len = c.sequence_length

        u_t = torch.tensor(u.reshape(-1, 1), dtype=self._dtype, device=self._device)
        y_t = torch.tensor(y, dtype=self._dtype, device=self._device)
        t_full = torch.arange(len(y_t), dtype=self._dtype, device=self._device) * self.dt
        max_start = max(1, len(y_t) - seq_len - 1)

        params = list(self.ode_func_.parameters())
        optimizer = optim.Adam(params, lr=c.learning_rate)
        history = []

        it = range(c.epochs)
        if c.verbose:
            it = tqdm(it, desc=f"Training {type(self).__name__} (subseq)", unit="epoch")

        for epoch in it:
            self.ode_func_.train(); optimizer.zero_grad()
            start = int(np.random.randint(0, max_start))
            end = start + seq_len
            y_seq = y_t[start:end]; t_seq = t_full[start:end]; u_seq = u_t[start:end]
            omega0 = (y_seq[1] - y_seq[0]).item() / self.dt if len(y_seq) > 1 else 0.0
            x0 = torch.tensor([[y_seq[0].item(), omega0]], dtype=self._dtype, device=self._device)
            pred = self._simulate(u_seq, x0, t_seq)
            loss = torch.mean((pred[:, 0] - y_seq) ** 2)
            if torch.isnan(loss) or torch.isinf(loss):
                history.append(float("nan")); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, DEFAULT_GRAD_CLIP)
            optimizer.step()
            loss_val = loss.item(); history.append(loss_val)
            if c.verbose and hasattr(it, "set_postfix"):
                it.set_postfix(loss=loss_val)
            if logger and c.wandb_log_every > 0 and (epoch + 1) % c.wandb_log_every == 0:
                logger.log_metrics({"train/loss": loss_val, "train/epoch": epoch + 1}, step=epoch + 1)
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
                preds.append(pred[-1, 0].cpu().item())
        return np.asarray(preds)

    def predict_free_run(self, u, y_initial):
        import torch
        u = np.asarray(u, dtype=float).flatten()
        y0 = np.asarray(y_initial, dtype=float).flatten()
        omega0 = (y0[1] - y0[0]) / self.dt if len(y0) > 1 else 0.0
        self.ode_func_.eval()
        with torch.no_grad():
            u_t = torch.tensor(u.reshape(-1, 1), dtype=self._dtype, device=self._device)
            x0 = torch.tensor([[y0[0], omega0]], dtype=self._dtype, device=self._device)
            pred = self._simulate(u_t, x0)
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
        self._device = self._resolve_torch_device("cpu")
        self._dtype = torch.float32
        self.ode_func_ = self.ode_factory().to(self._device)


# ── public classes ────────────────────────────────────────────────────

class LinearPhysics(_PhysODEModel):
    """Linear 2nd-order beam: J·θ̈ + R·θ̇ + K·(θ+δ) = τ·V"""

    def __init__(self, config: LinearPhysicsConfig | None = None, **kwargs):
        if config is None:
            config = LinearPhysicsConfig(**kwargs)
        super().__init__(config, ode_factory=_build_linear_ode)

    def __repr__(self):
        return f"LinearPhysics(dt={self.dt}, epochs={self.config.epochs})"


class StribeckPhysics(_PhysODEModel):
    """Linear beam + Stribeck friction."""

    def __init__(self, config: StribeckPhysicsConfig | None = None, **kwargs):
        if config is None:
            config = StribeckPhysicsConfig(**kwargs)
        super().__init__(config, ode_factory=_build_stribeck_ode)

    def __repr__(self):
        return f"StribeckPhysics(dt={self.dt}, epochs={self.config.epochs})"
