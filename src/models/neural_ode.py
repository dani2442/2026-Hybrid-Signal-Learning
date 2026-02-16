"""Neural ODE model solved with torchsde (zero diffusion).

Supports both subsequence-batched training and full-trajectory training.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import NeuralODEConfig
from .base import BaseModel, DEFAULT_GRAD_CLIP
from .torchsde_utils import train_sequence_batches


# ── internal ODE dynamics ─────────────────────────────────────────────

class _ODEFunc:
    """ODE dynamics f(t, x, u) with linear interpolation of inputs.

    Not an nn.Module — delegates parameter management to ``self.net``.
    Provides the ``f`` / ``g`` / ``noise_type`` / ``sde_type`` interface
    expected by ``torchsde.sdeint``.
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, state_dim: int, input_dim: int, hidden_layers: List[int],
                 activation: str = "selu"):
        import torch.nn as nn

        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        act_cls = {"selu": nn.SELU, "tanh": nn.Tanh, "relu": nn.ReLU}.get(
            activation.lower(), nn.SELU)

        layers: list[nn.Module] = []
        prev = self.state_dim + self.input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), act_cls()]
            prev = h
        layers.append(nn.Linear(prev, self.state_dim))
        self.net = nn.Sequential(*layers)

        nonlin = "selu" if activation.lower() == "selu" else "linear"
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                is_last = m is list(self.net.modules())[-1]
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear" if is_last else nonlin)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self._u_series = None
        self._t_series = None

    def to(self, device):
        self.net = self.net.to(device)
        return self

    def parameters(self):
        return self.net.parameters()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def set_control(self, u_path, t_path=None):
        import torch
        if u_path.ndim == 1:
            u_path = u_path.reshape(-1, 1)
        self._u_series = u_path
        self._t_series = t_path if t_path is not None else torch.arange(
            u_path.shape[0], dtype=u_path.dtype, device=u_path.device).float()

    def _u_at(self, t):
        import torch
        ts, us = self._t_series, self._u_series
        t_c = torch.clamp(t, ts[0], ts[-1])
        k = torch.clamp(torch.searchsorted(ts, t_c, right=True), 1, len(ts) - 1)
        t_lo, t_hi = ts[k - 1], ts[k]
        u_lo, u_hi = us[k - 1], us[k]
        d = t_hi - t_lo
        if d.abs() < 1e-9:
            return u_lo
        return u_lo + (t_c - t_lo) / d * (u_hi - u_lo)

    def f(self, t, y):
        """Drift: f(t, y) = net([y, u(t)])."""
        import torch
        u_t = self._u_at(t)
        if u_t.ndim == 1:
            u_t = u_t.unsqueeze(0).expand(y.shape[0], -1)
        return self.net(torch.cat([y, u_t], dim=-1))

    def g(self, t, y):
        """Zero diffusion (pure ODE)."""
        import torch
        return torch.zeros_like(y)


# ── public model ──────────────────────────────────────────────────────

class NeuralODE(BaseModel):
    """Neural ODE for continuous-time system identification.

    Dynamics: dx/dt = f_θ(x, u(t)), solved via torchsde (zero diffusion).
    """

    def __init__(self, config: NeuralODEConfig | None = None, **kwargs):
        if config is None:
            config = NeuralODEConfig(**kwargs)
        super().__init__(config)
        c = self.config
        self.state_dim = c.state_dim
        self.input_dim = c.input_dim
        self.dt = c.dt
        self.ode_func_ = None
        self._device = None
        self._dtype = None

    # ── build / simulate ──────────────────────────────────────────────

    def _build_ode_func(self):
        c = self.config
        self.ode_func_ = _ODEFunc(
            state_dim=c.state_dim, input_dim=c.input_dim,
            hidden_layers=c.hidden_layers, activation=c.activation,
        ).to(self._device)

    def _simulate_trajectory(self, u_path, x0, t_path=None):
        import torch, torchsde
        ts = t_path if t_path is not None else (
            torch.arange(u_path.shape[0], dtype=u_path.dtype, device=u_path.device) * self.dt)
        self.ode_func_.set_control(u_path, ts)
        x0b = x0 if x0.ndim == 2 else x0.reshape(1, -1)
        return torchsde.sdeint(
            self.ode_func_, x0b, ts, method="euler", dt=self.dt,
        )[:, 0, :]

    # ── training ──────────────────────────────────────────────────────

    def _fit(self, u, y, *, val_data=None, logger=None):
        import torch
        c = self.config
        self._device = self._resolve_torch_device()
        self._dtype = torch.float32
        self._build_ode_func()

        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y = np.asarray(y, dtype=float).reshape(-1, self.state_dim)

        if c.training_mode == "full":
            self.training_loss_ = self._train_full(u, y, logger)
        else:
            self.training_loss_ = train_sequence_batches(
                sde_func=self.ode_func_,
                simulate_fn=lambda u_seq, x0: self._simulate_trajectory(u_seq, x0),
                u=u, y=y,
                input_dim=self.input_dim, state_dim=self.state_dim,
                sequence_length=c.sequence_length, epochs=c.epochs,
                learning_rate=c.learning_rate,
                device=self._device, dtype=self._dtype,
                verbose=c.verbose,
                progress_desc="Training NeuralODE",
                logger=logger, log_every=c.wandb_log_every,
                sequences_per_epoch=c.sequences_per_epoch,
                early_stopping_patience=c.early_stopping_patience,
            )

    def _train_full(self, u, y, logger) -> list:
        import torch, torch.nn as nn, torch.optim as optim
        from tqdm.auto import tqdm
        c = self.config

        y_t = torch.tensor(y, dtype=self._dtype, device=self._device).reshape(-1, self.state_dim)
        u_t = torch.tensor(u, dtype=self._dtype, device=self._device).reshape(-1, self.input_dim)
        x0 = y_t[0:1]
        t_grid = torch.arange(len(y_t), dtype=self._dtype, device=self._device) * self.dt

        params = list(self.ode_func_.parameters())
        optimizer = optim.Adam(params, lr=c.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=200, min_lr=1e-6)
        criterion = nn.MSELoss()
        history, best_loss, best_state = [], float("inf"), None

        it = range(c.epochs)
        if c.verbose:
            it = tqdm(it, desc="Training NeuralODE (full)", unit="epoch")

        for epoch in it:
            self.ode_func_.train()
            optimizer.zero_grad()
            pred = self._simulate_trajectory(u_t, x0, t_grid)
            loss = criterion(pred, y_t)

            if torch.isnan(loss) or torch.isinf(loss):
                history.append(float("nan"))
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, DEFAULT_GRAD_CLIP)
            optimizer.step()
            loss_val = float(loss.item())
            history.append(loss_val)
            scheduler.step(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val
                best_state = {k: v.clone() for k, v in self.ode_func_.net.state_dict().items()}

            if c.verbose and hasattr(it, "set_postfix"):
                it.set_postfix(loss=loss_val, lr=f"{optimizer.param_groups[0]['lr']:.1e}")
            if logger and c.wandb_log_every > 0 and (epoch + 1) % c.wandb_log_every == 0:
                logger.log_metrics({"train/loss": loss_val, "train/epoch": epoch + 1}, step=epoch + 1)

        if best_state is not None:
            self.ode_func_.net.load_state_dict(best_state)
        return history

    # ── predict ───────────────────────────────────────────────────────

    def predict_osa(self, u, y):
        import torch
        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y = np.asarray(y, dtype=float).reshape(-1, self.state_dim)
        preds = []
        self.ode_func_.eval()
        with torch.no_grad():
            for t in range(len(y) - 1):
                x_t = torch.tensor(y[t], dtype=self._dtype, device=self._device)
                u_t = torch.tensor(u[t:t+1], dtype=self._dtype, device=self._device)
                u_p = torch.cat([u_t, u_t], dim=0)
                x_next = self._simulate_trajectory(u_p, x_t)[-1]
                preds.append(x_next.cpu().numpy())
        return np.asarray(preds).reshape(-1)

    def predict_free_run(self, u, y_initial):
        import torch
        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y0 = np.asarray(y_initial, dtype=float).reshape(-1, self.state_dim)
        self.ode_func_.eval()
        with torch.no_grad():
            u_t = torch.tensor(u, dtype=self._dtype, device=self._device)
            x0 = torch.tensor(y0[0], dtype=self._dtype, device=self._device)
            pred = self._simulate_trajectory(u_t, x0)
        return pred.cpu().numpy().reshape(-1)

    # ── save / load hooks ─────────────────────────────────────────────

    def _collect_state(self) -> Dict[str, Any]:
        if self.ode_func_ is None:
            return {}
        return {"ode_func_net": self.ode_func_.net.state_dict()}

    def _restore_state(self, state: Dict[str, Any]):
        if "ode_func_net" in state:
            self.ode_func_.net.load_state_dict(state["ode_func_net"])

    def _build_for_load(self):
        import torch
        self._device = self._resolve_torch_device("cpu")
        self._dtype = torch.float32
        self._build_ode_func()

    def __repr__(self) -> str:
        c = self.config
        return (f"NeuralODE(state={c.state_dim}, input={c.input_dim}, "
                f"hidden={c.hidden_layers})")
