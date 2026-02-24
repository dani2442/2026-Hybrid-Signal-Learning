"""Neural SDE model for 2-D state.

Adapted from models/continuous/neural_sde.py and models/blackbox/blackbox_sde.py.
Learns drift f(x, u) and diagonal diffusion g(x, u), integrated with torchsde.

Requires ``torchsde``: ``pip install torchsde``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import InterpNeuralODEBase, NN_VARIANTS, NnVariantConfig, _build_selu_mlp


class _SDEDrift(nn.Module):
    """Structured drift: dx0=x1, dx1 = NN(x, u)."""

    def __init__(self, variant: NnVariantConfig) -> None:
        super().__init__()
        self.accel_net = _build_selu_mlp(input_dim=3, output_dim=1, variant=variant)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        thd = x[..., 1:2]
        thdd = self.accel_net(torch.cat([x, u], dim=-1))
        return torch.cat([thd, thdd], dim=-1)


class _SDEDiffusion(nn.Module):
    """State-dependent diagonal diffusion σ(x, u) → [σ_pos, σ_vel]."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, u], dim=-1))


class BlackBoxSDE(nn.Module):
    """Neural SDE for 2-D state [theta, theta_dot].

    Uses ``torchsde.sdeint`` for integration.  Trained via the unified
    ``train_model`` function; the model’s ``predict_k_steps`` handles
    the SDE-specific integration.

    The model exposes ``set_series`` / ``set_batch_start_times`` for
    compatibility with the interpolation convention, plus a
    ``rollout(t, u, y0, device)`` method for prediction.
    """

    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, variant_name: str = "base") -> None:
        super().__init__()
        if variant_name not in NN_VARIANTS:
            raise ValueError(f"Unknown variant '{variant_name}'")
        self.variant_name = variant_name

        cfg = NN_VARIANTS[variant_name]
        self.drift = _SDEDrift(cfg)
        self.diffusion = _SDEDiffusion(hidden_dim=max(32, cfg.hidden_dim // 4))

        # Interpolation state (shared convention with ODE models)
        self.t_series: torch.Tensor | None = None
        self.u_series: torch.Tensor | None = None
        self._u_func = None

    # ── interpolation helpers ─────────────────────────────────────────

    def set_series(self, t_series: torch.Tensor, u_series: torch.Tensor) -> None:
        self.t_series = t_series
        self.u_series = u_series

    def set_batch_start_times(self, batch_start_times: torch.Tensor | None) -> None:
        pass  # not used for SDE training (full-trajectory)

    def _interp_u_scalar(self, t: torch.Tensor) -> torch.Tensor:
        """Interpolate u at scalar time *t*."""
        if self.t_series is None or self.u_series is None:
            raise RuntimeError("Call set_series(t, u) first")
        t_val = t.item() if isinstance(t, torch.Tensor) and t.dim() == 0 else float(t)
        idx = torch.searchsorted(self.t_series, torch.tensor([t_val], device=self.t_series.device))
        idx = idx.clamp(1, len(self.t_series) - 1)
        t1 = self.t_series[idx - 1]
        t2 = self.t_series[idx]
        u1 = self.u_series[idx - 1]
        u2 = self.u_series[idx]
        denom = (t2 - t1).clamp_min(1e-6)
        alpha = (t_val - t1) / denom
        return (u1 + alpha * (u2 - u1)).squeeze(0)  # [1]

    # ── torchsde interface ────────────────────────────────────────────

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        u = self._interp_u_scalar(t)
        if u.dim() == 1:
            u = u.unsqueeze(0).expand(y.shape[0], -1)
        return self.drift(y, u)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        u = self._interp_u_scalar(t)
        if u.dim() == 1:
            u = u.unsqueeze(0).expand(y.shape[0], -1)
        return self.diffusion(y, u)

    # ── k-step training prediction ─────────────────────────────────────

    def predict_k_steps(self, tensors, start_idx, k_steps: int, obs_dim: int) -> torch.Tensor:
        """Predict k-step trajectories via SDE integration.

        Returns
        -------
        Tensor of shape ``[K, B, obs_dim]``
        """
        import torchsde

        device = tensors.t.device
        dt = float((tensors.t[1] - tensors.t[0]).item())
        t_eval = torch.linspace(0, (k_steps - 1) * dt, k_steps, device=device)

        x0 = tensors.y[start_idx]  # [B, 2]
        preds = []
        for b in range(len(start_idx)):
            t_local = t_eval + tensors.t[start_idx[b]]
            pred = torchsde.sdeint(self, x0[b : b + 1], t_local, method="euler", dt=dt)
            preds.append(pred[:, 0, :obs_dim])  # [K, obs_dim]

        return torch.stack(preds, dim=1)  # [K, B, obs_dim]

    # ── convenience rollout ───────────────────────────────────────────

    def rollout(
        self,
        t: torch.Tensor,
        u: torch.Tensor,
        y0: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Integrate the SDE over the full time vector. Returns ``[T, obs_dim]``."""
        import torchsde

        self.eval()
        self.to(device)
        self.set_series(t, u)

        x0 = y0.reshape(1, -1)
        with torch.no_grad():
            pred = torchsde.sdeint(self, x0, t, method="euler", dt=float(t[1] - t[0]))
        return pred.squeeze(1)  # [T, state_dim]
