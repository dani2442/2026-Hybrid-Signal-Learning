"""Base ODE module and shared building blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class NnVariantConfig:
    hidden_dim: int
    depth: int
    dropout: float = 0.05


NN_VARIANTS: dict[str, NnVariantConfig] = {
    "compact": NnVariantConfig(hidden_dim=64, depth=2, dropout=0.05),
    "base": NnVariantConfig(hidden_dim=128, depth=3, dropout=0.05),
    "wide": NnVariantConfig(hidden_dim=256, depth=3, dropout=0.05),
    "deep": NnVariantConfig(hidden_dim=128, depth=5, dropout=0.05),
}


# Model keys that accept an NN variant
_NN_VARIANT_KEYS: set[str] = {
    "blackbox",
    "structured_blackbox",
    "adaptive_blackbox",
    "hybrid_joint",
    "hybrid_joint_stribeck",
    "hybrid_frozen",
    "hybrid_frozen_stribeck",
    "ude",
    "neural_sde",
    "gru",
    "lstm",
    "tcn",
    "mamba",
    "feedforward_nn",
}


def uses_nn_variant(model_key: str) -> bool:
    return model_key in _NN_VARIANT_KEYS


class InterpNeuralODEBase(nn.Module):
    """torchdiffeq-compatible base with piecewise-linear interpolation of u(t)."""

    def __init__(self) -> None:
        super().__init__()
        self.u_series: torch.Tensor | None = None
        self.t_series: torch.Tensor | None = None
        self.batch_start_times: torch.Tensor | None = None

    def set_series(self, t_series: torch.Tensor, u_series: torch.Tensor) -> None:
        if t_series.ndim != 1:
            raise ValueError("t_series must be 1D")
        if u_series.ndim != 2 or u_series.shape[1] != 1:
            raise ValueError("u_series must have shape (N,1)")
        if t_series.shape[0] != u_series.shape[0]:
            raise ValueError("t_series and u_series length mismatch")
        self.t_series = t_series
        self.u_series = u_series

    def set_batch_start_times(self, batch_start_times: torch.Tensor | None) -> None:
        self.batch_start_times = batch_start_times

    def _as_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.ndim == 2:
            return x, False
        if x.ndim == 1:
            return x.unsqueeze(0), True
        raise ValueError(f"Expected x with ndim 1 or 2, got {x.ndim}")

    def _interp_u(self, t: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
        if self.t_series is None or self.u_series is None:
            raise RuntimeError("Call set_series(t_series, u_series) before integration")

        if self.batch_start_times is not None:
            t_abs = self.batch_start_times + t
        else:
            t_abs = t * torch.ones_like(x_batch[:, 0:1])

        k_idx = torch.searchsorted(self.t_series, t_abs.reshape(-1), right=True)
        k_idx = torch.clamp(k_idx, 1, len(self.t_series) - 1)

        t1 = self.t_series[k_idx - 1].unsqueeze(1)
        t2 = self.t_series[k_idx].unsqueeze(1)
        u1 = self.u_series[k_idx - 1]
        u2 = self.u_series[k_idx]

        denom = t2 - t1
        denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
        alpha = (t_abs - t1) / denom
        return u1 + alpha * (u2 - u1)

    def predict_k_steps(self, tensors, start_idx, k_steps: int, obs_dim: int) -> torch.Tensor:
        """Predict k-step trajectories via ODE integration.

        Returns
        -------
        Tensor of shape ``[K, B, obs_dim]``
        """
        from torchdiffeq import odeint

        dt = float((tensors.t[1] - tensors.t[0]).item())
        t_eval = torch.arange(0, k_steps * dt, dt, device=tensors.t.device)

        x0 = tensors.y[start_idx]
        if getattr(self, "augmented_state", False):
            x0 = self.prepare_x0(x0)
        self.set_batch_start_times(tensors.t[start_idx].reshape(-1, 1))

        pred = odeint(self, x0, t_eval, method="rk4")
        return pred[..., :obs_dim]


def _build_selu_mlp(input_dim: int, output_dim: int, variant: NnVariantConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim

    for _ in range(variant.depth):
        layers.append(nn.Linear(in_dim, variant.hidden_dim))
        layers.append(nn.SELU())
        if variant.dropout > 0.0:
            layers.append(nn.AlphaDropout(variant.dropout))
        in_dim = variant.hidden_dim

    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)
