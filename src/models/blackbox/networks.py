"""Shared network blocks for blackbox 2-D ODE/SDE/CDE models."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def _selu_block(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """SELU-activated MLP block."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.SELU(),
        nn.Linear(hidden_dim, hidden_dim), nn.SELU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _tanh_block(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Tanh-activated MLP block."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, out_dim),
    )


# ─────────────────────────────────────────────────────────────────────
# ODE function variants (used by blackbox ODE and SDE drift)
# ─────────────────────────────────────────────────────────────────────

class VanillaODEFunc(nn.Module):
    """Single MLP: ``f(y, u) → ẏ``."""

    def __init__(self, state_dim: int, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _selu_block(state_dim + input_dim, hidden_dim, state_dim)

    def forward(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([y, u], dim=-1))


class StructuredODEFunc(nn.Module):
    """Structured: ``ẏ₁ = y₂``, ``ẏ₂ = f(y, u)``."""

    def __init__(self, state_dim: int, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.accel_net = _selu_block(state_dim + input_dim, hidden_dim, 1)
        self.state_dim = state_dim

    def forward(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        accel = self.accel_net(torch.cat([y, u], dim=-1))
        vel = y[..., 1:2]
        return torch.cat([vel, accel], dim=-1)


class AdaptiveODEFunc(nn.Module):
    """Adaptive: structured + learned damping/stiffness terms."""

    def __init__(self, state_dim: int, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.nn_term = _selu_block(state_dim + input_dim, hidden_dim, 1)
        self.damping = nn.Parameter(torch.tensor(0.1))
        self.stiffness = nn.Parameter(torch.tensor(1.0))
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        pos = y[..., 0:1]
        vel = y[..., 1:2]
        nn_out = self.nn_term(torch.cat([y, u], dim=-1))
        accel = (
            -self.damping * vel
            - self.stiffness * pos
            + self.gain * u
            + nn_out
        )
        return torch.cat([vel, accel], dim=-1)


# ─────────────────────────────────────────────────────────────────────
# Diffusion function variants (used by blackbox SDE)
# ─────────────────────────────────────────────────────────────────────

class DiagonalDiffusion(nn.Module):
    """State-dependent diagonal diffusion: ``σ(y, u) → [σ₁, σ₂]``."""

    def __init__(self, state_dim: int, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _tanh_block(state_dim + input_dim, hidden_dim, state_dim)

    def forward(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([y, u], dim=-1))
