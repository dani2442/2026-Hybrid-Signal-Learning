"""Mamba / selective state-space sequence model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import MambaConfig
from src.models.registry import register_model
from src.models.sequence.base import SequenceModel


# ─────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────

class _SelectiveSSM(nn.Module):
    """Simplified selective state-space block inspired by Mamba."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
    ) -> None:
        super().__init__()
        d_inner = d_model * expand_factor
        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner,
        )
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # Learnable log-scale A matrix
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        bdt = self.x_proj(x_conv)
        B_param, C_param = bdt.split(self.d_state, dim=-1)
        dt = F.softplus(self.dt_proj(x_conv))

        A = -torch.exp(self.A_log)
        y = self._ssm_scan(x_conv, dt, A, B_param, C_param)
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        return self.out_proj(y * F.silu(z))

    def _ssm_scan(self, x, dt, A, B, C):
        B_sz, L, D = x.shape
        N = self.d_state
        h = torch.zeros(B_sz, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            dt_i = dt[:, i, :].unsqueeze(-1)
            dA = torch.exp(dt_i * A.unsqueeze(0))
            dB = dt_i * B[:, i, :].unsqueeze(1)
            h = dA * h + dB * x[:, i, :].unsqueeze(-1)
            y_i = (h * C[:, i, :].unsqueeze(1)).sum(dim=-1)
            ys.append(y_i)
        return torch.stack(ys, dim=1)


class _MambaBlock(nn.Module):
    """Single Mamba block with layer norm."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand_factor: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = _SelectiveSSM(d_model, d_state, d_conv, expand_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


class _MambaNetwork(nn.Module):
    """Full Mamba: input projection → Mamba blocks → output projection."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        n_layers: int,
        expand_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList(
            [_MambaBlock(d_model, d_state, d_conv, expand_factor) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = self.dropout(block(h))
        return self.output_proj(self.norm(h))


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────

@register_model("mamba", MambaConfig)
class MambaModel(SequenceModel):
    """Mamba / selective SSM for system identification."""

    def __init__(self, config: MambaConfig | None = None) -> None:
        super().__init__(config or MambaConfig())
        self.config: MambaConfig

    def _build_network(self) -> nn.Module:
        cfg = self.config
        return _MambaNetwork(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            n_layers=cfg.n_layers,
            expand_factor=cfg.expand_factor,
            dropout=cfg.dropout,
        )
