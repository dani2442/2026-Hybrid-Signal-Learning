"""Temporal Convolutional Network (TCN) sequence model."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.config import TCNConfig
from src.models.registry import register_model
from src.models.sequence.base import SequenceModel


# ─────────────────────────────────────────────────────────────────────
# TCN building blocks
# ─────────────────────────────────────────────────────────────────────

class _CausalConv1d(nn.Module):
    """Causal (left-padded) 1-D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return self.dropout(self.relu(out))


class _TemporalBlock(nn.Module):
    """Residual block with two causal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(
            in_channels, out_channels, kernel_size, dilation, dropout
        )
        self.conv2 = _CausalConv1d(
            out_channels, out_channels, kernel_size, dilation, dropout
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return out + self.downsample(x)


class _TCNNetwork(nn.Module):
    """Full TCN: stack of temporal blocks → linear output."""

    def __init__(
        self,
        num_channels: List[int],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                _TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, 1) → permute to (batch, 1, seq) for Conv1d
        out = self.network(x.permute(0, 2, 1))
        out = out.permute(0, 2, 1)  # back to (batch, seq, channels)
        return self.fc(out)


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────

@register_model("tcn", TCNConfig)
class TCNModel(SequenceModel):
    """Temporal Convolutional Network for system identification."""

    def __init__(self, config: TCNConfig | None = None) -> None:
        super().__init__(config or TCNConfig())
        self.config: TCNConfig

    def _build_network(self) -> nn.Module:
        cfg = self.config
        return _TCNNetwork(cfg.num_channels, cfg.kernel_size, cfg.dropout)
