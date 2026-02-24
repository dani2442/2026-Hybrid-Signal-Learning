"""Sequence-to-sequence models: GRU, LSTM, TCN, Mamba.

All models map an input sequence u(t) of shape ``(batch, seq_len, 1)``
to a predicted state trajectory ``(batch, seq_len, 2)`` — i.e. [position, velocity].

These are *discrete-time* models trained with windowed MSE via the
unified ``train_model`` in ``train.py``.  Each model inherits
``predict_k_steps`` from ``SequenceModelBase``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import NN_VARIANTS, NnVariantConfig


# =====================================================================
# Base
# =====================================================================

class SequenceModelBase(nn.Module):
    """Thin base for all sequence models.  Input ``(B, T, 1)`` → ``(B, T, 2)``."""

    # Marker so training code can branch on model type.
    is_sequence_model: bool = True

    def set_series(self, t_series, u_series):  # noqa: ARG002
        """No-op (API compat with ODE models)."""

    def set_batch_start_times(self, batch_start_times):  # noqa: ARG002
        """No-op."""

    def predict_k_steps(self, tensors, start_idx, k_steps: int, obs_dim: int) -> torch.Tensor:
        """Predict k-step trajectories from input windows.

        Returns
        -------
        Tensor of shape ``[K, B, obs_dim]``
        """
        u_windows = torch.stack([tensors.u[i : i + k_steps] for i in start_idx])  # [B, K, 1]
        y_pred = self(u_windows)  # [B, K, 2]
        return y_pred.permute(1, 0, 2)[..., :obs_dim]


# =====================================================================
# GRU
# =====================================================================

class GRUSeqModel(SequenceModelBase):
    """GRU encoder → linear decoder for 2-D state prediction."""

    def __init__(self, variant_name: str = "base") -> None:
        super().__init__()
        if variant_name not in NN_VARIANTS:
            raise ValueError(f"Unknown variant '{variant_name}'")
        self.variant_name = variant_name
        cfg = NN_VARIANTS[variant_name]

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.depth,
            dropout=cfg.dropout if cfg.depth > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(cfg.hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out)


# =====================================================================
# LSTM
# =====================================================================

class LSTMSeqModel(SequenceModelBase):
    """LSTM encoder → linear decoder for 2-D state prediction."""

    def __init__(self, variant_name: str = "base") -> None:
        super().__init__()
        if variant_name not in NN_VARIANTS:
            raise ValueError(f"Unknown variant '{variant_name}'")
        self.variant_name = variant_name
        cfg = NN_VARIANTS[variant_name]

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.depth,
            dropout=cfg.dropout if cfg.depth > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(cfg.hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out)


# =====================================================================
# TCN
# =====================================================================

class _CausalConv1d(nn.Module):
    """Causal (left-padded) 1-D convolution."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=self.padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return self.dropout(self.relu(out))


class _TemporalBlock(nn.Module):
    """Residual block with two causal convolutions."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel_size, dilation, dropout)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel_size, dilation, dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + self.downsample(x)


class TCNSeqModel(SequenceModelBase):
    """Temporal Convolutional Network for 2-D state prediction."""

    def __init__(self, variant_name: str = "base") -> None:
        super().__init__()
        if variant_name not in NN_VARIANTS:
            raise ValueError(f"Unknown variant '{variant_name}'")
        self.variant_name = variant_name
        cfg = NN_VARIANTS[variant_name]

        kernel_size = 7
        channels = [cfg.hidden_dim] * cfg.depth

        layers: list[nn.Module] = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            layers.append(_TemporalBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=cfg.dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1) → Conv1d expects (B, C, T)
        out = self.network(x.permute(0, 2, 1))
        out = out.permute(0, 2, 1)  # → (B, T, C)
        return self.fc(out)


# =====================================================================
# Mamba (Selective SSM)
# =====================================================================

class _SelectiveSSM(nn.Module):
    """Simplified selective state-space block inspired by Mamba."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand_factor: int = 2) -> None:
        super().__init__()
        d_inner = d_model * expand_factor
        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

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
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand_factor: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = _SelectiveSSM(d_model, d_state, d_conv, expand_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


class MambaSeqModel(SequenceModelBase):
    """Mamba / selective SSM for 2-D state prediction."""

    def __init__(self, variant_name: str = "base") -> None:
        super().__init__()
        if variant_name not in NN_VARIANTS:
            raise ValueError(f"Unknown variant '{variant_name}'")
        self.variant_name = variant_name
        cfg = NN_VARIANTS[variant_name]

        d_model = cfg.hidden_dim
        d_state = 16
        d_conv = 4
        expand_factor = 2

        self.input_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList(
            [_MambaBlock(d_model, d_state, d_conv, expand_factor) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = self.dropout(block(h))
        return self.output_proj(self.norm(h))
