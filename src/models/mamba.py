"""Mamba (Selective State Space Model) for time series forecasting.

Pure-PyTorch implementation of the S6 / Mamba architecture.
Reference: Gu & Dao, arXiv:2312.00752
"""

from __future__ import annotations

from ..config import MambaConfig
from .sequence_base import SequenceModel


class Mamba(SequenceModel):
    """Mamba with selective state-space blocks.

    Architecture lives entirely in ``_build_model``; all training,
    prediction, normalisation, and save/load are in :class:`SequenceModel`.
    """

    def __init__(self, config: MambaConfig | None = None, **kwargs):
        if config is None:
            config = MambaConfig(**kwargs)
        super().__init__(config)

    def _build_model(self, input_size: int):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        cfg = self.config

        class SelectiveSSM(nn.Module):
            def __init__(self, d_inner, d_state):
                super().__init__()
                self.d_state = d_state
                self.A_log = nn.Parameter(
                    torch.log(
                        torch.arange(1, d_state + 1, dtype=torch.float32)
                        .unsqueeze(0).expand(d_inner, -1).clone()
                    )
                )
                self.D = nn.Parameter(torch.ones(d_inner))
                self.proj_delta = nn.Linear(d_inner, d_inner, bias=True)
                self.proj_B = nn.Linear(d_inner, d_state, bias=False)
                self.proj_C = nn.Linear(d_inner, d_state, bias=False)

            def forward(self, x):
                B_batch, L, D = x.shape
                A = -torch.exp(self.A_log)
                delta = F.softplus(self.proj_delta(x))
                B_mat = self.proj_B(x)
                C_mat = self.proj_C(x)
                deltaA = torch.exp(delta.unsqueeze(-1) * A)
                deltaB = delta.unsqueeze(-1) * B_mat.unsqueeze(2)
                ys = torch.zeros(B_batch, L, D, device=x.device, dtype=x.dtype)
                h = torch.zeros(B_batch, D, self.d_state, device=x.device, dtype=x.dtype)
                for t in range(L):
                    h = deltaA[:, t] * h + deltaB[:, t] * x[:, t].unsqueeze(-1)
                    ys[:, t] = (h * C_mat[:, t].unsqueeze(1)).sum(dim=-1)
                return ys + x * self.D

        class MambaBlock(nn.Module):
            def __init__(self, d_model, d_state, d_conv, expand, dropout):
                super().__init__()
                d_inner = d_model * expand
                self.norm = nn.LayerNorm(d_model)
                self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
                self.conv1d = nn.Conv1d(
                    d_inner, d_inner, kernel_size=d_conv,
                    padding=d_conv - 1, groups=d_inner, bias=True,
                )
                self.ssm = SelectiveSSM(d_inner, d_state)
                self.out_proj = nn.Linear(d_inner, d_model, bias=False)
                self.drop = nn.Dropout(dropout)

            def forward(self, x):
                residual = x
                x = self.norm(x)
                x_ssm, z = self.in_proj(x).chunk(2, dim=-1)
                x_ssm = x_ssm.permute(0, 2, 1)
                x_ssm = self.conv1d(x_ssm)[:, :, :residual.size(1)]
                x_ssm = F.silu(x_ssm.permute(0, 2, 1))
                x_ssm = self.ssm(x_ssm)
                out = self.out_proj(self.drop(x_ssm * F.silu(z)))
                return out + residual

        class MambaModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(input_size, cfg.d_model)
                self.blocks = nn.ModuleList([
                    MambaBlock(cfg.d_model, cfg.d_state, cfg.d_conv,
                               cfg.expand_factor, cfg.dropout)
                    for _ in range(cfg.n_layers)
                ])
                self.norm = nn.LayerNorm(cfg.d_model)
                self.head = nn.Linear(cfg.d_model, 1)

            def forward(self, x):
                x = self.embed(x)
                for block in self.blocks:
                    x = block(x)
                return self.head(self.norm(x)[:, -1, :])

        return MambaModel()
