"""GRU-based sequence model."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import GRUConfig
from src.models.registry import register_model
from src.models.sequence.base import SequenceModel


class _GRUNetwork(nn.Module):
    """GRU encoder → linear decoder."""

    def __init__(self, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out)


@register_model("gru", GRUConfig)
class GRUModel(SequenceModel):
    """GRU sequence model for system identification."""

    def __init__(self, config: GRUConfig | None = None) -> None:
        super().__init__(config or GRUConfig())
        self.config: GRUConfig

    def _build_network(self) -> nn.Module:
        cfg = self.config
        return _GRUNetwork(cfg.hidden_size, cfg.num_layers, cfg.dropout)
