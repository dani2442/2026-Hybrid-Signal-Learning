"""LSTM-based sequence model."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import LSTMConfig
from src.models.registry import register_model
from src.models.sequence.base import SequenceModel


class _LSTMNetwork(nn.Module):
    """LSTM encoder → linear decoder."""

    def __init__(self, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out)


@register_model("lstm", LSTMConfig)
class LSTMModel(SequenceModel):
    """LSTM sequence model for system identification."""

    def __init__(self, config: LSTMConfig | None = None) -> None:
        super().__init__(config or LSTMConfig())
        self.config: LSTMConfig

    def _build_network(self) -> nn.Module:
        cfg = self.config
        return _LSTMNetwork(cfg.hidden_size, cfg.num_layers, cfg.dropout)
