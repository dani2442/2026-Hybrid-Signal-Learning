"""LSTM (Long Short-Term Memory) model for time series forecasting."""

from __future__ import annotations

from ..config import LSTMConfig
from .sequence_base import SequenceModel


class LSTM(SequenceModel):
    """LSTM network for sequence-to-one system identification.

    Uses three gates (input, forget, output) and a separate cell state.
    All shared logic lives in :class:`SequenceModel`.
    """

    def __init__(self, config: LSTMConfig | None = None, **kwargs):
        if config is None:
            config = LSTMConfig(**kwargs)
        super().__init__(config)

    @property
    def _model_returns_hidden(self) -> bool:
        return True

    def _build_model(self, input_size: int):
        import torch.nn as nn

        cfg = self.config

        class LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=cfg.hidden_size,
                    num_layers=cfg.num_layers,
                    batch_first=True,
                    dropout=cfg.dropout if cfg.num_layers > 1 else 0,
                )
                self.fc = nn.Linear(cfg.hidden_size, 1)

            def forward(self, x, hidden=None):
                out, hidden = self.lstm(x, hidden)
                return self.fc(out[:, -1, :]), hidden

        return LSTMModel()

