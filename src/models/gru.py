"""GRU (Gated Recurrent Unit) model for time series forecasting."""

from __future__ import annotations

from ..config import GRUConfig
from .sequence_base import SequenceModel


class GRU(SequenceModel):
    """GRU network for sequence-to-one system identification.

    All training, prediction, normalisation, save/load, and W&B logging
    logic lives in :class:`SequenceModel`.  This class only defines the
    architecture.
    """

    def __init__(self, config: GRUConfig | None = None, **kwargs):
        if config is None:
            config = GRUConfig(**kwargs)
        super().__init__(config)

    @property
    def _model_returns_hidden(self) -> bool:
        return True

    def _build_model(self, input_size: int):
        import torch.nn as nn

        cfg = self.config

        class GRUModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(
                    input_size=input_size,
                    hidden_size=cfg.hidden_size,
                    num_layers=cfg.num_layers,
                    batch_first=True,
                    dropout=cfg.dropout if cfg.num_layers > 1 else 0,
                )
                self.fc = nn.Linear(cfg.hidden_size, 1)

            def forward(self, x, hidden=None):
                out, hidden = self.gru(x, hidden)
                return self.fc(out[:, -1, :]), hidden

        return GRUModel()

