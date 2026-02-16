"""Base class for sequence (RNN / TCN / Mamba) models.

All sequence models share the same ``_fit`` / ``_predict`` skeleton:
normalise → build network → train with sliding windows → predict
on full sequence.  Only ``_build_network()`` differs.
"""

from __future__ import annotations

import numpy as np
import torch

from src.config import BaseConfig
from src.models.base import BaseModel, PickleStateMixin
from src.models.training import train_sequence_model


class SequenceModel(PickleStateMixin, BaseModel):
    """Abstract base for sequence-to-sequence models.

    Subclasses must implement :meth:`_build_network` which returns an
    ``nn.Module`` mapping ``(batch, seq, 1) → (batch, seq, 1)``.
    """

    def __init__(self, config: BaseConfig) -> None:
        super().__init__(config)
        self.network: torch.nn.Module | None = None

    # ── abstract ──────────────────────────────────────────────────────

    def _build_network(self) -> torch.nn.Module:
        """Return an uninitialised ``nn.Module``."""
        raise NotImplementedError

    # ── shared fit / predict ──────────────────────────────────────────

    def _fit(self, u, y, *, val_data=None, logger=None) -> None:
        u_norm = self._normalize_u(u)
        y_norm = self._normalize_y(y)

        self.network = self._build_network()

        train_sequence_model(
            self.network,
            u_norm,
            y_norm,
            config=self.config,
            logger=logger,
            device=self.device,
            val_data=val_data,
        )

    def _predict(self, u, *, y0=None) -> np.ndarray:
        u_norm = self._normalize_u(u)
        u_tensor = (
            torch.tensor(u_norm, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(-1)
        )

        self.network.eval()
        self.network.to(self.device)

        with torch.no_grad():
            pred = self.network(u_tensor)

        y_pred_norm = pred.squeeze().cpu().numpy()
        return self._denormalize_y(y_pred_norm)
