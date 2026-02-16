"""NARX (Nonlinear AutoRegressive with eXogenous inputs) model.

Uses FROLS (Forward Regression Orthogonal Least Squares) for automatic
polynomial term selection.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.config import NARXConfig
from src.models.base import BaseModel
from src.models.predict_utils import autoregressive_free_run, one_step_ahead
from src.models.registry import register_model
from src.utils.frols import frols
from src.utils.regression import reg_mat_narx


@register_model("narx", NARXConfig)
class NARXModel(BaseModel):
    """NARX polynomial model with FROLS term selection."""

    def __init__(self, config: NARXConfig | None = None) -> None:
        super().__init__(config or NARXConfig())
        self.config: NARXConfig
        self.theta: Optional[np.ndarray] = None
        self.selected_terms: Optional[list] = None
        self.term_names: Optional[list] = None
        self._reg_matrix: Optional[np.ndarray] = None
        self._all_colnames: Optional[list] = None

    def _fit(self, u, y, *, val_data=None, logger=None) -> None:
        cfg = self.config
        reg_matrix, colnames, target = reg_mat_narx(
            u, y, cfg.nu, cfg.ny, cfg.poly_order
        )
        self._all_colnames = colnames

        if reg_matrix.shape[0] == 0:
            raise ValueError("Not enough data for the given lag orders")

        result = frols(reg_matrix, target, cfg.selection_criteria, colnames)
        self.selected_terms = result["selected_indices"]
        self.theta = result["theta"]
        self.term_names = result["selected_colnames"]

        # Store for potential inspection
        self._reg_matrix = reg_matrix

    @property
    def max_lag(self) -> int:
        return max(self.config.ny, self.config.nu)

    def _predict_one_fn(self):
        """Return the single-step prediction callable."""
        cfg = self.config

        def _predict_one(features: np.ndarray) -> float:
            raw = features.ravel()
            poly = [1.0]
            poly.extend(raw.tolist())

            if cfg.poly_order >= 2:
                from itertools import combinations_with_replacement
                n_base = len(raw)
                for order in range(2, cfg.poly_order + 1):
                    for indices in combinations_with_replacement(range(n_base), order):
                        poly.append(float(np.prod([raw[i] for i in indices])))

            poly = np.array(poly)
            return float(poly[self.selected_terms] @ self.theta)

        return _predict_one

    def _predict(self, u, *, y0=None) -> np.ndarray:
        cfg = self.config
        return autoregressive_free_run(
            self._predict_one_fn(), u, cfg.ny, cfg.nu, y0=y0
        )

    def _predict_osa(self, u, *, y=None, y0=None) -> np.ndarray:
        cfg = self.config
        if y is None:
            return self._predict(u, y0=y0)
        return one_step_ahead(
            self._predict_one_fn(), u, y, cfg.ny, cfg.nu
        )
