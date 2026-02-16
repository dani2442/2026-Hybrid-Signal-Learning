"""Random Forest regressor for system identification."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.config import RandomForestConfig
from src.models.base import BaseModel
from src.models.predict_utils import autoregressive_free_run, one_step_ahead
from src.models.registry import register_model
from src.utils.regression import create_lagged_features


@register_model("random_forest", RandomForestConfig)
class RandomForestModel(BaseModel):
    """Scikit-learn Random Forest with lagged I/O features."""

    def __init__(self, config: RandomForestConfig | None = None) -> None:
        super().__init__(config or RandomForestConfig())
        self.config: RandomForestConfig
        self._rf = None

    def _fit(self, u, y, *, val_data=None, logger=None) -> None:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:
            raise ImportError(
                "scikit-learn required: pip install scikit-learn"
            ) from exc

        cfg = self.config
        X, y_target = create_lagged_features(y, u, cfg.ny, cfg.nu)

        if X.shape[0] == 0:
            raise ValueError("Not enough data for the given lag orders")

        self._rf = RandomForestRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_split=cfg.min_samples_split,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.random_state,
        )
        self._rf.fit(X, y_target)

    @property
    def max_lag(self) -> int:
        return max(self.config.ny, self.config.nu)

    def _predict_one_fn(self):
        rf = self._rf
        def _predict_one(features: np.ndarray) -> float:
            return float(rf.predict(features.reshape(1, -1))[0])
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
