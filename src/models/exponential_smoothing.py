"""Exponential Smoothing models for time series forecasting."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from ..config import ExponentialSmoothingConfig
from .base import BaseModel, PickleStateMixin


class ExponentialSmoothing(PickleStateMixin, BaseModel):
    """Holt-Winters Exponential Smoothing wrapper using statsmodels."""

    _pickle_attr = "fitted_model_"
    _pickle_key = "statsmodel"

    def __init__(self, config: ExponentialSmoothingConfig | None = None, **kwargs):
        if config is None:
            config = ExponentialSmoothingConfig(**kwargs)
        super().__init__(config)
        self.fitted_model_ = None

    # ── training ──────────────────────────────────────────────────────

    def _fit(
        self,
        u: np.ndarray,
        y: np.ndarray,
        *,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        logger: Any = None,
    ) -> None:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as StatsES

        cfg = self.config
        model = StatsES(
            y,
            trend=cfg.trend,
            seasonal=cfg.seasonal,
            seasonal_periods=cfg.seasonal_periods,
        )
        self.fitted_model_ = model.fit(optimized=True)

    # ── prediction ────────────────────────────────────────────────────

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(self.fitted_model_.fittedvalues[self.max_lag:])

    def predict_free_run(self, u: np.ndarray, y_initial: np.ndarray) -> np.ndarray:
        n_forecast = len(u) - self.max_lag
        return np.asarray(self.fitted_model_.forecast(steps=n_forecast))

