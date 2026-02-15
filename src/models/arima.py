"""ARIMA model for time series forecasting."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from ..config import ARIMAConfig
from .base import BaseModel, PickleStateMixin


class ARIMA(PickleStateMixin, BaseModel):
    """ARIMA / ARIMAX wrapper using statsmodels."""

    _pickle_attr = "fitted_model_"
    _pickle_key = "statsmodel"

    def __init__(self, config: ARIMAConfig | None = None, **kwargs):
        if config is None:
            config = ARIMAConfig(**kwargs)
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
        from statsmodels.tsa.arima.model import ARIMA as StatsARIMA

        exog = u.reshape(-1, 1) if self.nu > 0 else None
        model = StatsARIMA(y, order=self.config.order, exog=exog)
        self.fitted_model_ = model.fit()

    # ── prediction ────────────────────────────────────────────────────

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        exog = u.reshape(-1, 1) if self.nu > 0 else None
        pred = self.fitted_model_.get_prediction(exog=exog)
        return pred.predicted_mean[self.max_lag:]

    def predict_free_run(self, u: np.ndarray, y_initial: np.ndarray) -> np.ndarray:
        n_forecast = len(u) - self.max_lag
        exog_future = u[self.max_lag:].reshape(-1, 1) if self.nu > 0 else None
        forecast = self.fitted_model_.get_forecast(steps=n_forecast, exog=exog_future)
        return forecast.predicted_mean

