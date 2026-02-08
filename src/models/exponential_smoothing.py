"""Exponential Smoothing models for time series forecasting."""

from typing import Optional, Tuple
import numpy as np

from .base import BaseModel


class ExponentialSmoothing(BaseModel):
    """
    Holt-Winters Exponential Smoothing model for system identification.
    
    Uses statsmodels ExponentialSmoothing under the hood.
    
    Args:
        trend: Type of trend component ('add', 'mul', or None)
        seasonal: Type of seasonal component ('add', 'mul', or None)
        seasonal_periods: Number of periods in a complete seasonal cycle
        nu: Number of input lags (for compatibility, inputs are not used directly)
    """

    def __init__(
        self,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        nu: int = 0,
    ):
        super().__init__(nu=nu, ny=1)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model_ = None
        self.fitted_model_ = None
        self._y_train: Optional[np.ndarray] = None

    def fit(self, u: np.ndarray, y: np.ndarray) -> "ExponentialSmoothing":
        """
        Fit Exponential Smoothing model.
        
        Args:
            u: Input signal (not used directly, for API compatibility)
            y: Output signal (endogenous variable)
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing as StatsES
        except ImportError:
            raise ImportError("statsmodels required. Install with: pip install statsmodels")

        y = np.asarray(y, dtype=float)
        self._y_train = y

        self.model_ = StatsES(
            y,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        )
        self.fitted_model_ = self.model_.fit(optimized=True)
        self._is_fitted = True

        return self

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction (in-sample fitted values)."""
        if self.fitted_model_ is None:
            raise RuntimeError("Model not fitted")

        y = np.asarray(y, dtype=float)
        
        # Get in-sample fitted values
        y_hat = self.fitted_model_.fittedvalues
        
        # Align with max_lag
        return y_hat[self.max_lag:]

    def predict_free_run(
        self, u: np.ndarray, y_initial: np.ndarray
    ) -> np.ndarray:
        """
        Free-run forecast.
        
        Uses the model's forecast method for out-of-sample prediction.
        """
        if self.fitted_model_ is None:
            raise RuntimeError("Model not fitted")

        u = np.asarray(u, dtype=float)
        n_forecast = len(u) - self.max_lag

        forecast = self.fitted_model_.forecast(steps=n_forecast)
        return np.asarray(forecast)

    def summary(self) -> str:
        """Print model summary."""
        if not self._is_fitted:
            return "Model not fitted"
        return str(self.fitted_model_.summary())

    def __repr__(self) -> str:
        return f"ExponentialSmoothing(trend={self.trend}, seasonal={self.seasonal})"
