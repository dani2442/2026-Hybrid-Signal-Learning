"""ARIMA model for time series forecasting."""

from typing import Tuple
import numpy as np

from .base import BaseModel
from ._dependency_utils import is_binary_incompatibility_error


class ARIMA(BaseModel):
    """
    ARIMA model wrapper for system identification.
    
    Uses statsmodels ARIMA under the hood. For ARIMAX (with exogenous inputs),
    the input u is treated as an exogenous regressor.
    
    Args:
        order: (p, d, q) tuple - AR order, differencing, MA order
        nu: Number of input lags (for exogenous inputs)
    """

    def __init__(self, order: Tuple[int, int, int] = (1, 0, 1), nu: int = 0):
        p, d, q = order
        super().__init__(nu=nu, ny=max(p, q))
        self.order = order
        self.model_ = None
        self.fitted_model_ = None

    def fit(self, u: np.ndarray, y: np.ndarray) -> "ARIMA":
        """
        Fit ARIMA/ARIMAX model.
        
        Args:
            u: Input signal (exogenous variable)
            y: Output signal (endogenous variable)
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
        except ImportError as exc:
            if is_binary_incompatibility_error(exc):
                raise RuntimeError(
                    "Binary incompatibility between NumPy and compiled dependencies "
                    "(statsmodels/pandas/pyarrow). Reinstall compatible versions, e.g.:\n"
                    "  python -m pip install --upgrade --force-reinstall "
                    "\"numpy<2\" pandas pyarrow statsmodels"
                ) from exc
            raise ImportError(
                "statsmodels required. Install with: pip install statsmodels"
            ) from exc
        except Exception as exc:
            if is_binary_incompatibility_error(exc):
                raise RuntimeError(
                    "Binary incompatibility between NumPy and compiled dependencies "
                    "(statsmodels/pandas/pyarrow). Reinstall compatible versions, e.g.:\n"
                    "  python -m pip install --upgrade --force-reinstall "
                    "\"numpy<2\" pandas pyarrow statsmodels"
                ) from exc
            raise

        y = np.asarray(y, dtype=float)
        u = np.asarray(u, dtype=float)

        # Use exogenous if nu > 0
        exog = u.reshape(-1, 1) if self.nu > 0 else None

        self.model_ = StatsARIMA(y, order=self.order, exog=exog)
        self.fitted_model_ = self.model_.fit()
        self._is_fitted = True

        return self

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction."""
        if self.fitted_model_ is None:
            raise RuntimeError("Model not fitted")

        y = np.asarray(y, dtype=float)
        u = np.asarray(u, dtype=float)
        
        # Get in-sample predictions
        exog = u.reshape(-1, 1) if self.nu > 0 else None
        
        # Use get_prediction for in-sample predictions
        pred = self.fitted_model_.get_prediction(exog=exog)
        y_hat = pred.predicted_mean
        
        # Align with max_lag
        return y_hat[self.max_lag:]

    def predict_free_run(
        self, u: np.ndarray, y_initial: np.ndarray
    ) -> np.ndarray:
        """
        Free-run forecast.
        
        Note: ARIMA free-run is inherently different from NARX - it uses
        the model's internal state rather than recursively applying predictions.
        """
        if self.fitted_model_ is None:
            raise RuntimeError("Model not fitted")

        u = np.asarray(u, dtype=float)
        n_forecast = len(u) - self.max_lag

        exog_future = u[self.max_lag:].reshape(-1, 1) if self.nu > 0 else None
        
        forecast = self.fitted_model_.get_forecast(steps=n_forecast, exog=exog_future)
        return forecast.predicted_mean

    def summary(self) -> str:
        """Print model summary."""
        if not self._is_fitted:
            return "Model not fitted"
        return str(self.fitted_model_.summary())

    def __repr__(self) -> str:
        return f"ARIMA(order={self.order}, nu={self.nu})"
