"""Abstract base class for system identification models."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all system identification models."""

    def __init__(self, nu: int = 1, ny: int = 1):
        """
        Initialize base model.
        
        Args:
            nu: Number of input lags
            ny: Number of output lags
        """
        self.nu = nu
        self.ny = ny
        self.max_lag = max(nu, ny)
        self._is_fitted = False

    @abstractmethod
    def fit(self, u: np.ndarray, y: np.ndarray) -> "BaseModel":
        """
        Fit the model to training data.
        
        Args:
            u: Input signal array
            y: Output signal array
            
        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        One-step-ahead prediction using actual past outputs.
        
        Args:
            u: Input signal array
            y: Actual output signal array (for lagged values)
            
        Returns:
            Predicted output array
        """
        pass

    @abstractmethod
    def predict_free_run(
        self, u: np.ndarray, y_initial: np.ndarray
    ) -> np.ndarray:
        """
        Free-run simulation using only initial conditions.
        
        Args:
            u: Input signal array for full simulation
            y_initial: Initial output conditions
            
        Returns:
            Simulated output array
        """
        pass

    def predict(
        self,
        u: np.ndarray,
        y: Optional[np.ndarray] = None,
        mode: str = "OSA",
    ) -> np.ndarray:
        """
        Unified prediction interface.
        
        Args:
            u: Input signal
            y: Output signal (required for OSA, initial conditions for FR)
            mode: 'OSA' for one-step-ahead, 'FR' for free-run simulation
            
        Returns:
            Predicted output array
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
            
        if mode == "OSA":
            if y is None:
                raise ValueError("y required for OSA prediction")
            return self.predict_osa(u, y)
        elif mode == "FR":
            if y is None:
                raise ValueError("Initial conditions y required for free-run")
            return self.predict_free_run(u, y)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'OSA' or 'FR'.")

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nu={self.nu}, ny={self.ny})"
