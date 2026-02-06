"""Control System Identification Library."""

from .models import (
    ARIMA,
    GRU,
    NARX,
    ExponentialSmoothing,
    HybridLinearBeam,
    HybridNonlinearCam,
    NeuralNetwork,
    NeuralODE,
    NeuralSDE,
    RandomForest,
)
from .data import Dataset
from .visualization import (
    plot_predictions,
    plot_spectrograms,
    plot_residuals,
    plot_signals,
    plot_model_comparison,
)
from .validation import Metrics

__version__ = "0.1.0"
__all__ = [
    "NARX",
    "ARIMA",
    "NeuralNetwork", 
    "NeuralODE",
    "NeuralSDE",
    "ExponentialSmoothing",
    "RandomForest",
    "GRU",
    "HybridLinearBeam",
    "HybridNonlinearCam",
    "Dataset",
    "Metrics",
    "plot_predictions",
    "plot_spectrograms",
    "plot_residuals",
    "plot_signals",
    "plot_model_comparison",
]
