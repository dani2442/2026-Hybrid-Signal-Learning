"""Control System Identification Library."""

from .models import NARX, ARIMA, NeuralNetwork, NeuralODE, ExponentialSmoothing, RandomForest, GRU
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
    "ExponentialSmoothing",
    "RandomForest",
    "GRU",
    "Dataset",
    "Metrics",
    "plot_predictions",
    "plot_spectrograms",
    "plot_residuals",
    "plot_signals",
    "plot_model_comparison",
]
