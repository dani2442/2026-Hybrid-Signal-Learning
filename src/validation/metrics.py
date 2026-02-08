"""Model validation metrics for system identification."""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class Metrics:
    """
    Validation metrics for system identification models.
    
    Computes and stores various error metrics for model evaluation.
    """

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        y_true, y_pred = _align_arrays(y_true, y_pred)
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(Metrics.mse(y_true, y_pred)))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        y_true, y_pred = _align_arrays(y_true, y_pred)
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared (coefficient of determination)."""
        y_true, y_pred = _align_arrays(y_true, y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    @staticmethod
    def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Normalized RMSE (by range)."""
        y_true, y_pred = _align_arrays(y_true, y_pred)
        y_range = np.max(y_true) - np.min(y_true)
        if y_range == 0:
            return 0.0
        return float(Metrics.rmse(y_true, y_pred) / y_range)

    @staticmethod
    def fit_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        MATLAB-style FIT (normalized to 0-1 range).
        
        FIT = 1 - norm(y-ŷ)/norm(y-mean(y))
        """
        y_true, y_pred = _align_arrays(y_true, y_pred)
        y_mean = np.mean(y_true)
        norm_err = np.linalg.norm(y_true - y_pred)
        norm_ref = np.linalg.norm(y_true - y_mean)
        if norm_ref == 0:
            return 1.0
        return float(1 - norm_err / norm_ref)

    @staticmethod
    def compute_all(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            y_true: Actual output
            y_pred: Predicted output
            
        Returns:
            Dictionary with all metrics
        """
        return {
            "MSE": Metrics.mse(y_true, y_pred),
            "RMSE": Metrics.rmse(y_true, y_pred),
            "MAE": Metrics.mae(y_true, y_pred),
            "R2": Metrics.r2(y_true, y_pred),
            "NRMSE": Metrics.nrmse(y_true, y_pred),
            "FIT%": Metrics.fit_percent(y_true, y_pred),
        }

    @staticmethod
    def summary(
        y_true: np.ndarray, y_pred: np.ndarray, name: str = "Model"
    ) -> str:
        """
        Print formatted metrics summary.
        
        Args:
            y_true: Actual output
            y_pred: Predicted output
            name: Model name for display
            
        Returns:
            Formatted string
        """
        metrics = Metrics.compute_all(y_true, y_pred)
        
        lines = [
            f"\n{'=' * 40}",
            f"Metrics for: {name}",
            f"{'=' * 40}",
            f"  MSE:    {metrics['MSE']:.6f}",
            f"  RMSE:   {metrics['RMSE']:.6f}",
            f"  MAE:    {metrics['MAE']:.6f}",
            f"  R²:     {metrics['R2']:.4f}",
            f"  NRMSE:  {metrics['NRMSE']:.4f}",
            f"  FIT%:   {metrics['FIT%']:.2f}%",
            f"{'=' * 40}",
        ]
        
        summary = "\n".join(lines)
        print(summary)
        return summary


def _align_arrays(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple:
    """Align arrays to same length."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = min(len(y_true), len(y_pred))
    return y_true[:n], y_pred[:n]


def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models.
    
    Args:
        y_true: Actual output
        predictions: Dict of {model_name: predicted_output}
        
    Returns:
        Dict of {model_name: {metric_name: value}}
    """
    results = {}
    for name, y_pred in predictions.items():
        results[name] = Metrics.compute_all(y_true, y_pred)
        Metrics.summary(y_true, y_pred, name)
    return results
