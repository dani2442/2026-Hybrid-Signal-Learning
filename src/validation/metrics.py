"""Model validation metrics for system identification."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _align_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    skip: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Align arrays to the shorter length, optionally skipping an initial transient.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Ground truth and prediction arrays.
    skip : int
        Number of leading samples to discard (e.g. ``max_lag`` for AR
        models where the first samples are initial conditions, not
        real predictions).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = min(len(y_true), len(y_pred))
    # Both arrays are compared over the SAME index range [skip:n]
    start = min(skip, n)
    return y_true[start:n], y_pred[start:n]


def mse(y_true: np.ndarray, y_pred: np.ndarray, skip: int = 0) -> float:
    """Mean Squared Error."""
    yt, yp = _align_arrays(y_true, y_pred, skip)
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray, skip: int = 0) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred, skip)))


def mae(y_true: np.ndarray, y_pred: np.ndarray, skip: int = 0) -> float:
    """Mean Absolute Error."""
    yt, yp = _align_arrays(y_true, y_pred, skip)
    return float(np.mean(np.abs(yt - yp)))


def r2(y_true: np.ndarray, y_pred: np.ndarray, skip: int = 0) -> float:
    """R-squared (coefficient of determination)."""
    yt, yp = _align_arrays(y_true, y_pred, skip)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, skip: int = 0) -> float:
    """Normalized RMSE (by range)."""
    yt, yp = _align_arrays(y_true, y_pred, skip)
    y_range = np.max(yt) - np.min(yt)
    if y_range == 0:
        return 0.0
    return float(rmse(yt, yp) / y_range)


def fit_index(y_true: np.ndarray, y_pred: np.ndarray, skip: int = 0) -> float:
    """MATLAB-style FIT index (0–1 range).

    ``FIT = 1 - ||y - ŷ|| / ||y - mean(y)||``
    """
    yt, yp = _align_arrays(y_true, y_pred, skip)
    norm_err = np.linalg.norm(yt - yp)
    norm_ref = np.linalg.norm(yt - np.mean(yt))
    if norm_ref == 0:
        return 1.0
    return float(1 - norm_err / norm_ref)


def compute_all(y_true: np.ndarray, y_pred: np.ndarray, skip: int = 0) -> Dict[str, float]:
    """Compute all metrics and return as a dict.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground truth and predictions (same length).
    skip : int
        Number of leading samples to discard from both arrays before
        computing metrics (e.g. ``model.max_lag``).
    """
    return {
        "MSE": mse(y_true, y_pred, skip),
        "RMSE": rmse(y_true, y_pred, skip),
        "MAE": mae(y_true, y_pred, skip),
        "R2": r2(y_true, y_pred, skip),
        "NRMSE": nrmse(y_true, y_pred, skip),
        "FIT": fit_index(y_true, y_pred, skip),
    }


def summary(y_true: np.ndarray, y_pred: np.ndarray, name: str = "Model", skip: int = 0) -> str:
    """Return a formatted metrics summary string."""
    m = compute_all(y_true, y_pred, skip)
    return (
        f"\n{'=' * 40}\n"
        f"Metrics for: {name}\n"
        f"{'=' * 40}\n"
        f"  MSE:    {m['MSE']:.6f}\n"
        f"  RMSE:   {m['RMSE']:.6f}\n"
        f"  MAE:    {m['MAE']:.6f}\n"
        f"  R²:     {m['R2']:.4f}\n"
        f"  NRMSE:  {m['NRMSE']:.4f}\n"
        f"  FIT:    {m['FIT']:.4f}\n"
        f"{'=' * 40}"
    )


# Backward-compatible class wrapper
class Metrics:
    """Static facade for metric functions (backward compatibility)."""

    mse = staticmethod(mse)
    rmse = staticmethod(rmse)
    mae = staticmethod(mae)
    r2 = staticmethod(r2)
    nrmse = staticmethod(nrmse)
    fit_index = staticmethod(fit_index)
    compute_all = staticmethod(compute_all)
    summary = staticmethod(summary)
