"""Autoregressive free-run prediction for discrete models."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def autoregressive_free_run(
    predict_fn: Callable[[np.ndarray], float],
    u: np.ndarray,
    ny: int,
    nu: int,
    y0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Free-run (multi-step-ahead) simulation for discrete AR-like models.

    The prediction function receives a feature vector
    ``[y(k-1), ..., y(k-ny), u(k-1), ..., u(k-nu)]`` and returns a
    scalar prediction for *y(k)*.

    Parameters
    ----------
    predict_fn : callable
        ``predict_fn(features_1d) -> float``
    u : np.ndarray
        Input signal (1-D, length *N*).
    ny : int
        Number of output lags.
    nu : int
        Number of input lags.
    y0 : np.ndarray | None
        Initial output values (length ``ny``).  Zero-filled when absent.

    Returns
    -------
    np.ndarray
        Predicted output, same length as *u*.
    """
    u = np.asarray(u, dtype=np.float64).ravel()
    N = len(u)
    max_lag = max(ny, nu) if (ny > 0 or nu > 0) else 0
    y_pred = np.zeros(N, dtype=np.float64)

    if y0 is not None:
        y0 = np.asarray(y0, dtype=np.float64).ravel()
        y_pred[: min(len(y0), N)] = y0[: min(len(y0), N)]

    for k in range(max_lag, N):
        features = []
        for j in range(1, ny + 1):
            features.append(y_pred[k - j])
        for j in range(1, nu + 1):
            features.append(u[k - j])
        y_pred[k] = float(predict_fn(np.array(features)))

    return y_pred


def one_step_ahead(
    predict_fn: Callable[[np.ndarray], float],
    u: np.ndarray,
    y: np.ndarray,
    ny: int,
    nu: int,
) -> np.ndarray:
    """One-step-ahead prediction using *true* past outputs.

    At each step ``k``, the feature vector is built from the *true*
    ``y[k-1], …, y[k-ny]`` and ``u[k-1], …, u[k-nu]``, yielding
    the best possible prediction at every time step.

    Parameters
    ----------
    predict_fn : callable
        ``predict_fn(features_1d) -> float``
    u : np.ndarray
        Input signal (1-D, length *N*).
    y : np.ndarray
        True output signal (1-D, same length as ``u``).
    ny, nu : int
        Number of output / input lags.

    Returns
    -------
    np.ndarray
        Predicted output, same length as *u*.
    """
    u = np.asarray(u, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    N = len(u)
    max_lag = max(ny, nu) if (ny > 0 or nu > 0) else 0
    y_pred = np.copy(y)  # copy true values (initial lags are kept)

    for k in range(max_lag, N):
        features = []
        for j in range(1, ny + 1):
            features.append(y[k - j])  # true past outputs
        for j in range(1, nu + 1):
            features.append(u[k - j])
        y_pred[k] = float(predict_fn(np.array(features)))

    return y_pred
