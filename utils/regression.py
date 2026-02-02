"""Regression matrix utilities for system identification."""

from itertools import combinations_with_replacement
from typing import List, Tuple
import numpy as np


def reg_mat_arx(
    y: np.ndarray, u: np.ndarray, ny: int, nu: int
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Create ARX regression matrix and target vector.
    
    Args:
        y: Output signal
        u: Input signal
        ny: Number of output lags
        nu: Number of input lags
        
    Returns:
        Tuple of (regressor_matrix, column_names, target_vector)
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)

    if len(y) != len(u):
        raise ValueError("Input signals must have same length")
    if ny < 0 or nu < 0:
        raise ValueError("Lags must be non-negative")

    max_lag = max(ny, nu) if (ny > 0 or nu > 0) else 0
    n_samples = len(y)
    n_rows = n_samples - max_lag

    # Column names
    colnames = []
    if ny > 0:
        colnames.extend([f"y(k-{i})" for i in range(1, ny + 1)])
    if nu > 0:
        colnames.extend([f"u(k-{i})" for i in range(1, nu + 1)])

    if n_rows <= 0:
        return np.empty((0, len(colnames))), colnames, np.empty(0)

    # Build regression matrix
    regressor = np.zeros((n_rows, ny + nu))
    for k in range(max_lag, n_samples):
        row_idx = k - max_lag
        col = 0
        for j in range(1, ny + 1):
            regressor[row_idx, col] = y[k - j]
            col += 1
        for j in range(1, nu + 1):
            regressor[row_idx, col] = u[k - j]
            col += 1

    target = y[max_lag:]
    return regressor, colnames, target


def reg_mat_narx(
    u: np.ndarray, y: np.ndarray, nu: int, ny: int, poly_order: int
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Create NARX polynomial regression matrix.
    
    Args:
        u: Input signal
        y: Output signal
        nu: Number of input lags
        ny: Number of output lags
        poly_order: Polynomial order (>= 1)
        
    Returns:
        Tuple of (regressor_matrix, column_names, target_vector)
    """
    if poly_order < 1:
        raise ValueError("Polynomial order must be >= 1")

    p0_data, p0_names, target = reg_mat_arx(y, u, ny, nu)
    n_rows = len(target)
    n_base = p0_data.shape[1]

    # Start with constant term
    columns = [np.ones((n_rows, 1))] if n_rows > 0 else [np.empty((0, 1))]
    colnames = ["constant"]

    # Add base terms
    columns.append(p0_data)
    colnames.extend(p0_names)

    # Add polynomial terms
    if poly_order >= 2 and n_base > 0 and n_rows > 0:
        for order in range(2, poly_order + 1):
            for indices in combinations_with_replacement(range(n_base), order):
                name = "".join(p0_names[i] for i in indices)
                colnames.append(name)
                term = np.prod(p0_data[:, list(indices)], axis=1, keepdims=True)
                columns.append(term)

    if n_rows == 0:
        return np.empty((0, len(colnames))), colnames, target

    return np.concatenate(columns, axis=1), colnames, target


def create_lagged_features(
    y: np.ndarray, u: np.ndarray, ny: int, nu: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lagged feature matrix for neural network models.
    
    Args:
        y: Output signal
        u: Input signal
        ny: Number of output lags
        nu: Number of input lags
        
    Returns:
        Tuple of (feature_matrix, target_vector)
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    
    max_lag = max(ny, nu)
    n_samples = len(y)
    n_rows = n_samples - max_lag

    if n_rows <= 0:
        return np.empty((0, ny + nu)), np.empty(0)

    features = np.zeros((n_rows, ny + nu))
    for k in range(max_lag, n_samples):
        row = k - max_lag
        col = 0
        for j in range(1, ny + 1):
            features[row, col] = y[k - j]
            col += 1
        for j in range(1, nu + 1):
            features[row, col] = u[k - j]
            col += 1

    target = y[max_lag:]
    return features, target
