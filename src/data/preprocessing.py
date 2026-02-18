"""Data preprocessing utilities — stateless functions."""

from __future__ import annotations

from typing import Optional

import numpy as np


def find_trigger_start(trigger: Optional[np.ndarray]) -> int:
    """Find first active trigger index."""
    if trigger is None:
        return 0
    idx = np.where(np.asarray(trigger).ravel() != 0)[0]
    return int(idx[0]) if idx.size > 0 else 0


def find_end_before_ref_zero(
    y_ref: Optional[np.ndarray],
    tolerance: float = 1e-8,
) -> int:
    """Find last index before reference goes to zero."""
    if y_ref is None:
        return -1
    y_ref = np.asarray(y_ref).ravel()
    for i in range(len(y_ref) - 1, -1, -1):
        if np.abs(y_ref[i]) > tolerance:
            return int(i + 1)
    return -1


def estimate_y_dot(
    y: np.ndarray,
    dt: float,
    method: str = "savgol",
    savgol_window: int = 51,
    savgol_poly: int = 3,
) -> np.ndarray:
    """Estimate output derivative from sampled output."""
    y = np.asarray(y, dtype=float).ravel()
    if len(y) < 3 or dt <= 0:
        return np.zeros_like(y)

    if method == "central":
        return np.gradient(y, dt)

    if method == "savgol":
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            return np.gradient(y, dt)

        w = max(5, int(savgol_window))
        if w % 2 == 0:
            w += 1
        if w >= len(y):
            w = len(y) - 1 if len(y) % 2 == 0 else len(y)
        if w < 5:
            return np.gradient(y, dt)
        poly = min(int(savgol_poly), w - 1)
        return savgol_filter(
            y,
            window_length=w,
            polyorder=poly,
            deriv=1,
            delta=dt,
            mode="interp",
        )

    raise ValueError(f"Unknown y_dot estimation method: {method}")


def slice_optional(arr: Optional[np.ndarray], start: int, end: int) -> Optional[np.ndarray]:
    """Slice an optional array safely."""
    if arr is None:
        return None
    return arr[start:end]


def downsample_optional(arr: Optional[np.ndarray], factor: int) -> Optional[np.ndarray]:
    """Downsample optional array when factor > 1."""
    if arr is None or factor <= 1:
        return arr
    return arr[::factor]


def shift_time_to_zero(t: np.ndarray) -> np.ndarray:
    """Shift time vector so it starts at zero."""
    if len(t) == 0:
        return t
    return t - t[0]


def estimate_dt_and_fs(t: np.ndarray, default_fs: float = 1.0) -> tuple[float, float]:
    """Estimate sampling time and rate from time vector."""
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    fs = 1.0 / dt if dt > 0 else float(default_fs)
    return dt, fs
