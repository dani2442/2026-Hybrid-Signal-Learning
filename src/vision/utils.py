"""Vision utilities: sync/alignment and image transforms.

Most dataset/label/registry helpers now live in ``src.data``.  This module
re-exports them for backward compatibility and adds vision-specific helpers
(frame normalisation, heatmap decoding, theta alignment).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Re-exports from src.data  (backward-compatible public names)
# ─────────────────────────────────────────────────────────────────────
from ..data.registry import (                       # noqa: F401
    DATASET_VIDEO_MAP,
    VIDEO_LED_FRAME_MAP,
    resolve_led_frame,
    resolve_video_name,
)
from ..data.labels import (                         # noqa: F401
    KEYPOINT_COLUMN_LABELS,
    SENSOR_STATE_LABELS,
    interpolate_missing_keypoints,
    keypoints_to_theta,
    load_theta_csv,
    parse_keypoint_labels_csv,
)


# ─────────────────────────────────────────────────────────────────────
# Frame ↔ sensor alignment
# ─────────────────────────────────────────────────────────────────────

def build_frame_index_map(
    n_sensor_samples: int,
    sensor_dt: float,
    video_fps: float,
    n_video_frames: int,
    frame_start: int = 0,
) -> np.ndarray:
    """Build a mapping from sensor sample index → nearest video frame index.

    Parameters
    ----------
    n_sensor_samples:
        Number of samples in the (cropped/resampled) sensor series.
    sensor_dt:
        Sampling period of the sensor series (seconds).
    video_fps:
        Video frame rate after sync (Hz).
    n_video_frames:
        Total number of synced video frames available.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_sensor_samples,)`` where entry *i*
        is the nearest video frame index for sensor sample *i*.
    """
    sensor_times = np.arange(n_sensor_samples) * sensor_dt
    frame_indices = np.round(sensor_times * video_fps).astype(int) + int(frame_start)
    return np.clip(frame_indices, 0, n_video_frames - 1)


# ─────────────────────────────────────────────────────────────────────
# Image normalisation helpers
# ─────────────────────────────────────────────────────────────────────

# ImageNet statistics used by torchvision pretrained models.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_frame(
    frame: np.ndarray,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD,
) -> np.ndarray:
    """Convert a uint8 HWC frame to float32 CHW with ImageNet normalisation.

    Parameters
    ----------
    frame:
        ``(H, W, 3)`` uint8 array.
    mean, std:
        Per-channel normalisation statistics.

    Returns
    -------
    np.ndarray
        ``(3, H, W)`` float32 tensor-ready array.
    """
    img = frame.astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean_arr) / std_arr
    return img.transpose(2, 0, 1)  # HWC → CHW


def heatmaps_to_keypoints(heatmaps: np.ndarray) -> np.ndarray:
    """Convert a batch of heatmaps to (x, y) keypoint coordinates via argmax.

    Parameters
    ----------
    heatmaps:
        ``(K, H, W)`` array of *K* heatmaps.

    Returns
    -------
    np.ndarray
        ``(K, 2)`` array of ``(x, y)`` coordinates (column, row).
    """
    K, H, W = heatmaps.shape
    flat = heatmaps.reshape(K, -1)
    max_idx = flat.argmax(axis=1)
    y_coords = max_idx // W
    x_coords = max_idx % W
    return np.column_stack([x_coords, y_coords]).astype(np.float64)


def align_and_calibrate_theta(
    sensor_t: np.ndarray,
    sensor_theta: np.ndarray,
    video_t: np.ndarray,
    video_theta: np.ndarray,
    *,
    offset_min_s: float = -12.0,
    offset_max_s: float = 12.0,
    estimate_offset: bool = True,
    estimate_sign: bool = True,
    calibrate_scale: bool = True,
    min_overlap: int = 200,
) -> Dict[str, np.ndarray | float | int]:
    """Align video theta to sensor theta via offset/sign and optional linear calibration."""
    sensor_t = np.asarray(sensor_t, dtype=float)
    sensor_theta = np.asarray(sensor_theta, dtype=float)
    video_t = np.asarray(video_t, dtype=float)
    video_theta = np.asarray(video_theta, dtype=float)

    sign_candidates = (-1, 1) if estimate_sign else (1,)
    best_sign = 1
    best_offset = 0.0
    best_corr = -np.inf

    for sign in sign_candidates:
        if estimate_offset:
            offset, corr = _search_offset(
                sensor_t=sensor_t,
                sensor_theta=sensor_theta,
                video_t=video_t,
                video_theta=sign * video_theta,
                offset_min_s=offset_min_s,
                offset_max_s=offset_max_s,
                min_overlap=min_overlap,
            )
        else:
            offset = 0.0
            corr = _corr_for_offset(
                sensor_t=sensor_t,
                sensor_theta=sensor_theta,
                video_t=video_t,
                video_theta=sign * video_theta,
                offset_s=0.0,
                min_overlap=min_overlap,
            )
        if corr > best_corr:
            best_corr = corr
            best_sign = int(sign)
            best_offset = float(offset)

    shifted_t = video_t - best_offset
    signed_theta = video_theta * float(best_sign)

    overlap = (shifted_t >= sensor_t[0]) & (shifted_t <= sensor_t[-1]) & np.isfinite(signed_theta)
    alpha, beta = 1.0, 0.0
    theta_cal = signed_theta.copy()
    if calibrate_scale and np.sum(overlap) >= max(3, min_overlap // 4):
        sensor_interp = np.interp(shifted_t[overlap], sensor_t, sensor_theta)
        A = np.column_stack([signed_theta[overlap], np.ones(np.sum(overlap))])
        try:
            (alpha, beta), _, _, _ = np.linalg.lstsq(A, sensor_interp, rcond=None)
            theta_cal = signed_theta * float(alpha) + float(beta)
        except np.linalg.LinAlgError:
            alpha, beta = 1.0, 0.0
            theta_cal = signed_theta

    sensor_theta_from_video = np.full_like(sensor_theta, np.nan, dtype=float)
    valid = np.isfinite(theta_cal)
    if np.any(valid):
        tt = shifted_t[valid]
        yy = theta_cal[valid]
        order = np.argsort(tt)
        tt = tt[order]
        yy = yy[order]
        in_range = (sensor_t >= tt[0]) & (sensor_t <= tt[-1])
        sensor_theta_from_video[in_range] = np.interp(sensor_t[in_range], tt, yy)

    # Also build a sparse version that has values only at sensor time-steps
    # nearest to the actual (non-interpolated) video label instants.
    sensor_theta_from_video_sparse = np.full_like(sensor_theta, np.nan, dtype=float)
    if np.any(valid):
        for t_v, y_v in zip(tt, yy):
            idx = int(np.argmin(np.abs(sensor_t - t_v)))
            sensor_theta_from_video_sparse[idx] = y_v

    # --- Compute quality metrics at the *actual* data points -----------
    # For sparse video data, computing metrics after dense interpolation
    # onto the sensor grid is misleading (linear interp can't reconstruct
    # high-frequency content between sparse points).  Instead, compare
    # the calibrated video values at the original sparse time-points
    # against the sensor signal interpolated to those same instants.
    cal_overlap = overlap & np.isfinite(theta_cal)
    point_corr = np.nan
    point_rmse = np.nan
    if np.sum(cal_overlap) > 2:
        sensor_at_pts = np.interp(shifted_t[cal_overlap], sensor_t, sensor_theta)
        video_at_pts = theta_cal[cal_overlap]
        point_corr = float(np.corrcoef(sensor_at_pts, video_at_pts)[0, 1])
        point_rmse = float(np.sqrt(np.mean((sensor_at_pts - video_at_pts) ** 2)))

    # Also provide grid-interpolated metrics (useful when video is dense).
    metrics_mask = np.isfinite(sensor_theta_from_video) & np.isfinite(sensor_theta)
    grid_corr = np.nan
    grid_rmse = np.nan
    if np.sum(metrics_mask) > 2:
        grid_corr = float(np.corrcoef(sensor_theta[metrics_mask], sensor_theta_from_video[metrics_mask])[0, 1])
        grid_rmse = float(np.sqrt(np.mean((sensor_theta[metrics_mask] - sensor_theta_from_video[metrics_mask]) ** 2)))

    # Report point-wise metrics as the primary figures; they are always
    # meaningful regardless of data density.
    return {
        "offset_s": float(best_offset),
        "sign": int(best_sign),
        "alpha": float(alpha),
        "beta": float(beta),
        "corr": float(point_corr),
        "rmse": float(point_rmse),
        "grid_corr": float(grid_corr),
        "grid_rmse": float(grid_rmse),
        "n_overlap_points": int(np.sum(cal_overlap)),
        "video_t_aligned": shifted_t,
        "theta_video_aligned": theta_cal,
        "theta_sensor_aligned": sensor_theta_from_video,
        "theta_sensor_aligned_sparse": sensor_theta_from_video_sparse,
    }


def _search_offset(
    *,
    sensor_t: np.ndarray,
    sensor_theta: np.ndarray,
    video_t: np.ndarray,
    video_theta: np.ndarray,
    offset_min_s: float,
    offset_max_s: float,
    min_overlap: int,
) -> Tuple[float, float]:
    try:
        from scipy.optimize import minimize_scalar
    except ImportError:
        minimize_scalar = None

    def _neg_corr(off: float) -> float:
        return -_corr_for_offset(
            sensor_t=sensor_t,
            sensor_theta=sensor_theta,
            video_t=video_t,
            video_theta=video_theta,
            offset_s=float(off),
            min_overlap=min_overlap,
        )

    # --- Coarse grid search (always run) --------------------------------
    span = float(offset_max_s) - float(offset_min_s)
    n_grid = max(41, int(span / 0.05) + 1)  # ~0.05 s resolution
    candidates = np.linspace(float(offset_min_s), float(offset_max_s), n_grid)
    corr_vals = np.array([-_neg_corr(float(c)) for c in candidates])
    best_grid_idx = int(np.nanargmax(corr_vals))
    best_off = float(candidates[best_grid_idx])
    best_corr = float(corr_vals[best_grid_idx])

    # --- Refine around the best grid point with Brent -------------------
    if minimize_scalar is not None:
        half = span / n_grid  # one grid step
        lo = max(float(offset_min_s), best_off - 3 * half)
        hi = min(float(offset_max_s), best_off + 3 * half)
        res = minimize_scalar(
            _neg_corr,
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-4},
        )
        if res.success:
            refined_off = float(res.x)
            refined_corr = -float(res.fun)
            if refined_corr >= best_corr:
                best_off = refined_off
                best_corr = refined_corr

    return best_off, best_corr


def _corr_for_offset(
    *,
    sensor_t: np.ndarray,
    sensor_theta: np.ndarray,
    video_t: np.ndarray,
    video_theta: np.ndarray,
    offset_s: float,
    min_overlap: int,
) -> float:
    shifted = video_t - float(offset_s)
    mask = (shifted >= sensor_t[0]) & (shifted <= sensor_t[-1]) & np.isfinite(video_theta)
    if np.sum(mask) < int(min_overlap):
        return -np.inf
    yy = video_theta[mask]
    xx = shifted[mask]
    sensor_interp = np.interp(xx, sensor_t, sensor_theta)
    if np.std(sensor_interp) < 1e-12 or np.std(yy) < 1e-12:
        return -np.inf
    return float(np.corrcoef(sensor_interp, yy)[0, 1])
