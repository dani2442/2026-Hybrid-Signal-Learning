"""Vision utilities: dataset↔video mapping, frame alignment, image transforms."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Dataset ↔ video name mapping
# ─────────────────────────────────────────────────────────────────────

DATASET_VIDEO_MAP: Dict[str, str] = {
    "multisine_05": "multisine_01",
    "multisine_06": "multisine_02",
    "random_steps_01": "random_steps_W",
    "random_steps_02": "random_steps_X",
    "random_steps_03": "random_steps_Y",
    "random_steps_04": "random_steps_Z",
}


def resolve_video_name(
    dataset_name: str,
    override_map: Optional[Dict[str, str]] = None,
) -> str:
    """Map a sensor dataset name to the corresponding video name.

    Parameters
    ----------
    dataset_name:
        Canonical sensor dataset key (e.g. ``"multisine_05"``).
    override_map:
        Optional user-provided mapping that takes priority.

    Returns
    -------
    str
        Video name (e.g. ``"multisine_01"``).
    """
    if override_map and dataset_name in override_map:
        return override_map[dataset_name]
    if dataset_name in DATASET_VIDEO_MAP:
        return DATASET_VIDEO_MAP[dataset_name]
    # Fallback: assume the dataset name matches the video name.
    return dataset_name


# ─────────────────────────────────────────────────────────────────────
# Frame ↔ sensor alignment
# ─────────────────────────────────────────────────────────────────────

def build_frame_index_map(
    n_sensor_samples: int,
    sensor_dt: float,
    video_fps: float,
    n_video_frames: int,
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
    frame_indices = np.round(sensor_times * video_fps).astype(int)
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


def keypoints_to_theta(kp_left: np.ndarray, kp_right: np.ndarray) -> np.ndarray:
    """Compute beam angle θ (degrees) from left/right keypoint coordinates.

    θ = arctan2(y_right − y_left, x_right − x_left)

    Parameters
    ----------
    kp_left:
        ``(N, 2)`` or ``(2,)`` array of ``(x, y)`` for the left keypoint.
    kp_right:
        ``(N, 2)`` or ``(2,)`` array of ``(x, y)`` for the right keypoint.

    Returns
    -------
    np.ndarray
        Angle in degrees, shape ``(N,)`` or scalar.
    """
    kp_left = np.atleast_2d(kp_left)
    kp_right = np.atleast_2d(kp_right)
    dx = kp_right[:, 0] - kp_left[:, 0]
    dy = kp_right[:, 1] - kp_left[:, 1]
    return np.degrees(np.arctan2(dy, dx)).squeeze()
