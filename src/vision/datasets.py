"""PyTorch datasets for multimodal BAB video + sensor training.

Provides:
- ``load_bab_with_video`` — load sensor data and aligned video frames.
- ``FrameStateDataset`` — per-frame encoder training.
- ``WindowedSequenceDataset`` — windowed Enc→ODE→Dec training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from ..data import Dataset as SensorDataset
from .utils import build_frame_index_map, normalize_frame, resolve_video_name


# ─────────────────────────────────────────────────────────────────────
# Helper: load sensor + video together
# ─────────────────────────────────────────────────────────────────────

def _try_load_frames_opencv(video_path: str) -> np.ndarray:
    """Load all frames from a video file using OpenCV."""
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for video loading. "
            "Install with: pip install hybrid-modeling[vision]"
        )
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    return np.stack(frames)


def _try_load_frames_imageio(video_path: str) -> np.ndarray:
    """Load all frames from a video file using imageio."""
    try:
        import imageio.v3 as iio
    except ImportError:
        raise ImportError(
            "imageio is required for video loading. "
            "Install with: pip install hybrid-modeling[vision]"
        )
    frames = iio.imread(video_path, plugin="pyav")
    return np.asarray(frames)


def _load_video_frames(video_path: str) -> np.ndarray:
    """Load video frames, trying OpenCV first then imageio."""
    try:
        return _try_load_frames_opencv(video_path)
    except ImportError:
        return _try_load_frames_imageio(video_path)


def load_bab_with_video(
    dataset_name: str,
    *,
    video_dir: Optional[str] = None,
    video_path: Optional[str] = None,
    video_map: Optional[Dict[str, str]] = None,
    resample_factor: int = 50,
    video_fps: float = 30.0,
    frame_height: int = 224,
    frame_width: int = 224,
    preprocess: bool = True,
    data_dir: Optional[str] = None,
) -> Tuple[SensorDataset, np.ndarray, np.ndarray]:
    """Load a BAB sensor dataset together with aligned video frames.

    Parameters
    ----------
    dataset_name:
        Sensor dataset key (e.g. ``"multisine_05"``).
    video_dir:
        Directory containing video files. Videos are looked up as
        ``<video_dir>/<video_name>.mp4`` (or ``.avi``).
    video_path:
        Explicit path to a single video file (overrides *video_dir*).
    video_map:
        Optional ``{dataset_name: video_name}`` override mapping.
    resample_factor:
        Passed through to ``SensorDataset.from_bab_experiment``.
    video_fps:
        Frame rate of the synced video (default 30 Hz).
    frame_height, frame_width:
        Target frame dimensions for resizing.
    preprocess:
        Whether to preprocess the sensor data (trigger crop, resample).
    data_dir:
        Local directory for sensor ``.mat`` files.

    Returns
    -------
    data:
        Loaded ``SensorDataset`` (preprocessed).
    frames:
        ``(N_frames, H, W, 3)`` uint8 array of RGB video frames.
    frame_index_map:
        ``(N_sensor,)`` int array mapping sensor index → frame index.
    """
    # Load sensor data
    data = SensorDataset.from_bab_experiment(
        dataset_name,
        preprocess=preprocess,
        resample_factor=resample_factor,
        data_dir=data_dir,
    )

    # Resolve video file path
    if video_path is None:
        if video_dir is None:
            raise ValueError("Provide either video_path or video_dir.")
        video_name = resolve_video_name(dataset_name, override_map=video_map)
        video_dir_path = Path(video_dir)
        # Try common extensions
        for ext in (".mp4", ".avi", ".mkv"):
            candidate = video_dir_path / f"{video_name}{ext}"
            if candidate.exists():
                video_path = str(candidate)
                break
        if video_path is None:
            raise FileNotFoundError(
                f"No video found for '{video_name}' in {video_dir} "
                f"(tried .mp4, .avi, .mkv)"
            )

    # Load video frames
    frames = _load_video_frames(video_path)

    # Optionally resize
    if frames.shape[1] != frame_height or frames.shape[2] != frame_width:
        try:
            import cv2

            resized = np.empty(
                (len(frames), frame_height, frame_width, 3), dtype=np.uint8
            )
            for i, f in enumerate(frames):
                resized[i] = cv2.resize(f, (frame_width, frame_height))
            frames = resized
        except ImportError:
            from PIL import Image

            resized = np.empty(
                (len(frames), frame_height, frame_width, 3), dtype=np.uint8
            )
            for i, f in enumerate(frames):
                img = Image.fromarray(f).resize((frame_width, frame_height))
                resized[i] = np.asarray(img)
            frames = resized

    # Build alignment map
    sensor_dt = 1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0
    frame_index_map = build_frame_index_map(
        n_sensor_samples=len(data),
        sensor_dt=sensor_dt,
        video_fps=video_fps,
        n_video_frames=len(frames),
    )

    return data, frames, frame_index_map


# ─────────────────────────────────────────────────────────────────────
# FrameStateDataset — per-frame encoder training
# ─────────────────────────────────────────────────────────────────────

class FrameStateDataset(TorchDataset):
    """Per-frame (image, state) dataset for encoder training.

    Each sample yields:
      - ``frame``: float32 tensor ``(3, H, W)`` (normalised).
      - ``state``: float32 tensor ``(2,)`` — ``[θ, θ̇]``.
      - ``meta``: dict with ``dataset_name``, sample index ``i``, time ``t``.
    """

    def __init__(
        self,
        data: SensorDataset,
        frames: np.ndarray,
        frame_index_map: np.ndarray,
        *,
        indices: Optional[np.ndarray] = None,
    ) -> None:
        self.data = data
        self.frames = frames
        self.frame_index_map = frame_index_map
        # If indices not provided, use all sensor samples with valid frames.
        self.indices = (
            np.arange(len(data)) if indices is None else np.asarray(indices)
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        i = int(self.indices[idx])
        frame_idx = int(self.frame_index_map[i])

        # Normalise frame: uint8 HWC → float32 CHW
        frame = normalize_frame(self.frames[frame_idx])
        frame_tensor = torch.from_numpy(frame)

        # State: [theta, theta_dot]
        theta = float(self.data.y[i])
        theta_dot = float(self.data.y_dot[i]) if self.data.y_dot is not None else 0.0
        state = torch.tensor([theta, theta_dot], dtype=torch.float32)

        meta = {
            "dataset_name": self.data.name,
            "i": i,
            "t": float(self.data.t[i]),
            "frame_idx": frame_idx,
        }
        return frame_tensor, state, meta

    def split(
        self, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple["FrameStateDataset", "FrameStateDataset", "FrameStateDataset"]:
        """Split into train/val/test by contiguous time slices."""
        n = len(self.indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx = self.indices[:n_train]
        val_idx = self.indices[n_train : n_train + n_val]
        test_idx = self.indices[n_train + n_val :]
        return (
            FrameStateDataset(self.data, self.frames, self.frame_index_map, indices=train_idx),
            FrameStateDataset(self.data, self.frames, self.frame_index_map, indices=val_idx),
            FrameStateDataset(self.data, self.frames, self.frame_index_map, indices=test_idx),
        )


# ─────────────────────────────────────────────────────────────────────
# WindowedSequenceDataset — Enc→ODE→Dec training
# ─────────────────────────────────────────────────────────────────────

class WindowedSequenceDataset(TorchDataset):
    """Windowed sequence dataset for Enc→ODE and Enc→ODE→Dec training.

    Each sample yields:
      - ``y0``: initial frame tensor ``(3, H, W)`` or window ``(L, 3, H, W)``.
      - ``u_seq``: control input sequence ``(K, 1)``.
      - ``x_seq``: sensor state sequence ``(K, 2)`` — targets for ODE rollout.
      - ``start_idx``: starting sensor sample index.
    """

    def __init__(
        self,
        data: SensorDataset,
        frames: np.ndarray,
        frame_index_map: np.ndarray,
        *,
        k_steps: int = 20,
        encoder_window: int = 1,
        stride: int = 1,
        indices: Optional[np.ndarray] = None,
    ) -> None:
        self.data = data
        self.frames = frames
        self.frame_index_map = frame_index_map
        self.k_steps = k_steps
        self.encoder_window = encoder_window
        self.stride = stride

        max_start = len(data) - k_steps
        if indices is not None:
            self.start_indices = np.asarray(indices)
        else:
            self.start_indices = np.arange(0, max(1, max_start), stride)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        i = int(self.start_indices[idx])
        K = self.k_steps
        L = self.encoder_window

        # Initial frame(s)
        if L == 1:
            frame_idx = int(self.frame_index_map[i])
            y0 = torch.from_numpy(normalize_frame(self.frames[frame_idx]))
        else:
            window_frames = []
            for offset in range(L):
                j = min(i + offset, len(self.data) - 1)
                fj = int(self.frame_index_map[j])
                window_frames.append(normalize_frame(self.frames[fj]))
            y0 = torch.from_numpy(np.stack(window_frames))

        # Control input sequence: (K, 1)
        u_seq = torch.tensor(
            self.data.u[i : i + K].reshape(-1, 1),
            dtype=torch.float32,
        )

        # State target sequence: (K, 2)  — [theta, theta_dot]
        theta = self.data.y[i : i + K]
        theta_dot = (
            self.data.y_dot[i : i + K]
            if self.data.y_dot is not None
            else np.zeros_like(theta)
        )
        x_seq = torch.tensor(
            np.column_stack([theta, theta_dot]),
            dtype=torch.float32,
        )

        return {
            "y0": y0,
            "u_seq": u_seq,
            "x_seq": x_seq,
            "start_idx": i,
        }

    def split(
        self, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple["WindowedSequenceDataset", "WindowedSequenceDataset", "WindowedSequenceDataset"]:
        """Split into train/val/test by contiguous start-index ranges."""
        n = len(self.start_indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            WindowedSequenceDataset(
                self.data, self.frames, self.frame_index_map,
                k_steps=self.k_steps, encoder_window=self.encoder_window,
                stride=self.stride, indices=self.start_indices[:n_train],
            ),
            WindowedSequenceDataset(
                self.data, self.frames, self.frame_index_map,
                k_steps=self.k_steps, encoder_window=self.encoder_window,
                stride=self.stride, indices=self.start_indices[n_train : n_train + n_val],
            ),
            WindowedSequenceDataset(
                self.data, self.frames, self.frame_index_map,
                k_steps=self.k_steps, encoder_window=self.encoder_window,
                stride=self.stride, indices=self.start_indices[n_train + n_val :],
            ),
        )
