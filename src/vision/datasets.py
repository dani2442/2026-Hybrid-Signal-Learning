"""PyTorch datasets for multimodal BAB video + sensor training.

Provides:
- ``load_bab_with_video`` — sensor + video loading with optional label alignment.
- ``FrameStateDataset`` — per-frame encoder training.
- ``WindowedSequenceDataset`` — windowed Enc→ODE→Dec training.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from ..data import Dataset as SensorDataset
from ..data.labels import (
    KEYPOINT_COLUMN_LABELS,
    SENSOR_STATE_LABELS,
    interpolate_missing_keypoints,
    load_theta_csv,
    parse_keypoint_labels_csv,
)
from ..data.registry import (
    ensure_sensor as _ensure_sensor,
    resolve_led_frame,
    resolve_video_path as _resolve_video_path_registry,
)
from ..data.video import get_video_fps, get_video_resolution, load_video_frames
from .utils import (
    align_and_calibrate_theta,
    build_frame_index_map,
    normalize_frame,
)

def _resolve_video_path(
    dataset_name: str,
    *,
    video_dir: Optional[str],
    video_path: Optional[str],
    video_map: Optional[Dict[str, str]],
    data_root: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve final video path and canonical video name."""
    return _resolve_video_path_registry(
        dataset_name,
        video_dir=video_dir,
        video_path=video_path,
        video_map=video_map,
        data_root=data_root,
    )


def _map_frame_labels_to_video_segment(
    frame_labels: np.ndarray,
    frame_numbers: np.ndarray,
    *,
    n_video_frames: int,
    frame_start: int,
) -> np.ndarray:
    """Map labels indexed by full-video frame ID to trigger-cropped segment indices."""
    out = np.full((n_video_frames, frame_labels.shape[1]), np.nan, dtype=float)
    local = frame_numbers.astype(int) - int(frame_start)
    valid = (local >= 0) & (local < n_video_frames)
    if np.any(valid):
        out[local[valid]] = frame_labels[valid]
    return out


def _theta_series_for_segment(
    theta_t: np.ndarray,
    theta_v: np.ndarray,
    *,
    n_video_frames: int,
    video_fps: float,
    frame_start: int,
    keep_sparse: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert theta(t) from full-video timebase to trigger-cropped timebase.

    Parameters
    ----------
    keep_sparse : bool
        If True, return only the shifted sparse (t, theta) pairs without
        interpolating onto a regular frame grid.  Preferred when the source
        data has very few labelled frames (e.g. DLC CollectedData).
    """
    src_t = np.asarray(theta_t, dtype=float) - (float(frame_start) / float(video_fps))
    src_v = np.asarray(theta_v, dtype=float)
    valid = np.isfinite(src_t) & np.isfinite(src_v)
    if np.sum(valid) < 2:
        target_t = np.arange(n_video_frames, dtype=float) / float(video_fps)
        return target_t, np.full_like(target_t, np.nan, dtype=float)

    src_t = src_t[valid]
    src_v = src_v[valid]
    order = np.argsort(src_t)
    src_t = src_t[order]
    src_v = src_v[order]

    if keep_sparse:
        # Keep only points within the segment timespan
        seg_dur = float(n_video_frames) / float(video_fps)
        in_seg = (src_t >= 0) & (src_t <= seg_dur)
        return src_t[in_seg], src_v[in_seg]

    target_t = np.arange(n_video_frames, dtype=float) / float(video_fps)
    out = np.full_like(target_t, np.nan, dtype=float)
    in_range = (target_t >= src_t[0]) & (target_t <= src_t[-1])
    out[in_range] = np.interp(target_t[in_range], src_t, src_v)
    return target_t, out


# ---------------------------------------------------------------------
# Public loading API
# ---------------------------------------------------------------------

def load_bab_with_video(
    dataset_name: str,
    *,
    video_dir: Optional[str] = None,
    video_path: Optional[str] = None,
    video_map: Optional[Dict[str, str]] = None,
    resample_factor: int = 50,
    video_fps: Optional[float] = None,
    frame_height: Optional[int] = 224,
    frame_width: Optional[int] = 224,
    preprocess: bool = True,
    data_dir: Optional[str] = None,
    led_frame: Optional[int] = None,
    use_led_sync: bool = True,
    keypoint_labels_csv: Optional[str] = None,
    theta_labels_csv: Optional[str] = None,
    align_theta: bool = True,
    alignment_offset_min_s: float = -12.0,
    alignment_offset_max_s: float = 12.0,
    auto_match_video_fps: bool = True,
    return_aux: bool = False,
) -> (
    Tuple[SensorDataset, np.ndarray, np.ndarray]
    | Tuple[SensorDataset, np.ndarray, np.ndarray, Dict[str, Any]]
):
    """Load BAB sensor data with synced video frames.

    Parameters
    ----------
    video_fps : float or None
        Video frame rate.  When *None* (default) the FPS is read from the
        video file metadata (falls back to 30.0).
    auto_match_video_fps : bool
        When True (default), override *resample_factor* so that the sensor
        data rate matches the video FPS as closely as possible (like the
        reference notebook approach).  Set False to use the raw
        *resample_factor*.

    Returns
    -------
    ``(data, frames, frame_index_map)`` by default, or
    ``(data, frames, frame_index_map, aux)`` when ``return_aux=True``.
    """

    # --- Resolve video first so we can read its FPS ----------------
    video_path_resolved, video_name = _resolve_video_path(
        dataset_name,
        video_dir=video_dir,
        video_path=video_path,
        video_map=video_map,
        data_root=data_dir,
    )

    # Auto-detect FPS from the video file when not explicitly provided
    if video_fps is None:
        video_fps = get_video_fps(video_path_resolved)

    # When requested, compute resample_factor to approximately match video FPS.
    # The sensor data is at ~1 kHz; to get ~30 Hz we need factor ≈ 33.
    effective_resample_factor = resample_factor
    if auto_match_video_fps and preprocess:
        import scipy.io

        sensor_path = str(_ensure_sensor(dataset_name, data_root=data_dir))
        raw_t = scipy.io.loadmat(sensor_path)["time"].flatten()
        raw_dt = float(np.median(np.diff(raw_t))) if len(raw_t) > 1 else 0.001
        raw_fs = 1.0 / raw_dt if raw_dt > 0 else 1000.0
        effective_resample_factor = max(1, round(raw_fs / video_fps))

    data = SensorDataset.from_bab_experiment(
        dataset_name,
        preprocess=preprocess,
        resample_factor=effective_resample_factor,
        data_dir=data_dir,
    )

    frame_start = resolve_led_frame(video_name, override=led_frame) if use_led_sync else 0
    frame_start = int(max(0, frame_start))

    # --- Determine original resolution for keypoint scaling --------
    orig_h, orig_w = get_video_resolution(video_path_resolved)

    frames = load_video_frames(
        video_path_resolved,
        frame_start=frame_start,
        frame_height=frame_height,
        frame_width=frame_width,
    )

    # Compute scale factors if frames were resized (for keypoint adjustment)
    loaded_h, loaded_w = frames.shape[1], frames.shape[2]
    kp_scale_x = loaded_w / orig_w if orig_w > 0 else 1.0
    kp_scale_y = loaded_h / orig_h if orig_h > 0 else 1.0

    sensor_dt = 1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0
    frame_index_map = build_frame_index_map(
        n_sensor_samples=len(data),
        sensor_dt=sensor_dt,
        video_fps=video_fps,
        n_video_frames=len(frames),
        frame_start=0,  # frames are already cropped at frame_start
    )

    if (
        not return_aux
        and keypoint_labels_csv is None
        and theta_labels_csv is None
    ):
        return data, frames, frame_index_map

    aux: Dict[str, Any] = {
        "video_name": video_name,
        "video_path": video_path_resolved,
        "video_fps": float(video_fps),
        "frame_start": int(frame_start),
        "sensor_resample_factor": effective_resample_factor,
        "keypoint_labels": list(KEYPOINT_COLUMN_LABELS),
        "sensor_state_labels": list(SENSOR_STATE_LABELS),
    }

    parsed_keypoints: Optional[Dict[str, np.ndarray]] = None
    if keypoint_labels_csv:
        parsed_keypoints = parse_keypoint_labels_csv(keypoint_labels_csv, fps=video_fps)

        # Compute local (segment-relative) indices of frames with actual labels.
        raw_frame_nums = parsed_keypoints["frame"].astype(int)
        eff_frame_start = frame_start if use_led_sync else 0
        local_indices = raw_frame_nums - int(eff_frame_start)
        in_segment = (local_indices >= 0) & (local_indices < len(frames))
        labeled_frame_indices = local_indices[in_segment].astype(int)

        kp_video = _map_frame_labels_to_video_segment(
            parsed_keypoints["keypoints"],
            parsed_keypoints["frame"],
            n_video_frames=len(frames),
            frame_start=eff_frame_start,
        )

        # Store raw (pre-interpolation) keypoints for frames that have
        # actual labels — consumers can use these for accurate overlays.
        kp_video_sparse = kp_video.copy()

        kp_video = interpolate_missing_keypoints(kp_video)

        # Scale keypoints to match the (possibly resized) frame dimensions.
        # DLC keypoints are in original-resolution pixel coords (xl, yl, xr, yr).
        if kp_scale_x != 1.0 or kp_scale_y != 1.0:
            kp_video[:, 0] *= kp_scale_x  # xl
            kp_video[:, 1] *= kp_scale_y  # yl
            kp_video[:, 2] *= kp_scale_x  # xr
            kp_video[:, 3] *= kp_scale_y  # yr
            kp_video_sparse[:, 0] *= kp_scale_x
            kp_video_sparse[:, 1] *= kp_scale_y
            kp_video_sparse[:, 2] *= kp_scale_x
            kp_video_sparse[:, 3] *= kp_scale_y

        kp_sensor = kp_video[frame_index_map]
        aux["keypoints_video"] = kp_video
        aux["keypoints_video_sparse"] = kp_video_sparse
        aux["keypoints_sensor"] = kp_sensor
        aux["labeled_frame_indices"] = labeled_frame_indices
        aux["keypoint_coverage"] = float(np.mean(np.isfinite(kp_video[:, 0])))
        aux["keypoint_scale"] = (kp_scale_x, kp_scale_y)
        aux["original_resolution"] = (orig_h, orig_w)

    theta_t = None
    theta_v = None
    is_sparse_labels = False
    if theta_labels_csv:
        theta_csv = load_theta_csv(theta_labels_csv, fps=video_fps)
        theta_t = theta_csv["t_s"]
        theta_v = theta_csv["theta_deg"]
    elif parsed_keypoints is not None:
        theta_t = parsed_keypoints["t_s"]
        theta_v = parsed_keypoints["theta_deg"]
        # Detect sparse labels: if we have <25% of total video frames labelled
        n_labels = len(theta_t)
        n_total = len(frames) + frame_start
        is_sparse_labels = n_labels < 0.25 * n_total

    if theta_t is not None and theta_v is not None:
        # For sparse hand-labeled data, pass through without interpolating
        # to a regular grid — alignment works directly on the sparse points.
        t_video_segment, theta_video_segment = _theta_series_for_segment(
            theta_t,
            theta_v,
            n_video_frames=len(frames),
            video_fps=video_fps,
            frame_start=frame_start if use_led_sync else 0,
            keep_sparse=is_sparse_labels,
        )

        # When LED sync is used, tighten the offset search range.
        # The LED frame already provides good temporal alignment; we only
        # need a small correction (at most a few frames).
        eff_offset_min = alignment_offset_min_s
        eff_offset_max = alignment_offset_max_s
        if use_led_sync and align_theta:
            # LED sync should be accurate to within ~1 second.
            eff_offset_min = max(alignment_offset_min_s, -2.0)
            eff_offset_max = min(alignment_offset_max_s, 2.0)

        theta_info = align_and_calibrate_theta(
            sensor_t=data.t,
            sensor_theta=data.y,
            video_t=t_video_segment,
            video_theta=theta_video_segment,
            offset_min_s=eff_offset_min,
            offset_max_s=eff_offset_max,
            estimate_offset=align_theta,
            estimate_sign=align_theta,
            calibrate_scale=align_theta,
            min_overlap=max(20, int(2 * video_fps)),
        )

        # Build the regular-grid versions for visualization even when
        # alignment was done on sparse data.
        if is_sparse_labels:
            reg_t = np.arange(len(frames), dtype=float) / float(video_fps)
            reg_theta = np.full_like(reg_t, np.nan, dtype=float)
            in_range = (reg_t >= t_video_segment[0]) & (reg_t <= t_video_segment[-1])
            if np.any(in_range):
                reg_theta[in_range] = np.interp(
                    reg_t[in_range], t_video_segment, theta_video_segment
                )
            aux["theta_video_t_segment"] = reg_t
            aux["theta_video_raw_segment"] = reg_theta
        else:
            aux["theta_video_t_segment"] = t_video_segment
            aux["theta_video_raw_segment"] = theta_video_segment

        aux["theta_video_aligned"] = np.asarray(theta_info["theta_video_aligned"], dtype=float)
        aux["theta_sensor_from_video"] = np.asarray(theta_info["theta_sensor_aligned"], dtype=float)
        aux["theta_sensor_from_video_sparse"] = np.asarray(theta_info["theta_sensor_aligned_sparse"], dtype=float)
        aux["theta_alignment"] = {
            "offset_s": float(theta_info["offset_s"]),
            "sign": int(theta_info["sign"]),
            "alpha": float(theta_info["alpha"]),
            "beta": float(theta_info["beta"]),
            "corr": float(theta_info["corr"]),
            "rmse": float(theta_info["rmse"]),
        }
        aux["is_sparse_labels"] = is_sparse_labels

    if return_aux:
        return data, frames, frame_index_map, aux
    return data, frames, frame_index_map


# ---------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------

class FrameStateDataset(TorchDataset):
    """Per-frame (image, θ) dataset for encoder training.

    The encoder is trained to predict position ``θ`` only; velocity ``θ̇``
    is obtained from finite differences of consecutive encoder outputs in
    the composite ``EncOdeDecModel``.
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
        self.indices = np.arange(len(data)) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        i = int(self.indices[idx])
        frame_idx = int(self.frame_index_map[i])
        frame_tensor = torch.from_numpy(normalize_frame(self.frames[frame_idx]))
        theta = float(self.data.y[i])
        # Encoder target is θ only (1-D); velocity comes from finite
        # differences of two consecutive encoder outputs.
        state = torch.tensor([theta], dtype=torch.float32)
        meta = {
            "dataset_name": self.data.name,
            "i": i,
            "t": float(self.data.t[i]),
            "frame_idx": frame_idx,
        }
        return frame_tensor, state, meta

    def split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple["FrameStateDataset", "FrameStateDataset", "FrameStateDataset"]:
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


class WindowedSequenceDataset(TorchDataset):
    """Windowed sequence dataset for Enc→ODE and Enc→ODE→Dec training.

    Each sample yields:
      - ``y0`` initial frame,
      - ``y_prev`` previous frame (for finite-difference velocity),
      - ``u_seq`` control sequence ``(K, 1)``,
      - ``x_seq`` state sequence ``(K, 2)`` with ``[θ, θ̇]``,
      - optional ``y_seq`` keypoint sequence ``(K, D)``.
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
        keypoints_sensor: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
    ) -> None:
        self.data = data
        self.frames = frames
        self.frame_index_map = frame_index_map
        self.k_steps = k_steps
        self.encoder_window = encoder_window
        self.stride = stride

        self.keypoints_sensor = None
        if keypoints_sensor is not None:
            kp = np.asarray(keypoints_sensor, dtype=float)
            if kp.shape[0] != len(data):
                raise ValueError("keypoints_sensor must have shape (N_sensor, D).")
            self.keypoints_sensor = kp

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

        # Previous frame for finite-difference velocity computation.
        i_prev = max(0, i - 1)
        frame_idx_prev = int(self.frame_index_map[i_prev])
        y_prev = torch.from_numpy(normalize_frame(self.frames[frame_idx_prev]))

        u_seq = torch.tensor(self.data.u[i : i + K].reshape(-1, 1), dtype=torch.float32)

        theta = self.data.y[i : i + K]
        theta_dot = (
            self.data.y_dot[i : i + K]
            if self.data.y_dot is not None
            else np.zeros_like(theta)
        )
        x_seq = torch.tensor(np.column_stack([theta, theta_dot]), dtype=torch.float32)

        sample: Dict[str, Any] = {
            "y0": y0,
            "y_prev": y_prev,
            "u_seq": u_seq,
            "x_seq": x_seq,
            "start_idx": i,
        }
        if self.keypoints_sensor is not None:
            sample["y_seq"] = torch.tensor(self.keypoints_sensor[i : i + K], dtype=torch.float32)
        return sample

    def split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple["WindowedSequenceDataset", "WindowedSequenceDataset", "WindowedSequenceDataset"]:
        n = len(self.start_indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            WindowedSequenceDataset(
                self.data,
                self.frames,
                self.frame_index_map,
                k_steps=self.k_steps,
                encoder_window=self.encoder_window,
                stride=self.stride,
                keypoints_sensor=self.keypoints_sensor,
                indices=self.start_indices[:n_train],
            ),
            WindowedSequenceDataset(
                self.data,
                self.frames,
                self.frame_index_map,
                k_steps=self.k_steps,
                encoder_window=self.encoder_window,
                stride=self.stride,
                keypoints_sensor=self.keypoints_sensor,
                indices=self.start_indices[n_train : n_train + n_val],
            ),
            WindowedSequenceDataset(
                self.data,
                self.frames,
                self.frame_index_map,
                k_steps=self.k_steps,
                encoder_window=self.encoder_window,
                stride=self.stride,
                keypoints_sensor=self.keypoints_sensor,
                indices=self.start_indices[n_train + n_val :],
            ),
        )
