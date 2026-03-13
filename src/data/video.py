"""Video frame loading utilities.

Provides ``load_video_frames`` that tries OpenCV first, then falls back
to imageio/pyav.  Used by both the training pipeline and diagnostic scripts.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _try_load_frames_opencv(
    video_path: str,
    *,
    frame_start: int = 0,
    frame_height: Optional[int] = None,
    frame_width: Optional[int] = None,
) -> np.ndarray:
    """Load video frames using OpenCV with optional start offset and resizing."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for video loading. "
            "Install with: pip install hybrid-modeling[vision]"
        ) from exc

    cap = cv2.VideoCapture(video_path)
    if frame_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_start))
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_height is not None and frame_width is not None:
            if frame.shape[0] != frame_height or frame.shape[1] != frame_width:
                frame = cv2.resize(frame, (frame_width, frame_height))
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    return np.stack(frames)


def _try_load_frames_imageio(
    video_path: str, *, frame_start: int = 0
) -> np.ndarray:
    """Load all frames from a video file using imageio."""
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError(
            "imageio is required for video loading. "
            "Install with: pip install hybrid-modeling[vision]"
        ) from exc
    frames = np.asarray(iio.imread(video_path, plugin="pyav"))
    if frame_start > 0:
        frames = frames[frame_start:]
    return frames


def resize_frames(
    frames: np.ndarray,
    frame_height: int,
    frame_width: int,
) -> np.ndarray:
    """Resize frames only when dimensions differ."""
    if frames.shape[1] == frame_height and frames.shape[2] == frame_width:
        return frames

    try:
        import cv2

        resized = np.empty(
            (len(frames), frame_height, frame_width, 3), dtype=np.uint8
        )
        for i, f in enumerate(frames):
            resized[i] = cv2.resize(f, (frame_width, frame_height))
        return resized
    except ImportError:
        from PIL import Image

        resized = np.empty(
            (len(frames), frame_height, frame_width, 3), dtype=np.uint8
        )
        for i, f in enumerate(frames):
            resized[i] = np.asarray(
                Image.fromarray(f).resize((frame_width, frame_height))
            )
        return resized


def get_video_fps(video_path: str) -> float:
    """Return the frame rate (fps) of a video from its metadata.

    Falls back to 30.0 if FPS cannot be determined.
    """
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps > 0:
            return float(fps)
    except ImportError:
        pass

    return 30.0


def get_video_resolution(video_path: str) -> tuple[int, int]:
    """Return ``(height, width)`` of a video without loading frames.

    Falls back to loading 1 frame with imageio when OpenCV is unavailable.
    """
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if h > 0 and w > 0:
            return h, w
    except ImportError:
        pass

    # fallback: load one frame
    try:
        import imageio.v3 as iio

        frame = iio.imread(video_path, index=0, plugin="pyav")
        return frame.shape[0], frame.shape[1]
    except Exception:
        pass

    raise RuntimeError(f"Cannot determine resolution of {video_path}")


def load_video_frames(
    video_path: str,
    *,
    frame_start: int = 0,
    frame_height: Optional[int] = None,
    frame_width: Optional[int] = None,
) -> np.ndarray:
    """Load video frames, trying OpenCV first then imageio.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    frame_start : int
        Frame index to start reading from (for LED-sync crop).
    frame_height, frame_width : int or None
        If both given, resize every frame.  Pass ``None`` to keep
        the original resolution (useful for diagnostic overlays).

    Returns
    -------
    np.ndarray
        ``(N, H, W, 3)`` uint8 array of RGB frames.
    """
    try:
        return _try_load_frames_opencv(
            video_path,
            frame_start=frame_start,
            frame_height=frame_height,
            frame_width=frame_width,
        )
    except ImportError:
        frames = _try_load_frames_imageio(video_path, frame_start=frame_start)
        if frame_height is not None and frame_width is not None:
            frames = resize_frames(
                frames, frame_height=frame_height, frame_width=frame_width
            )
        return frames
