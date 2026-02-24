import os
from urllib.request import urlretrieve
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable, Union

import numpy as np

from .core import (
    DATASET_REGISTRY,
    _download_if_needed,
    _find_trigger_start,
    _load_mat,
    _resolve_name,
)

VIDEO_REGISTRY: Dict[str, Dict[str, str]] = {
    "rampa_positiva": {
        "filename": "rampa_positiva.MOV",
        "url": "https://osf.io/download/6988598ecc6c050425c72a2b/",
    },
    "rampa_negativa": {
        "filename": "rampa_negativa.MOV",
        "url": "https://osf.io/download/698859902f1925341e85bb72/",
    },
    "swept_sine": {
        "filename": "swept_sine.MOV",
        "url": "https://osf.io/download/6988598dcc6c050425c72a29/",
    },
    "multisine_01": {
        "filename": "multisine_01.MOV",
        "url": "https://osf.io/download/6wuxp/",
    },
    "multisine_02": {
        "filename": "multisine_02.MOV",
        "url": "https://osf.io/download/698859903a9f3f11e6c72c5b/",
    },
    "random_steps_W": {
        "filename": "random_steps_W.MOV",
        "url": "https://osf.io/download/6988598efa846778d990453a/",
    },
    "random_steps_X": {
        "filename": "random_steps_X.MOV",
        "url": "https://osf.io/download/698859902f1925341e85bb74/",
    },
    "random_steps_Y": {
        "filename": "random_steps_Y.MOV",
        "url": "https://osf.io/download/wujtg/",
    },
    "random_steps_Z": {
        "filename": "random_steps_Z.MOV",
        "url": "https://osf.io/download/69885990fa846778d990453c/",
    },
}

VIDEO_ALIASES = {
    "multisine_05": "multisine_01",
    "multisine_06": "multisine_02",
}

LED_FRAME_MAP = {
    "swept_sine": 291,
    "rampa_positiva": 415,
    "rampa_negativa": 266,
    "random_steps_W": 273,
    "random_steps_X": 375,
    "random_steps_Y": 238,
    "random_steps_Z": 288,
    "multisine_01": 279,
    "multisine_02": 260,
}


def list_videos():
    return sorted(VIDEO_REGISTRY.keys())


def _resolve_video_name(name: str) -> str:
    if name in VIDEO_REGISTRY:
        return name
    if name in VIDEO_ALIASES:
        return VIDEO_ALIASES[name]
    return name


def _default_video_dir() -> str:
    env_dir = os.getenv("BAB_DATASETS_VIDEO_DIR")
    if env_dir:
        return os.path.abspath(env_dir)
    pkg_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(pkg_dir, "..", "..", "videos_BAB"))


def _download_video_if_needed(url: str, filename: str, video_dir: str) -> str:
    os.makedirs(video_dir, exist_ok=True)
    path = os.path.join(video_dir, filename)
    if not os.path.exists(path):
        if not url:
            raise FileNotFoundError(
                f"Video not found at {path}. Set BAB_DATASETS_VIDEO_DIR or provide video_dir."
            )
        _download_with_progress(url, path)
    return path


def _download_with_progress(url: str, path: str) -> None:
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        tqdm = None

    if tqdm is None:
        urlretrieve(url, path)
        return

    pbar = None

    def _hook(block_num: int, block_size: int, total_size: int):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading video")
        downloaded = block_num * block_size
        pbar.update(max(0, downloaded - pbar.n))

    try:
        urlretrieve(url, path, reporthook=_hook)
    finally:
        if pbar is not None:
            pbar.close()


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - import-time guard
        raise RuntimeError(
            "OpenCV is required for video loading. Install with: pip install opencv-python"
        ) from exc
    return cv2


def load_video_path(name: str, video_dir: Optional[str] = None) -> str:
    name = _resolve_video_name(name)
    if name not in VIDEO_REGISTRY:
        raise ValueError(f"Unknown video '{name}'. Available: {list_videos()}")
    entry = VIDEO_REGISTRY[name]
    video_dir = video_dir or _default_video_dir()
    return _download_video_if_needed(entry.get("url", ""), entry["filename"], video_dir)


def load_video_metadata(path: str) -> Tuple[int, float, int, int]:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frame_count, fps, width, height


def iter_video_frames(
    path: str,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    grayscale: bool = True,
) -> Iterable[Tuple[int, np.ndarray]]:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = start_frame
    read_count = 0
    while True:
        if max_frames is not None and read_count >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yield idx, frame
        idx += 1
        read_count += 1
    cap.release()


def _roi_bounds(
    width: int,
    height: int,
    roi: str,
    roi_frac: float,
    roi_bounds: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[int, int, int, int]:
    if roi_bounds is not None:
        x0, x1, y0, y1 = roi_bounds
        return int(x0), int(x1), int(y0), int(y1)
    roi_frac = float(max(min(roi_frac, 1.0), 0.05))
    w = int(width * roi_frac)
    h = int(height * roi_frac)
    if roi == "lower_left":
        return 0, w, height - h, height
    if roi == "lower_right":
        return width - w, width, height - h, height
    if roi == "upper_left":
        return 0, w, 0, h
    if roi == "upper_right":
        return width - w, width, 0, h
    raise ValueError(f"Unknown roi '{roi}'.")


def extract_led_trace(
    path: str,
    roi: str = "lower_left",
    roi_frac: float = 0.25,
    roi_bounds: Optional[Tuple[int, int, int, int]] = None,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    smoothing_window: int = 5,
) -> Tuple[np.ndarray, float]:
    frame_count, fps, width, height = load_video_metadata(path)
    x0, x1, y0, y1 = _roi_bounds(width, height, roi, roi_frac, roi_bounds)
    trace = []
    for _, frame in iter_video_frames(
        path, start_frame=start_frame, max_frames=max_frames, grayscale=True
    ):
        roi_frame = frame[y0:y1, x0:x1]
        trace.append(float(np.mean(roi_frame)))
    trace = np.asarray(trace, dtype=np.float32)
    if smoothing_window and smoothing_window > 1 and trace.size >= smoothing_window:
        kernel = np.ones(int(smoothing_window), dtype=np.float32) / float(smoothing_window)
        trace = np.convolve(trace, kernel, mode="same")
    return trace, fps


def _trace_from_frames(
    frames: np.ndarray,
    roi: str,
    roi_frac: float,
    roi_bounds: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    if frames.size == 0:
        return np.zeros((0,), dtype=np.float32)
    height = frames.shape[1]
    width = frames.shape[2]
    x0, x1, y0, y1 = _roi_bounds(width, height, roi, roi_frac, roi_bounds)
    if frames.ndim == 4:
        roi_frames = frames[:, y0:y1, x0:x1, :]
        trace = roi_frames.mean(axis=(1, 2, 3))
    else:
        roi_frames = frames[:, y0:y1, x0:x1]
        trace = roi_frames.mean(axis=(1, 2))
    return trace.astype(np.float32)


def detect_led_onset(
    trace: np.ndarray,
    fps: float,
    threshold: Optional[float] = None,
    baseline_seconds: float = 1.0,
    min_seconds: float = 0.0,
) -> int:
    if trace.size == 0:
        return 0
    min_idx = int(max(0, np.floor(min_seconds * fps)))
    if threshold is None:
        n0 = int(max(5, min(trace.size, baseline_seconds * fps)))
        baseline = trace[:n0]
        mean = float(np.mean(baseline))
        std = float(np.std(baseline))
        threshold = mean + 5.0 * std
        if std < 1e-3:
            threshold = mean + 0.05 * float(np.max(trace) - np.min(trace) + 1.0)
    idx = np.where(trace >= threshold)[0]
    if idx.size == 0:
        return 0
    idx = idx[idx >= min_idx]
    if idx.size == 0:
        return 0
    return int(idx[0])


@dataclass
class VideoSync:
    video_name: str
    dataset_name: str
    fps: float
    sampling_time: float
    frame_start: int
    sample_start: int
    frame_count: int
    roi: str
    roi_frac: float
    led_threshold: Optional[float]

    def frame_to_time(self, frame_idx: int) -> float:
        return (frame_idx - self.frame_start) / self.fps

    def sample_to_time(self, sample_idx: int) -> float:
        return (sample_idx - self.sample_start) * self.sampling_time

    def frame_time_vector(self, n_frames: Optional[int] = None) -> np.ndarray:
        if n_frames is None:
            n_frames = self.frame_count
        return (np.arange(int(n_frames)) - self.frame_start) / self.fps

    def sample_time_vector(self, n_samples: int) -> np.ndarray:
        return (np.arange(int(n_samples)) - self.sample_start) * self.sampling_time


def sync_video_with_dataset(
    video_name: str,
    dataset_name: str,
    video_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    led_frame: Optional[int] = None,
) -> VideoSync:
    video_path = load_video_path(video_name, video_dir=video_dir)

    dataset_name = _resolve_name(dataset_name)
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )
    entry = DATASET_REGISTRY[dataset_name]
    data_dir = data_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    mat_path = _download_if_needed(entry["url"], entry["filename"], data_dir)
    t, _, _, trig, _, _ = _load_mat(mat_path)

    frame_count, fps, _, _ = load_video_metadata(video_path)
    if led_frame is None:
        led_frame = LED_FRAME_MAP.get(video_name)
    if led_frame is None:
        raise ValueError(f"No LED frame provided for video '{video_name}'.")
    frame_start = int(led_frame)
    sample_start = 0
    sampling_time = float(np.average(np.diff(t)))

    return VideoSync(
        video_name=video_name,
        dataset_name=dataset_name,
        fps=fps,
        sampling_time=sampling_time,
        frame_start=frame_start,
        sample_start=sample_start,
        frame_count=frame_count,
        roi="manual",
        roi_frac=0.0,
        led_threshold=None,
    )


def load_synced_video_frames(
    video_name: str,
    data,
    video_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    led_frame: Optional[int] = None,
    max_frames: Optional[int] = None,
    grayscale: bool = False,
    verbose: bool = False,
    debug_plots: bool = False,
    preallocate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, VideoSync]:
    """
    Load video frames aligned to the preprocessed dataset timebase.
    Returns (video_t, frames, sync), where video_t starts at 0 (LED onset).
    """
    if verbose:
        print("Loading video sync metadata...")
    video_path = load_video_path(video_name, video_dir=video_dir)

    sync = sync_video_with_dataset(
        video_name=video_name,
        dataset_name=data.name,
        video_dir=video_dir,
        data_dir=data_dir,
        led_frame=led_frame,
    )
    # use preprocessed dataset sampling time for alignment
    sync.sampling_time = float(data.sampling_time)
    if verbose:
        print(
            f"Sync done. frame_start={sync.frame_start} sample_start={sync.sample_start} "
            f"fps={sync.fps:.3f} sampling_time={sync.sampling_time:.6f}"
        )

    if debug_plots:
        try:
            import matplotlib.pyplot as plt

            for _, frame in iter_video_frames(
                video_path,
                start_frame=sync.frame_start,
                max_frames=1,
                grayscale=False,
            ):
                plt.figure(figsize=(6, 3))
                plt.imshow(frame[:, :, ::-1])
                plt.title(f"Frame at LED onset (frame {sync.frame_start})")
                plt.axis("off")
                plt.show()
                break
        except Exception as exc:  # pragma: no cover - debug only
            if verbose:
                print(f"Debug plots failed: {exc}")

    duration = float(len(data.y) * data.sampling_time)
    start = sync.frame_start
    end = int(min(sync.frame_count, (sync.frame_start + duration * sync.fps)))
    if max_frames is not None:
        end = min(end, start + int(max_frames))
    if verbose:
        print(f"Loading frames {start}..{end} (total {max(0, end-start)})")

    total = max(0, end - start)
    frames = []
    iterator = iter_video_frames(
        video_path,
        start_frame=start,
        max_frames=end - start,
        grayscale=grayscale,
    )

    if preallocate and total > 0:
        first = None
        for _, frame in iterator:
            first = frame
            break
        if first is None:
            frames = np.zeros((0,), dtype=np.uint8)
        else:
            # allocate and fill
            if first.ndim == 2:
                frames = np.zeros((total, first.shape[0], first.shape[1]), dtype=first.dtype)
            else:
                frames = np.zeros((total, first.shape[0], first.shape[1], first.shape[2]), dtype=first.dtype)
            frames[0] = first
            idx = 1
            next_mark = 10
            for _, frame in iterator:
                if idx >= total:
                    break
                frames[idx] = frame
                idx += 1
                if verbose and total > 0:
                    pct = int(idx * 100 / total)
                    if pct >= next_mark:
                        print(f"  loaded {idx}/{total} frames ({pct}%)")
                        next_mark += 10
            if idx < total:
                frames = frames[:idx]
    else:
        for _, frame in iterator:
            frames.append(frame)
            if verbose and len(frames) % 300 == 0:
                print(f"  loaded {len(frames)} frames...")
        if frames:
            try:
                from tqdm.auto import tqdm  # type: ignore
                stack_iter = tqdm([0], desc="Stacking frames")
            except Exception:
                stack_iter = [0]
            for _ in stack_iter:
                frames = np.stack(frames, axis=0)
        else:
            frames = np.zeros((0,), dtype=np.uint8)
    video_t = np.arange(len(frames)) / sync.fps

    if verbose:
        if len(video_t) > 0:
            print(f"Loaded {len(frames)} frames. video_t range {video_t[0]:.2f}s..{video_t[-1]:.2f}s")
        print(f"Done. frames.shape={frames.shape}")
    return video_t, frames, sync


def create_sync_gif(
    video_t: np.ndarray,
    frames: np.ndarray,
    data,
    sync: VideoSync,
    window_s: float = 5.0,
    fps: Optional[float] = None,
    out_path: str = "/tmp/bab_sync_preview.gif",
    verbose: bool = False,
) -> str:
    """
    Create a 2-row GIF: top is video frame, bottom is u/y with a time marker.
    Returns the GIF path and displays it if running in a notebook.
    """
    import matplotlib.pyplot as plt

    try:
        import imageio
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("create_sync_gif requires imageio. Install with pip install imageio") from exc

    t = sync.sample_time_vector(len(data.y))
    t_mid = 0.5 * (t[0] + t[-1])
    # clamp window to data range and keep exact window_s length
    t0 = max(t[0], t_mid - window_s / 2)
    t1 = min(t[-1], t0 + window_s)
    t0 = max(t[0], t1 - window_s)

    vmask = (video_t >= t0) & (video_t <= t1)
    vid_idx = np.where(vmask)[0]
    dmask = (t >= t0) & (t <= t1)

    if vid_idx.size == 0:
        raise ValueError("No video frames found in the requested window.")

    fps_out = float(fps if fps is not None else min(sync.fps, 15))
    frames_out = []
    if verbose:
        print("create_sync_gif")
        print(f"  window_s={window_s:.2f}  t0={t0:.2f}s  t1={t1:.2f}s")
        print(f"  video frames in window={len(vid_idx)}  fps_out={fps_out:.2f}")
        print(f"  data samples in window={int(np.sum(dmask))}")

    for i in vid_idx:
        if verbose and (len(frames_out) == 0 or (len(frames_out) + 1) % 100 == 0):
            print(f"  rendering frame {len(frames_out)+1}/{len(vid_idx)}")
        fig, ax = plt.subplots(
            2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 2]}
        )

        frame = frames[i]
        if frame.ndim == 3:
            ax[0].imshow(frame[:, :, ::-1])  # BGR -> RGB
        else:
            ax[0].imshow(frame, cmap="gray")
        ax[0].set_title(f"t={video_t[i]:.2f}s")
        ax[0].axis("off")

        # plot only the 0..window_s segment and fill as time advances
        tw = t[dmask] - t0
        u_w = data.u[dmask]
        y_w = data.y[dmask]
        current_t = video_t[i] - t0
        mask_cur = tw <= current_t

        ax[1].plot(tw[mask_cur], u_w[mask_cur], label="u", alpha=0.7)
        ax[1].plot(tw[mask_cur], y_w[mask_cur], label="y", alpha=0.7)
        ax[1].axvline(current_t, color="k", lw=1)
        ax[1].set_xlim(0, window_s)
        ax[1].set_xlabel("time (s)")
        ax[1].set_title("u / y")
        ax[1].legend()
        ax[1].grid(True)

        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        img = img[:, :, :3].copy()
        frames_out.append(img)
        plt.close(fig)

    imageio.mimsave(out_path, frames_out, duration=1.0 / fps_out, loop=0)

    try:
        from IPython.display import Image, display  # type: ignore

        display(Image(filename=out_path))
    except Exception:
        pass

    return out_path


def make_sync_preview(
    frames: np.ndarray,
    video_t: np.ndarray,
    data,
    out_path: str = "/tmp/sync_preview.mp4",
    window_s: float = 5.0,
    window_start: Union[float, str] = 0.0,
    fps: Optional[float] = None,
    include_y_dot: bool = False,
) -> str:
    """
    Create a preview video (MP4 or GIF) showing video frames over u/y signals.
    Uses only frames/video_t/data; no re-reading of video from disk.
    """
    import matplotlib.pyplot as plt

    if window_start == "mid":
        t = data.time_vector()
        window_start = float(0.5 * (t[0] + t[-1]) - window_s / 2)
    window_start = float(window_start)
    t0, t1 = window_start, window_start + window_s

    t = data.time_vector()
    mask = (t >= t0) & (t <= t1)
    tw = t[mask] - t0
    u_win = data.u[mask]
    y_win = data.y[mask]
    ydot_win = data.y_dot[mask] if include_y_dot and data.y_dot is not None else None

    vmask = (video_t >= t0) & (video_t <= t1)
    vid_idx = np.where(vmask)[0]
    if vid_idx.size == 0:
        raise ValueError("No frames found in the requested window.")

    if fps is None:
        if video_t.size > 1:
            fps = float(1.0 / np.median(np.diff(video_t)))
        else:
            fps = 30.0
    fps = float(fps)

    def _render_plot(t_now_win):
        n_rows = 2 if ydot_win is not None else 1
        fig, ax = plt.subplots(
            n_rows, 1, figsize=(8, 4), gridspec_kw={"height_ratios": [2] * n_rows}
        )
        if n_rows == 1:
            ax = [ax]

        ax[0].plot(tw, u_win, label="u", alpha=0.7)
        ax[0].plot(tw, y_win, label="y", alpha=0.7)
        ax[0].axvline(t_now_win, color="k", lw=1)
        ax[0].set_xlim(0, window_s)
        ax[0].set_xlabel("time (s)")
        ax[0].set_title(f"u / y ({window_s:.2f}s window)")
        ax[0].legend()
        ax[0].grid(True)

        if ydot_win is not None:
            ax[1].plot(tw, ydot_win, label="y_dot", color="tab:green", alpha=0.8)
            ax[1].axvline(t_now_win, color="k", lw=1)
            ax[1].set_xlim(0, window_s)
            ax[1].set_xlabel("time (s)")
            ax[1].set_title(f"y_dot ({window_s:.2f}s window)")
            ax[1].legend()
            ax[1].grid(True)

        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return img

    if out_path.lower().endswith(".gif"):
        try:
            import imageio
        except Exception as exc:
            raise RuntimeError("GIF output requires imageio. Install with pip install imageio") from exc
        frames_out = []
        try:
            from tqdm.auto import tqdm  # type: ignore
            iterator = tqdm(vid_idx, desc="Rendering GIF")
        except Exception:
            iterator = vid_idx
        for i in iterator:
            t_now_win = video_t[i] - t0
            img = _render_plot(t_now_win)
            frames_out.append(img)
        imageio.mimsave(out_path, frames_out, duration=1.0 / fps, loop=0)
        return out_path

    cv2 = _require_cv2()
    writer = None
    try:
        from tqdm.auto import tqdm  # type: ignore
        iterator = tqdm(vid_idx, desc="Rendering MP4")
    except Exception:
        iterator = vid_idx
    for i in iterator:
        t_now_win = video_t[i] - t0
        plot_img = _render_plot(t_now_win)
        # resize plot to match video width
        plot_img = cv2.resize(plot_img, (frames[i].shape[1], int(frames[i].shape[0] * 0.5)))
        combined = np.vstack([frames[i], plot_img])
        if writer is None:
            h, w = combined.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        writer.write(combined)
    if writer:
        writer.release()
    return out_path
