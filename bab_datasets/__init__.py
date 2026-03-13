"""bab_datasets: lightweight loaders for nonlinear system ID datasets."""

from .core import InputOutputData, load_experiment, list_experiments

try:  # optional video utilities
    from .video import (
        VideoSync,
        create_sync_gif,
        make_sync_preview,
        list_videos,
        load_video_metadata,
        load_video_path,
        load_synced_video_frames,
        sync_video_with_dataset,
    )

    __all__ = [
        "InputOutputData",
        "load_experiment",
        "list_experiments",
        "VideoSync",
        "create_sync_gif",
        "make_sync_preview",
        "list_videos",
        "load_video_metadata",
        "load_video_path",
        "load_synced_video_frames",
        "sync_video_with_dataset",
    ]
except Exception:  # pragma: no cover - optional dependency guard
    __all__ = ["InputOutputData", "load_experiment", "list_experiments"]

__all__ = ["InputOutputData", "load_experiment", "list_experiments"]
