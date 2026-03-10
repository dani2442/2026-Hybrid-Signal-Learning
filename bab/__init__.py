"""Ball-and-Beam DLC-to-PyTorch pipeline.

Submodules with heavy dependencies (cv2, deeplabcut, tensorflow) are
imported lazily so that lightweight imports like ``from bab import
BABAutoencoder`` work even when those packages are not installed.
"""


def __getattr__(name: str):
    # ── models (no heavy deps) ──────────────────────────────────────
    _model_names = {
        "DLCResNet50", "PoseResNet50",
        "BABEncoder", "BABDecoder", "BABAutoencoder",
    }
    if name in _model_names:
        from . import models
        return getattr(models, name)

    # ── convert (needs tensorflow) ──────────────────────────────────
    _convert_names = {"convert_dlc_tf_to_pytorch", "inspect_tf_checkpoint"}
    if name in _convert_names:
        from . import convert
        return getattr(convert, name)

    # ── inference (needs cv2, torch) ────────────────────────────────
    _inference_names = {
        "analyze_video_pytorch", "predict_frame_dlc",
        "predict_frame", "compute_theta",
    }
    if name in _inference_names:
        from . import inference
        return getattr(inference, name)

    # ── dlc_pipeline (needs deeplabcut) ─────────────────────────────
    _dlc_names = {"run_dlc_training", "run_dlc_analysis", "extract_theta_from_dlc"}
    if name in _dlc_names:
        from . import dlc_pipeline
        return getattr(dlc_pipeline, name)

    # ── video dataset (needs cv2) ───────────────────────────────────
    if name == "BABVideoDataset":
        from .video_dataset import BABVideoDataset
        return BABVideoDataset

    # ── autoencoder training ────────────────────────────────────────
    if name == "train_autoencoder":
        from .training import train_autoencoder
        return train_autoencoder

    raise AttributeError(f"module 'bab' has no attribute {name!r}")
