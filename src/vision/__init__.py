"""Lazy public API for the multimodal BAB vision pipeline."""

from __future__ import annotations

import importlib as _importlib

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # models
    "EncoderThetaNet": (".models", "EncoderThetaNet"),
    "PoseResNet50": (".models", "PoseResNet50"),
    "DecoderKeypointsMLP": (".models", "DecoderKeypointsMLP"),
    "DecoderFrameDeconv": (".models", "DecoderFrameDeconv"),
    "EncOdeDecModel": (".models", "EncOdeDecModel"),
    # datasets
    "FrameStateDataset": (".datasets", "FrameStateDataset"),
    "WindowedSequenceDataset": (".datasets", "WindowedSequenceDataset"),
    "load_bab_with_video": (".datasets", "load_bab_with_video"),
    # losses
    "mse_state": (".losses", "mse_state"),
    "mse_keypoints": (".losses", "mse_keypoints"),
    "compute_losses": (".losses", "compute_losses"),
    # training
    "train_encoder": (".train", "train_encoder"),
    "train_decoder": (".train", "train_decoder"),
    "train_image_decoder": (".train", "train_image_decoder"),
    "train_end_to_end": (".train", "train_end_to_end"),
    # evaluation
    "evaluate_encoder_framewise": (".eval", "evaluate_encoder_framewise"),
    "evaluate_ode_rollout": (".eval", "evaluate_ode_rollout"),
    "evaluate_decoder": (".eval", "evaluate_decoder"),
    "evaluate_image_decoder": (".eval", "evaluate_image_decoder"),
    "evaluate_end2end": (".eval", "evaluate_end2end"),
}

_LAZY_SUBMODULES = {"datasets", "eval", "losses", "models", "pipeline_utils", "train", "utils"}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    if name in _LAZY_SUBMODULES:
        return _importlib.import_module(f".{name}", __package__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS) + list(_LAZY_SUBMODULES)
