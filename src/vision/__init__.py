"""Multimodal encoder–NeuralODE–decoder vision pipeline.

Provides image-based encoder, keypoint decoder, and end-to-end
training/evaluation for BAB video + sensor data.
"""

from .models import EncoderThetaNet, PoseResNet50, DecoderKeypointsMLP, EncOdeDecModel
from .datasets import FrameStateDataset, WindowedSequenceDataset, load_bab_with_video
from .losses import mse_state, mse_keypoints, compute_losses
from .train import train_encoder, train_decoder, train_end_to_end
from .eval import (
    evaluate_encoder_framewise,
    evaluate_ode_rollout,
    evaluate_decoder,
    evaluate_end2end,
)

__all__ = [
    # models
    "EncoderThetaNet",
    "PoseResNet50",
    "DecoderKeypointsMLP",
    "EncOdeDecModel",
    # datasets
    "FrameStateDataset",
    "WindowedSequenceDataset",
    "load_bab_with_video",
    # losses
    "mse_state",
    "mse_keypoints",
    "compute_losses",
    # training
    "train_encoder",
    "train_decoder",
    "train_end_to_end",
    # evaluation
    "evaluate_encoder_framewise",
    "evaluate_ode_rollout",
    "evaluate_decoder",
    "evaluate_end2end",
]
