#!/usr/bin/env python
"""Train/evaluate the BAB multimodal pipeline.

Most operational defaults live in ``src.config.BabVideoPipelineConfig`` so
the example CLI exposes only the main run controls.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    BAB_VIDEO_ENCODERS,
    BAB_VIDEO_ODE_MODELS,
    BAB_VIDEO_PIPELINE_MODES,
    BabVideoPipelineConfig,
)
from src.data.registry import ensure_keypoint_labels, ensure_theta_labels, ensure_true_labels
from src.vision.datasets import load_bab_with_video
from src.vision.pipeline_utils import save_json
from src.vision.utils.train import MODE_RUNNERS, mode_requires_encoder


def _build_parser(defaults: BabVideoPipelineConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BAB video + sensor multimodal training pipeline.")
    parser.add_argument("--dataset", default=defaults.dataset, help="BAB sensor dataset key.")
    parser.add_argument("--mode", default=defaults.mode, choices=BAB_VIDEO_PIPELINE_MODES, help="Training mode.")
    parser.add_argument("--encoder", default=defaults.encoder, choices=BAB_VIDEO_ENCODERS, help="Encoder family.")
    parser.add_argument("--ode-model", default=defaults.ode_model, choices=BAB_VIDEO_ODE_MODELS, help="ODE dynamics family.")
    parser.add_argument("--wandb", default=defaults.wandb_project, help="W&B project name.")
    parser.add_argument("--run-name", default=defaults.run_name, help="Optional run name override.")
    parser.add_argument("--encoder-checkpoint", default=defaults.encoder_checkpoint, help="Checkpoint to reuse for encoder-based ODE training.")
    parser.add_argument("--ode-checkpoint", default=defaults.ode_checkpoint, help="Optional checkpoint for joint encoder/ODE modes.")
    parser.add_argument(
        "--ode-use-encoder-labels",
        default=defaults.ode_use_encoder_labels,
        action=argparse.BooleanOptionalAction,
        help="Train the ODE against encoder predictions instead of ground-truth sensor theta.",
    )
    return parser


def _build_config(args) -> BabVideoPipelineConfig:
    return BabVideoPipelineConfig(
        dataset=args.dataset,
        mode=args.mode,
        encoder=args.encoder,
        ode_model=args.ode_model,
        wandb_project=args.wandb,
        run_name=args.run_name,
        encoder_checkpoint=args.encoder_checkpoint,
        ode_checkpoint=args.ode_checkpoint,
        ode_use_encoder_labels=args.ode_use_encoder_labels,
    )


def _load_video_map(config: BabVideoPipelineConfig) -> dict | None:
    return None if config.video_map_json is None else json.loads(Path(config.video_map_json).read_text())


def _resolve_label_csvs(config: BabVideoPipelineConfig) -> tuple[str | None, str | None]:
    true_labels = ensure_true_labels(config.dataset)
    keypoint_path = config.keypoint_labels_csv or ensure_keypoint_labels(config.dataset) or true_labels
    theta_path = config.theta_labels_csv or ensure_theta_labels(config.dataset) or true_labels
    return tuple(None if path is None else str(path) for path in (keypoint_path, theta_path))


def _build_encoder(config: BabVideoPipelineConfig):
    from src.vision.models import EncoderThetaNet, PoseResNet50

    return {
        "theta_regression": lambda: EncoderThetaNet(pretrained=config.pretrained),
        "pose_heatmap": lambda: PoseResNet50(num_keypoints=2, pretrained=config.pretrained),
    }[config.encoder]()


def main() -> None:
    defaults = BabVideoPipelineConfig()
    parser = _build_parser(defaults)
    args = parser.parse_args()
    config = _build_config(args)

    if config.mode == "ode_retrain" and config.ode_use_encoder_labels and not config.encoder_checkpoint:
        parser.error(
            "--encoder-checkpoint is required for --mode ode_retrain "
            "when --ode-use-encoder-labels is enabled"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_root) / f"{config.resolved_run_name()}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    save_json(config.to_dict(), run_dir / "config.json")

    keypoint_csv, theta_csv = _resolve_label_csvs(config)
    data, frames, frame_idx_map, aux = load_bab_with_video(
        config.dataset,
        video_dir=config.video_dir,
        video_path=config.video_path,
        video_map=_load_video_map(config),
        resample_factor=config.resample_factor,
        video_fps=config.video_fps,
        frame_height=config.frame_height,
        frame_width=config.frame_width,
        preprocess=True,
        led_frame=config.led_frame,
        use_led_sync=config.use_led_sync,
        keypoint_labels_csv=keypoint_csv,
        theta_labels_csv=theta_csv,
        align_theta=config.align_theta,
        alignment_offset_min_s=config.alignment_offset_min_s,
        alignment_offset_max_s=config.alignment_offset_max_s,
        auto_match_video_fps=config.auto_match_video_fps,
        return_aux=True,
    )
    print(f"Sensor samples: {len(data)}, Video frames (aligned): {len(frames)}")
    if "theta_alignment" in aux:
        print("Theta alignment:", aux["theta_alignment"])
        save_json(aux["theta_alignment"], run_dir / "theta_alignment.json")

    encoder = _build_encoder(config) if mode_requires_encoder(config) else None
    MODE_RUNNERS[config.mode](config, encoder, data, frames, frame_idx_map, aux, run_dir)
    print(f"\nDone. Results in {run_dir}")


if __name__ == "__main__":
    main()
