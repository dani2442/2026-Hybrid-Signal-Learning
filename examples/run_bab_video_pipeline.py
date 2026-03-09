#!/usr/bin/env python
"""Train and evaluate the multimodal encoder–NeuralODE–decoder pipeline.

Modes
-----
``encoder_only``   Train encoder (image → state) and evaluate per-frame.
``decoder_only``   Train decoder (state → keypoints) given labelled data.
``enc_ode``        Train encoder + ODE end-to-end (freeze ODE optionally).
``ode_dec``        Train ODE + decoder (sensor init).
``enc_ode_dec``    Full pipeline: encoder → ODE → decoder.

Usage::

    python examples/run_bab_video_pipeline.py \\
        --mode encoder_only \\
        --dataset multisine_05 \\
        --video-dir /path/to/videos \\
        --epochs 50

    python examples/run_bab_video_pipeline.py \\
        --mode enc_ode \\
        --dataset multisine_05 \\
        --video-dir /path/to/videos \\
        --ode-checkpoint checkpoints/structured_node.pt \\
        --freeze-ode-epochs 10 \\
        --epochs 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def _make_run_dir(output_root: str, run_name: str) -> Path:
    """Create a timestamped run directory."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_metrics_csv(metrics: dict, path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in sorted(metrics.items()):
            writer.writerow([k, f"{v:.6f}"])


def main():
    parser = argparse.ArgumentParser(
        description="BAB video + sensor multimodal training pipeline."
    )

    # Data
    parser.add_argument(
        "--dataset", default="multisine_05",
        help="BAB sensor dataset key (default: multisine_05).",
    )
    parser.add_argument(
        "--video-dir", default=None,
        help="Directory containing video files.",
    )
    parser.add_argument(
        "--video-path", default=None,
        help="Explicit path to a single video file (overrides --video-dir).",
    )
    parser.add_argument(
        "--video-map-json", default=None,
        help="JSON file with dataset→video name mapping overrides.",
    )
    parser.add_argument(
        "--resample-factor", type=int, default=50,
        help="Sensor resample factor (default: 50).",
    )
    parser.add_argument(
        "--video-fps", type=float, default=30.0,
        help="Video frame rate (default: 30).",
    )

    # Mode
    parser.add_argument(
        "--mode",
        default="encoder_only",
        choices=["encoder_only", "decoder_only", "enc_ode", "ode_dec", "enc_ode_dec"],
        help="Training mode.",
    )
    parser.add_argument(
        "--encoder", default="theta_regression",
        choices=["theta_regression", "pose_heatmap"],
        help="Encoder architecture.",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--k-steps", type=int, default=20, help="ODE rollout length.")
    parser.add_argument(
        "--freeze-ode-epochs", type=int, default=0,
        help="Epochs to freeze ODE params (enc_ode / enc_ode_dec modes).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    # ODE checkpoint (for modes that compose with a pretrained ODE)
    parser.add_argument(
        "--ode-checkpoint", default=None,
        help="Path to a pretrained ODE model checkpoint (.pt).",
    )

    # Output
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb", default=None, help="W&B project name.")
    parser.add_argument(
        "--no-pretrained", action="store_true",
        help="Don't use ImageNet-pretrained encoder backbone.",
    )

    args = parser.parse_args()

    # ── Imports (deferred to keep --help fast) ────────────────────────
    from src.vision.datasets import (
        FrameStateDataset,
        WindowedSequenceDataset,
        load_bab_with_video,
    )
    from src.vision.models import (
        DecoderKeypointsMLP,
        EncOdeDecModel,
        EncoderThetaNet,
        PoseResNet50,
    )
    from src.vision.train import train_decoder, train_encoder, train_end_to_end
    from src.vision import (
        evaluate_encoder_framewise,
        evaluate_ode_rollout,
        evaluate_decoder,
        evaluate_end2end,
    )

    # ── Run dir ───────────────────────────────────────────────────────
    run_name = args.run_name or f"{args.mode}_{args.dataset}"
    run_dir = _make_run_dir(args.output_root, run_name)
    print(f"Run directory: {run_dir}")

    # Save config
    config = vars(args)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Load data ─────────────────────────────────────────────────────
    video_map = None
    if args.video_map_json:
        with open(args.video_map_json) as f:
            video_map = json.load(f)

    data, frames, frame_idx_map = load_bab_with_video(
        args.dataset,
        video_dir=args.video_dir,
        video_path=args.video_path,
        video_map=video_map,
        resample_factor=args.resample_factor,
        video_fps=args.video_fps,
    )
    print(f"Sensor samples: {len(data)}, Video frames: {len(frames)}")

    # ── Build encoder ─────────────────────────────────────────────────
    pretrained = not args.no_pretrained
    if args.encoder == "theta_regression":
        encoder = EncoderThetaNet(pretrained=pretrained)
    else:
        encoder = PoseResNet50(num_keypoints=2, pretrained=pretrained)

    # ── Mode dispatch ─────────────────────────────────────────────────
    if args.mode == "encoder_only":
        _run_encoder_only(args, encoder, data, frames, frame_idx_map, run_dir)
    elif args.mode == "decoder_only":
        _run_decoder_only(args, data, run_dir)
    elif args.mode in ("enc_ode", "enc_ode_dec"):
        _run_enc_ode(args, encoder, data, frames, frame_idx_map, run_dir)
    elif args.mode == "ode_dec":
        _run_ode_dec(args, data, frames, frame_idx_map, run_dir)

    print(f"\nDone. Results in {run_dir}")


# ─────────────────────────────────────────────────────────────────────
# Mode implementations
# ─────────────────────────────────────────────────────────────────────

def _run_encoder_only(args, encoder, data, frames, frame_idx_map, run_dir):
    from src.vision.datasets import FrameStateDataset
    from src.vision.train import train_encoder
    from src.vision.eval import evaluate_encoder_framewise

    ds = FrameStateDataset(data, frames, frame_idx_map)
    train_ds, val_ds, test_ds = ds.split()
    print(f"Splits: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    result = train_encoder(
        encoder, train_ds, val_ds,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device, seed=args.seed,
        wandb_project=args.wandb, wandb_run_name=f"enc_{args.dataset}",
        checkpoint_dir=str(run_dir),
    )
    print(f"Best val loss: {result['best_val_loss']:.6f}")

    metrics = evaluate_encoder_framewise(encoder, test_ds, device=args.device)
    _save_metrics_csv(metrics, run_dir / "metrics_encoder.csv")
    print("Encoder test metrics:", metrics)


def _run_decoder_only(args, data, run_dir):
    """Decoder-only requires pre-existing keypoint labels (not yet wired)."""
    print(
        "decoder_only mode requires keypoint label arrays. "
        "Provide them by extending this script or loading from CSV."
    )


def _run_enc_ode(args, encoder, data, frames, frame_idx_map, run_dir):
    from src.vision.datasets import WindowedSequenceDataset
    from src.vision.models import EncOdeDecModel, DecoderKeypointsMLP
    from src.vision.train import train_end_to_end
    from src.vision.eval import evaluate_end2end

    ds = WindowedSequenceDataset(
        data, frames, frame_idx_map, k_steps=args.k_steps,
    )
    train_ds, val_ds, test_ds = ds.split()
    print(f"Splits: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    # Load or create ODE func
    ode_func = _load_ode_func(args)

    decoder = None
    if args.mode == "enc_ode_dec":
        decoder = DecoderKeypointsMLP()

    composite = EncOdeDecModel(
        encoder=encoder, ode_func=ode_func, decoder=decoder,
    )

    result = train_end_to_end(
        composite, train_ds, val_ds,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device, seed=args.seed,
        freeze_ode_epochs=args.freeze_ode_epochs,
        wandb_project=args.wandb, wandb_run_name=f"{args.mode}_{args.dataset}",
        checkpoint_dir=str(run_dir),
    )
    print(f"Best val loss: {result['best_val_loss']:.6f}")

    metrics = evaluate_end2end(composite, test_ds, device=args.device)
    _save_metrics_csv(metrics, run_dir / "metrics_end2end.csv")
    print("End-to-end test metrics:", metrics)


def _run_ode_dec(args, data, frames, frame_idx_map, run_dir):
    """ODE→Dec mode: train decoder from ODE rollout using sensor init."""
    print(
        "ode_dec mode: train a decoder from ODE-predicted states. "
        "Requires keypoint labels; extend this script or load from CSV."
    )


def _load_ode_func(args):
    """Load ODE dynamics func from checkpoint or create a default."""
    from src.models.blackbox_ode import _build_structured

    if args.ode_checkpoint:
        import torch
        from src.models.base import BaseModel

        model = BaseModel.load(args.ode_checkpoint)
        if hasattr(model, "func_") and model.func_ is not None:
            return model.func_
        raise ValueError(
            f"Checkpoint {args.ode_checkpoint} does not contain an ODE func."
        )

    # Default: create a fresh StructuredNODE func
    print("No --ode-checkpoint given; creating a fresh StructuredNODE dynamics.")
    return _build_structured(hidden_dim=128)


if __name__ == "__main__":
    main()
