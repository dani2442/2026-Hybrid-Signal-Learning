#!/usr/bin/env python
"""Train/evaluate the BAB multimodal pipeline (video, sensor, decoder labels)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

# Allow `python examples/run_bab_video_pipeline.py` from repo root.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vision.pipeline_utils import (
    evaluate_and_plot_encoder,
    evaluate_and_plot_ode_free_run,
    load_ode_func,
    make_run_dir,
    predict_encoder_framewise,
    save_json,
    save_metrics_csv,
    select_evenly_spaced,
    split_indices,
    states_from_sensor,
    train_and_evaluate_image_decoder,
    train_ode_model_separate,
)
from src.visualization.pipeline_plots import (
    plot_sensor_to_future_sensor,
    plot_sensor_to_video_coords,
    plot_sensor_to_video_overlay,
)


def _run_encoder_only(args, encoder, data, frames, frame_idx_map, aux, run_dir):
    from src.vision.datasets import FrameStateDataset
    from src.vision.train import train_encoder

    ds = FrameStateDataset(data, frames, frame_idx_map)
    train_ds, val_ds, test_ds = ds.split()
    print(f"Splits: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    result = train_encoder(
        encoder,
        train_ds,
        val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        wandb_project=args.wandb,
        wandb_run_name=f"enc_{args.dataset}",
        checkpoint_dir=str(run_dir),
    )
    print(f"Best val loss: {result['best_val_loss']:.6f}")

    metrics, _ = evaluate_and_plot_encoder(encoder, test_ds, aux, run_dir, device=args.device)
    print("Video->sensor metrics:", metrics)


def _require_keypoints(aux: Dict[str, object]) -> np.ndarray:
    kp = aux.get("keypoints_sensor")
    if kp is None:
        raise ValueError(
            "Keypoint labels are required. Provide --keypoint-labels-csv "
            "with beam_left/beam_right labels."
        )
    arr = np.asarray(kp, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("Aligned keypoints must have shape (N, 4).")
    return arr[:, :4]


def _run_decoder_only(args, data, frames, frame_idx_map, aux, run_dir):
    from src.vision.models import DecoderFrameDeconv

    states = states_from_sensor(data)
    frames_sensor = np.asarray(frames[frame_idx_map], dtype=np.float32) / 255.0
    t = np.asarray(data.t, dtype=float)

    decoder = DecoderFrameDeconv(
        frame_height=int(frames_sensor.shape[1]),
        frame_width=int(frames_sensor.shape[2]),
    )
    decoder_run = train_and_evaluate_image_decoder(
        decoder,
        states,
        frames_sensor,
        t,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        wandb_project=args.wandb,
        wandb_run_name=f"dec_img_{args.dataset}",
        run_dir=run_dir,
        plot_count=args.plot_count,
    )
    print(f"Decoder best val loss: {decoder_run['result']['best_val_loss']:.6f}")
    print("Sensor->video metrics:", decoder_run["metrics"])


def _run_enc_ode(args, encoder, data, frames, frame_idx_map, aux, run_dir):
    from src.vision.datasets import FrameStateDataset, WindowedSequenceDataset
    from src.vision.eval import evaluate_decoder, evaluate_end2end, evaluate_ode_rollout
    from src.vision.models import DecoderKeypointsMLP, EncOdeDecModel
    from src.vision.train import train_end_to_end

    keypoints_sensor = None
    if args.mode == "enc_ode_dec":
        keypoints_sensor = _require_keypoints(aux)

    seq_ds = WindowedSequenceDataset(
        data,
        frames,
        frame_idx_map,
        k_steps=args.k_steps,
        keypoints_sensor=keypoints_sensor,
    )
    train_ds, val_ds, test_ds = seq_ds.split()
    print(f"Sequence splits: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    ode_func = load_ode_func(args)
    decoder = DecoderKeypointsMLP() if args.mode == "enc_ode_dec" else None
    composite = EncOdeDecModel(encoder=encoder, ode_func=ode_func, decoder=decoder)

    result = train_end_to_end(
        composite,
        train_ds,
        val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        freeze_ode_epochs=args.freeze_ode_epochs,
        wandb_project=args.wandb,
        wandb_run_name=f"{args.mode}_{args.dataset}",
        checkpoint_dir=str(run_dir),
    )
    print(f"Best val loss: {result['best_val_loss']:.6f}")

    # 1) video -> sensor
    frame_ds = FrameStateDataset(data, frames, frame_idx_map)
    _, _, frame_test_ds = frame_ds.split()
    enc_metrics, _ = evaluate_and_plot_encoder(encoder, frame_test_ds, aux, run_dir, device=args.device)

    # 2) sensor -> future sensor (NeuralODE rollout)
    ode_metrics = evaluate_ode_rollout(composite, test_ds, init_from="sensor", device=args.device)
    save_metrics_csv(ode_metrics, run_dir / "metrics_sensor_to_future_sensor.csv")
    if len(test_ds) > 0:
        dt = 1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0
        plot_sensor_to_future_sensor(
            run_dir / "plot_sensor_to_future_sensor.png",
            composite,
            test_ds[0],
            dt=dt,
            device=args.device,
        )

    # 3) sensor -> video (decoder) when labels are available
    if decoder is not None and keypoints_sensor is not None:
        states = np.column_stack([
            data.y,
            data.y_dot if data.y_dot is not None else np.zeros_like(data.y),
        ]).astype(float)
        _, _, test_idx = split_indices(len(states))
        dec_metrics = evaluate_decoder(decoder, states[test_idx], keypoints_sensor[test_idx], device=args.device)
        save_metrics_csv(dec_metrics, run_dir / "metrics_sensor_to_video.csv")
        plot_sensor_to_video_coords(
            run_dir / "plot_sensor_to_video_coords.png",
            decoder,
            states[test_idx],
            keypoints_sensor[test_idx],
            t=np.asarray(data.t, dtype=float)[test_idx],
            device=args.device,
        )
        overlay_sel = select_evenly_spaced(test_idx, n=args.plot_count)
        plot_sensor_to_video_overlay(
            run_dir / "plot_sensor_to_video_overlay.png",
            decoder,
            states,
            keypoints_sensor,
            frames,
            frame_idx_map,
            overlay_sel,
            device=args.device,
        )

    e2e_metrics = evaluate_end2end(composite, test_ds, device=args.device)
    save_metrics_csv(e2e_metrics, run_dir / "metrics_end2end.csv")
    print("Video->sensor metrics:", enc_metrics)
    print("Sensor->future sensor metrics:", ode_metrics)
    if decoder is not None and keypoints_sensor is not None:
        print("Sensor->video metrics saved to metrics_sensor_to_video.csv")


def _run_ode_dec(args, data, frames, frame_idx_map, aux, run_dir):
    from src.vision.datasets import WindowedSequenceDataset
    from src.vision.eval import evaluate_ode_rollout

    # Train decoder first from state->video labels.
    _run_decoder_only(args, data, frames, frame_idx_map, aux, run_dir)

    # Report sensor->future sensor from ODE model (loaded/fresh).
    ode_func = load_ode_func(args)
    ds = WindowedSequenceDataset(data, frames, frame_idx_map, k_steps=args.k_steps)
    _, _, test_ds = ds.split()
    ode_metrics = evaluate_ode_rollout(ode_func, test_ds, init_from="sensor", device=args.device)
    save_metrics_csv(ode_metrics, run_dir / "metrics_sensor_to_future_sensor.csv")
    if len(test_ds) > 0:
        dt = 1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0
        plot_sensor_to_future_sensor(
            run_dir / "plot_sensor_to_future_sensor.png",
            ode_func,
            test_ds[0],
            dt=dt,
            device=args.device,
        )
    print("Sensor->future sensor metrics:", ode_metrics)


def _run_separate(args, encoder, data, frames, frame_idx_map, aux, run_dir):
    """Train encoder, ODE, and image decoder separately; then evaluate all."""
    from src.vision.datasets import FrameStateDataset
    from src.vision.models import DecoderFrameDeconv
    from src.vision.train import train_encoder

    # 1) Train encoder (video -> sensor)
    frame_ds = FrameStateDataset(data, frames, frame_idx_map)
    enc_train_ds, enc_val_ds, enc_test_ds = frame_ds.split()
    print(
        f"Encoder splits: {len(enc_train_ds)} train / "
        f"{len(enc_val_ds)} val / {len(enc_test_ds)} test"
    )

    enc_result = train_encoder(
        encoder,
        enc_train_ds,
        enc_val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        wandb_project=args.wandb,
        wandb_run_name=f"enc_sep_{args.dataset}",
        checkpoint_dir=str(run_dir),
    )
    print(f"Encoder best val loss: {enc_result['best_val_loss']:.6f}")

    enc_metrics, _ = evaluate_and_plot_encoder(encoder, enc_test_ds, aux, run_dir, device=args.device)

    # 2) Train ODE separately and evaluate free-run simulation.
    #    Optionally use encoder-predicted theta as ODE training targets.
    y_override = None
    if args.ode_use_encoder_labels:
        full_pred = predict_encoder_framewise(encoder, frame_ds, device=args.device)
        y_override = np.asarray(data.y, dtype=float).copy()
        y_override[full_pred["idx"]] = full_pred["pred"][:, 0]
        print(f"Using encoder-predicted theta for ODE training ({len(full_pred['idx'])} samples)")
    else:
        print("Using sensor theta labels for ODE training (ground truth).")
    ode_model, tr, va, te = train_ode_model_separate(args, data, run_dir, y_override=y_override)
    if args.ode_init_from_y_dot:
        print("Using sensor y_dot for ODE rollout initial theta_dot.")
    else:
        print("Using finite-difference theta for ODE rollout initial theta_dot.")
    ode_eval = evaluate_and_plot_ode_free_run(
        ode_model,
        data,
        te,
        run_dir,
        use_sensor_y_dot=args.ode_init_from_y_dot,
    )
    ode_metrics = ode_eval["metrics"]

    # 3) Train image decoder separately (sensor state -> frame).
    states = states_from_sensor(data)
    frames_sensor = np.asarray(frames[frame_idx_map], dtype=np.float32) / 255.0
    t_all = np.asarray(data.t, dtype=float)

    decoder = DecoderFrameDeconv(
        frame_height=int(frames_sensor.shape[1]),
        frame_width=int(frames_sensor.shape[2]),
    )
    decoder_run = train_and_evaluate_image_decoder(
        decoder,
        states,
        frames_sensor,
        t_all,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        wandb_project=args.wandb,
        wandb_run_name=f"dec_sep_{args.dataset}",
        run_dir=run_dir,
        plot_count=args.plot_count,
    )
    print(f"Decoder best val loss: {decoder_run['result']['best_val_loss']:.6f}")
    dec_metrics = decoder_run["metrics"]

    save_json(
        {
            "encoder_best_val_loss": float(enc_result["best_val_loss"]),
            "ode_train_points": int(len(tr)),
            "ode_val_points": int(len(va)),
            "ode_test_points": int(len(te)),
            "decoder_best_val_loss": float(decoder_run["result"]["best_val_loss"]),
        },
        run_dir / "separate_training_summary.json",
    )
    print("Video->sensor metrics:", enc_metrics)
    print("Sensor->future sensor (free-run) metrics:", ode_metrics)
    print("Sensor->video metrics:", dec_metrics)


def _run_ode_retrain(args, encoder, data, frames, frame_idx_map, aux, run_dir):
    """Train only the ODE, reusing an already-trained encoder."""
    import torch
    from src.vision.datasets import FrameStateDataset

    # Load encoder weights from checkpoint
    print(f"Loading encoder checkpoint: {args.encoder_checkpoint}")
    encoder.load_state_dict(
        torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=True)
    )
    encoder.eval()

    # Optionally generate encoder-predicted theta for ODE training targets
    y_override = None
    if args.ode_use_encoder_labels:
        frame_ds = FrameStateDataset(data, frames, frame_idx_map)
        full_pred = predict_encoder_framewise(encoder, frame_ds, device=args.device)
        y_override = np.asarray(data.y, dtype=float).copy()
        y_override[full_pred["idx"]] = full_pred["pred"][:, 0]
        print(f"Using encoder-predicted theta for ODE training "
              f"({len(full_pred['idx'])} samples)")
    else:
        print("Using sensor theta labels for ODE training (ground truth).")

    # Train ODE
    ode_model, tr, va, te = train_ode_model_separate(
        args, data, run_dir, y_override=y_override,
    )
    if args.ode_init_from_y_dot:
        print("Using sensor y_dot for ODE rollout initial theta_dot.")
    else:
        print("Using finite-difference theta for ODE rollout initial theta_dot.")
    ode_eval = evaluate_and_plot_ode_free_run(
        ode_model,
        data,
        te,
        run_dir,
        use_sensor_y_dot=args.ode_init_from_y_dot,
    )
    ode_metrics = ode_eval["metrics"]

    # Print learned parameters if physics model
    if hasattr(ode_model, "ode_func_") and ode_model.ode_func_ is not None:
        print("\nLearned ODE parameters:")
        for name, param in ode_model.ode_func_.named_parameters():
            val = param.detach().cpu().numpy()
            if "log_" in name:
                print(f"  {name} = {val:.4f}  (exp = {np.exp(val):.6f})")
            else:
                print(f"  {name} = {val:.4f}")

    print("\nSensor->future sensor (free-run) metrics:", ode_metrics)


def main():
    parser = argparse.ArgumentParser(description="BAB video + sensor multimodal training pipeline.")

    # Data
    parser.add_argument("--dataset", default="multisine_05", help="BAB sensor dataset key.")
    parser.add_argument("--video-dir", default=None, help="Directory containing video files.")
    parser.add_argument("--video-path", default=None, help="Single explicit video path.")
    parser.add_argument("--video-map-json", default=None, help="JSON dataset->video override map.")
    parser.add_argument("--resample-factor", type=int, default=50, help="Sensor resample factor.")
    parser.add_argument("--video-fps", type=float, default=30.0, help="Video fps.")
    parser.add_argument(
        "--no-auto-match-video-fps",
        action="store_true",
        help="Disable automatic sensor resampling to match video FPS.",
    )
    parser.add_argument("--frame-height", type=int, default=96, help="Loaded video frame height.")
    parser.add_argument("--frame-width", type=int, default=96, help="Loaded video frame width.")
    parser.add_argument("--keypoint-labels-csv", default=None, help="CSV with beam_left/beam_right labels.")
    parser.add_argument("--theta-labels-csv", default=None, help="CSV with t_s,theta_deg labels.")
    parser.add_argument("--led-frame", type=int, default=None, help="Manual LED-on frame override.")
    parser.add_argument("--no-led-sync", action="store_true", help="Disable LED trigger sync crop.")
    parser.add_argument("--no-theta-align", action="store_true", help="Disable theta offset/sign/scale alignment.")
    parser.add_argument("--alignment-offset-min-s", type=float, default=-12.0)
    parser.add_argument("--alignment-offset-max-s", type=float, default=12.0)

    # Mode
    parser.add_argument(
        "--mode",
        default="encoder_only",
        choices=["encoder_only", "decoder_only", "enc_ode", "ode_dec", "enc_ode_dec", "separate", "ode_retrain"],
        help="Training mode.",
    )
    parser.add_argument(
        "--encoder",
        default="theta_regression",
        choices=["theta_regression", "pose_heatmap"],
        help="Encoder architecture.",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--k-steps", type=int, default=20, help="ODE rollout length.")
    parser.add_argument("--freeze-ode-epochs", type=int, default=0)
    parser.add_argument(
        "--ode-model",
        default="linear_physics",
        choices=["linear_physics", "structured_node", "stribeck_physics"],
        help="ODE dynamics family (default: linear_physics).",
    )
    parser.add_argument("--ode-hidden-dim", type=int, default=128, help="Hidden width for structured ODE.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plot-count", type=int, default=6, help="Number of frame overlays for sensor->video plot.")
    parser.add_argument("--ode-checkpoint", default=None, help="Path to pretrained ODE checkpoint (.pt).")
    parser.add_argument("--encoder-checkpoint", default=None,
                        help="Path to trained encoder checkpoint (.pt) for --mode ode_retrain.")
    parser.add_argument("--ode-use-encoder-labels", action="store_true",
                        help="In --mode separate/ode_retrain, use encoder-predicted theta as ODE targets "
                             "(default: sensor ground-truth theta labels).")
    parser.add_argument(
        "--ode-init-from-y-dot",
        action="store_true",
        help="Use sensor y_dot to set ODE initial theta_dot (training and free-run). "
             "Default: finite difference from theta.",
    )
    parser.add_argument("--ode-training-mode", default=None,
                        choices=["full", "subsequence"],
                        help="ODE training strategy: 'full' (entire trajectory) or "
                             "'subsequence' (random windows). Default: model-specific.")

    # ODE-specific overrides
    parser.add_argument("--ode-batch-size", type=int, default=128)
    parser.add_argument(
        "--ode-lr",
        type=float,
        default=0.01,
        help="ODE learning rate override. Default: model config default "
             "(linear_physics/stribeck_physics: 1e-2).",
    )

    # Output
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb", default=None, help="W&B project name.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained encoder.")

    args = parser.parse_args()

    from src.data.registry import ensure_keypoint_labels, ensure_theta_labels
    from src.vision.datasets import load_bab_with_video
    from src.vision.models import DecoderKeypointsMLP, EncOdeDecModel, EncoderThetaNet, PoseResNet50

    run_name = args.run_name or f"{args.mode}_{args.dataset}"
    run_dir = make_run_dir(args.output_root, run_name)
    print(f"Run directory: {run_dir}")

    config = vars(args)
    save_json(config, run_dir / "config.json")

    video_map = None
    if args.video_map_json:
        with open(args.video_map_json) as f:
            video_map = json.load(f)

    # Auto-resolve label CSVs from registry when not explicitly provided
    kp_csv = args.keypoint_labels_csv
    if kp_csv is None:
        kp_path = ensure_keypoint_labels(args.dataset)
        kp_csv = str(kp_path) if kp_path else None
    theta_csv = args.theta_labels_csv
    if theta_csv is None:
        th_path = ensure_theta_labels(args.dataset)
        theta_csv = str(th_path) if th_path else None

    loaded = load_bab_with_video(
        args.dataset,
        video_dir=args.video_dir,
        video_path=args.video_path,
        video_map=video_map,
        resample_factor=args.resample_factor,
        video_fps=args.video_fps,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
        preprocess=True,
        led_frame=args.led_frame,
        use_led_sync=not args.no_led_sync,
        keypoint_labels_csv=kp_csv,
        theta_labels_csv=theta_csv,
        align_theta=not args.no_theta_align,
        alignment_offset_min_s=args.alignment_offset_min_s,
        alignment_offset_max_s=args.alignment_offset_max_s,
        auto_match_video_fps=not args.no_auto_match_video_fps,
        return_aux=True,
    )
    data, frames, frame_idx_map, aux = loaded
    print(f"Sensor samples: {len(data)}, Video frames (aligned): {len(frames)}")
    if "theta_alignment" in aux:
        print("Theta alignment:", aux["theta_alignment"])
        save_json(aux["theta_alignment"], run_dir / "theta_alignment.json")

    pretrained = not args.no_pretrained
    if args.encoder == "theta_regression":
        encoder = EncoderThetaNet(pretrained=pretrained)
    else:
        encoder = PoseResNet50(num_keypoints=2, pretrained=pretrained)

    if args.mode == "ode_retrain" and not args.encoder_checkpoint:
        parser.error("--encoder-checkpoint is required for --mode ode_retrain")

    if args.mode == "encoder_only":
        _run_encoder_only(args, encoder, data, frames, frame_idx_map, aux, run_dir)
    elif args.mode == "decoder_only":
        _run_decoder_only(args, data, frames, frame_idx_map, aux, run_dir)
    elif args.mode in ("enc_ode", "enc_ode_dec"):
        _run_enc_ode(args, encoder, data, frames, frame_idx_map, aux, run_dir)
    elif args.mode == "ode_dec":
        _run_ode_dec(args, data, frames, frame_idx_map, aux, run_dir)
    elif args.mode == "separate":
        _run_separate(args, encoder, data, frames, frame_idx_map, aux, run_dir)
    elif args.mode == "ode_retrain":
        _run_ode_retrain(args, encoder, data, frames, frame_idx_map, aux, run_dir)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(f"\nDone. Results in {run_dir}")


if __name__ == "__main__":
    main()
