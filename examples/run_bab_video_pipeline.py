#!/usr/bin/env python
"""Train/evaluate the BAB multimodal pipeline.

Most operational defaults live in ``src.config.BabVideoPipelineConfig`` so
the example CLI exposes only the main run controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

# Allow `python examples/run_bab_video_pipeline.py` from repo root.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import BAB_VIDEO_ODE_MODELS, BAB_VIDEO_PIPELINE_MODES, BabVideoPipelineConfig
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


def _build_parser() -> argparse.ArgumentParser:
    defaults = BabVideoPipelineConfig()
    parser = argparse.ArgumentParser(
        description="BAB video + sensor multimodal training pipeline."
    )
    parser.add_argument("--dataset", default=defaults.dataset, help="BAB sensor dataset key.")
    parser.add_argument(
        "--mode",
        default=defaults.mode,
        choices=BAB_VIDEO_PIPELINE_MODES,
        help="Training mode.",
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs, help="Training epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults.batch_size,
        help="Mini-batch size.",
    )
    parser.add_argument("--lr", type=float, default=defaults.lr, help="Learning rate.")
    parser.add_argument(
        "--k-steps",
        type=int,
        default=defaults.k_steps,
        help="ODE rollout length.",
    )
    parser.add_argument(
        "--ode-model",
        default=defaults.ode_model,
        choices=BAB_VIDEO_ODE_MODELS,
        help="ODE dynamics family.",
    )
    parser.add_argument("--run-name", default=defaults.run_name, help="Run name.")
    parser.add_argument(
        "--encoder-checkpoint",
        default=defaults.encoder_checkpoint,
        help="Required for --mode ode_retrain.",
    )
    parser.add_argument(
        "--output-root",
        default=defaults.output_root,
        help="Output directory for run artifacts.",
    )
    parser.add_argument("--wandb", default=defaults.wandb_project, help="W&B project name.")
    return parser


def _config_from_args(args: argparse.Namespace) -> BabVideoPipelineConfig:
    return BabVideoPipelineConfig(
        dataset=args.dataset,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        k_steps=args.k_steps,
        ode_model=args.ode_model,
        run_name=args.run_name,
        encoder_checkpoint=args.encoder_checkpoint,
        output_root=args.output_root,
        wandb_project=args.wandb,
    )


def _load_video_map(config: BabVideoPipelineConfig) -> dict | None:
    return None if config.video_map_json is None else json.loads(Path(config.video_map_json).read_text())


def _resolve_label_paths(config: BabVideoPipelineConfig) -> tuple[str | None, str | None]:
    from src.data.registry import ensure_keypoint_labels, ensure_theta_labels, ensure_true_labels

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


def _run_encoder_only(config, encoder, data, frames, frame_idx_map, aux, run_dir):
    from src.vision.datasets import FrameStateDataset
    from src.vision.train import train_encoder

    assert encoder is not None
    frame_ds = FrameStateDataset(data, frames, frame_idx_map)
    train_ds, val_ds, test_ds = frame_ds.split()
    print(f"Splits: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    result = train_encoder(
        encoder,
        train_ds,
        val_ds,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        device=config.device,
        seed=config.seed,
        wandb_project=config.wandb_project,
        wandb_run_name=f"enc_{config.dataset}",
        checkpoint_dir=str(run_dir),
    )
    print(f"Best val loss: {result['best_val_loss']:.6f}")

    metrics, _ = evaluate_and_plot_encoder(
        encoder,
        test_ds,
        aux,
        run_dir,
        plot_dataset=frame_ds,
        device=config.device,
    )
    print("Video->sensor metrics:", metrics)


def _require_keypoints(aux: dict) -> np.ndarray:
    kp = aux.get("keypoints_sensor")
    if kp is None:
        raise ValueError(
            "Keypoint labels are required. Configure keypoint_labels_csv "
            "or provide a dataset with true labels."
        )
    arr = np.asarray(kp, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("Aligned keypoints must have shape (N, 4).")
    return arr[:, :4]


def _sensor_video_data(data, frames, frame_idx_map):
    return (
        states_from_sensor(data),
        np.asarray(frames[frame_idx_map], dtype=np.float32) / 255.0,
        np.asarray(data.t, dtype=float),
    )


def _train_decoder(config, data, frames, frame_idx_map, run_dir, *, run_name: str):
    from src.vision.models import DecoderFrameDeconv

    states, frames_sensor, t = _sensor_video_data(data, frames, frame_idx_map)
    decoder = DecoderFrameDeconv(
        frame_height=int(frames_sensor.shape[1]),
        frame_width=int(frames_sensor.shape[2]),
    )
    return train_and_evaluate_image_decoder(
        decoder,
        states,
        frames_sensor,
        t,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        device=config.device,
        seed=config.seed,
        wandb_project=config.wandb_project,
        wandb_run_name=run_name,
        run_dir=run_dir,
        plot_count=config.plot_count,
        plot_states=states,
        plot_frames=frames_sensor,
        plot_t=t,
    )


def _evaluate_ode_test_rollout(config, data, test_ds, model_or_func, run_dir):
    from src.vision.eval import evaluate_ode_rollout

    ode_metrics = evaluate_ode_rollout(
        model_or_func,
        test_ds,
        init_from="sensor",
        device=config.device,
    )
    save_metrics_csv(ode_metrics, run_dir / "metrics_sensor_to_future_sensor.csv")
    if len(test_ds):
        plot_sensor_to_future_sensor(
            run_dir / "plot_sensor_to_future_sensor.png",
            model_or_func,
            test_ds[0],
            dt=1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0,
            device=config.device,
        )
    return ode_metrics


def _run_decoder_only(config, _encoder, data, frames, frame_idx_map, _aux, run_dir):
    decoder_run = _train_decoder(
        config,
        data,
        frames,
        frame_idx_map,
        run_dir,
        run_name=f"dec_img_{config.dataset}",
    )
    print(f"Decoder best val loss: {decoder_run['result']['best_val_loss']:.6f}")
    print("Sensor->video metrics:", decoder_run["metrics"])


def _run_enc_ode(config, encoder, data, frames, frame_idx_map, aux, run_dir):
    from src.vision.datasets import FrameStateDataset, WindowedSequenceDataset
    from src.vision.eval import evaluate_decoder, evaluate_end2end
    from src.vision.models import DecoderKeypointsMLP, EncOdeDecModel
    from src.vision.train import train_end_to_end

    assert encoder is not None
    keypoints_sensor = _require_keypoints(aux) if config.mode == "enc_ode_dec" else None

    seq_ds = WindowedSequenceDataset(
        data,
        frames,
        frame_idx_map,
        k_steps=config.k_steps,
        keypoints_sensor=keypoints_sensor,
    )
    train_ds, val_ds, test_ds = seq_ds.split()
    print(f"Sequence splits: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    ode_func = load_ode_func(config)
    decoder = DecoderKeypointsMLP() if config.mode == "enc_ode_dec" else None
    composite = EncOdeDecModel(encoder=encoder, ode_func=ode_func, decoder=decoder)

    result = train_end_to_end(
        composite,
        train_ds,
        val_ds,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        device=config.device,
        seed=config.seed,
        freeze_ode_epochs=config.freeze_ode_epochs,
        wandb_project=config.wandb_project,
        wandb_run_name=f"{config.mode}_{config.dataset}",
        checkpoint_dir=str(run_dir),
    )
    print(f"Best val loss: {result['best_val_loss']:.6f}")

    frame_ds = FrameStateDataset(data, frames, frame_idx_map)
    _, _, frame_test_ds = frame_ds.split()
    enc_metrics, _ = evaluate_and_plot_encoder(
        encoder,
        frame_test_ds,
        aux,
        run_dir,
        plot_dataset=frame_ds,
        device=config.device,
    )

    ode_metrics = _evaluate_ode_test_rollout(config, data, test_ds, composite, run_dir)

    if decoder is not None and keypoints_sensor is not None:
        states = states_from_sensor(data)
        _, _, test_idx = split_indices(len(states))
        dec_metrics = evaluate_decoder(
            decoder,
            states[test_idx],
            keypoints_sensor[test_idx],
            device=config.device,
        )
        save_metrics_csv(dec_metrics, run_dir / "metrics_sensor_to_video.csv")
        plot_sensor_to_video_coords(
            run_dir / "plot_sensor_to_video_coords.png",
            decoder,
            states[test_idx],
            keypoints_sensor[test_idx],
            t=np.asarray(data.t, dtype=float)[test_idx],
            device=config.device,
        )
        overlay_sel = select_evenly_spaced(test_idx, n=config.plot_count)
        plot_sensor_to_video_overlay(
            run_dir / "plot_sensor_to_video_overlay.png",
            decoder,
            states,
            keypoints_sensor,
            frames,
            frame_idx_map,
            overlay_sel,
            device=config.device,
        )

    e2e_metrics = evaluate_end2end(composite, test_ds, device=config.device)
    save_metrics_csv(e2e_metrics, run_dir / "metrics_end2end.csv")
    print("Video->sensor metrics:", enc_metrics)
    print("Sensor->future sensor metrics:", ode_metrics)
    if decoder is not None and keypoints_sensor is not None:
        print("Sensor->video metrics saved to metrics_sensor_to_video.csv")


def _run_ode_dec(config, _encoder, data, frames, frame_idx_map, aux, run_dir):
    from src.vision.datasets import WindowedSequenceDataset

    _run_decoder_only(config, None, data, frames, frame_idx_map, aux, run_dir)

    ode_func = load_ode_func(config)
    ds = WindowedSequenceDataset(data, frames, frame_idx_map, k_steps=config.k_steps)
    _, _, test_ds = ds.split()
    ode_metrics = _evaluate_ode_test_rollout(config, data, test_ds, ode_func, run_dir)
    print("Sensor->future sensor metrics:", ode_metrics)


def _resolve_ode_targets(config, encoder, frame_ds, data) -> np.ndarray | None:
    if not config.ode_use_encoder_labels:
        print("Using sensor theta labels for ODE training (ground truth).")
        return None

    assert encoder is not None
    full_pred = predict_encoder_framewise(encoder, frame_ds, device=config.device)
    y_override = np.asarray(data.y, dtype=float).copy()
    y_override[full_pred["idx"]] = full_pred["pred"][:, 0]
    print(f"Using encoder-predicted theta for ODE training ({len(full_pred['idx'])} samples)")
    return y_override


def _print_ode_init_source(config: BabVideoPipelineConfig) -> None:
    source = ("finite-difference theta", "sensor y_dot")[config.ode_init_from_y_dot]
    print(f"Using {source} for ODE rollout initial theta_dot.")


def _run_separate(config, encoder, data, frames, frame_idx_map, aux, run_dir):
    """Train encoder, ODE, and image decoder separately; then evaluate all."""
    from src.vision.datasets import FrameStateDataset
    from src.vision.train import train_encoder

    assert encoder is not None
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
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        device=config.device,
        seed=config.seed,
        wandb_project=config.wandb_project,
        wandb_run_name=f"enc_sep_{config.dataset}",
        checkpoint_dir=str(run_dir),
    )
    print(f"Encoder best val loss: {enc_result['best_val_loss']:.6f}")

    enc_metrics, _ = evaluate_and_plot_encoder(
        encoder,
        enc_test_ds,
        aux,
        run_dir,
        plot_dataset=frame_ds,
        device=config.device,
    )

    y_override = _resolve_ode_targets(config, encoder, frame_ds, data)
    ode_model, tr, va, te = train_ode_model_separate(
        config,
        data,
        run_dir,
        y_override=y_override,
    )
    _print_ode_init_source(config)
    ode_eval = evaluate_and_plot_ode_free_run(
        ode_model,
        data,
        te,
        run_dir,
        use_sensor_y_dot=config.ode_init_from_y_dot,
    )
    ode_metrics = ode_eval["metrics"]

    decoder_run = _train_decoder(
        config,
        data,
        frames,
        frame_idx_map,
        run_dir,
        run_name=f"dec_sep_{config.dataset}",
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


def _run_ode_retrain(config, encoder, data, frames, frame_idx_map, _aux, run_dir):
    """Train only the ODE, reusing an already-trained encoder."""
    import torch
    from src.vision.datasets import FrameStateDataset

    assert encoder is not None
    print(f"Loading encoder checkpoint: {config.encoder_checkpoint}")
    encoder.load_state_dict(
        torch.load(config.encoder_checkpoint, map_location="cpu", weights_only=True)
    )
    encoder.eval()

    frame_ds = FrameStateDataset(data, frames, frame_idx_map)
    y_override = _resolve_ode_targets(config, encoder, frame_ds, data)

    ode_model, _tr, _va, te = train_ode_model_separate(
        config,
        data,
        run_dir,
        y_override=y_override,
    )
    _print_ode_init_source(config)
    ode_eval = evaluate_and_plot_ode_free_run(
        ode_model,
        data,
        te,
        run_dir,
        use_sensor_y_dot=config.ode_init_from_y_dot,
    )
    ode_metrics = ode_eval["metrics"]

    if hasattr(ode_model, "ode_func_") and ode_model.ode_func_ is not None:
        print("\nLearned ODE parameters:")
        for name, param in ode_model.ode_func_.named_parameters():
            val = param.detach().cpu().numpy()
            suffix = f"  (exp = {np.exp(val):.6f})" if "log_" in name else ""
            print(f"  {name} = {val:.4f}{suffix}")

    print("\nSensor->future sensor (free-run) metrics:", ode_metrics)


MODE_RUNNERS: dict[str, Callable] = {
    "encoder_only": _run_encoder_only,
    "decoder_only": _run_decoder_only,
    "enc_ode": _run_enc_ode,
    "enc_ode_dec": _run_enc_ode,
    "ode_dec": _run_ode_dec,
    "separate": _run_separate,
    "ode_retrain": _run_ode_retrain,
}

MODES_WITH_ENCODER = {"encoder_only", "enc_ode", "enc_ode_dec", "separate", "ode_retrain"}


def main():
    parser = _build_parser()
    config = _config_from_args(parser.parse_args())
    if config.mode == "ode_retrain" and not config.encoder_checkpoint:
        parser.error("--encoder-checkpoint is required for --mode ode_retrain")

    from src.vision.datasets import load_bab_with_video

    run_dir = make_run_dir(config.output_root, config.resolved_run_name())
    print(f"Run directory: {run_dir}")
    save_json(config.to_dict(), run_dir / "config.json")

    video_map = _load_video_map(config)
    keypoint_csv, theta_csv = _resolve_label_paths(config)

    data, frames, frame_idx_map, aux = load_bab_with_video(
        config.dataset,
        video_dir=config.video_dir,
        video_path=config.video_path,
        video_map=video_map,
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

    encoder = _build_encoder(config) if config.mode in MODES_WITH_ENCODER else None
    MODE_RUNNERS[config.mode](config, encoder, data, frames, frame_idx_map, aux, run_dir)
    print(f"\nDone. Results in {run_dir}")


if __name__ == "__main__":
    main()
