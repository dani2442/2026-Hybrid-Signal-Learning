#!/usr/bin/env python
"""Train/evaluate the BAB multimodal pipeline (video, sensor, decoder labels)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Allow `python examples/run_bab_video_pipeline.py` from repo root.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vision.pipeline_utils import (
    load_ode_func,
    make_run_dir,
    predict_encoder_framewise,
    predict_image_decoder,
    save_json,
    save_metrics_csv,
    select_evenly_spaced,
    split_indices,
    states_from_sensor,
    train_ode_model_separate,
)
from src.visualization.pipeline_plots import (
    plot_sensor_to_future_sensor,
    plot_sensor_to_future_sensor_freerun_full,
    plot_sensor_to_video_coords,
    plot_sensor_to_video_image_errors,
    plot_sensor_to_video_image_montage,
    plot_sensor_to_video_overlay,
    plot_video_to_sensor,
)


def _run_encoder_only(args, encoder, data, frames, frame_idx_map, aux, run_dir):
    from src.vision.datasets import FrameStateDataset
    from src.vision.eval import evaluate_encoder_framewise
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

    metrics = evaluate_encoder_framewise(encoder, test_ds, device=args.device)
    save_metrics_csv(metrics, run_dir / "metrics_video_to_sensor.csv")
    print("Video->sensor metrics:", metrics)

    pred = predict_encoder_framewise(encoder, test_ds, device=args.device)
    theta_video = None
    if "theta_sensor_from_video_sparse" in aux:
        theta_video = np.asarray(aux["theta_sensor_from_video_sparse"])[pred["idx"]]
    elif "theta_sensor_from_video" in aux:
        theta_video = np.asarray(aux["theta_sensor_from_video"])[pred["idx"]]
    plot_video_to_sensor(
        run_dir / "plot_video_to_sensor.png",
        pred["t"],
        pred["true"][:, 0],
        pred["pred"][:, 0],
        theta_video_label=theta_video,
    )


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
    from src.vision.eval import evaluate_image_decoder
    from src.vision.models import DecoderFrameDeconv
    from src.vision.train import train_image_decoder

    states = states_from_sensor(data)
    frames_sensor = np.asarray(frames[frame_idx_map], dtype=np.float32) / 255.0
    t = np.asarray(data.t, dtype=float)

    tr, va, te = split_indices(len(states))
    decoder = DecoderFrameDeconv(
        frame_height=int(frames_sensor.shape[1]),
        frame_width=int(frames_sensor.shape[2]),
    )
    result = train_image_decoder(
        decoder,
        states[tr],
        frames_sensor[tr],
        val_states=states[va],
        val_frames=frames_sensor[va],
        epochs=args.decoder_epochs if args.decoder_epochs is not None else args.epochs,
        batch_size=args.decoder_batch_size if args.decoder_batch_size is not None else args.batch_size,
        lr=args.decoder_lr if args.decoder_lr is not None else args.lr,
        device=args.device,
        seed=args.seed,
        wandb_project=args.wandb,
        wandb_run_name=f"dec_img_{args.dataset}",
        checkpoint_dir=str(run_dir),
    )
    print(f"Decoder best val loss: {result['best_val_loss']:.6f}")

    metrics = evaluate_image_decoder(decoder, states[te], frames_sensor[te], device=args.device)
    save_metrics_csv(metrics, run_dir / "metrics_sensor_to_video.csv")
    print("Sensor->video metrics:", metrics)

    pred = predict_image_decoder(decoder, states[te], device=args.device)
    mse_per_frame = np.mean((pred - frames_sensor[te]) ** 2, axis=(1, 2, 3))
    plot_sensor_to_video_image_errors(
        run_dir / "plot_sensor_to_video_error_timeline.png",
        t[te],
        mse_per_frame,
    )
    sample_sel = select_evenly_spaced(np.arange(len(te), dtype=int), n=args.plot_count)
    plot_sensor_to_video_image_montage(
        run_dir / "plot_sensor_to_video_frames.png",
        t[te],
        frames_sensor[te],
        pred,
        sample_sel,
    )
    save_json(
        {
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "n_test": int(len(te)),
            "prediction_shape": list(pred.shape),
        },
        run_dir / "decoder_summary.json",
    )


def _run_enc_ode(args, encoder, data, frames, frame_idx_map, aux, run_dir):
    from src.vision.datasets import FrameStateDataset, WindowedSequenceDataset
    from src.vision.eval import evaluate_decoder, evaluate_encoder_framewise, evaluate_end2end, evaluate_ode_rollout
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
    enc_metrics = evaluate_encoder_framewise(encoder, frame_test_ds, device=args.device)
    save_metrics_csv(enc_metrics, run_dir / "metrics_video_to_sensor.csv")
    pred_enc = predict_encoder_framewise(encoder, frame_test_ds, device=args.device)
    theta_video = None
    if "theta_sensor_from_video_sparse" in aux:
        theta_video = np.asarray(aux["theta_sensor_from_video_sparse"])[pred_enc["idx"]]
    elif "theta_sensor_from_video" in aux:
        theta_video = np.asarray(aux["theta_sensor_from_video"])[pred_enc["idx"]]
    plot_video_to_sensor(
        run_dir / "plot_video_to_sensor.png",
        pred_enc["t"],
        pred_enc["true"][:, 0],
        pred_enc["pred"][:, 0],
        theta_video_label=theta_video,
    )

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
    from src.validation.metrics import Metrics
    from src.vision.datasets import FrameStateDataset
    from src.vision.eval import evaluate_encoder_framewise, evaluate_image_decoder
    from src.vision.models import DecoderFrameDeconv
    from src.vision.train import train_encoder, train_image_decoder

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
        epochs=args.encoder_epochs if args.encoder_epochs is not None else args.epochs,
        batch_size=args.encoder_batch_size if args.encoder_batch_size is not None else args.batch_size,
        lr=args.encoder_lr if args.encoder_lr is not None else args.lr,
        device=args.device,
        seed=args.seed,
        wandb_project=args.wandb,
        wandb_run_name=f"enc_sep_{args.dataset}",
        checkpoint_dir=str(run_dir),
    )
    print(f"Encoder best val loss: {enc_result['best_val_loss']:.6f}")

    enc_metrics = evaluate_encoder_framewise(encoder, enc_test_ds, device=args.device)
    save_metrics_csv(enc_metrics, run_dir / "metrics_video_to_sensor.csv")
    pred_enc = predict_encoder_framewise(encoder, enc_test_ds, device=args.device)
    # Use sparse (non-interpolated) video labels for the plot.
    theta_video = None
    if "theta_sensor_from_video_sparse" in aux:
        theta_video = np.asarray(aux["theta_sensor_from_video_sparse"])[pred_enc["idx"]]
    elif "theta_sensor_from_video" in aux:
        theta_video = np.asarray(aux["theta_sensor_from_video"])[pred_enc["idx"]]
    plot_video_to_sensor(
        run_dir / "plot_video_to_sensor.png",
        pred_enc["t"],
        pred_enc["true"][:, 0],
        pred_enc["pred"][:, 0],
        theta_video_label=theta_video,
    )

    # 2) Train ODE separately and evaluate free-run simulation.
    #    Optionally use encoder-predicted theta as ODE training targets.
    y_override = None
    if args.ode_use_encoder_labels:
        full_pred = predict_encoder_framewise(encoder, frame_ds, device=args.device)
        y_override = np.asarray(data.y, dtype=float).copy()
        y_override[full_pred["idx"]] = full_pred["pred"][:, 0]
        print(f"Using encoder-predicted theta for ODE training ({len(full_pred['idx'])} samples)")
    ode_model, tr, va, te = train_ode_model_separate(args, data, run_dir, y_override=y_override)
    dt = 1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0

    # --- Free-run on the FULL trajectory (t=0 → end) ----------------
    u_all = np.asarray(data.u, dtype=float)
    y_all = np.asarray(data.y, dtype=float)
    t_all_ode = np.asarray(data.t, dtype=float)

    # Use model's own theta_dot from full-state prediction
    full_state_pred = np.asarray(
        ode_model.predict_free_run(u_all, y_all, return_full_state=True), dtype=float
    )
    if full_state_pred.ndim == 2 and full_state_pred.shape[1] >= 2:
        y_pred_fr_full = full_state_pred[:, 0]
        td_pred_full = full_state_pred[:, 1]
    else:
        y_pred_fr_full = full_state_pred.flatten()
        td_pred_full = np.gradient(y_pred_fr_full, dt) if len(y_pred_fr_full) > 1 else np.zeros_like(y_pred_fr_full)

    n_fr_full = min(len(y_pred_fr_full), len(y_all))
    y_true_fr_full = y_all[:n_fr_full]
    y_pred_fr_full = y_pred_fr_full[:n_fr_full]
    td_pred_full = td_pred_full[:n_fr_full]
    t_fr_full = t_all_ode[:n_fr_full]

    # Use consistent derivative method for ground truth: Savgol from data.y_dot
    # Convert to deg/s to match model's theta_dot (model state is [theta_deg, theta_dot_deg_per_s])
    if data.y_dot is not None:
        td_true_full = np.asarray(data.y_dot, dtype=float)[:n_fr_full]
    else:
        td_true_full = np.gradient(y_true_fr_full, dt)

    # Metrics on test portion only (fair evaluation)
    u_test = u_all[te]
    y_test = y_all[te]
    full_state_test = np.asarray(
        ode_model.predict_free_run(u_test, y_test, return_full_state=True), dtype=float
    )
    if full_state_test.ndim == 2 and full_state_test.shape[1] >= 2:
        y_pred_fr_test = full_state_test[:, 0]
        td_pred_te = full_state_test[:, 1]
    else:
        y_pred_fr_test = full_state_test.flatten()
        td_pred_te = np.gradient(y_pred_fr_test, dt) if len(y_pred_fr_test) > 1 else np.zeros_like(y_pred_fr_test)

    n_fr_te = min(len(y_pred_fr_test), len(y_test))
    if data.y_dot is not None:
        td_true_te = np.asarray(data.y_dot, dtype=float)[te][:n_fr_te]
    else:
        td_true_te = np.gradient(y_test[:n_fr_te], dt)
    td_pred_te = td_pred_te[:n_fr_te]

    ode_metrics = {
        "rmse_theta": Metrics.rmse(y_test[:n_fr_te], y_pred_fr_test[:n_fr_te]),
        "mae_theta": Metrics.mae(y_test[:n_fr_te], y_pred_fr_test[:n_fr_te]),
        "r2_theta": Metrics.r2(y_test[:n_fr_te], y_pred_fr_test[:n_fr_te]),
        "fit_theta": Metrics.fit_percent(y_test[:n_fr_te], y_pred_fr_test[:n_fr_te]),
        "rmse_theta_dot": Metrics.rmse(td_true_te, td_pred_te),
    }
    save_metrics_csv(ode_metrics, run_dir / "metrics_sensor_to_future_sensor.csv")

    # Plot uses the FULL trajectory (t=0 → end)
    plot_sensor_to_future_sensor_freerun_full(
        run_dir / "plot_sensor_to_future_sensor_freerun.png",
        t_fr_full,
        y_true_fr_full,
        y_pred_fr_full,
        u=u_all[:n_fr_full],
        theta_dot_true=td_true_full,
        theta_dot_pred=td_pred_full,
    )

    # 3) Train image decoder separately (sensor state -> frame).
    states = states_from_sensor(data)
    frames_sensor = np.asarray(frames[frame_idx_map], dtype=np.float32) / 255.0
    t_all = np.asarray(data.t, dtype=float)

    decoder = DecoderFrameDeconv(
        frame_height=int(frames_sensor.shape[1]),
        frame_width=int(frames_sensor.shape[2]),
    )
    dec_result = train_image_decoder(
        decoder,
        states[tr],
        frames_sensor[tr],
        val_states=states[va],
        val_frames=frames_sensor[va],
        epochs=args.decoder_epochs if args.decoder_epochs is not None else args.epochs,
        batch_size=args.decoder_batch_size if args.decoder_batch_size is not None else args.batch_size,
        lr=args.decoder_lr if args.decoder_lr is not None else args.lr,
        device=args.device,
        seed=args.seed,
        wandb_project=args.wandb,
        wandb_run_name=f"dec_sep_{args.dataset}",
        checkpoint_dir=str(run_dir),
    )
    print(f"Decoder best val loss: {dec_result['best_val_loss']:.6f}")

    dec_metrics = evaluate_image_decoder(decoder, states[te], frames_sensor[te], device=args.device)
    save_metrics_csv(dec_metrics, run_dir / "metrics_sensor_to_video.csv")
    pred_frames = predict_image_decoder(decoder, states[te], device=args.device)
    mse_per_frame = np.mean((pred_frames - frames_sensor[te]) ** 2, axis=(1, 2, 3))
    plot_sensor_to_video_image_errors(
        run_dir / "plot_sensor_to_video_error_timeline.png",
        t_all[te],
        mse_per_frame,
    )
    sample_sel = select_evenly_spaced(np.arange(len(te), dtype=int), n=args.plot_count)
    plot_sensor_to_video_image_montage(
        run_dir / "plot_sensor_to_video_frames.png",
        t_all[te],
        frames_sensor[te],
        pred_frames,
        sample_sel,
    )

    save_json(
        {
            "encoder_best_val_loss": float(enc_result["best_val_loss"]),
            "ode_train_points": int(len(tr)),
            "ode_val_points": int(len(va)),
            "ode_test_points": int(len(te)),
            "decoder_best_val_loss": float(dec_result["best_val_loss"]),
        },
        run_dir / "separate_training_summary.json",
    )
    print("Video->sensor metrics:", enc_metrics)
    print("Sensor->future sensor (free-run) metrics:", ode_metrics)
    print("Sensor->video metrics:", dec_metrics)


def _run_ode_retrain(args, encoder, data, frames, frame_idx_map, aux, run_dir):
    """Train only the ODE, reusing an already-trained encoder."""
    import torch
    from src.validation.metrics import Metrics
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

    # Train ODE
    ode_model, tr, va, te = train_ode_model_separate(
        args, data, run_dir, y_override=y_override,
    )
    dt = 1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0

    # Free-run on FULL trajectory
    u_all = np.asarray(data.u, dtype=float)
    y_all = np.asarray(data.y, dtype=float)
    t_all = np.asarray(data.t, dtype=float)

    full_state_pred = np.asarray(
        ode_model.predict_free_run(u_all, y_all, return_full_state=True), dtype=float
    )
    if full_state_pred.ndim == 2 and full_state_pred.shape[1] >= 2:
        y_pred_full = full_state_pred[:, 0]
        td_pred_full = full_state_pred[:, 1]
    else:
        y_pred_full = full_state_pred.flatten()
        td_pred_full = (
            np.gradient(y_pred_full, dt)
            if len(y_pred_full) > 1
            else np.zeros_like(y_pred_full)
        )

    n_full = min(len(y_pred_full), len(y_all))
    y_true_full = y_all[:n_full]
    y_pred_full = y_pred_full[:n_full]
    td_pred_full = td_pred_full[:n_full]
    t_full = t_all[:n_full]
    td_true_full = (
        np.asarray(data.y_dot, dtype=float)[:n_full]
        if data.y_dot is not None
        else np.gradient(y_true_full, dt)
    )

    # Metrics on test portion
    full_state_test = np.asarray(
        ode_model.predict_free_run(u_all[te], y_all[te], return_full_state=True),
        dtype=float,
    )
    if full_state_test.ndim == 2 and full_state_test.shape[1] >= 2:
        y_pred_test = full_state_test[:, 0]
        td_pred_test = full_state_test[:, 1]
    else:
        y_pred_test = full_state_test.flatten()
        td_pred_test = (
            np.gradient(y_pred_test, dt)
            if len(y_pred_test) > 1
            else np.zeros_like(y_pred_test)
        )

    n_te = min(len(y_pred_test), len(y_all[te]))
    td_true_te = (
        np.asarray(data.y_dot, dtype=float)[te][:n_te]
        if data.y_dot is not None
        else np.gradient(y_all[te][:n_te], dt)
    )

    ode_metrics = {
        "rmse_theta": Metrics.rmse(y_all[te][:n_te], y_pred_test[:n_te]),
        "mae_theta": Metrics.mae(y_all[te][:n_te], y_pred_test[:n_te]),
        "r2_theta": Metrics.r2(y_all[te][:n_te], y_pred_test[:n_te]),
        "fit_theta": Metrics.fit_percent(y_all[te][:n_te], y_pred_test[:n_te]),
        "rmse_theta_dot": Metrics.rmse(td_true_te, td_pred_test[:n_te]),
    }
    save_metrics_csv(ode_metrics, run_dir / "metrics_sensor_to_future_sensor.csv")

    plot_sensor_to_future_sensor_freerun_full(
        run_dir / "plot_sensor_to_future_sensor_freerun.png",
        t_full,
        y_true_full,
        y_pred_full,
        u=u_all[:n_full],
        theta_dot_true=td_true_full,
        theta_dot_pred=td_pred_full,
    )

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
                        help="In --mode separate, use encoder-predicted theta as ODE training targets.")
    parser.add_argument("--ode-training-mode", default=None,
                        choices=["full", "subsequence"],
                        help="ODE training strategy: 'full' (entire trajectory) or "
                             "'subsequence' (random windows). Default: model-specific.")

    # Optional per-stage overrides for --mode separate
    parser.add_argument("--encoder-epochs", type=int, default=None)
    parser.add_argument("--encoder-batch-size", type=int, default=None)
    parser.add_argument("--encoder-lr", type=float, default=None)
    parser.add_argument("--ode-epochs", type=int, default=None)
    parser.add_argument("--ode-batch-size", type=int, default=None)
    parser.add_argument("--ode-lr", type=float, default=1e-5)
    parser.add_argument("--decoder-epochs", type=int, default=None)
    parser.add_argument("--decoder-batch-size", type=int, default=None)
    parser.add_argument("--decoder-lr", type=float, default=None)

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
