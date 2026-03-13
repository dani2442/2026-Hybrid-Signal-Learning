"""Helpers for the BAB multimodal pipeline."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np


# ── IO helpers ──────────────────────────────────────────────────────

def make_run_dir(output_root: str, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_metrics_csv(metrics: dict, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in sorted(metrics.items()):
            writer.writerow([k, f"{float(v):.6f}"])


def save_json(data: dict, path: Path) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)


# ── Data helpers ────────────────────────────────────────────────────

def split_indices(
    n: int, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(n, dtype=int)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        all_idx[:n_train],
        all_idx[n_train : n_train + n_val],
        all_idx[n_train + n_val :],
    )


def collate_frame_state(batch):
    import torch

    frames = torch.stack([b[0] for b in batch])
    states = torch.stack([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return frames, states, metas


def states_from_sensor(data) -> np.ndarray:
    theta_dot = data.y_dot if data.y_dot is not None else np.zeros_like(data.y)
    return np.column_stack([data.y, theta_dot]).astype(float)


def prepare_ode_initial_segment(
    theta: np.ndarray, theta_dot: np.ndarray | None, dt: float
) -> np.ndarray:
    """Build a short theta segment that encodes the desired initial theta_dot."""
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    if theta_arr.size == 0:
        raise ValueError("theta must contain at least one sample.")
    if theta_dot is None:
        return theta_arr

    theta_dot_arr = np.asarray(theta_dot, dtype=float).reshape(-1)
    if theta_dot_arr.size == 0 or not np.isfinite(theta_dot_arr[0]):
        return theta_arr

    theta0 = float(theta_arr[0])
    return np.asarray([theta0, theta0 + float(theta_dot_arr[0]) * float(dt)], dtype=float)


def select_evenly_spaced(indices: np.ndarray, n: int) -> np.ndarray:
    if len(indices) == 0:
        return np.array([], dtype=int)
    n = int(max(1, min(n, len(indices))))
    if n == 1:
        return np.array([int(indices[len(indices) // 2])], dtype=int)
    sel = np.linspace(0, len(indices) - 1, n, dtype=int)
    return indices[sel]


# ── Prediction / rollout helpers ────────────────────────────────────

def predict_encoder_framewise(
    encoder, dataset, *, device: str = "auto", batch_size: int = 64
):
    import torch
    from torch.utils.data import DataLoader
    from src.models.base import resolve_device

    dev = resolve_device(device)
    encoder = encoder.to(dev).eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_frame_state
    )

    all_t = []
    all_true = []
    all_pred = []
    all_i = []
    with torch.no_grad():
        for frames, states, metas in loader:
            pred = encoder(frames.to(dev)).cpu().numpy()
            all_pred.append(pred)
            all_true.append(states.numpy())
            all_t.extend(float(m["t"]) for m in metas)
            all_i.extend(int(m["i"]) for m in metas)
    return {
        "t": np.asarray(all_t, dtype=float),
        "idx": np.asarray(all_i, dtype=int),
        "true": np.concatenate(all_true, axis=0),
        "pred": np.concatenate(all_pred, axis=0),
    }


def predict_image_decoder(
    decoder, states: np.ndarray, *, device: str = "auto", batch_size: int = 128
) -> np.ndarray:
    import torch
    from src.models.base import resolve_device

    dev = resolve_device(device)
    decoder = decoder.to(dev).eval()
    x = torch.tensor(states, dtype=torch.float32, device=dev)
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            pred = decoder(x[start : start + batch_size]).cpu().numpy()  # (B, C, H, W)
            out.append(pred)
    pred_chw = np.concatenate(out, axis=0)
    pred_hwc = np.transpose(pred_chw, (0, 2, 3, 1))
    return np.clip(pred_hwc, 0.0, 1.0)


def evaluate_and_plot_encoder(encoder, dataset, aux: dict, run_dir: Path, *, device: str = "auto"):
    from src.vision.eval import evaluate_encoder_framewise
    from src.visualization.pipeline_plots import plot_video_to_sensor

    metrics = evaluate_encoder_framewise(encoder, dataset, device=device)
    save_metrics_csv(metrics, run_dir / "metrics_video_to_sensor.csv")

    pred = predict_encoder_framewise(encoder, dataset, device=device)
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
    return metrics, pred


def train_and_evaluate_image_decoder(
    decoder,
    states: np.ndarray,
    frames_sensor: np.ndarray,
    t: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int | None,
    wandb_project: str | None,
    wandb_run_name: str | None,
    run_dir: Path,
    plot_count: int,
) -> dict:
    from src.vision.eval import evaluate_image_decoder
    from src.vision.train import train_image_decoder
    from src.visualization.pipeline_plots import (
        plot_sensor_to_video_image_errors,
        plot_sensor_to_video_image_montage,
    )

    tr, va, te = split_indices(len(states))
    result = train_image_decoder(
        decoder,
        states[tr],
        frames_sensor[tr],
        val_states=states[va],
        val_frames=frames_sensor[va],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        checkpoint_dir=str(run_dir),
    )
    metrics = evaluate_image_decoder(decoder, states[te], frames_sensor[te], device=device)
    save_metrics_csv(metrics, run_dir / "metrics_sensor_to_video.csv")

    pred = predict_image_decoder(decoder, states[te], device=device)
    mse_per_frame = np.mean((pred - frames_sensor[te]) ** 2, axis=(1, 2, 3))
    plot_sensor_to_video_image_errors(
        run_dir / "plot_sensor_to_video_error_timeline.png",
        t[te],
        mse_per_frame,
    )
    sample_sel = select_evenly_spaced(np.arange(len(te), dtype=int), n=plot_count)
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
    return {
        "result": result,
        "metrics": metrics,
        "predictions": pred,
        "train_idx": tr,
        "val_idx": va,
        "test_idx": te,
    }


def rollout_states(model_or_func, x0, u_seq, k_steps: int, dt: float):
    import torch

    if hasattr(model_or_func, "rollout"):
        with torch.no_grad():
            x_hat = model_or_func.rollout(x0, u_seq, k_steps)  # (K, B, 2)
        return x_hat

    ode_func = model_or_func
    dev = x0.device
    t_eval = torch.arange(k_steps, dtype=x0.dtype, device=dev) * float(dt)
    if u_seq.ndim == 3:
        u_flat = u_seq[0]
    else:
        u_flat = u_seq
    ode_func.u_series = u_flat.to(dev)
    ode_func.t_series = t_eval
    ode_func.batch_start_times = None
    x = x0.clone()
    traj = [x]
    with torch.no_grad():
        for step in range(1, k_steps):
            dx = ode_func.f(t_eval[step - 1], x)
            x = x + dx * float(dt)
            traj.append(x)
    return torch.stack(traj, dim=0)


def _split_rollout_state(full_state_pred: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(full_state_pred, dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 0], arr[:, 1]

    theta = arr.reshape(-1)
    theta_dot = np.gradient(theta, dt) if len(theta) > 1 else np.zeros_like(theta)
    return theta, theta_dot


def evaluate_and_plot_ode_free_run(
    ode_model,
    data,
    test_idx: np.ndarray,
    run_dir: Path,
    *,
    use_sensor_y_dot: bool = False,
) -> dict[str, object]:
    from src.validation.metrics import Metrics
    from src.visualization.pipeline_plots import plot_sensor_to_future_sensor_freerun_full

    dt = 1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0
    u_all = np.asarray(data.u, dtype=float)
    y_all = np.asarray(data.y, dtype=float)
    y_dot_all = np.asarray(data.y_dot, dtype=float) if data.y_dot is not None else None
    t_all = np.asarray(data.t, dtype=float)

    y_init_full = prepare_ode_initial_segment(
        y_all,
        y_dot_all if use_sensor_y_dot else None,
        dt,
    )
    full_state_pred = np.asarray(
        ode_model.predict_free_run(u_all, y_init_full, return_full_state=True),
        dtype=float,
    )
    y_pred_full, theta_dot_pred_full = _split_rollout_state(full_state_pred, dt)

    n_full = min(len(y_pred_full), len(y_all))
    y_true_full = y_all[:n_full]
    y_pred_full = y_pred_full[:n_full]
    theta_dot_pred_full = theta_dot_pred_full[:n_full]
    t_full = t_all[:n_full]
    theta_dot_true_full = (
        np.asarray(data.y_dot, dtype=float)[:n_full]
        if data.y_dot is not None
        else np.gradient(y_true_full, dt)
    )

    y_test = y_all[test_idx]
    y_dot_test = y_dot_all[test_idx] if y_dot_all is not None else None
    y_init_test = prepare_ode_initial_segment(
        y_test,
        y_dot_test if use_sensor_y_dot else None,
        dt,
    )
    full_state_test = np.asarray(
        ode_model.predict_free_run(u_all[test_idx], y_init_test, return_full_state=True),
        dtype=float,
    )
    y_pred_test, theta_dot_pred_test = _split_rollout_state(full_state_test, dt)

    n_test = min(len(y_pred_test), len(y_test))
    theta_dot_true_test = (
        np.asarray(data.y_dot, dtype=float)[test_idx][:n_test]
        if data.y_dot is not None
        else np.gradient(y_test[:n_test], dt)
    )
    metrics = {
        "rmse_theta": Metrics.rmse(y_test[:n_test], y_pred_test[:n_test]),
        "mae_theta": Metrics.mae(y_test[:n_test], y_pred_test[:n_test]),
        "r2_theta": Metrics.r2(y_test[:n_test], y_pred_test[:n_test]),
        "fit_theta": Metrics.fit_percent(y_test[:n_test], y_pred_test[:n_test]),
        "rmse_theta_dot": Metrics.rmse(theta_dot_true_test, theta_dot_pred_test[:n_test]),
    }
    save_metrics_csv(metrics, run_dir / "metrics_sensor_to_future_sensor.csv")

    plot_sensor_to_future_sensor_freerun_full(
        run_dir / "plot_sensor_to_future_sensor_freerun.png",
        t_full,
        y_true_full,
        y_pred_full,
        u=u_all[:n_full],
        theta_dot_true=theta_dot_true_full,
        theta_dot_pred=theta_dot_pred_full,
    )
    return {"metrics": metrics}


# ── ODE loading / training helpers ──────────────────────────────────

def load_ode_func(args):
    if args.ode_checkpoint:
        from src.models.base import load_model

        model = load_model(args.ode_checkpoint)
        if hasattr(model, "func_") and model.func_ is not None:
            return model.func_
        if hasattr(model, "ode_func_") and model.ode_func_ is not None:
            return model.ode_func_
        raise ValueError(
            f"Checkpoint {args.ode_checkpoint} does not expose 'func_' or 'ode_func_'."
        )

    if args.ode_model == "linear_physics":
        from src.models.physics_ode import _build_linear_ode

        print("No --ode-checkpoint given; creating a fresh LinearPhysics dynamics.")
        return _build_linear_ode()
    if args.ode_model == "stribeck_physics":
        from src.models.physics_ode import _build_stribeck_ode

        print("No --ode-checkpoint given; creating a fresh StribeckPhysics dynamics.")
        return _build_stribeck_ode()
    if args.ode_model == "structured_node":
        from src.models.blackbox_ode import _build_structured

        print("No --ode-checkpoint given; creating a fresh StructuredNODE dynamics.")
        return _build_structured(hidden_dim=args.ode_hidden_dim)
    raise ValueError(f"Unsupported --ode-model: {args.ode_model}")


def train_ode_model_separate(args, data, run_dir: Path, *, y_override=None):
    n = len(data)
    tr, va, te = split_indices(n)
    y = y_override if y_override is not None else np.asarray(data.y)
    y_dot = np.asarray(data.y_dot) if data.y_dot is not None else None

    dt = 1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0
    # Keep ODE LR model-specific by default; only override when explicitly set.
    default_ode_lr = None if args.ode_model in ("linear_physics", "stribeck_physics") else args.lr
    ode_lr = getattr(args, "ode_lr", None)
    if ode_lr is None:
        ode_lr = default_ode_lr

    common_cfg = dict(
        dt=dt,
        epochs=1000, #previously args.epochs
        device=args.device,
        seed=args.seed,
        verbose=True,
        wandb_project=args.wandb,
        wandb_run_name=f"ode_{args.dataset}",
    )
    if getattr(args, "ode_batch_size", None) is not None:
        common_cfg["batch_size"] = args.ode_batch_size

    if args.ode_model == "linear_physics":
        from src.config import LinearPhysicsConfig
        from src.models.physics_ode import LinearPhysics

        ode_cfg = LinearPhysicsConfig(**common_cfg)
        if ode_lr is not None:
            ode_cfg.learning_rate = float(ode_lr)
        # Reuse --k-steps as subsequence window length for physics ODEs.
        ode_cfg.sequence_length = max(2, int(args.k_steps))
        if args.ode_training_mode:
            ode_cfg.training_mode = args.ode_training_mode
        model = LinearPhysics(ode_cfg)
    elif args.ode_model == "stribeck_physics":
        from src.config import StribeckPhysicsConfig
        from src.models.physics_ode import StribeckPhysics

        ode_cfg = StribeckPhysicsConfig(**common_cfg)
        if ode_lr is not None:
            ode_cfg.learning_rate = float(ode_lr)
        # Reuse --k-steps as subsequence window length for physics ODEs.
        ode_cfg.sequence_length = max(2, int(args.k_steps))
        if args.ode_training_mode:
            ode_cfg.training_mode = args.ode_training_mode
        model = StribeckPhysics(ode_cfg)
    elif args.ode_model == "structured_node":
        from src.config import BlackboxODE2DConfig
        from src.models.blackbox_ode import StructuredNODE

        ode_cfg = BlackboxODE2DConfig(
            hidden_dim=args.ode_hidden_dim,
            learning_rate=ode_lr if ode_lr is not None else args.lr,
            k_steps=args.k_steps,
            **common_cfg,
        )
        if args.ode_training_mode:
            ode_cfg.training_mode = args.ode_training_mode
        model = StructuredNODE(ode_cfg)
    else:
        raise ValueError(f"Unsupported --ode-model: {args.ode_model}")

    print(
        f"ODE training mode: {ode_cfg.training_mode} | "
        f"lr={ode_cfg.learning_rate} | batch_size={ode_cfg.batch_size}"
    )

    y_fit = y
    y_val = y[va] if len(va) > 0 else None
    if (
        getattr(args, "ode_init_from_y_dot", False)
        and args.ode_model in ("linear_physics", "stribeck_physics")
        and y_dot is not None
    ):
        y_fit = np.column_stack([np.asarray(y, dtype=float), y_dot]).astype(float)
        if len(va) > 0:
            y_val = y_fit[va]
        print("Using sensor y_dot for ODE initial theta_dot during training.")

    model.fit(
        train_data=(np.asarray(data.u)[tr], y_fit[tr]),
        val_data=(np.asarray(data.u)[va], y_val) if len(va) > 0 else None,
    )
    model.save(run_dir / "ode_model.pt")
    return model, tr, va, te
