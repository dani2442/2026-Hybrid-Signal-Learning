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
    return np.column_stack([data.y, np.zeros_like(data.y) if data.y_dot is None else data.y_dot]).astype(float)


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


def _select_theta_video_labels(aux: dict, indices: np.ndarray) -> np.ndarray | None:
    for key in ("theta_sensor_from_video_sparse", "theta_sensor_from_video"):
        values = aux.get(key)
        if values is not None:
            return np.asarray(values, dtype=float)[indices]
    return None


def _maybe_array(values, *, dtype=float):
    return None if values is None else np.asarray(values, dtype=dtype)


def evaluate_and_plot_encoder(
    encoder,
    dataset,
    aux: dict,
    run_dir: Path,
    *,
    plot_dataset=None,
    device: str = "auto",
):
    from src.vision.eval import evaluate_encoder_framewise
    from src.visualization.pipeline_plots import plot_video_to_sensor

    metrics = evaluate_encoder_framewise(encoder, dataset, device=device)
    save_metrics_csv(metrics, run_dir / "metrics_video_to_sensor.csv")

    plot_source = dataset if plot_dataset is None else plot_dataset
    pred = predict_encoder_framewise(encoder, plot_source, device=device)
    theta_video = _select_theta_video_labels(aux, pred["idx"])

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
    plot_states: np.ndarray | None = None,
    plot_frames: np.ndarray | None = None,
    plot_t: np.ndarray | None = None,
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

    eval_pred = predict_image_decoder(decoder, states[te], device=device)

    plot_states_arr = states if plot_states is None else np.asarray(plot_states, dtype=float)
    plot_frames_arr = frames_sensor if plot_frames is None else np.asarray(plot_frames, dtype=np.float32)
    plot_t_arr = t if plot_t is None else np.asarray(plot_t, dtype=float)
    plot_pred = predict_image_decoder(decoder, plot_states_arr, device=device)
    mse_per_frame = np.mean((plot_pred - plot_frames_arr) ** 2, axis=(1, 2, 3))
    plot_sensor_to_video_image_errors(
        run_dir / "plot_sensor_to_video_error_timeline.png",
        plot_t_arr,
        mse_per_frame,
    )
    sample_sel = select_evenly_spaced(np.arange(len(plot_states_arr), dtype=int), n=plot_count)
    plot_sensor_to_video_image_montage(
        run_dir / "plot_sensor_to_video_frames.png",
        plot_t_arr,
        plot_frames_arr,
        plot_pred,
        sample_sel,
    )
    save_json(
        {
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "n_test": int(len(te)),
            "prediction_shape": list(eval_pred.shape),
            "plot_prediction_shape": list(plot_pred.shape),
        },
        run_dir / "decoder_summary.json",
    )
    return {
        "result": result,
        "metrics": metrics,
        "predictions": eval_pred,
        "plot_predictions": plot_pred,
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


def _free_run_segment(
    ode_model,
    u: np.ndarray,
    theta: np.ndarray,
    theta_dot: np.ndarray | None,
    dt: float,
    *,
    use_sensor_y_dot: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    theta_dot = _maybe_array(theta_dot)
    init = prepare_ode_initial_segment(theta, theta_dot if use_sensor_y_dot else None, dt)
    pred_theta, pred_theta_dot = _split_rollout_state(
        np.asarray(ode_model.predict_free_run(u, init, return_full_state=True), dtype=float),
        dt,
    )
    n = min(len(theta), len(pred_theta))
    true_theta_dot = (
        np.gradient(theta[:n], dt)
        if theta_dot is None
        else np.asarray(theta_dot[:n], dtype=float)
    )
    return theta[:n], pred_theta[:n], true_theta_dot, pred_theta_dot[:n]


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
    y_dot_all = _maybe_array(data.y_dot)
    t_all = np.asarray(data.t, dtype=float)

    y_test = y_all[test_idx]
    y_dot_test = y_dot_all[test_idx] if y_dot_all is not None else None
    y_true_full, y_pred_full, theta_dot_true_full, theta_dot_pred_full = _free_run_segment(
        ode_model,
        u_all,
        y_all,
        y_dot_all,
        dt,
        use_sensor_y_dot=use_sensor_y_dot,
    )
    y_true_test, y_pred_test, theta_dot_true_test, theta_dot_pred_test = _free_run_segment(
        ode_model,
        u_all[test_idx],
        y_test,
        y_dot_test,
        dt,
        use_sensor_y_dot=use_sensor_y_dot,
    )
    metrics = {
        "rmse_theta": Metrics.rmse(y_true_test, y_pred_test),
        "mae_theta": Metrics.mae(y_true_test, y_pred_test),
        "r2_theta": Metrics.r2(y_true_test, y_pred_test),
        "fit_theta": Metrics.fit_percent(y_true_test, y_pred_test),
        "rmse_theta_dot": Metrics.rmse(theta_dot_true_test, theta_dot_pred_test),
    }
    save_metrics_csv(metrics, run_dir / "metrics_sensor_to_future_sensor.csv")

    plot_sensor_to_future_sensor_freerun_full(
        run_dir / "plot_sensor_to_future_sensor_freerun.png",
        t_all[: len(y_true_full)],
        y_true_full,
        y_pred_full,
        u=u_all[: len(y_true_full)],
        theta_dot_true=theta_dot_true_full,
        theta_dot_pred=theta_dot_pred_full,
    )
    return {"metrics": metrics}


# ── ODE loading / training helpers ──────────────────────────────────

def load_ode_func(config):
    if config.ode_checkpoint:
        from src.models.base import load_model

        model = load_model(config.ode_checkpoint)
        if hasattr(model, "func_") and model.func_ is not None:
            return model.func_
        if hasattr(model, "ode_func_") and model.ode_func_ is not None:
            return model.ode_func_
        raise ValueError(
            f"Checkpoint {config.ode_checkpoint} does not expose 'func_' or 'ode_func_'."
        )

    from src.models.blackbox_ode import _build_structured
    from src.models.physics_ode import _build_linear_ode, _build_stribeck_ode

    builders = {
        "linear_physics": ("LinearPhysics", _build_linear_ode),
        "stribeck_physics": ("StribeckPhysics", _build_stribeck_ode),
        "structured_node": (
            "StructuredNODE",
            lambda: _build_structured(hidden_dim=config.ode_hidden_dim),
        ),
    }
    label, builder = builders[config.ode_model]
    print(f"No --ode-checkpoint given; creating a fresh {label} dynamics.")
    return builder()


def _ode_learning_rate(config) -> float | None:
    default = None if config.ode_model in ("linear_physics", "stribeck_physics") else config.lr
    return default if config.ode_lr is None else config.ode_lr


def _build_ode_model(config, common_cfg: dict):
    from src.config import BlackboxODE2DConfig, LinearPhysicsConfig, StribeckPhysicsConfig
    from src.models.blackbox_ode import StructuredNODE
    from src.models.physics_ode import LinearPhysics, StribeckPhysics

    ode_lr = _ode_learning_rate(config)
    training_mode = common_cfg.pop("training_mode_override")

    def _configure(ode_cfg, *, physics: bool):
        if ode_lr is not None:
            ode_cfg.learning_rate = float(ode_lr)
        if physics:
            ode_cfg.sequence_length = max(2, int(config.k_steps))
        if training_mode is not None:
            ode_cfg.training_mode = training_mode
        return ode_cfg

    return {
        "linear_physics": lambda: LinearPhysics(
            _configure(LinearPhysicsConfig(**common_cfg), physics=True)
        ),
        "stribeck_physics": lambda: StribeckPhysics(
            _configure(StribeckPhysicsConfig(**common_cfg), physics=True)
        ),
        "structured_node": lambda: StructuredNODE(
            _configure(
                BlackboxODE2DConfig(
                    hidden_dim=config.ode_hidden_dim,
                    learning_rate=config.lr if ode_lr is None else ode_lr,
                    k_steps=config.k_steps,
                    **common_cfg,
                ),
                physics=False,
            )
        ),
    }[config.ode_model]()


def train_ode_model_separate(config, data, run_dir: Path, *, y_override=None):
    n = len(data)
    tr, va, te = split_indices(n)
    y = y_override if y_override is not None else np.asarray(data.y)
    y_dot = np.asarray(data.y_dot) if data.y_dot is not None else None

    dt = 1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0
    common_cfg = dict(
        dt=dt,
        epochs=config.epochs_ode,
        device=config.device,
        seed=config.seed,
        verbose=True,
        wandb_project=config.wandb_project,
        wandb_run_name=f"ode_{config.dataset}",
        training_mode_override=config.ode_training_mode,
    )
    if config.ode_batch_size is not None:
        common_cfg["batch_size"] = config.ode_batch_size
    model = _build_ode_model(config, common_cfg)
    ode_cfg = model.config

    print(
        f"ODE training mode: {ode_cfg.training_mode} | "
        f"lr={ode_cfg.learning_rate} | batch_size={ode_cfg.batch_size}"
    )

    y_fit = y
    y_val = y[va] if len(va) > 0 else None
    if (
        config.ode_init_from_y_dot
        and config.ode_model in ("linear_physics", "stribeck_physics")
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
