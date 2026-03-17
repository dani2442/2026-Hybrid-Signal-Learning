"""Shared helpers for the BAB multimodal training pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Tuple

import numpy as np


def sensor_dt(data) -> float:
    """Return the effective sensor sample period for the current dataset."""
    return 1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0


def save_metrics_csv(metrics: dict[str, float], path: Path) -> None:
    """Persist a flat metric dict as a two-column CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in sorted(metrics.items()):
            writer.writerow([key, f"{float(value):.6f}"])


def save_json(data: Any, path: Path) -> None:
    """Write JSON with light NumPy/Path normalization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_json_ready(data), f, indent=2)


def split_indices(
    n: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic train/val/test index splits."""
    all_idx = np.arange(n, dtype=int)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        all_idx[:n_train],
        all_idx[n_train : n_train + n_val],
        all_idx[n_train + n_val :],
    )


def collate_frame_state(batch):
    """Collate ``FrameStateDataset`` samples while keeping metadata as a list."""
    import torch

    frames = torch.stack([sample[0] for sample in batch])
    states = torch.stack([sample[1] for sample in batch])
    metas = [sample[2] for sample in batch]
    return frames, states, metas


def states_from_sensor(data) -> np.ndarray:
    """Build the ``[theta, theta_dot]`` state array from sensor signals."""
    theta_dot = np.zeros_like(data.y) if data.y_dot is None else data.y_dot
    return np.column_stack([data.y, theta_dot]).astype(float)


def require_keypoints_sensor(aux: dict[str, Any]) -> np.ndarray:
    """Fetch aligned sensor-space keypoints and validate their shape."""
    keypoints = aux.get("keypoints_sensor")
    if keypoints is None:
        raise ValueError(
            "Keypoint labels are required. Configure keypoint_labels_csv or "
            "provide a dataset with true labels."
        )

    arr = np.asarray(keypoints, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("Aligned keypoints must have shape (N, 4).")
    return arr[:, :4]


def prepare_ode_initial_segment(
    theta: np.ndarray,
    theta_dot: np.ndarray | None,
    dt: float,
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
    """Pick up to ``n`` roughly even samples from an index array."""
    if len(indices) == 0:
        return np.array([], dtype=int)
    n = int(max(1, min(n, len(indices))))
    if n == 1:
        return np.array([int(indices[len(indices) // 2])], dtype=int)
    sel = np.linspace(0, len(indices) - 1, n, dtype=int)
    return indices[sel]


def predict_encoder_framewise(
    encoder,
    dataset,
    *,
    device: str = "auto",
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Run the encoder across an entire per-frame dataset."""
    import torch
    from torch.utils.data import DataLoader

    from src.models.base import resolve_device

    dev = resolve_device(device)
    encoder = encoder.to(dev).eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_frame_state,
    )

    all_t: list[float] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_i: list[int] = []
    with torch.no_grad():
        for frames, states, metas in loader:
            pred = encoder(frames.to(dev)).cpu().numpy()
            all_pred.append(pred)
            all_true.append(states.numpy())
            all_t.extend(float(meta["t"]) for meta in metas)
            all_i.extend(int(meta["i"]) for meta in metas)

    return {
        "t": np.asarray(all_t, dtype=float),
        "idx": np.asarray(all_i, dtype=int),
        "true": np.concatenate(all_true, axis=0),
        "pred": np.concatenate(all_pred, axis=0),
    }


def predict_image_decoder(
    decoder,
    states: np.ndarray,
    *,
    device: str = "auto",
    batch_size: int = 128,
) -> np.ndarray:
    """Run an image decoder on a full state array and return HWC frames."""
    import torch

    from src.models.base import resolve_device

    dev = resolve_device(device)
    decoder = decoder.to(dev).eval()
    x = torch.tensor(states, dtype=torch.float32, device=dev)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            pred = decoder(x[start : start + batch_size]).cpu().numpy()
            out.append(pred)
    pred_chw = np.concatenate(out, axis=0)
    pred_hwc = np.transpose(pred_chw, (0, 2, 3, 1))
    return np.clip(pred_hwc, 0.0, 1.0)


def evaluate_and_plot_encoder(
    encoder,
    dataset,
    aux: dict[str, Any],
    run_dir: Path,
    *,
    plot_dataset=None,
    device: str = "auto",
):
    """Evaluate an encoder and save the standard video-to-sensor plot."""
    from src.visualization.pipeline_plots import plot_video_to_sensor

    from .eval import evaluate_encoder_framewise

    metrics = evaluate_encoder_framewise(encoder, dataset, device=device)
    save_metrics_csv(metrics, run_dir / "metrics_video_to_sensor.csv")

    plot_source = dataset if plot_dataset is None else plot_dataset
    pred = predict_encoder_framewise(encoder, plot_source, device=device)
    theta_video = _select_theta_video_labels(aux, pred["idx"])
    theta_dot_true = _select_theta_dot_true(plot_source, aux, pred["idx"])
    theta_dot_fd = _select_theta_dot_fd(aux, pred["idx"])
    theta_dot_pred = _select_theta_dot_pred(
        pred["pred"],
        pred["t"],
        encoder_velocity_mode=getattr(plot_source, "encoder_velocity_mode", "encoder_fd"),
    )
    theta_dot_pred_label = (
        "encoder theta_dot (video->sensor)"
        if pred["pred"].ndim == 2 and pred["pred"].shape[1] >= 2
        else "encoder theta_dot (FD from theta pred)"
    )

    plot_video_to_sensor(
        run_dir / "plot_video_to_sensor.png",
        pred["t"],
        pred["true"][:, 0],
        pred["pred"][:, 0],
        theta_video_label=theta_video,
        theta_dot_true=theta_dot_true,
        theta_dot_pred=theta_dot_pred,
        theta_dot_fd=theta_dot_fd,
        theta_dot_pred_label=theta_dot_pred_label,
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
) -> dict[str, Any]:
    """Train an image decoder, evaluate it, and write the standard artifacts."""
    from src.visualization.pipeline_plots import (
        plot_sensor_to_video_image_errors,
        plot_sensor_to_video_image_montage,
    )

    from .eval import evaluate_image_decoder
    from .train import train_image_decoder

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


def train_image_decoder_from_sensor(
    config,
    data,
    frames: np.ndarray,
    frame_idx_map: np.ndarray,
    run_dir: Path,
    *,
    run_name: str,
) -> dict[str, Any]:
    """Create the image decoder and train it from sensor-aligned frames."""
    from .models import DecoderFrameDeconv

    states = states_from_sensor(data)
    frames_sensor = np.asarray(frames[frame_idx_map], dtype=np.float32)
    if frames_sensor.max() > 1.0 or frames_sensor.min() < 0.0:
        frames_sensor = np.clip(frames_sensor / 255.0, 0.0, 1.0)
    t = np.asarray(data.t, dtype=float)

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


def resolve_ode_training_targets(config, encoder, frame_ds, data) -> np.ndarray | None:
    """Optionally replace ODE training targets with encoder predictions."""
    if not config.ode_use_encoder_labels:
        print("Using sensor theta labels for ODE training (ground truth).")
        return None

    full_pred = predict_encoder_framewise(encoder, frame_ds, device=config.device)
    y_override = np.asarray(data.y, dtype=float).copy()
    y_override[full_pred["idx"]] = full_pred["pred"][:, 0]
    print(f"Using encoder-predicted theta for ODE training ({len(full_pred['idx'])} samples)")
    return y_override


def evaluate_ode_test_rollout(
    data,
    test_ds,
    model_or_func,
    run_dir: Path,
    *,
    device: str = "auto",
) -> dict[str, float]:
    """Evaluate ODE rollout on the held-out windowed test set."""
    from src.visualization.pipeline_plots import plot_sensor_to_future_sensor

    from .eval import evaluate_ode_rollout

    ode_metrics = evaluate_ode_rollout(model_or_func, test_ds, init_from="sensor", device=device)
    save_metrics_csv(ode_metrics, run_dir / "metrics_sensor_to_future_sensor.csv")
    if len(test_ds):
        plot_sensor_to_future_sensor(
            run_dir / "plot_sensor_to_future_sensor.png",
            model_or_func,
            test_ds[0],
            dt=sensor_dt(data),
            device=device,
        )
    return ode_metrics


def rollout_states(model_or_func, x0, u_seq, k_steps: int, dt: float):
    """Roll out either a composite model or a bare ODE function."""
    import torch

    if hasattr(model_or_func, "rollout"):
        with torch.no_grad():
            return model_or_func.rollout(x0, u_seq, k_steps)

    ode_func = model_or_func
    dev = x0.device
    t_eval = torch.arange(k_steps, dtype=x0.dtype, device=dev) * float(dt)
    u_flat = u_seq[0] if u_seq.ndim == 3 else u_seq
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


def evaluate_and_plot_ode_free_run(
    ode_model,
    data,
    test_idx: np.ndarray,
    run_dir: Path,
    *,
    use_sensor_y_dot: bool = False,
    encoder=None,
    frame_dataset=None,
    device: str = "auto",
) -> dict[str, object]:
    """Evaluate a free-running ODE model and save trajectory plots."""
    from src.validation.metrics import Metrics
    from src.visualization.pipeline_plots import plot_sensor_to_future_sensor_freerun_full

    dt = sensor_dt(data)
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

    result: dict[str, object] = {"metrics": metrics}
    if encoder is not None and frame_dataset is not None:
        theta_video = _predict_encoder_theta_series(
            encoder,
            frame_dataset,
            len(y_all),
            device=device,
        )
        theta_dot_video = np.gradient(theta_video, dt) if len(theta_video) > 1 else np.zeros_like(theta_video)
        y_true_video, y_pred_video, theta_dot_true_video, theta_dot_pred_video = _free_run_segment(
            ode_model,
            u_all,
            y_all,
            y_dot_all,
            dt,
            use_sensor_y_dot=False,
            initial_theta=float(theta_video[0]),
            initial_theta_dot=float(theta_dot_video[0]),
        )
        plot_sensor_to_future_sensor_freerun_full(
            run_dir / "plot_video_to_sensor_to_future_sensor_freerun.png",
            t_all[: len(y_true_video)],
            y_true_video,
            y_pred_video,
            u=u_all[: len(y_true_video)],
            theta_dot_true=theta_dot_true_video,
            theta_dot_pred=theta_dot_pred_video,
            title="Video -> Sensor -> Future Sensor (NeuralODE, free-run simulation)",
            theta_pred_label="pred theta (video->sensor->future sensor)",
            theta_dot_pred_label="pred theta_dot (video->sensor->future sensor)",
        )

        valid_test_idx = np.asarray(test_idx, dtype=int)
        valid_test_idx = valid_test_idx[valid_test_idx < len(y_true_video)]
        if len(valid_test_idx) > 0:
            video_metrics = {
                "rmse_theta": Metrics.rmse(y_true_video[valid_test_idx], y_pred_video[valid_test_idx]),
                "mae_theta": Metrics.mae(y_true_video[valid_test_idx], y_pred_video[valid_test_idx]),
                "r2_theta": Metrics.r2(y_true_video[valid_test_idx], y_pred_video[valid_test_idx]),
                "fit_theta": Metrics.fit_percent(y_true_video[valid_test_idx], y_pred_video[valid_test_idx]),
                "rmse_theta_dot": Metrics.rmse(
                    theta_dot_true_video[valid_test_idx],
                    theta_dot_pred_video[valid_test_idx],
                ),
            }
            save_metrics_csv(
                video_metrics,
                run_dir / "metrics_video_to_sensor_to_future_sensor.csv",
            )
            result["video_chain_metrics"] = video_metrics
    return result


def load_ode_func(config):
    """Load ODE dynamics from a checkpoint or create a fresh one."""
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


def train_ode_model_separate(config, data, run_dir: Path, *, y_override=None):
    """Train an ODE model directly on sensor trajectories."""
    n = len(data)
    tr, va, te = split_indices(n)
    y = np.asarray(data.y if y_override is None else y_override, dtype=float)
    y_dot = np.asarray(data.y_dot, dtype=float) if data.y_dot is not None else None

    common_cfg = dict(
        dt=sensor_dt(data),
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
        train_data=(np.asarray(data.u, dtype=float)[tr], y_fit[tr]),
        val_data=(np.asarray(data.u, dtype=float)[va], y_val) if len(va) > 0 else None,
    )
    model.save(run_dir / "ode_model.pt")
    return model, tr, va, te


def _json_ready(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _select_theta_video_labels(aux: dict[str, Any], indices: np.ndarray) -> np.ndarray | None:
    for key in ("theta_sensor_from_video_sparse", "theta_sensor_from_video"):
        values = aux.get(key)
        if values is not None:
            return np.asarray(values, dtype=float)[indices]
    return None


def _select_theta_dot_true(plot_source, aux: dict[str, Any], indices: np.ndarray) -> np.ndarray | None:
    data = getattr(plot_source, "data", None)
    if data is not None and getattr(data, "y_dot", None) is not None:
        return np.asarray(data.y_dot, dtype=float)[indices]

    theta_dot_fd = aux.get("theta_dot_sensor_fd")
    if theta_dot_fd is not None:
        return np.asarray(theta_dot_fd, dtype=float)[indices]
    return None


def _select_theta_dot_fd(aux: dict[str, Any], indices: np.ndarray) -> np.ndarray | None:
    theta_dot_fd = aux.get("theta_dot_sensor_fd")
    if theta_dot_fd is None:
        return None
    return np.asarray(theta_dot_fd, dtype=float)[indices]


def _select_theta_dot_pred(
    pred: np.ndarray,
    t: np.ndarray,
    *,
    encoder_velocity_mode: str,
) -> np.ndarray | None:
    pred_arr = np.asarray(pred, dtype=float)
    if pred_arr.ndim != 2 or pred_arr.shape[0] == 0:
        return None
    if pred_arr.shape[1] >= 2 and encoder_velocity_mode in {"full_encoder", "late_fusion"}:
        return pred_arr[:, 1]
    return _finite_difference_series(pred_arr[:, 0], np.asarray(t, dtype=float))


def _finite_difference_series(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    if values.size != t.size:
        raise ValueError("values and t must have matching lengths.")
    out = np.full_like(values, np.nan, dtype=float)
    if values.size < 2:
        return np.nan_to_num(out, nan=0.0)

    dt = np.diff(t)
    dy = np.diff(values)
    valid = np.abs(dt) > 1e-12
    out[1:][valid] = dy[valid] / dt[valid]
    finite = np.isfinite(out)
    if np.any(finite[1:]):
        first_valid = int(np.where(finite)[0][0])
        out[0] = out[first_valid]
    elif np.isfinite(out[1]):
        out[0] = out[1]
    else:
        out[:] = 0.0
    return out


def _maybe_array(values, *, dtype=float):
    return None if values is None else np.asarray(values, dtype=dtype)


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
    initial_theta: float | None = None,
    initial_theta_dot: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    theta_dot = _maybe_array(theta_dot)
    init_theta = theta if initial_theta is None else np.asarray([initial_theta], dtype=float)
    if initial_theta_dot is not None:
        init_theta_dot = np.asarray([initial_theta_dot], dtype=float)
    elif use_sensor_y_dot:
        init_theta_dot = theta_dot
    else:
        init_theta_dot = None
    init = prepare_ode_initial_segment(init_theta, init_theta_dot, dt)
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


def _predict_encoder_theta_series(
    encoder,
    frame_dataset,
    n_samples: int,
    *,
    device: str = "auto",
) -> np.ndarray:
    pred = predict_encoder_framewise(encoder, frame_dataset, device=device)
    theta = np.full(int(n_samples), np.nan, dtype=float)
    theta[pred["idx"]] = pred["pred"][:, 0]
    if np.any(~np.isfinite(theta)):
        raise ValueError("Encoder predictions are missing for some sensor samples.")
    return theta


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
