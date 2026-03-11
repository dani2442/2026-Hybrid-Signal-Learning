#!/usr/bin/env python
"""Train/evaluate the BAB multimodal pipeline (video, sensor, decoder labels)."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

# Allow `python examples/run_bab_video_pipeline.py` from repo root.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _make_run_dir(output_root: str, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_metrics_csv(metrics: dict, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in sorted(metrics.items()):
            writer.writerow([k, f"{float(v):.6f}"])


def _save_json(data: dict, path: Path) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def _split_indices(n: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(n, dtype=int)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        all_idx[:n_train],
        all_idx[n_train : n_train + n_val],
        all_idx[n_train + n_val :],
    )


def _collate_frame_state(batch):
    import torch

    frames = torch.stack([b[0] for b in batch])
    states = torch.stack([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return frames, states, metas


def _predict_encoder_framewise(encoder, dataset, *, device: str = "auto", batch_size: int = 64):
    import torch
    from torch.utils.data import DataLoader
    from src.models.base import resolve_device

    dev = resolve_device(device)
    encoder = encoder.to(dev).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_frame_state)

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


def _states_from_sensor(data) -> np.ndarray:
    theta_dot = data.y_dot if data.y_dot is not None else np.zeros_like(data.y)
    return np.column_stack([data.y, theta_dot]).astype(float)


def _predict_image_decoder(decoder, states: np.ndarray, *, device: str = "auto", batch_size: int = 128) -> np.ndarray:
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


def _rollout_states(model_or_func, x0, u_seq, k_steps: int, dt: float):
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


def _plot_video_to_sensor(
    out_path: Path,
    t: np.ndarray,
    theta_true: np.ndarray,
    theta_pred: np.ndarray,
    theta_video_label: Optional[np.ndarray] = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, theta_true, label="sensor theta (true)", linewidth=1.0)
    ax.plot(t, theta_pred, label="encoder theta (video->sensor)", linewidth=1.0, linestyle="--")
    if theta_video_label is not None and np.any(np.isfinite(theta_video_label)):
        ax.plot(t, theta_video_label, label="video theta label (aligned)", linewidth=0.9, alpha=0.75)
    ax.set_title("Video -> Sensor")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("theta [deg]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_sensor_to_future_sensor(
    out_path: Path,
    model_or_func,
    sample: dict,
    dt: float,
    device: str = "auto",
) -> None:
    import torch
    import matplotlib.pyplot as plt
    from src.models.base import resolve_device

    dev = resolve_device(device)
    u_seq = sample["u_seq"].unsqueeze(0).to(dev)  # (1, K, 1)
    x_true = sample["x_seq"].to(dev)  # (K, 2)
    k_steps = x_true.shape[0]
    x0 = x_true[0:1, :]
    x_hat = _rollout_states(model_or_func, x0, u_seq, k_steps, dt=dt)  # (K, 1, 2)
    x_hat_np = x_hat[:, 0, :].cpu().numpy()
    x_true_np = x_true.cpu().numpy()
    tt = np.arange(k_steps, dtype=float) * float(dt)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(tt, x_true_np[:, 0], label="true theta", linewidth=1.0)
    axes[0].plot(tt, x_hat_np[:, 0], label="pred theta (sensor->future sensor)", linewidth=1.0, linestyle="--")
    axes[0].set_ylabel("theta [deg]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(tt, x_true_np[:, 1], label="true theta_dot", linewidth=1.0)
    axes[1].plot(tt, x_hat_np[:, 1], label="pred theta_dot", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("theta_dot")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_sensor_to_future_sensor_osa_full(
    out_path: Path,
    t: np.ndarray,
    theta_true: np.ndarray,
    theta_pred: np.ndarray,
    *,
    theta_dot_true: Optional[np.ndarray] = None,
    theta_dot_pred: Optional[np.ndarray] = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, theta_true, label="true theta", linewidth=1.0)
    axes[0].plot(t, theta_pred, label="pred theta (OSA)", linewidth=1.0, linestyle="--")
    axes[0].set_ylabel("theta [deg]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if theta_dot_true is not None and theta_dot_pred is not None:
        axes[1].plot(t, theta_dot_true, label="true theta_dot", linewidth=1.0)
        axes[1].plot(t, theta_dot_pred, label="pred theta_dot (from OSA theta)", linewidth=1.0, linestyle="--")
        axes[1].set_ylabel("theta_dot")
    else:
        resid = theta_true - theta_pred
        axes[1].plot(t, resid, label="theta residual (true - pred)", linewidth=1.0)
        axes[1].set_ylabel("residual")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Sensor -> Future Sensor (NeuralODE, complete OSA run)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_sensor_to_video_coords(
    out_path: Path,
    decoder,
    states: np.ndarray,
    keypoints_true: np.ndarray,
    *,
    t: Optional[np.ndarray] = None,
    device: str = "auto",
) -> np.ndarray:
    import torch
    import matplotlib.pyplot as plt
    from src.models.base import resolve_device

    dev = resolve_device(device)
    decoder = decoder.to(dev).eval()
    with torch.no_grad():
        pred = decoder(torch.tensor(states, dtype=torch.float32, device=dev)).cpu().numpy()

    if t is None:
        t = np.arange(len(pred), dtype=float)

    labels = ("beam_left_x", "beam_left_y", "beam_right_x", "beam_right_y")
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(t, keypoints_true[:, k], linewidth=0.9, label=f"true {labels[k]}")
        ax.plot(t, pred[:, k], linewidth=0.9, linestyle="--", label=f"pred {labels[k]}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Sensor -> Video (decoder coordinates)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return pred


def _select_evenly_spaced(indices: np.ndarray, n: int) -> np.ndarray:
    if len(indices) == 0:
        return np.array([], dtype=int)
    n = int(max(1, min(n, len(indices))))
    if n == 1:
        return np.array([int(indices[len(indices) // 2])], dtype=int)
    sel = np.linspace(0, len(indices) - 1, n, dtype=int)
    return indices[sel]


def _plot_sensor_to_video_overlay(
    out_path: Path,
    decoder,
    states: np.ndarray,
    keypoints_true: np.ndarray,
    frames: np.ndarray,
    frame_index_map: np.ndarray,
    sample_indices: Iterable[int],
    *,
    device: str = "auto",
) -> None:
    import torch
    import matplotlib.pyplot as plt
    from src.models.base import resolve_device

    idx = np.asarray(list(sample_indices), dtype=int)
    if len(idx) == 0:
        return

    dev = resolve_device(device)
    decoder = decoder.to(dev).eval()
    with torch.no_grad():
        pred = decoder(torch.tensor(states[idx], dtype=torch.float32, device=dev)).cpu().numpy()

    fig, axes = plt.subplots(1, len(idx), figsize=(4 * len(idx), 4), squeeze=False)
    for j, i in enumerate(idx):
        ax = axes[0, j]
        fidx = int(frame_index_map[i])
        ax.imshow(frames[fidx])
        gt = keypoints_true[i].reshape(2, 2)
        pd = pred[j].reshape(2, 2)
        ax.scatter(gt[:, 0], gt[:, 1], c="lime", s=45, label="true")
        ax.scatter(pd[:, 0], pd[:, 1], c="red", s=45, marker="x", label="pred")
        ax.set_title(f"sensor idx {i}")
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Sensor -> Video (decoder overlay on frames)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_sensor_to_video_image_errors(
    out_path: Path,
    t: np.ndarray,
    mse_per_frame: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, mse_per_frame, linewidth=0.9)
    ax.set_title("Sensor -> Video (decoder) reconstruction error over complete run")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("MSE per frame")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_sensor_to_video_image_montage(
    out_path: Path,
    t: np.ndarray,
    frames_true: np.ndarray,
    frames_pred: np.ndarray,
    sample_indices: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    if len(sample_indices) == 0:
        return

    n = len(sample_indices)
    fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 8), squeeze=False)
    for col, i in enumerate(sample_indices):
        gt = np.clip(frames_true[i], 0.0, 1.0)
        pd = np.clip(frames_pred[i], 0.0, 1.0)
        err = np.abs(gt - pd)

        axes[0, col].imshow(gt)
        axes[0, col].set_title(f"true t={t[i]:.2f}s", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(pd)
        axes[1, col].set_title("pred", fontsize=8)
        axes[1, col].axis("off")

        axes[2, col].imshow(err)
        axes[2, col].set_title("abs error", fontsize=8)
        axes[2, col].axis("off")

    fig.suptitle("Sensor -> Video (true vs predicted frames, complete run samples)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _load_ode_func(args):
    from src.models.blackbox_ode import _build_structured

    if args.ode_checkpoint:
        from src.models.base import load_model

        model = load_model(args.ode_checkpoint)
        if hasattr(model, "func_") and model.func_ is not None:
            return model.func_
        raise ValueError(f"Checkpoint {args.ode_checkpoint} does not expose 'func_'.")

    print("No --ode-checkpoint given; creating a fresh StructuredNODE dynamics.")
    return _build_structured(hidden_dim=128)


def _train_ode_model_separate(args, data, run_dir: Path):
    from src.config import BlackboxODE2DConfig
    from src.models.blackbox_ode import StructuredNODE

    n = len(data)
    tr, va, te = _split_indices(n)

    ode_cfg = BlackboxODE2DConfig(
        hidden_dim=args.ode_hidden_dim,
        dt=(1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0),
        k_steps=args.k_steps,
        epochs=args.ode_epochs if args.ode_epochs is not None else args.epochs,
        batch_size=args.ode_batch_size if args.ode_batch_size is not None else args.batch_size,
        learning_rate=args.ode_lr if args.ode_lr is not None else args.lr,
        device=args.device,
        seed=args.seed,
        verbose=True,
        wandb_project=args.wandb,
        wandb_run_name=f"ode_{args.dataset}",
    )
    model = StructuredNODE(ode_cfg)
    model.fit(
        train_data=(np.asarray(data.u)[tr], np.asarray(data.y)[tr]),
        val_data=(np.asarray(data.u)[va], np.asarray(data.y)[va]) if len(va) > 0 else None,
    )
    model.save(run_dir / "ode_model.pt")
    return model, tr, va, te


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
    _save_metrics_csv(metrics, run_dir / "metrics_video_to_sensor.csv")
    print("Video->sensor metrics:", metrics)

    pred = _predict_encoder_framewise(encoder, test_ds, device=args.device)
    theta_video = None
    if "theta_sensor_from_video" in aux:
        theta_video = np.asarray(aux["theta_sensor_from_video"])[pred["idx"]]
    _plot_video_to_sensor(
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

    states = _states_from_sensor(data)
    frames_sensor = np.asarray(frames[frame_idx_map], dtype=np.float32) / 255.0
    t = np.asarray(data.t, dtype=float)

    tr, va, te = _split_indices(len(states))
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
    _save_metrics_csv(metrics, run_dir / "metrics_sensor_to_video.csv")
    print("Sensor->video metrics:", metrics)

    pred = _predict_image_decoder(decoder, states[te], device=args.device)
    mse_per_frame = np.mean((pred - frames_sensor[te]) ** 2, axis=(1, 2, 3))
    _plot_sensor_to_video_image_errors(
        run_dir / "plot_sensor_to_video_error_timeline.png",
        t[te],
        mse_per_frame,
    )
    sample_sel = _select_evenly_spaced(np.arange(len(te), dtype=int), n=args.plot_count)
    _plot_sensor_to_video_image_montage(
        run_dir / "plot_sensor_to_video_frames.png",
        t[te],
        frames_sensor[te],
        pred,
        sample_sel,
    )
    _save_json(
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

    ode_func = _load_ode_func(args)
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
    _save_metrics_csv(enc_metrics, run_dir / "metrics_video_to_sensor.csv")
    pred_enc = _predict_encoder_framewise(encoder, frame_test_ds, device=args.device)
    theta_video = None
    if "theta_sensor_from_video" in aux:
        theta_video = np.asarray(aux["theta_sensor_from_video"])[pred_enc["idx"]]
    _plot_video_to_sensor(
        run_dir / "plot_video_to_sensor.png",
        pred_enc["t"],
        pred_enc["true"][:, 0],
        pred_enc["pred"][:, 0],
        theta_video_label=theta_video,
    )

    # 2) sensor -> future sensor (NeuralODE rollout)
    ode_metrics = evaluate_ode_rollout(composite, test_ds, init_from="sensor", device=args.device)
    _save_metrics_csv(ode_metrics, run_dir / "metrics_sensor_to_future_sensor.csv")
    if len(test_ds) > 0:
        dt = 1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0
        _plot_sensor_to_future_sensor(
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
        _, _, test_idx = _split_indices(len(states))
        dec_metrics = evaluate_decoder(decoder, states[test_idx], keypoints_sensor[test_idx], device=args.device)
        _save_metrics_csv(dec_metrics, run_dir / "metrics_sensor_to_video.csv")
        _plot_sensor_to_video_coords(
            run_dir / "plot_sensor_to_video_coords.png",
            decoder,
            states[test_idx],
            keypoints_sensor[test_idx],
            t=np.asarray(data.t, dtype=float)[test_idx],
            device=args.device,
        )
        overlay_sel = _select_evenly_spaced(test_idx, n=args.plot_count)
        _plot_sensor_to_video_overlay(
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
    _save_metrics_csv(e2e_metrics, run_dir / "metrics_end2end.csv")
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
    ode_func = _load_ode_func(args)
    ds = WindowedSequenceDataset(data, frames, frame_idx_map, k_steps=args.k_steps)
    _, _, test_ds = ds.split()
    ode_metrics = evaluate_ode_rollout(ode_func, test_ds, init_from="sensor", device=args.device)
    _save_metrics_csv(ode_metrics, run_dir / "metrics_sensor_to_future_sensor.csv")
    if len(test_ds) > 0:
        dt = 1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0
        _plot_sensor_to_future_sensor(
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
    _save_metrics_csv(enc_metrics, run_dir / "metrics_video_to_sensor.csv")
    pred_enc = _predict_encoder_framewise(encoder, enc_test_ds, device=args.device)
    theta_video = None
    if "theta_sensor_from_video" in aux:
        theta_video = np.asarray(aux["theta_sensor_from_video"])[pred_enc["idx"]]
    _plot_video_to_sensor(
        run_dir / "plot_video_to_sensor.png",
        pred_enc["t"],
        pred_enc["true"][:, 0],
        pred_enc["pred"][:, 0],
        theta_video_label=theta_video,
    )

    # 2) Train ODE separately and evaluate complete OSA run.
    ode_model, tr, va, te = _train_ode_model_separate(args, data, run_dir)
    u_test = np.asarray(data.u, dtype=float)[te]
    y_test = np.asarray(data.y, dtype=float)[te]
    t_test = np.asarray(data.t, dtype=float)[te]
    y_pred_osa = np.asarray(ode_model.predict_osa(u_test, y_test), dtype=float)

    n_osa = min(len(y_pred_osa), max(0, len(y_test) - 1))
    y_true_osa = y_test[1 : 1 + n_osa]
    y_pred_osa = y_pred_osa[:n_osa]
    t_osa = t_test[1 : 1 + n_osa]

    dt = 1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0
    if data.y_dot is not None:
        td_true_full = np.asarray(data.y_dot, dtype=float)[te]
    else:
        td_true_full = np.gradient(y_test, dt)
    td_true_osa = td_true_full[1 : 1 + n_osa]
    td_pred_osa = np.gradient(y_pred_osa, dt) if n_osa > 1 else np.zeros_like(y_pred_osa)

    ode_metrics = {
        "rmse_theta": Metrics.rmse(y_true_osa, y_pred_osa),
        "mae_theta": Metrics.mae(y_true_osa, y_pred_osa),
        "r2_theta": Metrics.r2(y_true_osa, y_pred_osa),
        "fit_theta": Metrics.fit_percent(y_true_osa, y_pred_osa),
        "rmse_theta_dot_approx": Metrics.rmse(td_true_osa, td_pred_osa),
    }
    _save_metrics_csv(ode_metrics, run_dir / "metrics_sensor_to_future_sensor.csv")
    _plot_sensor_to_future_sensor_osa_full(
        run_dir / "plot_sensor_to_future_sensor_osa_full.png",
        t_osa,
        y_true_osa,
        y_pred_osa,
        theta_dot_true=td_true_osa,
        theta_dot_pred=td_pred_osa,
    )

    # 3) Train image decoder separately (sensor state -> frame).
    states = _states_from_sensor(data)
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
    _save_metrics_csv(dec_metrics, run_dir / "metrics_sensor_to_video.csv")
    pred_frames = _predict_image_decoder(decoder, states[te], device=args.device)
    mse_per_frame = np.mean((pred_frames - frames_sensor[te]) ** 2, axis=(1, 2, 3))
    _plot_sensor_to_video_image_errors(
        run_dir / "plot_sensor_to_video_error_timeline.png",
        t_all[te],
        mse_per_frame,
    )
    sample_sel = _select_evenly_spaced(np.arange(len(te), dtype=int), n=args.plot_count)
    _plot_sensor_to_video_image_montage(
        run_dir / "plot_sensor_to_video_frames.png",
        t_all[te],
        frames_sensor[te],
        pred_frames,
        sample_sel,
    )

    _save_json(
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
    print("Sensor->future sensor (complete OSA) metrics:", ode_metrics)
    print("Sensor->video metrics:", dec_metrics)


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
        choices=["encoder_only", "decoder_only", "enc_ode", "ode_dec", "enc_ode_dec", "separate"],
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
    parser.add_argument("--ode-hidden-dim", type=int, default=128, help="Hidden width for structured ODE.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plot-count", type=int, default=6, help="Number of frame overlays for sensor->video plot.")
    parser.add_argument("--ode-checkpoint", default=None, help="Path to pretrained ODE checkpoint (.pt).")

    # Optional per-stage overrides for --mode separate
    parser.add_argument("--encoder-epochs", type=int, default=None)
    parser.add_argument("--encoder-batch-size", type=int, default=None)
    parser.add_argument("--encoder-lr", type=float, default=None)
    parser.add_argument("--ode-epochs", type=int, default=None)
    parser.add_argument("--ode-batch-size", type=int, default=None)
    parser.add_argument("--ode-lr", type=float, default=None)
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
    run_dir = _make_run_dir(args.output_root, run_name)
    print(f"Run directory: {run_dir}")

    config = vars(args)
    _save_json(config, run_dir / "config.json")

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
        _save_json(aux["theta_alignment"], run_dir / "theta_alignment.json")

    pretrained = not args.no_pretrained
    if args.encoder == "theta_regression":
        encoder = EncoderThetaNet(pretrained=pretrained)
    else:
        encoder = PoseResNet50(num_keypoints=2, pretrained=pretrained)

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
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(f"\nDone. Results in {run_dir}")


if __name__ == "__main__":
    main()
