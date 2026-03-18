"""Plotting helpers for the BAB multimodal pipeline.

Moved from ``examples/run_bab_video_pipeline.py`` to keep the main
pipeline script lean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# ── Video → Sensor ──────────────────────────────────────────────────

def plot_video_to_sensor(
    out_path: Path,
    t: np.ndarray,
    theta_true: np.ndarray,
    theta_pred: np.ndarray,
    theta_video_label: Optional[np.ndarray] = None,
    *,
    theta_dot_true: Optional[np.ndarray] = None,
    theta_dot_pred: Optional[np.ndarray] = None,
    theta_dot_fd: Optional[np.ndarray] = None,
    theta_dot_fd_label: str = "video theta_dot label (aligned FD)",
    theta_dot_pred_label: str = "encoder theta_dot (video->sensor)",
) -> None:
    import matplotlib.pyplot as plt

    has_theta_dot = any(
        arr is not None for arr in (theta_dot_true, theta_dot_pred, theta_dot_fd)
    )
    if has_theta_dot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        theta_ax, theta_dot_ax = axes
    else:
        fig, theta_ax = plt.subplots(figsize=(12, 4))
        theta_dot_ax = None

    theta_ax.plot(t, theta_true, label="sensor theta (true)", linewidth=1.0)
    theta_ax.plot(
        t,
        theta_pred,
        label="encoder theta (video->sensor)",
        linewidth=1.0,
        linestyle="--",
    )
    if theta_video_label is not None and np.any(np.isfinite(theta_video_label)):
        finite = np.isfinite(theta_video_label)
        theta_ax.scatter(
            t[finite],
            theta_video_label[finite],
            label="video theta label (aligned)",
            marker="x",
            s=28,
            linewidths=0.9,
            alpha=0.8,
            zorder=3,
            c="tab:green",
        )
    theta_ax.set_title("Video -> Sensor")
    theta_ax.set_ylabel("theta [deg]")
    theta_ax.grid(True, alpha=0.3)
    theta_ax.legend()

    if theta_dot_ax is not None:
        if theta_dot_true is not None:
            theta_dot_ax.plot(t, theta_dot_true, label="sensor theta_dot (true)", linewidth=1.0)
        if theta_dot_pred is not None:
            theta_dot_ax.plot(
                t,
                theta_dot_pred,
                label=theta_dot_pred_label,
                linewidth=1.0,
                linestyle="--",
            )
        if theta_dot_fd is not None:
            finite = np.isfinite(theta_dot_fd)
            theta_dot_ax.scatter(
                t[finite],
                theta_dot_fd[finite],
                label=theta_dot_fd_label,
                marker="x",
                s=28,
                linewidths=0.9,
                alpha=0.8,
                zorder=3,
                c="tab:green",
            )
        theta_dot_ax.set_ylabel("theta_dot [deg/s]")
        theta_dot_ax.set_xlabel("time [s]")
        theta_dot_ax.grid(True, alpha=0.3)
        theta_dot_ax.legend()
    else:
        theta_ax.set_xlabel("time [s]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ── Sensor → Future Sensor (windowed rollout) ──────────────────────

def plot_sensor_to_future_sensor(
    out_path: Path,
    model_or_func,
    sample: dict,
    dt: float,
    device: str = "auto",
) -> None:
    import torch
    import matplotlib.pyplot as plt
    from src.models.base import resolve_device
    from src.vision.pipeline_utils import rollout_states

    dev = resolve_device(device)
    u_seq = sample["u_seq"].unsqueeze(0).to(dev)  # (1, K, 1)
    x_true = sample["x_seq"].to(dev)  # (K, 2)
    k_steps = x_true.shape[0]
    x0 = x_true[0:1, :]
    x_hat = rollout_states(model_or_func, x0, u_seq, k_steps, dt=dt)  # (K, 1, 2)
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


# ── Sensor → Future Sensor (free-run, full trajectory) ─────────────

def plot_sensor_to_future_sensor_freerun_full(
    out_path: Path,
    t: np.ndarray,
    theta_true: np.ndarray,
    theta_pred: np.ndarray,
    *,
    u: Optional[np.ndarray] = None,
    theta_dot_true: Optional[np.ndarray] = None,
    theta_dot_pred: Optional[np.ndarray] = None,
    title: str = "Sensor -> Future Sensor (NeuralODE, free-run simulation)",
    theta_pred_label: str = "pred theta (free-run)",
    theta_dot_pred_label: str = "pred theta_dot (free-run)",
) -> None:
    import matplotlib.pyplot as plt

    if u is not None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(t, u, label="input u", linewidth=1.0, color="tab:green")
        axes[0].set_ylabel("u")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        theta_ax = axes[1]
        lower_ax = axes[2]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        theta_ax = axes[0]
        lower_ax = axes[1]

    theta_ax.plot(t, theta_true, label="true theta", linewidth=1.0)
    theta_ax.plot(t, theta_pred, label=theta_pred_label, linewidth=1.0, linestyle="--")
    theta_ax.set_ylabel("theta [deg]")
    theta_ax.grid(True, alpha=0.3)
    theta_ax.legend()

    if theta_dot_true is not None and theta_dot_pred is not None:
        lower_ax.plot(t, theta_dot_true, label="true theta_dot", linewidth=1.0)
        lower_ax.plot(
            t,
            theta_dot_pred,
            label=theta_dot_pred_label,
            linewidth=1.0,
            linestyle="--",
        )
        lower_ax.set_ylabel("theta_dot")
    else:
        resid = theta_true - theta_pred
        lower_ax.plot(t, resid, label="theta residual (true - pred)", linewidth=1.0)
        lower_ax.set_ylabel("residual")
    lower_ax.set_xlabel("time [s]")
    lower_ax.grid(True, alpha=0.3)
    lower_ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ── Sensor → Video (keypoint coordinates) ──────────────────────────

def plot_sensor_to_video_coords(
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


# ── Sensor → Video (overlay on frames) ─────────────────────────────

def plot_sensor_to_video_overlay(
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


# ── Sensor → Video (image reconstruction errors) ───────────────────

def plot_sensor_to_video_image_errors(
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


# ── Sensor → Video (image montage: true / pred / error) ────────────

def plot_sensor_to_video_image_montage(
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
