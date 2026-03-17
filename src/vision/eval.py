"""Evaluation routines for the multimodal vision pipeline.

Each function returns a flat ``dict[str, float]`` of metrics, consistent
with the ``Metrics`` class for state-level metrics (RMSE, R², FIT%).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..validation import Metrics
from ..models.base import resolve_device
from .datasets import FrameStateDataset, WindowedSequenceDataset

if TYPE_CHECKING:
    from .models import EncOdeDecModel


# ─────────────────────────────────────────────────────────────────────
# 1. Encoder evaluation (per-frame)
# ─────────────────────────────────────────────────────────────────────

def _collate_frame_state(batch):
    frames = torch.stack([b[0] for b in batch])
    states = torch.stack([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return frames, states, metas


def evaluate_encoder_framewise(
    encoder: nn.Module,
    dataset: FrameStateDataset,
    *,
    device: str = "auto",
    batch_size: int = 64,
) -> Dict[str, float]:
    """Evaluate encoder on per-frame state prediction.

    Returns
    -------
    Dict with ``rmse_theta``, ``rmse_theta_dot``, ``r2_theta``,
    ``r2_theta_dot``, and overall ``rmse``, ``r2``, ``fit_pct``.
    """
    dev = resolve_device(device)
    encoder = encoder.to(dev).eval()

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_frame_state,
    )

    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

    with torch.no_grad():
        for frames, states, _meta in loader:
            pred = encoder(frames.to(dev)).cpu().numpy()
            all_pred.append(pred)
            all_true.append(states.numpy())

    pred_arr = np.concatenate(all_pred, axis=0)  # (N, 2)
    true_arr = np.concatenate(all_true, axis=0)  # (N, 2)

    return _state_metrics(true_arr, pred_arr)


# ─────────────────────────────────────────────────────────────────────
# 2. ODE rollout evaluation (two init modes)
# ─────────────────────────────────────────────────────────────────────

def evaluate_ode_rollout(
    ode_model,
    dataset: "WindowedSequenceDataset",
    *,
    encoder: Optional[nn.Module] = None,
    init_from: str = "sensor",
    device: str = "auto",
    batch_size: int = 16,
) -> Dict[str, float]:
    """Evaluate ODE rollout accuracy.

    Parameters
    ----------
    ode_model:
        Either an ``EncOdeDecModel`` (uses its ``rollout``), or a raw ODE
        func module (will be Euler-integrated manually).
    dataset:
        ``WindowedSequenceDataset``.
    encoder:
        Required when ``init_from="encoder"``.  Maps images → x0.
    init_from:
        ``"sensor"`` (ground-truth x0) or ``"encoder"`` (predicted x0).
    device:
        Device string.
    batch_size:
        Evaluation batch size.

    Returns
    -------
    Dict with ``rmse_theta``, ``rmse_theta_dot``, ``r2_theta``,
    ``r2_theta_dot``, ``rmse``, ``r2``, ``fit_pct``.
    """
    dev = resolve_device(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    control_series = torch.tensor(
        np.asarray(dataset.data.u, dtype=np.float32).reshape(-1, 1),
        dtype=torch.float32,
        device=dev,
    )

    # Determine rollout callable
    from .models import EncOdeDecModel

    if isinstance(ode_model, EncOdeDecModel):
        composite = ode_model.to(dev).eval()
        rollout_fn = composite.rollout
        ode_dt = composite.dt
    else:
        # Bare ODE func — wrap in a simple Euler integrator
        ode_func = ode_model.to(dev).eval()
        ode_dt = getattr(ode_model, "dt", 0.05)

        def rollout_fn(x0, u_seq, k_steps, *, control_series=None, start_idx=None):
            t_eval = torch.arange(k_steps, dtype=x0.dtype, device=dev) * ode_dt
            if start_idx is not None:
                ode_func.u_series = control_series
                ode_func.t_series = (
                    torch.arange(
                        control_series.shape[0],
                        dtype=x0.dtype,
                        device=dev,
                    )
                    * ode_dt
                )
                ode_func.batch_start_times = start_idx.to(
                    device=dev,
                    dtype=x0.dtype,
                ).reshape(-1, 1) * ode_dt
                x = x0.clone()
                traj = [x]
                for step in range(1, k_steps):
                    dx = ode_func.f(t_eval[step - 1], x)
                    x = x + dx * ode_dt
                    traj.append(x)
                ode_func.batch_start_times = None
                return torch.stack(traj, dim=0)

            if u_seq.ndim == 2:
                ode_func.u_series = u_seq.to(dev)
                ode_func.t_series = t_eval
                ode_func.batch_start_times = None
                x = x0.clone()
                traj = [x]
                for step in range(1, k_steps):
                    dx = ode_func.f(t_eval[step - 1], x)
                    x = x + dx * ode_dt
                    traj.append(x)
                return torch.stack(traj, dim=0)

            if u_seq.ndim != 3:
                raise ValueError(f"u_seq must have shape (K, 1) or (B, K, 1); got {tuple(u_seq.shape)}")

            batch_traj = []
            for b in range(x0.shape[0]):
                ode_func.u_series = u_seq[b].to(dev)
                ode_func.t_series = t_eval
                ode_func.batch_start_times = None
                x = x0[b : b + 1].clone()
                traj_b = [x]
                for step in range(1, k_steps):
                    dx = ode_func.f(t_eval[step - 1], x)
                    x = x + dx * ode_dt
                    traj_b.append(x)
                batch_traj.append(torch.stack(traj_b, dim=0))  # (K, 1, 2)
            return torch.cat(batch_traj, dim=1)  # (K, B, 2)

    if encoder is not None:
        encoder = encoder.to(dev).eval()

    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            u_seq = batch["u_seq"].to(dev)
            x_seq = batch["x_seq"].to(dev)  # (B, K, 2)
            start_idx = batch["start_idx"].to(dev)
            K = x_seq.shape[1]

            if init_from == "encoder":
                if encoder is None:
                    raise ValueError("encoder is required when init_from='encoder'")
                y0 = batch["y0"].to(dev)
                y_prev = batch["y_prev"].to(dev)
                # Encode two consecutive frames for finite-diff velocity.
                theta_0 = encoder(y0)        # (B, 1)
                theta_prev = encoder(y_prev)  # (B, 1)
                theta_dot_0 = (theta_0 - theta_prev) / ode_dt
                x0 = torch.cat([theta_0, theta_dot_0], dim=-1)  # (B, 2)
            else:
                x0 = x_seq[:, 0, :]  # (B, 2)

            x_hat = rollout_fn(
                x0,
                u_seq,
                K,
                control_series=control_series,
                start_idx=start_idx,
            )  # (K, B, 2)
            # Convert to (B, K, 2) for comparison
            x_hat_np = x_hat.permute(1, 0, 2).cpu().numpy()
            x_true_np = x_seq.cpu().numpy()
            all_pred.append(x_hat_np.reshape(-1, 2))
            all_true.append(x_true_np.reshape(-1, 2))

    pred_arr = np.concatenate(all_pred, axis=0)
    true_arr = np.concatenate(all_true, axis=0)
    return _state_metrics(true_arr, pred_arr)


# ─────────────────────────────────────────────────────────────────────
# 3. Decoder evaluation
# ─────────────────────────────────────────────────────────────────────

def evaluate_decoder(
    decoder: nn.Module,
    states: np.ndarray,
    keypoints_true: np.ndarray,
    *,
    device: str = "auto",
) -> Dict[str, float]:
    """Evaluate decoder (state → keypoints) accuracy.

    Parameters
    ----------
    decoder:
        Decoder module (e.g. ``DecoderKeypointsMLP``).
    states:
        ``(N, 2)`` input states.
    keypoints_true:
        ``(N, 4)`` ground-truth keypoint coordinates.

    Returns
    -------
    Dict with ``rmse_keypoints`` and per-coordinate RMSE.
    """
    dev = resolve_device(device)
    decoder = decoder.to(dev).eval()

    x = torch.tensor(states, dtype=torch.float32, device=dev)
    with torch.no_grad():
        y_hat = decoder(x).cpu().numpy()

    y_true = np.asarray(keypoints_true)
    rmse_all = float(np.sqrt(np.mean((y_hat - y_true) ** 2)))
    result: Dict[str, float] = {"rmse_keypoints": rmse_all}

    labels = ["x_l", "y_l", "x_r", "y_r"]
    for i, lbl in enumerate(labels[: y_true.shape[1]]):
        result[f"rmse_{lbl}"] = float(
            np.sqrt(np.mean((y_hat[:, i] - y_true[:, i]) ** 2))
        )
    return result


def _prepare_frame_tensor(frames: np.ndarray, device) -> torch.Tensor:
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"frames must be 4D, got shape {arr.shape}")
    if arr.shape[-1] in (1, 3):
        arr = np.transpose(arr, (0, 3, 1, 2))
    elif arr.shape[1] not in (1, 3):
        raise ValueError(
            "frames must be shaped (N,H,W,C) or (N,C,H,W) with C in {1,3}; "
            f"got {arr.shape}"
        )
    if arr.max() > 1.0 or arr.min() < 0.0:
        arr = np.clip(arr / 255.0, 0.0, 1.0)
    return torch.tensor(arr, dtype=torch.float32, device=device)


def evaluate_image_decoder(
    decoder: nn.Module,
    states: np.ndarray,
    frames_true: np.ndarray,
    *,
    device: str = "auto",
) -> Dict[str, float]:
    """Evaluate image decoder (state → RGB frame)."""
    dev = resolve_device(device)
    decoder = decoder.to(dev).eval()

    x = torch.tensor(np.asarray(states, dtype=np.float32), dtype=torch.float32, device=dev)
    y_true = _prepare_frame_tensor(frames_true, dev)
    with torch.no_grad():
        y_hat = decoder(x)

    mse = F.mse_loss(y_hat, y_true).item()
    mae = F.l1_loss(y_hat, y_true).item()
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    return {
        "mse_image": float(mse),
        "mae_image": float(mae),
        "psnr_db": psnr,
    }


# ─────────────────────────────────────────────────────────────────────
# 4. End-to-end evaluation
# ─────────────────────────────────────────────────────────────────────

def evaluate_end2end(
    enc_ode_dec: "EncOdeDecModel",
    dataset: "WindowedSequenceDataset",
    *,
    device: str = "auto",
    batch_size: int = 16,
) -> Dict[str, float]:
    """Evaluate the full Enc → ODE → (Dec) pipeline.

    Returns metrics for:
      - State rollout (init from encoder):  ``rmse_theta``, ``r2_theta``, …
      - Decoder observations (if decoder present): ``rmse_keypoints``.
    """
    dev = resolve_device(device)
    enc_ode_dec = enc_ode_dec.to(dev).eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    control_series = torch.tensor(
        np.asarray(dataset.data.u, dtype=np.float32).reshape(-1, 1),
        dtype=torch.float32,
        device=dev,
    )

    all_pred_x: list[np.ndarray] = []
    all_true_x: list[np.ndarray] = []
    all_pred_y: list[np.ndarray] = []
    all_true_y: list[np.ndarray] = []
    has_decoder = enc_ode_dec.decoder is not None

    with torch.no_grad():
        for batch in loader:
            y0 = batch["y0"].to(dev)
            y_prev = batch["y_prev"].to(dev)
            u_seq = batch["u_seq"].to(dev)
            x_seq = batch["x_seq"].to(dev)
            start_idx = batch["start_idx"].to(dev)
            K = x_seq.shape[1]

            outputs = enc_ode_dec(
                y0,
                u_seq,
                K,
                y_prev=y_prev,
                control_series=control_series,
                start_idx=start_idx,
            )
            x_hat = outputs["x_seq_hat"].permute(1, 0, 2).cpu().numpy()  # (B, K, 2)
            all_pred_x.append(x_hat.reshape(-1, 2))
            all_true_x.append(x_seq.cpu().numpy().reshape(-1, 2))

            if has_decoder and "y_seq" in batch:
                y_hat = outputs["y_seq_hat"].permute(1, 0, 2).cpu().numpy()
                y_true = batch["y_seq"].cpu().numpy()
                all_pred_y.append(y_hat.reshape(-1, y_hat.shape[-1]))
                all_true_y.append(y_true.reshape(-1, y_true.shape[-1]))

    pred_x = np.concatenate(all_pred_x, axis=0)
    true_x = np.concatenate(all_true_x, axis=0)
    result = _state_metrics(true_x, pred_x)

    if all_pred_y:
        pred_y = np.concatenate(all_pred_y, axis=0)
        true_y = np.concatenate(all_true_y, axis=0)
        result["rmse_keypoints"] = float(np.sqrt(np.mean((pred_y - true_y) ** 2)))

    return result


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────

def _state_metrics(true_arr: np.ndarray, pred_arr: np.ndarray) -> Dict[str, float]:
    """Compute per-component and overall state metrics.

    Parameters
    ----------
    true_arr, pred_arr:
        ``(N, D)`` arrays.  When ``D=2`` the columns are ``[θ, θ̇]``.
        When ``D=1`` (encoder-only evaluation) only θ metrics are returned.

    Returns
    -------
    Dict with ``rmse_theta``, ``r2_theta``, and (when D≥2)
    ``rmse_theta_dot``, ``r2_theta_dot``, plus overall ``rmse``, ``r2``,
    ``fit_pct``.
    """
    if true_arr.ndim == 1:
        true_arr = true_arr[:, np.newaxis]
    if pred_arr.ndim == 1:
        pred_arr = pred_arr[:, np.newaxis]

    theta_true, theta_pred = true_arr[:, 0], pred_arr[:, 0]

    result: Dict[str, float] = {
        "rmse_theta": Metrics.rmse(theta_true, theta_pred),
        "r2_theta": Metrics.r2(theta_true, theta_pred),
    }

    if true_arr.shape[1] >= 2 and pred_arr.shape[1] >= 2:
        td_true, td_pred = true_arr[:, 1], pred_arr[:, 1]
        result["rmse_theta_dot"] = Metrics.rmse(td_true, td_pred)
        result["r2_theta_dot"] = Metrics.r2(td_true, td_pred)

    result["rmse"] = Metrics.rmse(true_arr.ravel(), pred_arr.ravel())
    result["r2"] = Metrics.r2(true_arr.ravel(), pred_arr.ravel())
    result["fit_pct"] = Metrics.fit_percent(true_arr.ravel(), pred_arr.ravel())
    return result
