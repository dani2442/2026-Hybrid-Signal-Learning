"""Non-plotting helpers for the BAB multimodal pipeline.

Moved from ``examples/run_bab_video_pipeline.py`` so the main script
only contains mode-dispatch logic and CLI parsing.
"""

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

    dt = 1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0
    common_cfg = dict(
        dt=dt,
        epochs=args.ode_epochs if args.ode_epochs is not None else args.epochs,
        batch_size=args.ode_batch_size if args.ode_batch_size is not None else args.batch_size,
        learning_rate=args.ode_lr if args.ode_lr is not None else args.lr,
        device=args.device,
        seed=args.seed,
        verbose=True,
        wandb_project=args.wandb,
        wandb_run_name=f"ode_{args.dataset}",
    )

    if args.ode_model == "linear_physics":
        from src.config import LinearPhysicsConfig
        from src.models.physics_ode import LinearPhysics

        ode_cfg = LinearPhysicsConfig(**common_cfg)
        if args.ode_training_mode:
            ode_cfg.training_mode = args.ode_training_mode
        model = LinearPhysics(ode_cfg)
    elif args.ode_model == "stribeck_physics":
        from src.config import StribeckPhysicsConfig
        from src.models.physics_ode import StribeckPhysics

        ode_cfg = StribeckPhysicsConfig(**common_cfg)
        if args.ode_training_mode:
            ode_cfg.training_mode = args.ode_training_mode
        model = StribeckPhysics(ode_cfg)
    elif args.ode_model == "structured_node":
        from src.config import BlackboxODE2DConfig
        from src.models.blackbox_ode import StructuredNODE

        ode_cfg = BlackboxODE2DConfig(
            hidden_dim=args.ode_hidden_dim,
            k_steps=args.k_steps,
            **common_cfg,
        )
        if args.ode_training_mode:
            ode_cfg.training_mode = args.ode_training_mode
        model = StructuredNODE(ode_cfg)
    else:
        raise ValueError(f"Unsupported --ode-model: {args.ode_model}")

    print(f"ODE training mode: {ode_cfg.training_mode}")

    model.fit(
        train_data=(np.asarray(data.u)[tr], y[tr]),
        val_data=(np.asarray(data.u)[va], y[va]) if len(va) > 0 else None,
    )
    model.save(run_dir / "ode_model.pt")
    return model, tr, va, te
