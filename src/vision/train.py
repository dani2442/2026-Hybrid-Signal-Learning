"""Training loops for the vision pipeline.

Three standalone entry-points plus a combined end-to-end loop:

1. ``train_encoder``  — image → state regression (EncoderThetaNet).
2. ``train_decoder``  — state → keypoints (DecoderKeypointsMLP).
3. ``train_image_decoder`` — state → RGB image (transpose-conv decoder).
4. ``train_end_to_end`` — Enc → ODE → (Dec) with multi-loss training.

The ODE-only sensor training should use the existing
``BaseModel.fit()`` / ``_BlackboxODE2D._fit()`` path; this module does
**not** duplicate that functionality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..logging import WandbLogger
from ..models.base import DEFAULT_GRAD_CLIP, resolve_device
from .losses import compute_losses, mse_state, mse_keypoints

if TYPE_CHECKING:
    from .models import EncOdeDecModel


# ─────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────

def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────
# 1. Encoder-only training
# ─────────────────────────────────────────────────────────────────────

def _collate_frame_state(batch):
    """Custom collate that handles the meta-dict third element."""
    frames = torch.stack([b[0] for b in batch])
    states = torch.stack([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return frames, states, metas


def train_encoder(
    model: nn.Module,
    train_dataset,
    val_dataset=None,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "auto",
    seed: Optional[int] = None,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train an encoder (image → state) using per-frame supervision.

    Parameters
    ----------
    model:
        Encoder ``nn.Module`` (e.g. ``EncoderThetaNet``).
    train_dataset:
        ``FrameStateDataset`` yielding (frame, state, meta).
    val_dataset:
        Optional validation dataset (same type).
    epochs, batch_size, lr:
        Standard training hyper-parameters.
    device:
        Device string ("auto", "cpu", "cuda").
    seed:
        RNG seed for reproducibility.
    grad_clip:
        Max gradient norm.
    wandb_project, wandb_run_name:
        W&B logging (None → disabled).
    checkpoint_dir:
        If given, save ``encoder_best.pt`` and ``encoder_last.pt`` here.
    verbose:
        Show tqdm progress bar.

    Returns
    -------
    Dict with ``"train_loss"``, ``"val_loss"`` histories and ``"best_val_loss"``.
    """
    _set_seed(seed)
    dev = resolve_device(device)
    model = model.to(dev)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=_collate_frame_state, drop_last=True,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=_collate_frame_state,
        )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    logger = WandbLogger(project=wandb_project, run_name=wandb_run_name)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")

    it = range(epochs)
    if verbose:
        it = tqdm(it, desc="Encoder training", unit="epoch")

    for epoch in it:
        # ── train ─────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for frames, states, _meta in train_loader:
            frames, states = frames.to(dev), states.to(dev)
            optimizer.zero_grad()
            pred = model(frames)
            loss = mse_state(pred, states)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # ── validate ──────────────────────────────────────────────────
        avg_val = float("nan")
        if val_loader is not None:
            model.eval()
            val_sum, val_n = 0.0, 0
            with torch.no_grad():
                for frames, states, _meta in val_loader:
                    frames, states = frames.to(dev), states.to(dev)
                    pred = model(frames)
                    val_sum += mse_state(pred, states).item()
                    val_n += 1
            avg_val = val_sum / max(val_n, 1)
            val_losses.append(avg_val)
            if avg_val < best_val:
                best_val = avg_val
                if checkpoint_dir:
                    _save_module(model, Path(checkpoint_dir) / "encoder_best.pt")

        # ── log ───────────────────────────────────────────────────────
        metrics = {"train/loss": avg_train, "train/epoch": epoch + 1}
        if val_loader is not None:
            metrics["val/loss"] = avg_val
        logger.log_metrics(metrics, step=epoch + 1)
        if verbose and hasattr(it, "set_postfix"):
            it.set_postfix(train=f"{avg_train:.5f}", val=f"{avg_val:.5f}")

    if checkpoint_dir:
        _save_module(model, Path(checkpoint_dir) / "encoder_last.pt")

    logger.finish()
    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_val_loss": best_val,
    }


# ─────────────────────────────────────────────────────────────────────
# 2. Decoder-only training
# ─────────────────────────────────────────────────────────────────────

def train_decoder(
    model: nn.Module,
    states: np.ndarray,
    keypoints: np.ndarray,
    *,
    val_states: Optional[np.ndarray] = None,
    val_keypoints: Optional[np.ndarray] = None,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "auto",
    seed: Optional[int] = None,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a decoder (state → keypoints) with simple tensor supervision.

    Parameters
    ----------
    model:
        Decoder ``nn.Module`` (e.g. ``DecoderKeypointsMLP``).
    states:
        ``(N, 2)`` training states ``[θ, θ̇]``.
    keypoints:
        ``(N, 4)`` training targets ``[x_l, y_l, x_r, y_r]``.
    val_states, val_keypoints:
        Optional validation arrays.

    Returns
    -------
    Dict with ``"train_loss"``, ``"val_loss"`` histories.
    """
    _set_seed(seed)
    dev = resolve_device(device)
    model = model.to(dev)

    x_train = torch.tensor(states, dtype=torch.float32, device=dev)
    y_train = torch.tensor(keypoints, dtype=torch.float32, device=dev)
    n = len(x_train)

    has_val = val_states is not None and val_keypoints is not None
    if has_val:
        x_val = torch.tensor(val_states, dtype=torch.float32, device=dev)
        y_val = torch.tensor(val_keypoints, dtype=torch.float32, device=dev)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    logger = WandbLogger(project=wandb_project, run_name=wandb_run_name)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")

    it = range(epochs)
    if verbose:
        it = tqdm(it, desc="Decoder training", unit="epoch")

    for epoch in it:
        model.train()
        perm = torch.randperm(n, device=dev)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            pred = model(x_train[idx])
            loss = mse_keypoints(pred, y_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        avg_val = float("nan")
        if has_val:
            model.eval()
            with torch.no_grad():
                pred_val = model(x_val)
                avg_val = mse_keypoints(pred_val, y_val).item()
            val_losses.append(avg_val)
            if avg_val < best_val:
                best_val = avg_val
                if checkpoint_dir:
                    _save_module(model, Path(checkpoint_dir) / "decoder_best.pt")

        logger.log_metrics(
            {"train/loss": avg_train, "val/loss": avg_val, "train/epoch": epoch + 1},
            step=epoch + 1,
        )
        if verbose and hasattr(it, "set_postfix"):
            it.set_postfix(train=f"{avg_train:.5f}", val=f"{avg_val:.5f}")

    if checkpoint_dir:
        _save_module(model, Path(checkpoint_dir) / "decoder_last.pt")

    logger.finish()
    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_val_loss": best_val,
    }


def _prepare_frame_tensor(frames: np.ndarray, device) -> torch.Tensor:
    """Convert frame arrays to ``(N, 3, H, W)`` float tensor in ``[0, 1]``."""
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"frames must be 4D, got shape {arr.shape}")

    # Accept HWC or CHW format.
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


def train_image_decoder(
    model: nn.Module,
    states: np.ndarray,
    frames: np.ndarray,
    *,
    val_states: Optional[np.ndarray] = None,
    val_frames: Optional[np.ndarray] = None,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "auto",
    seed: Optional[int] = None,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a true decoder (state → RGB image) with pixel MSE loss."""
    _set_seed(seed)
    dev = resolve_device(device)
    model = model.to(dev)

    x_train = torch.tensor(np.asarray(states, dtype=np.float32), dtype=torch.float32, device=dev)
    y_train = _prepare_frame_tensor(frames, dev)
    n = len(x_train)

    has_val = val_states is not None and val_frames is not None
    if has_val:
        x_val = torch.tensor(np.asarray(val_states, dtype=np.float32), dtype=torch.float32, device=dev)
        y_val = _prepare_frame_tensor(val_frames, dev)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    logger = WandbLogger(project=wandb_project, run_name=wandb_run_name)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")

    it = range(epochs)
    if verbose:
        it = tqdm(it, desc="Image decoder training", unit="epoch")

    for epoch in it:
        model.train()
        perm = torch.randperm(n, device=dev)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            pred = model(x_train[idx])
            loss = F.mse_loss(pred, y_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        avg_val = float("nan")
        if has_val:
            model.eval()
            with torch.no_grad():
                pred_val = model(x_val)
                avg_val = F.mse_loss(pred_val, y_val).item()
            val_losses.append(avg_val)
            if avg_val < best_val:
                best_val = avg_val
                if checkpoint_dir:
                    _save_module(model, Path(checkpoint_dir) / "decoder_image_best.pt")

        logger.log_metrics(
            {"train/loss": avg_train, "val/loss": avg_val, "train/epoch": epoch + 1},
            step=epoch + 1,
        )
        if verbose and hasattr(it, "set_postfix"):
            it.set_postfix(train=f"{avg_train:.5f}", val=f"{avg_val:.5f}")

    if checkpoint_dir:
        _save_module(model, Path(checkpoint_dir) / "decoder_image_last.pt")

    logger.finish()
    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_val_loss": best_val,
    }


# ─────────────────────────────────────────────────────────────────────
# 3. End-to-end Enc → ODE → (Dec) training
# ─────────────────────────────────────────────────────────────────────

def train_end_to_end(
    enc_ode_dec: "EncOdeDecModel",
    train_dataset,
    val_dataset=None,
    *,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    loss_weights: Optional[Dict[str, float]] = None,
    device: str = "auto",
    seed: Optional[int] = None,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    freeze_ode_epochs: int = 0,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """End-to-end training: Enc → ODE → (Dec).

    Parameters
    ----------
    enc_ode_dec:
        ``EncOdeDecModel`` instance.
    train_dataset:
        ``WindowedSequenceDataset`` yielding dict samples.
    val_dataset:
        Optional validation dataset.
    epochs, batch_size, lr:
        Standard hyper-parameters.
    loss_weights:
        ``{"enc": w, "ode": w, "dec": w, "smooth": w}`` (defaults to 1.0).
    freeze_ode_epochs:
        Number of initial epochs where ODE parameters are frozen (train
        encoder only to match ODE expectations).
    wandb_project, wandb_run_name:
        W&B logging.
    checkpoint_dir:
        Save checkpoints here.
    verbose:
        tqdm progress.

    Returns
    -------
    Dict with per-loss histories and best validation loss.
    """
    _set_seed(seed)
    dev = resolve_device(device)
    enc_ode_dec = enc_ode_dec.to(dev)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
        )

    optimizer = optim.Adam(enc_ode_dec.parameters(), lr=lr)
    logger = WandbLogger(project=wandb_project, run_name=wandb_run_name)

    history: Dict[str, list[float]] = {"train_total": [], "val_total": []}
    best_val = float("inf")

    it = range(epochs)
    if verbose:
        it = tqdm(it, desc="End-to-end training", unit="epoch")

    for epoch in it:
        # Optionally freeze ODE for early epochs
        _set_requires_grad(
            enc_ode_dec.ode_func,
            requires_grad=(epoch >= freeze_ode_epochs),
        )

        # ── train ─────────────────────────────────────────────────────
        enc_ode_dec.train()
        epoch_losses: Dict[str, float] = {}
        n_batches = 0

        for batch in train_loader:
            y0 = batch["y0"].to(dev)
            y_prev = batch["y_prev"].to(dev)
            u_seq = batch["u_seq"].to(dev)
            x_seq = batch["x_seq"].to(dev)
            K = x_seq.shape[1]

            optimizer.zero_grad()
            outputs = enc_ode_dec(y0, u_seq, K, y_prev=y_prev)

            # Build targets dict
            targets: Dict[str, torch.Tensor] = {
                "x0": x_seq[:, 0, :],                     # (B, 2)
                "x_seq": x_seq.permute(1, 0, 2),          # (K, B, 2)
            }
            if "y_seq" in batch:
                targets["y_seq"] = batch["y_seq"].to(dev).permute(1, 0, 2)

            losses = compute_losses(outputs, targets, weights=loss_weights)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(enc_ode_dec.parameters(), grad_clip)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            n_batches += 1

        # Average
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
        history["train_total"].append(epoch_losses.get("total", float("nan")))

        # ── validate ──────────────────────────────────────────────────
        val_total = float("nan")
        if val_loader is not None:
            enc_ode_dec.eval()
            val_sum, val_n = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    y0 = batch["y0"].to(dev)
                    y_prev = batch["y_prev"].to(dev)
                    u_seq = batch["u_seq"].to(dev)
                    x_seq = batch["x_seq"].to(dev)
                    K = x_seq.shape[1]
                    outputs = enc_ode_dec(y0, u_seq, K, y_prev=y_prev)
                    targets = {
                        "x0": x_seq[:, 0, :],
                        "x_seq": x_seq.permute(1, 0, 2),
                    }
                    if "y_seq" in batch:
                        targets["y_seq"] = batch["y_seq"].to(dev).permute(1, 0, 2)
                    losses = compute_losses(outputs, targets, weights=loss_weights)
                    val_sum += losses["total"].item()
                    val_n += 1
            val_total = val_sum / max(val_n, 1)
            history["val_total"].append(val_total)
            if val_total < best_val:
                best_val = val_total
                if checkpoint_dir:
                    _save_composite(enc_ode_dec, Path(checkpoint_dir) / "enc_ode_dec_best.pt")

        # ── log ───────────────────────────────────────────────────────
        log_dict = {f"train/{k}": v for k, v in epoch_losses.items()}
        log_dict["val/total"] = val_total
        log_dict["train/epoch"] = epoch + 1
        logger.log_metrics(log_dict, step=epoch + 1)
        if verbose and hasattr(it, "set_postfix"):
            it.set_postfix(
                train=f"{epoch_losses.get('total', 0):.5f}",
                val=f"{val_total:.5f}",
            )

    if checkpoint_dir:
        _save_composite(enc_ode_dec, Path(checkpoint_dir) / "enc_ode_dec_last.pt")

    logger.finish()
    return {
        "history": history,
        "best_val_loss": best_val,
    }


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _save_module(module: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), path)


def _save_composite(model: "EncOdeDecModel", path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "encoder": model.encoder.state_dict(),
        "ode_func": model.ode_func.state_dict(),
    }
    if model.decoder is not None:
        checkpoint["decoder"] = model.decoder.state_dict()
    torch.save(checkpoint, path)


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad
