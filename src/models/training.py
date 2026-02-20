"""Shared training utilities for PyTorch-based models.

Provides composable building blocks (optimizer setup, scheduler, early
stopping, gradient clipping) and a *generic training loop* that every
neural model can call with a custom ``step_fn``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import BaseConfig
from src.models.constants import DEFAULT_GRAD_CLIP


# ─────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────

def inverse_softplus(x: float) -> float:
    """Inverse of ``softplus(x) = log(1 + exp(x))``.

    Used by hybrid models to initialise raw parameters so that
    ``softplus(raw)`` yields a desired positive value.
    """
    if x <= 0:
        raise ValueError(f"inverse_softplus requires x > 0, got {x}")
    return math.log(math.exp(x) - 1)


def gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    """Return total ℓ₂ gradient norm across all parameters."""
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


# ─────────────────────────────────────────────────────────────────────
# Optimizer / scheduler / early-stopping factories
# ─────────────────────────────────────────────────────────────────────

def make_optimizer(
    params: Iterable[nn.Parameter],
    lr: float = 1e-3,
) -> torch.optim.Adam:
    """Create an Adam optimiser."""
    return torch.optim.Adam(params, lr=lr)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    patience: int = 200,
    factor: float = 0.5,
    min_lr: float = 1e-6,
) -> ReduceLROnPlateau:
    """Create a ReduceLROnPlateau scheduler."""
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=patience,
        factor=factor,
        min_lr=min_lr,
    )


class EarlyStopper:
    """Tracks validation loss and signals when to stop.

    Parameters
    ----------
    patience : int | None
        Number of epochs without improvement to tolerate.
        ``None`` disables early stopping.
    """

    def __init__(self, patience: Optional[int] = None) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    @property
    def enabled(self) -> bool:
        return self.patience is not None and self.patience > 0

    def step(self, loss: float) -> bool:
        """Return *True* if training should stop."""
        if not self.enabled:
            return False
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ─────────────────────────────────────────────────────────────────────
# Generic training loop
# ─────────────────────────────────────────────────────────────────────

def train_loop(
    step_fn: Callable[[int], float],
    *,
    epochs: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: ReduceLROnPlateau | None = None,
    early_stopper: EarlyStopper | None = None,
    logger=None,
    log_every: int = 1,
    verbose: bool = True,
    desc: str = "Training",
    val_step_fn: Optional[Callable[[], float]] = None,
    val_metrics_fn: Optional[Callable[[], Dict[str, float]]] = None,
    model_params=None,
) -> List[float]:
    """Run a generic epoch loop.

    Parameters
    ----------
    step_fn : callable(epoch) -> float
        Must execute one epoch and return the loss.  The function is
        responsible for forward, backward, and ``optimizer.step()``.
    epochs : int
        Maximum number of epochs.
    optimizer : Optimizer | None
        If provided, ``scheduler`` will inspect it for lr.
    scheduler : ReduceLROnPlateau | None
        Stepped every epoch with the loss.
    early_stopper : EarlyStopper | None
        Checked every epoch.
    logger : WandbLogger | None
        Receives ``train_loss``, ``val_loss``, ``lr``, ``grad_norm`` per epoch.
    log_every : int
        Log every N epochs.
    verbose : bool
        Show tqdm progress bar.
    desc : str
        Label for the progress bar.
    val_step_fn : callable() -> float | None
        If provided, called every ``log_every`` epochs to compute validation
        loss logged as ``val_loss``.
    val_metrics_fn : callable() -> dict | None
        If provided, called every ``log_every`` epochs to compute a full
        set of validation metrics (R2, FIT …).  The returned
        dict is merged directly into the per-epoch log, so use prefixed
        keys such as ``val/R2``, ``val/FIT``, etc.
    model_params : iterable | None
        If provided, gradient norm is computed and logged as ``grad_norm``.

    Returns
    -------
    list[float]
        Per-epoch loss values.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    losses: List[float] = []
    iterator = range(epochs)
    pbar = tqdm(iterator, desc=desc, disable=not verbose) if tqdm else iterator

    for epoch in pbar:
        loss = step_fn(epoch)
        losses.append(loss)

        if scheduler is not None:
            scheduler.step(loss)

        if early_stopper is not None and early_stopper.step(loss):
            if verbose and tqdm is not None:
                pbar.set_postfix(loss=f"{loss:.6f}", status="early_stop")
            break

        if verbose and tqdm is not None and hasattr(pbar, "set_postfix"):
            lr_str = ""
            if optimizer is not None:
                lr_str = f"{optimizer.param_groups[0]['lr']:.2e}"
            pbar.set_postfix(loss=f"{loss:.6f}", lr=lr_str)

        if logger is not None and (epoch + 1) % log_every == 0:
            metrics: Dict[str, float] = {"train_loss": loss, "epoch": epoch}
            if optimizer is not None:
                metrics["lr"] = optimizer.param_groups[0]["lr"]
            if val_step_fn is not None:
                metrics["val_loss"] = val_step_fn()
            if val_metrics_fn is not None:
                metrics.update(val_metrics_fn())
            if model_params is not None:
                metrics["grad_norm"] = gradient_norm(model_params)
            logger.log_metrics(metrics, step=epoch)

    return losses


# ─────────────────────────────────────────────────────────────────────
# Supervised training (NARX-like / feedforward)
# ─────────────────────────────────────────────────────────────────────

def train_supervised_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: BaseConfig,
    logger=None,
    device: str = "cpu",
    val_data: Optional[tuple] = None,
) -> List[float]:
    """Batch-SGD training for feed-forward PyTorch models.

    Parameters
    ----------
    model : nn.Module
        The network to train.
    X_train, y_train : np.ndarray
        Feature matrix and targets.
    config : BaseConfig
        Must have ``learning_rate``, ``epochs``, ``batch_size``,
        ``grad_clip``, ``scheduler_*``, ``early_stopping_patience``,
        ``verbose``.
    logger : WandbLogger | None
    device : str
    val_data : tuple | None
        (X_val, y_val) feature matrices for validation loss logging.

    Returns
    -------
    list[float]
        Per-epoch training losses.
    """
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(-1)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True
    )

    model.to(device)
    optimizer = make_optimizer(model.parameters(), lr=config.learning_rate)
    scheduler = make_scheduler(
        optimizer,
        patience=config.scheduler_patience,
        factor=config.scheduler_factor,
        min_lr=config.scheduler_min_lr,
    )
    stopper = EarlyStopper(config.early_stopping_patience)
    criterion = nn.MSELoss()

    # Build validation step + metrics functions from pre-built feature matrices
    val_sfn = None
    val_metrics_fn = None
    if val_data is not None:
        X_val, y_val = val_data
        X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_v = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(-1)

        def val_sfn() -> float:
            model.eval()
            with torch.no_grad():
                return criterion(model(X_v), y_v).item()

        def val_metrics_fn() -> Dict[str, float]:
            from src.validation.metrics import compute_all
            model.eval()
            with torch.no_grad():
                pred_np = model(X_v).cpu().numpy().ravel()
            y_np = y_v.cpu().numpy().ravel()
            return {f"val/{k}": float(v) for k, v in compute_all(y_np, pred_np).items()}

    def step_fn(_epoch: int) -> float:
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    return train_loop(
        step_fn,
        epochs=config.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopper=stopper,
        logger=logger,
        log_every=getattr(config, "wandb_log_every", 1),
        verbose=config.verbose,
        desc=f"Training (supervised)",
        val_step_fn=val_sfn,
        val_metrics_fn=val_metrics_fn,
        model_params=list(model.parameters()),
    )


# ─────────────────────────────────────────────────────────────────────
# Sequence training (GRU / LSTM / TCN / Mamba)
# ─────────────────────────────────────────────────────────────────────

def create_sequence_dataset(
    u: np.ndarray,
    y: np.ndarray,
    window_size: int,
) -> torch.utils.data.TensorDataset:
    """Create sliding-window ``TensorDataset`` for sequence models.

    Parameters
    ----------
    u, y : np.ndarray
        1-D signals.
    window_size : int
        Sub-sequence length.

    Returns
    -------
    TensorDataset
        Pairs of ``(u_window, y_window)`` with shape ``(window, 1)``.
    """
    u = np.asarray(u, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
    n = len(u)

    if n < window_size:
        # If signal is shorter than window, use full signal as one sample
        u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        return torch.utils.data.TensorDataset(u_t, y_t)

    n_windows = n - window_size + 1
    u_windows = np.lib.stride_tricks.sliding_window_view(u, window_size)
    y_windows = np.lib.stride_tricks.sliding_window_view(y, window_size)

    u_t = torch.tensor(u_windows[:n_windows], dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y_windows[:n_windows], dtype=torch.float32).unsqueeze(-1)
    return torch.utils.data.TensorDataset(u_t, y_t)


def train_sequence_model(
    model: nn.Module,
    u_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: BaseConfig,
    logger=None,
    device: str = "cpu",
    val_data: tuple | None = None,
) -> List[float]:
    """Train a sequence model (GRU/LSTM/TCN/Mamba) with sliding windows.

    Parameters
    ----------
    model : nn.Module
        Sequence network.
    u_train, y_train : np.ndarray
        Training signals.
    config : BaseConfig
        Must have ``train_window_size``, ``batch_size``, ``learning_rate``,
        ``epochs``, ``grad_clip``, ``scheduler_*``, ``verbose``.
    logger : WandbLogger | None
    device : str
    val_data : tuple | None
        (u_val, y_val) for validation loss tracking.

    Returns
    -------
    list[float]
        Per-epoch training losses.
    """
    window_size = getattr(config, "train_window_size", 20)
    dataset = create_sequence_dataset(u_train, y_train, window_size)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )

    model.to(device)
    optimizer = make_optimizer(model.parameters(), lr=config.learning_rate)
    scheduler = make_scheduler(
        optimizer,
        patience=config.scheduler_patience,
        factor=config.scheduler_factor,
        min_lr=config.scheduler_min_lr,
    )
    stopper = EarlyStopper(config.early_stopping_patience)
    criterion = nn.MSELoss()

    # Build validation step + metrics functions (val_data is already normalised by the caller)
    val_sfn = None
    val_metrics_fn = None
    if val_data is not None:
        u_val_arr, y_val_arr = val_data
        u_val_t = (
            torch.tensor(np.asarray(u_val_arr, dtype=np.float32).ravel(),
                         dtype=torch.float32, device=device)
            .unsqueeze(0).unsqueeze(-1)
        )
        y_val_t = (
            torch.tensor(np.asarray(y_val_arr, dtype=np.float32).ravel(),
                         dtype=torch.float32, device=device)
            .unsqueeze(0).unsqueeze(-1)
        )

        def val_sfn() -> float:
            model.eval()
            with torch.no_grad():
                pred_v = model(u_val_t)
                return criterion(pred_v, y_val_t).item()

        def val_metrics_fn() -> Dict[str, float]:
            from src.validation.metrics import compute_all
            model.eval()
            with torch.no_grad():
                pred_np = model(u_val_t).cpu().numpy().ravel()
            y_np = y_val_t.cpu().numpy().ravel()
            return {f"val/{k}": float(v) for k, v in compute_all(y_np, pred_np).items()}

    def step_fn(_epoch: int) -> float:
        model.train()
        total_loss = 0.0
        count = 0
        for u_batch, y_batch in loader:
            u_batch = u_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(u_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            count += 1
        return total_loss / max(count, 1)

    return train_loop(
        step_fn,
        epochs=config.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopper=stopper,
        logger=logger,
        log_every=getattr(config, "wandb_log_every", 1),
        verbose=config.verbose,
        desc=f"Training (sequence)",
        val_step_fn=val_sfn,
        val_metrics_fn=val_metrics_fn,
        model_params=list(model.parameters()),
    )


# ─────────────────────────────────────────────────────────────────────
# Continuous-time subsequence training (NeuralODE / SDE / UDE / Physics)
# ─────────────────────────────────────────────────────────────────────

def train_sequence_batches(
    step_fn: Callable,
    *,
    u: np.ndarray,
    y: np.ndarray,
    config: BaseConfig,
    t: np.ndarray | None = None,
    optimizer: torch.optim.Optimizer,
    logger=None,
    device: str = "cpu",
    val_step_fn: Optional[Callable[[], float]] = None,
    model_params=None,
) -> List[float]:
    """Epoch loop for continuous-time models with subsequence batching.

    ``step_fn(u_sub, y_sub, t_sub)`` is called with random subsequences
    and must return the scalar loss (already backpropagated).

    Parameters
    ----------
    step_fn : callable(u_sub, y_sub, t_sub) -> float
        One-subsequence forward/backward pass.
    u, y : np.ndarray
        Full training signals.
    config : BaseConfig
    t : np.ndarray | None
        Time vector; generated from ``config.dt`` if absent.
    optimizer : Optimizer
    logger : WandbLogger | None
    device : str
    val_step_fn : callable() -> float | None
        If provided, called every ``log_every`` epochs to compute validation
        loss, logged as ``val_loss``.
    model_params : iterable | None
        If provided, gradient norm is computed and logged as ``grad_norm``.

    Returns
    -------
    list[float]
        Per-epoch losses.
    """
    dt = getattr(config, "dt", 0.05)
    window = getattr(config, "train_window_size", 50)
    seqs_per_epoch = getattr(config, "sequences_per_epoch", 24)
    grad_clip = getattr(config, "grad_clip", DEFAULT_GRAD_CLIP)

    N = len(u)
    if t is None:
        t = np.arange(N) * dt

    scheduler = make_scheduler(
        optimizer,
        patience=config.scheduler_patience,
        factor=config.scheduler_factor,
        min_lr=config.scheduler_min_lr,
    )
    stopper = EarlyStopper(config.early_stopping_patience)

    def epoch_step(_epoch: int) -> float:
        total = 0.0
        for _ in range(seqs_per_epoch):
            max_start = max(N - window, 1)
            start = np.random.randint(0, max_start)
            end = min(start + window, N)
            u_sub = u[start:end]
            y_sub = y[start:end]
            t_sub = t[start:end] - t[start]

            optimizer.zero_grad()
            loss_val = step_fn(u_sub, y_sub, t_sub)
            nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                grad_clip,
            )
            optimizer.step()
            total += loss_val
        return total / seqs_per_epoch

    return train_loop(
        epoch_step,
        epochs=config.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopper=stopper,
        logger=logger,
        log_every=getattr(config, "wandb_log_every", 1),
        verbose=config.verbose,
        desc="Training (continuous)",
        val_step_fn=val_step_fn,
        model_params=model_params,
    )


# ─────────────────────────────────────────────────────────────────────
# K-step shooting (blackbox ODE/SDE/CDE)
# ─────────────────────────────────────────────────────────────────────

def prepare_shooting_windows(
    u: np.ndarray,
    y: np.ndarray,
    k_steps: int,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice ``(u, y)`` into k-step windows for shooting training.

    Returns
    -------
    u_windows : Tensor [n_windows, k_steps, 1]
    y_windows : Tensor [n_windows, k_steps, state_dim]
    t_span : Tensor [k_steps]
    """
    N = len(u)
    n_windows = N - k_steps
    if n_windows <= 0:
        raise ValueError(
            f"Signal length ({N}) must exceed k_steps ({k_steps})"
        )

    u_arr = np.asarray(u, dtype=np.float32).ravel()
    y_arr = np.asarray(y, dtype=np.float32).ravel()

    u_wins = np.lib.stride_tricks.sliding_window_view(u_arr, k_steps)[:n_windows]
    y_wins = np.lib.stride_tricks.sliding_window_view(y_arr, k_steps + 1)[:n_windows]

    # y_windows includes the initial condition at [:, 0] and targets at [:, 1:]
    u_t = torch.tensor(u_wins, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y_wins, dtype=torch.float32).unsqueeze(-1)
    t_span = torch.linspace(0, k_steps * dt, k_steps + 1)

    return u_t, y_t, t_span
