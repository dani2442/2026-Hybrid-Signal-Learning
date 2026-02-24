from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint

from .data import ExperimentData

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tqdm = None


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 1000
    lr: float = 0.01
    batch_size: int = 128
    k_steps: int = 20
    obs_dim: int = 2
    position_only_loss: bool = True


@dataclass(slots=True)
class TensorBundle:
    t: torch.Tensor
    u: torch.Tensor
    y: torch.Tensor


def to_tensor_bundle(
    *,
    t: np.ndarray,
    u: np.ndarray,
    y_sim: np.ndarray,
    device: torch.device,
) -> TensorBundle:
    return TensorBundle(
        t=torch.tensor(t, dtype=torch.float32, device=device),
        u=torch.tensor(u, dtype=torch.float32, device=device).reshape(-1, 1),
        y=torch.tensor(y_sim, dtype=torch.float32, device=device),
    )


def train_model(
    *,
    model: nn.Module,
    tensors: TensorBundle,
    valid_start_indices: np.ndarray,
    cfg: TrainConfig,
    show_progress: bool = False,
    progress_desc: str = "",
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Unified training loop for all model types.

    Each model must implement ``predict_k_steps(tensors, start_idx,
    k_steps, obs_dim) -> Tensor[K, B, obs_dim]``.  The training loop
    samples random windows, delegates prediction to the model, and
    computes MSE against the ground truth.
    """
    model.train()
    model.to(tensors.t.device)

    if hasattr(model, "set_series"):
        model.set_series(tensors.t, tensors.u)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: list[dict[str, float]] = []

    epoch_iter = range(cfg.epochs + 1)
    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(epoch_iter, desc=progress_desc or "train", leave=False)
        epoch_iter = pbar

    for epoch in epoch_iter:
        optimizer.zero_grad()

        start_idx = np.random.choice(valid_start_indices, size=cfg.batch_size, replace=True)

        # Model-specific k-step prediction
        pred_obs = model.predict_k_steps(tensors, start_idx, cfg.k_steps, cfg.obs_dim)

        # Ground-truth target windows  [K, B, obs_dim]
        y_target = torch.stack(
            [tensors.y[i : i + cfg.k_steps] for i in start_idx], dim=1
        )

        if cfg.position_only_loss:
            loss = torch.mean((pred_obs[..., 0:1] - y_target[..., 0:1]) ** 2)
        else:
            loss = torch.mean((pred_obs - y_target) ** 2)

        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        history.append({"epoch": float(epoch), "loss": loss_value})
        if pbar is not None:
            pbar.set_postfix(loss=f"{loss_value:.3e}")

    if pbar is not None:
        pbar.close()

    if hasattr(model, "set_batch_start_times"):
        model.set_batch_start_times(None)
    model.eval()
    return model, history


def simulate_full_rollout(
    *,
    model: nn.Module,
    t: np.ndarray,
    u: np.ndarray,
    y0: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Simulate a full rollout, dispatching to the correct backend.

    Handles ODE (torchdiffeq), SDE (torchsde),
    sequence, and feedforward models transparently.
    """
    # ── sequence models ───────────────────────────────────────────────
    if getattr(model, "is_sequence_model", False):
        return _simulate_sequence_rollout(model=model, u=u, device=device)

    # ── feedforward (autoregressive) ──────────────────────────────────
    if getattr(model, "is_feedforward_model", False):
        return model.predict_ar(u=u, y0=y0.reshape(-1, 2) if y0.ndim == 2 else np.tile(y0, (model.lag, 1)), device=device)

    # ── SDE models ────────────────────────────────────────────────────
    if hasattr(model, "sde_type"):
        return _simulate_sde_rollout(model=model, t=t, u=u, y0=y0, device=device)

    # ── standard ODE models (default) ─────────────────────────────────
    return _simulate_ode_rollout(model=model, t=t, u=u, y0=y0, device=device)


def _simulate_ode_rollout(
    *,
    model: nn.Module,
    t: np.ndarray,
    u: np.ndarray,
    y0: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Standard ODE rollout via torchdiffeq.odeint."""
    model.eval()
    model.to(device)

    t_t = torch.tensor(t, dtype=torch.float32, device=device)
    u_t = torch.tensor(u, dtype=torch.float32, device=device).reshape(-1, 1)
    x0 = torch.tensor(y0, dtype=torch.float32, device=device).reshape(1, -1)
    if getattr(model, "augmented_state", False):
        x0 = model.prepare_x0(x0)

    model.set_series(t_t, u_t)
    model.set_batch_start_times(torch.zeros(1, 1, device=device))

    with torch.no_grad():
        pred = odeint(model, x0, t_t, method="rk4").squeeze(1).detach().cpu().numpy()

    model.set_batch_start_times(None)
    obs_dim = len(y0) if y0.ndim == 1 else y0.shape[-1]
    return pred[:, :obs_dim]


def _simulate_sequence_rollout(
    *,
    model: nn.Module,
    u: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Feed full u sequence through a sequence model → [N, 2]."""
    model.eval()
    model.to(device)

    u_t = torch.tensor(u, dtype=torch.float32, device=device).reshape(1, -1, 1)
    with torch.no_grad():
        pred = model(u_t)
    return pred.squeeze(0).detach().cpu().numpy()


def _simulate_sde_rollout(
    *,
    model: nn.Module,
    t: np.ndarray,
    u: np.ndarray,
    y0: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """SDE rollout via torchsde.sdeint."""
    t_t = torch.tensor(t, dtype=torch.float32, device=device)
    u_t = torch.tensor(u, dtype=torch.float32, device=device).reshape(-1, 1)
    x0 = torch.tensor(y0, dtype=torch.float32, device=device).reshape(1, -1)

    pred = model.rollout(t_t, u_t, x0, device)
    obs_dim = len(y0) if y0.ndim == 1 else y0.shape[-1]
    return pred[:, :obs_dim].detach().cpu().numpy()


def compute_split_metrics(y_true: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    res_pos = y_true[:, 0] - y_hat[:, 0]
    res_vel = y_true[:, 1] - y_hat[:, 1]

    rmse_pos = float(np.sqrt(np.mean(res_pos**2)))
    rmse_vel = float(np.sqrt(np.mean(res_vel**2)))

    ss_res_pos = float(np.sum(res_pos**2))
    ss_tot_pos = float(np.sum((y_true[:, 0] - np.mean(y_true[:, 0])) ** 2))
    ss_res_vel = float(np.sum(res_vel**2))
    ss_tot_vel = float(np.sum((y_true[:, 1] - np.mean(y_true[:, 1])) ** 2))

    r2_pos = float(1.0 - ss_res_pos / ss_tot_pos) if ss_tot_pos > 0 else float("nan")
    r2_vel = float(1.0 - ss_res_vel / ss_tot_vel) if ss_tot_vel > 0 else float("nan")

    fit_pos = float(100.0 * (1.0 - np.linalg.norm(res_pos) / np.linalg.norm(y_true[:, 0] - np.mean(y_true[:, 0]))))
    fit_vel = float(100.0 * (1.0 - np.linalg.norm(res_vel) / np.linalg.norm(y_true[:, 1] - np.mean(y_true[:, 1]))))

    return {
        "rmse_pos": rmse_pos,
        "rmse_vel": rmse_vel,
        "r2_pos": r2_pos,
        "r2_vel": r2_vel,
        "fit_pos": fit_pos,
        "fit_vel": fit_vel,
    }


def evaluate_model_on_dataset(
    *,
    model: nn.Module,
    ds: ExperimentData,
    device: torch.device,
) -> dict[str, Any]:
    y_pred = simulate_full_rollout(
        model=model,
        t=ds.t,
        u=ds.u,
        y0=ds.y_sim[0],
        device=device,
    )

    metrics = {"train": None, "test": None}
    residuals = {"train": None, "test": None}

    for split_name, idx in (("train", ds.train_idx), ("test", ds.test_idx)):
        if idx.size < 2:
            continue
        y_true_split = ds.y_sim[idx]
        y_pred_split = y_pred[idx]

        metrics[split_name] = compute_split_metrics(y_true_split, y_pred_split)
        residuals[split_name] = {
            "pos": y_true_split[:, 0] - y_pred_split[:, 0],
            "vel": y_true_split[:, 1] - y_pred_split[:, 1],
        }

    return {
        "y_pred": y_pred,
        "metrics": metrics,
        "residuals": residuals,
    }


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
