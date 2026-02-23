from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint

from .bab_data import ExperimentData

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
    model.train()
    model.to(tensors.t.device)

    if not hasattr(model, "set_series") or not hasattr(model, "set_batch_start_times"):
        raise TypeError("Model must implement set_series() and set_batch_start_times()")

    model.set_series(tensors.t, tensors.u)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    dt_local = float((tensors.t[1] - tensors.t[0]).item())
    t_eval = torch.arange(0, cfg.k_steps * dt_local, dt_local, device=tensors.t.device)

    history: list[dict[str, float]] = []

    epoch_iter = range(cfg.epochs + 1)
    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(epoch_iter, desc=progress_desc or "train", leave=False)
        epoch_iter = pbar

    for epoch in epoch_iter:
        optimizer.zero_grad()

        start_idx = np.random.choice(valid_start_indices, size=cfg.batch_size, replace=True)
        x0 = tensors.y[start_idx]
        model.set_batch_start_times(tensors.t[start_idx].reshape(-1, 1))

        pred_state = odeint(model, x0, t_eval, method="rk4")
        pred_obs = pred_state[..., : cfg.obs_dim]

        target_list = [tensors.y[i : i + cfg.k_steps] for i in start_idx]
        y_target = torch.stack(target_list, dim=1)  # [K, B, obs_dim]

        if cfg.position_only_loss:
            pred_for_loss = pred_obs[..., 0:1]
            target_for_loss = y_target[..., 0:1]
        else:
            pred_for_loss = pred_obs
            target_for_loss = y_target

        loss = torch.mean((pred_for_loss - target_for_loss) ** 2)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        history.append({"epoch": float(epoch), "loss": loss_value})
        if pbar is not None:
            pbar.set_postfix(loss=f"{loss_value:.3e}")

    if pbar is not None:
        pbar.close()

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
    model.eval()
    model.to(device)

    t_t = torch.tensor(t, dtype=torch.float32, device=device)
    u_t = torch.tensor(u, dtype=torch.float32, device=device).reshape(-1, 1)
    x0 = torch.tensor(y0, dtype=torch.float32, device=device).reshape(1, -1)

    model.set_series(t_t, u_t)
    model.set_batch_start_times(torch.zeros(1, 1, device=device))

    with torch.no_grad():
        pred = odeint(model, x0, t_t, method="rk4").squeeze(1).detach().cpu().numpy()

    model.set_batch_start_times(None)
    return pred


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
