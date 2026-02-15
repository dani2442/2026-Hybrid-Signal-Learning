"""Shared utilities for controlled TorchSDE-based models."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np


def inverse_softplus(x: float, minimum: float = 1e-8) -> float:
    """Map positive value to its unconstrained softplus pre-image."""
    x = max(float(x), minimum)
    if x > 30.0:
        return x
    return float(np.log(np.expm1(x)))


# ─────────────────────────────────────────────────────────────────────
# Control interpolation helpers
# ─────────────────────────────────────────────────────────────────────

def interp_u(model, t, x, *, return_du: bool = False):
    """Linear-interpolate u(t) from stored ``(t_series, u_series)``.

    Works with both single-trajectory and batch-mode (``batch_start_times``).
    When *return_du* is ``True``, also returns the time-derivative du/dt
    (used by CDE-inspired models).

    Expects *model* to have attributes:
      - ``t_series``: 1-D time grid
      - ``u_series``: control input tensor aligned with ``t_series``
      - ``batch_start_times``: ``None`` or ``[B, 1]`` tensor of offsets
    """
    import torch

    if model.batch_start_times is not None:
        t_abs = model.batch_start_times + t
    else:
        t_abs = t * torch.ones_like(x[:, 0:1])

    k_idx = torch.searchsorted(model.t_series, t_abs.reshape(-1), right=True)
    k_idx = torch.clamp(k_idx, 1, len(model.t_series) - 1)
    t1 = model.t_series[k_idx - 1].unsqueeze(1)
    t2 = model.t_series[k_idx].unsqueeze(1)
    u1, u2 = model.u_series[k_idx - 1], model.u_series[k_idx]
    denom = (t2 - t1).clone()
    denom[denom < 1e-6] = 1.0
    alpha = (t_abs - t1) / denom
    u_t = u1 + alpha * (u2 - u1)

    if return_du:
        return u_t, (u2 - u1) / denom
    return u_t


class ControlledPathMixin:
    """Mixin implementing piecewise-constant control interpolation."""

    dt: float
    _u_path: object

    def _init_control_path(self, dt: float, input_dim: int, device, dtype):
        import torch

        self.dt = float(dt)
        self._u_path = torch.zeros(2, int(input_dim), dtype=dtype, device=device)

    def set_control(self, u_path):
        if u_path.ndim == 1:
            u_path = u_path.reshape(-1, 1)
        self._u_path = u_path

    def _u_at(self, t, batch_size: int):
        import torch

        idx = torch.clamp((t / self.dt).long(), min=0, max=self._u_path.shape[0] - 1)
        u_t = self._u_path[idx]
        if u_t.ndim == 1:
            u_t = u_t.unsqueeze(0)
        return u_t.expand(batch_size, -1)


def simulate_controlled_sde(
    sde_func,
    u_path,
    x0,
    dt: float,
    method: str = "euler",
    integration_dt: float | None = None,
):
    """Simulate a controlled trajectory and return shape [T, state_dim].

    Args:
        sde_func: SDE function implementing f() and g().
        u_path: Control input path of shape [T, input_dim].
        x0: Initial state.
        dt: Sampling time (used to build the output time grid).
        method: Integration method for torchsde.
        integration_dt: Internal integration step size. When *None* (default)
            the sampling time *dt* is used. For stiff systems pass a value
            smaller than *dt* (e.g. ``dt / 10``) so that the integrator
            takes multiple sub-steps between output time points.
    """
    # When sub-stepping is requested, use a lightweight pure-PyTorch Euler
    # loop instead of torchsde.sdeint.  This avoids the Brownian tree
    # recursion limit that torchsde hits when integration_dt << dt.
    if integration_dt is not None and integration_dt < dt:
        return _euler_ode_integrate(sde_func, u_path, x0, dt, integration_dt)

    try:
        import torch
        import torchsde
    except ImportError as exc:
        raise ImportError("torchsde required. Install with: pip install torchsde") from exc

    sde_func.set_control(u_path)
    ts = torch.arange(u_path.shape[0], dtype=u_path.dtype, device=u_path.device) * float(dt)
    x0_batch = x0 if x0.ndim == 2 else x0.reshape(1, -1)
    step_dt = float(integration_dt) if integration_dt is not None else float(dt)
    path = torchsde.sdeint(sde_func, x0_batch, ts, method=method, dt=step_dt)
    return path[:, 0, :]


def _euler_ode_integrate(sde_func, u_path, x0, dt: float, integration_dt: float):
    """Pure-PyTorch Euler integration for zero-diffusion (ODE) dynamics.

    This avoids the Brownian-tree overhead in *torchsde* and supports
    arbitrary sub-stepping without recursion-depth issues.

    Args:
        sde_func: Object with ``f(t, y)`` and ``set_control(u_path)`` methods.
        u_path: Control input tensor of shape ``[T, input_dim]``.
        x0: Initial state tensor of shape ``[state_dim]`` or ``[1, state_dim]``.
        dt: Sampling time (spacing of the output grid).
        integration_dt: Internal Euler step size (must be <= *dt*).

    Returns:
        Tensor of shape ``[T, state_dim]`` with the state at each output time.
    """
    import torch

    sde_func.set_control(u_path)
    n_steps = u_path.shape[0]
    substeps = max(1, round(dt / integration_dt))
    dt_sub = dt / substeps

    x = x0.reshape(1, -1).clone()            # [1, state_dim]
    trajectory = [x.squeeze(0).clone()]       # store t=0

    for i in range(1, n_steps):
        # Within each sampling interval, the control is piecewise-constant
        # at the value for time-index i-1.  The f() lookup uses real time.
        t_start = (i - 1) * dt
        for j in range(substeps):
            t_j = t_start + j * dt_sub
            t_tensor = torch.tensor(t_j, dtype=u_path.dtype, device=u_path.device)
            dx = sde_func.f(t_tensor, x)
            x = x + dx * dt_sub
        trajectory.append(x.squeeze(0).clone())

    return torch.stack(trajectory, dim=0)     # [T, state_dim]


def optimize_with_adam(
    parameters: Iterable,
    loss_fn: Callable[[], object],
    epochs: int,
    learning_rate: float,
    on_epoch_end: Callable[[int, float, float], None] | None = None,
    max_grad_norm: float | None = 1.0,
    verbose: bool = False,
    progress_desc: str = "Optimizing",
) -> list[float]:
    """Run a generic Adam loop and return per-epoch loss history.

    Args:
        parameters: Iterable of parameters to optimise.
        loss_fn: Callable returning the scalar loss tensor.
        epochs: Number of optimisation epochs.
        learning_rate: Adam learning rate.
        on_epoch_end: Optional callback ``(epoch, loss, grad_norm) -> None``.
        max_grad_norm: If not *None*, clip the global gradient norm to this
            value every step to prevent NaN propagation in stiff systems.
        verbose: Show tqdm progress bar.
        progress_desc: Description for the progress bar.
    """
    import torch
    import torch.optim as optim

    params = list(parameters)
    history: list[float] = []
    if not params:
        return history

    optimizer = optim.Adam(params, lr=float(learning_rate))

    epoch_iter = range(int(epochs))
    if verbose:
        from tqdm.auto import tqdm
        epoch_iter = tqdm(epoch_iter, desc=progress_desc, unit="epoch")

    for epoch in epoch_iter:
        optimizer.zero_grad()
        loss = loss_fn()

        # Guard against NaN / Inf loss – skip optimiser step to avoid
        # corrupting the running Adam statistics.
        if torch.isnan(loss) or torch.isinf(loss):
            loss_value = float(loss.detach().cpu().item())
            history.append(loss_value)
            if on_epoch_end is not None:
                on_epoch_end(epoch + 1, loss_value, 0.0)
            continue

        loss.backward()

        # Discard the entire step when any gradient is NaN/Inf – this can
        # happen when back-propagating through stiff ODE/SDE trajectories
        # even when the forward loss is finite.
        if any(
            p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in params
        ):
            optimizer.zero_grad()
            loss_value = float(loss.detach().cpu().item())
            history.append(loss_value)
            if on_epoch_end is not None:
                on_epoch_end(epoch + 1, loss_value, 0.0)
            continue

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))

        grad_norm = gradient_norm(params)
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        history.append(loss_value)
        if verbose and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(loss=loss_value)
        if on_epoch_end is not None:
            on_epoch_end(epoch + 1, loss_value, grad_norm)
    return history


def gradient_norm(parameters: Iterable) -> float:
    """Compute global L2 gradient norm."""
    import torch

    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(torch.sum(param.grad.detach() ** 2).cpu().item())
    return float(np.sqrt(total))


def _prepare_subsequence_datasets(
    u,
    y,
    input_dim: int,
    state_dim: int,
    sequence_length: int,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Normalise single/multi-dataset inputs for subsequence training.

    Returns a list of tuples ``(u_ds, y_ds, n_sequences)`` where:
      - ``u_ds`` has shape ``[N, input_dim]``
      - ``y_ds`` has shape ``[N, state_dim]``
      - ``n_sequences = N - sequence_length`` is the number of valid windows
    """
    seq_len = int(sequence_length)

    def _coerce_pair(u_arr, y_arr) -> tuple[np.ndarray, np.ndarray]:
        u_np = np.asarray(u_arr, dtype=float).reshape(-1, int(input_dim))
        y_np = np.asarray(y_arr, dtype=float).reshape(-1, int(state_dim))
        if len(u_np) != len(y_np):
            raise ValueError("Each dataset pair must have u and y with equal length")
        return u_np, y_np

    datasets: list[tuple[np.ndarray, np.ndarray, int]] = []

    is_multi = isinstance(u, Sequence) and not isinstance(u, np.ndarray)
    if is_multi:
        if not (isinstance(y, Sequence) and not isinstance(y, np.ndarray)):
            raise ValueError("When u is a sequence of datasets, y must also be a sequence")
        if len(u) != len(y):
            raise ValueError("u and y dataset lists must have the same number of elements")

        for u_ds, y_ds in zip(u, y):
            u_np, y_np = _coerce_pair(u_ds, y_ds)
            n_sequences = len(y_np) - seq_len
            if n_sequences > 0:
                datasets.append((u_np, y_np, int(n_sequences)))
    else:
        u_np, y_np = _coerce_pair(u, y)
        n_sequences = len(y_np) - seq_len
        if n_sequences > 0:
            datasets.append((u_np, y_np, int(n_sequences)))

    if not datasets:
        raise ValueError("Not enough data for the requested sequence_length")
    return datasets


def train_sequence_batches(
    sde_func,
    simulate_fn: Callable,
    u: np.ndarray | Sequence[np.ndarray],
    y: np.ndarray | Sequence[np.ndarray],
    input_dim: int,
    state_dim: int,
    sequence_length: int,
    epochs: int,
    learning_rate: float,
    device,
    dtype,
    verbose: bool,
    progress_desc: str,
    logger=None,
    log_every: int = 1,
    sequences_per_epoch: int | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    state_loss_weights: Sequence[float] | None = None,
) -> list[float]:
    """Train on random subsequences and return average loss per epoch.

    Supports:
      - single dataset: ``u`` and ``y`` as numpy arrays
      - multi-dataset: ``u`` and ``y`` as equally-sized lists/tuples of arrays
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm.auto import tqdm

    datasets = _prepare_subsequence_datasets(
        u=u,
        y=y,
        input_dim=input_dim,
        state_dim=state_dim,
        sequence_length=sequence_length,
    )
    n_total_sequences = int(sum(n_seq for _, _, n_seq in datasets))
    seq_weights = np.asarray([n_seq for _, _, n_seq in datasets], dtype=float)
    seq_weights = seq_weights / np.sum(seq_weights)
    if sequences_per_epoch is None:
        sequences_per_epoch = max(1, min(100, n_total_sequences))
    else:
        sequences_per_epoch = max(1, min(int(sequences_per_epoch), n_total_sequences))

    params = list(sde_func.parameters())
    optimizer = optim.Adam(params, lr=float(learning_rate))
    criterion = nn.MSELoss()
    loss_weights_t = None
    if state_loss_weights is not None:
        loss_weights_np = np.asarray(state_loss_weights, dtype=float).reshape(-1)
        if loss_weights_np.shape[0] != int(state_dim):
            raise ValueError("state_loss_weights must have length == state_dim")
        if np.any(loss_weights_np < 0):
            raise ValueError("state_loss_weights must be non-negative")
        loss_weights_t = torch.tensor(loss_weights_np, dtype=dtype, device=device).reshape(
            1, int(state_dim)
        )
    best_loss = float("inf")
    best_state = None
    bad_epochs = 0

    epoch_iter = range(int(epochs))
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc=progress_desc, unit="epoch")

    loss_history: list[float] = []
    for epoch in epoch_iter:
        sde_func.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        for _ in range(sequences_per_epoch):
            ds_idx = int(np.random.choice(len(datasets), p=seq_weights))
            u_ds, y_ds, n_seq_ds = datasets[ds_idx]
            idx = int(np.random.randint(0, n_seq_ds))

            y_seq = torch.tensor(
                y_ds[idx : idx + sequence_length], dtype=dtype, device=device
            ).reshape(-1, state_dim)
            u_seq = torch.tensor(
                u_ds[idx : idx + sequence_length], dtype=dtype, device=device
            ).reshape(-1, input_dim)

            optimizer.zero_grad()
            pred_seq = simulate_fn(u_seq, y_seq[0])
            if loss_weights_t is None:
                loss = criterion(pred_seq, y_seq)
            else:
                sq_err = (pred_seq - y_seq) ** 2
                loss = torch.mean(sq_err * loss_weights_t)
            loss.backward()
            total_grad_norm += gradient_norm(params)
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())

        avg_loss = total_loss / sequences_per_epoch
        avg_grad_norm = total_grad_norm / sequences_per_epoch
        loss_history.append(avg_loss)

        improved = avg_loss < (best_loss - float(early_stopping_min_delta))
        if improved:
            best_loss = avg_loss
            best_state = [p.detach().clone() for p in params]
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(loss=avg_loss)
        if logger is not None and log_every > 0 and (epoch + 1) % log_every == 0:
            logger.log_metrics(
                {
                    "train/epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "train/grad_norm": avg_grad_norm,
                    "train/sequences_per_epoch": sequences_per_epoch,
                    "train/best_loss": best_loss,
                },
                step=epoch + 1,
            )

        if (
            early_stopping_patience is not None
            and int(early_stopping_patience) > 0
            and bad_epochs >= int(early_stopping_patience)
        ):
            if verbose:
                print(
                    f"Early stopping: no improvement in {int(early_stopping_patience)} epochs "
                    f"(best={best_loss:.6f})"
                )
            break

    if best_state is not None:
        for p, p_best in zip(params, best_state):
            p.data.copy_(p_best)

    return loss_history
