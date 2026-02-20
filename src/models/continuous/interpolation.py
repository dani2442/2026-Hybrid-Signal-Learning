"""Control-signal interpolation for continuous-time models.

Provides a shared ``interp_u`` function and a ``ControlledPathMixin``
that continuous-time model classes can inherit.
"""

from __future__ import annotations

import numpy as np
import torch


def interp_u(
    t_eval: torch.Tensor,
    t_data: torch.Tensor,
    u_data: torch.Tensor,
) -> torch.Tensor:
    """Piece-wise linear interpolation of a control signal.

    Parameters
    ----------
    t_eval : Tensor [*batch, M]
        Times at which to evaluate.
    t_data : Tensor [T]
        Sample times of *u_data*.
    u_data : Tensor [T] or [T, D]
        Sampled control values.

    Returns
    -------
    Tensor  [*batch, M] or [*batch, M, D]
        Interpolated control values.
    """
    t_data = t_data.float()
    u_data = u_data.float()
    t_eval = t_eval.float()

    # Clamp to data range
    t_clamped = t_eval.clamp(t_data[0], t_data[-1])

    # Find interval indices
    idx = torch.searchsorted(t_data, t_clamped).clamp(1, len(t_data) - 1)
    t0 = t_data[idx - 1]
    t1 = t_data[idx]
    alpha = ((t_clamped - t0) / (t1 - t0 + 1e-12)).unsqueeze(-1) if u_data.dim() > 1 else (t_clamped - t0) / (t1 - t0 + 1e-12)

    if u_data.dim() == 1:
        return u_data[idx - 1] * (1 - alpha) + u_data[idx] * alpha
    return u_data[idx - 1] * (1 - alpha) + u_data[idx] * alpha


def make_u_func(
    u: "np.ndarray | torch.Tensor",
    t: "np.ndarray | torch.Tensor | None" = None,
    dt: float = 0.05,
    device: str = "cpu",
):
    """Build a callable ``u_func(t_eval) -> Tensor`` from data.

    Parameters
    ----------
    u : np.ndarray | Tensor
        Control signal (1-D).
    t : np.ndarray | Tensor | None
        Time vector.  Generated from *dt* when absent.
    dt : float
        Sampling interval (used if *t* is None).
    device : str

    Returns
    -------
    callable
        ``u_func(t_eval: Tensor) -> Tensor``
    """
    if isinstance(u, torch.Tensor):
        u_t = u.float().ravel().to(device)
    else:
        u_t = torch.tensor(
            np.asarray(u, dtype=np.float32).ravel(),
            dtype=torch.float32, device=device,
        )
    N = u_t.shape[0]

    if t is not None:
        if isinstance(t, torch.Tensor):
            t_t = t.float().ravel().to(device)
        else:
            t_t = torch.tensor(
                np.asarray(t, dtype=np.float32).ravel(),
                dtype=torch.float32, device=device,
            )
    else:
        t_t = torch.arange(N, dtype=torch.float32, device=device) * dt

    def _u_func(t_eval: torch.Tensor) -> torch.Tensor:
        return interp_u(t_eval, t_t, u_t)

    return _u_func
