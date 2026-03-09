"""Loss functions for the multimodal encoder–ODE–decoder pipeline.

All losses return plain scalar tensors so they compose naturally with
``torch.optim`` and the training loops in ``train.py``.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def mse_state(x_hat: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and true state trajectories.

    Works for:
      - single-step ``(B, 2)`` encoder outputs, or
      - sequence ``(K, B, 2)`` ODE rollouts.
    """
    return F.mse_loss(x_hat, x_true)


def mse_keypoints(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and true keypoint coordinates.

    Shapes: ``(B, 4)`` or ``(K, B, 4)`` for sequence-level comparison.
    """
    return F.mse_loss(y_hat, y_true)


def smoothness_loss(x_seq: torch.Tensor) -> torch.Tensor:
    """Penalise large jumps in the predicted state sequence.

    Parameters
    ----------
    x_seq:
        ``(K, B, D)`` predicted trajectory.

    Returns
    -------
    Scalar mean squared first-difference.
    """
    if x_seq.shape[0] < 2:
        return torch.tensor(0.0, device=x_seq.device, dtype=x_seq.dtype)
    diff = x_seq[1:] - x_seq[:-1]
    return torch.mean(diff ** 2)


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    weights: Dict[str, float] | None = None,
) -> Dict[str, torch.Tensor]:
    """Compute all applicable losses from model outputs and targets.

    Parameters
    ----------
    outputs:
        Keys produced by ``EncOdeDecModel.forward()`` and/or individual
        modules:
          - ``"x0_hat"``: encoded initial state ``(B, 2)``
          - ``"x_seq_hat"``: ODE trajectory ``(K, B, 2)``
          - ``"y_seq_hat"``: decoded observations ``(K, B, out_dim)``
    targets:
        Ground-truth tensors:
          - ``"x0"``: initial sensor state ``(B, 2)``
          - ``"x_seq"``: sensor state sequence ``(K, B, 2)``
          - ``"y_seq"``: observation sequence ``(K, B, out_dim)`` (optional)
    weights:
        Loss-weight overrides.  Recognised keys:
        ``"enc"``, ``"ode"``, ``"dec"``, ``"smooth"``.  Defaults to 1.0.

    Returns
    -------
    Dict with individual named losses plus a ``"total"`` entry.
    """
    w = {"enc": 1.0, "ode": 1.0, "dec": 1.0, "smooth": 0.0}
    if weights:
        w.update(weights)

    losses: Dict[str, torch.Tensor] = {}
    total = torch.tensor(0.0, device=_pick_device(outputs))

    # Encoder loss: predicted x0 vs. sensor x0
    if "x0_hat" in outputs and "x0" in targets:
        l = mse_state(outputs["x0_hat"], targets["x0"])
        losses["loss_enc"] = l
        total = total + w["enc"] * l

    # ODE rollout loss: predicted trajectory vs. sensor trajectory
    if "x_seq_hat" in outputs and "x_seq" in targets:
        l = mse_state(outputs["x_seq_hat"], targets["x_seq"])
        losses["loss_ode"] = l
        total = total + w["ode"] * l

    # Decoder loss: predicted observations vs. true observations
    if "y_seq_hat" in outputs and "y_seq" in targets:
        l = mse_keypoints(outputs["y_seq_hat"], targets["y_seq"])
        losses["loss_dec"] = l
        total = total + w["dec"] * l

    # Smoothness regulariser on ODE trajectory
    if "x_seq_hat" in outputs and w["smooth"] > 0:
        l = smoothness_loss(outputs["x_seq_hat"])
        losses["loss_smooth"] = l
        total = total + w["smooth"] * l

    losses["total"] = total
    return losses


def _pick_device(tensors: Dict[str, torch.Tensor]) -> torch.device:
    """Return the device of the first tensor found."""
    for v in tensors.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")
