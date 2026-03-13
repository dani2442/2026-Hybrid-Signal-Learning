"""Feedforward neural network with lagged I/O features.

Adapted from models/feedforward/neural_network.py.  Operates in
discrete time: at each step *k* the model receives a feature vector
``[y(k-1), ẏ(k-1), ..., y(k-lag), ẏ(k-lag), u(k-1), ..., u(k-lag)]``
and predicts ``[y(k), ẏ(k)]``.

For free-run (multi-step-ahead) simulation the predicted outputs are
fed back as lagged features, producing a fully autoregressive rollout.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .base import NN_VARIANTS, NnVariantConfig, _build_selu_mlp


class FeedForwardNN(nn.Module):
    """MLP with lagged I/O features for 2-D state prediction.

    Parameters
    ----------
    variant_name : str
        NN variant key (hidden_dim, depth, dropout).
    lag : int
        Number of past time-steps used as features.  The feature vector
        at step *k* has dimension ``lag * (state_dim + input_dim)``
        = ``lag * 3`` (for state_dim=2, input_dim=1).
    """

    # Marker for dispatch in training / simulation helpers.
    is_feedforward_model: bool = True

    def __init__(self, variant_name: str = "base", *, lag: int = 10) -> None:
        super().__init__()
        if variant_name not in NN_VARIANTS:
            raise ValueError(f"Unknown variant '{variant_name}'")
        self.variant_name = variant_name
        self.lag = lag

        state_dim = 2
        input_dim = 1
        feat_dim = lag * (state_dim + input_dim)
        self.net = _build_selu_mlp(input_dim=feat_dim, output_dim=state_dim, variant=NN_VARIANTS[variant_name])

    # API compat stubs
    def set_series(self, t_series, u_series):  # noqa: ARG002
        pass

    def set_batch_start_times(self, batch_start_times):  # noqa: ARG002
        pass

    def predict_k_steps(self, tensors, start_idx, k_steps: int, obs_dim: int) -> torch.Tensor:
        """Predict k-step trajectories using teacher-forced lagged features.

        Returns
        -------
        Tensor of shape ``[K, B, obs_dim]``
        """
        lag = self.lag
        device = tensors.t.device
        n_total = tensors.y.shape[0]

        lag_offsets = np.arange(1, lag + 1)  # [lag]
        preds = []
        for k in range(k_steps):
            indices = start_idx + k  # [B]
            lag_indices = indices[:, None] - lag_offsets[None, :]  # [B, lag]
            lag_indices = np.clip(lag_indices, 0, n_total - 1)

            lag_idx_t = torch.tensor(lag_indices, device=device, dtype=torch.long)
            y_lagged = tensors.y[lag_idx_t]  # [B, lag, 2]
            u_lagged = tensors.u[lag_idx_t]  # [B, lag, 1]

            # Interleave [pos, vel, u] per lag step → [B, lag*3]
            feat = torch.cat([y_lagged, u_lagged], dim=2).reshape(len(start_idx), -1)
            preds.append(self.net(feat))

        return torch.stack(preds, dim=0)[..., :obs_dim]  # [K, B, obs_dim]

    # ── feature construction ──────────────────────────────────────────

    @staticmethod
    def build_features(
        y_sim: np.ndarray | torch.Tensor,
        u: np.ndarray | torch.Tensor,
        lag: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build lagged feature matrix from (y_sim, u).

        Parameters
        ----------
        y_sim : array [N, 2]  (position, velocity)
        u     : array [N] or [N, 1]
        lag   : int

        Returns
        -------
        X : array [N - lag, lag * 3]
        Y : array [N - lag, 2]
        """
        if isinstance(y_sim, torch.Tensor):
            y_sim = y_sim.detach().cpu().numpy()
        if isinstance(u, torch.Tensor):
            u = u.detach().cpu().numpy()
        y_sim = np.asarray(y_sim, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64).ravel()

        N = len(u)
        rows_x: list[np.ndarray] = []
        rows_y: list[np.ndarray] = []

        for k in range(lag, N):
            feat: list[float] = []
            for j in range(1, lag + 1):
                feat.append(float(y_sim[k - j, 0]))
                feat.append(float(y_sim[k - j, 1]))
                feat.append(float(u[k - j]))
            rows_x.append(np.array(feat, dtype=np.float64))
            rows_y.append(y_sim[k])

        return np.array(rows_x), np.array(rows_y)

    # ── autoregressive rollout ────────────────────────────────────────

    def predict_ar(
        self,
        u: np.ndarray,
        y0: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        """Autoregressive free-run prediction.

        Parameters
        ----------
        u  : array [N] or [N, 1]
        y0 : array [lag, 2]  initial conditions (position, velocity)

        Returns
        -------
        y_pred : array [N, 2]
        """
        self.eval()
        self.to(device)

        u = np.asarray(u, dtype=np.float64).ravel()
        y0 = np.asarray(y0, dtype=np.float64)
        N = len(u)
        lag = self.lag

        y_pred = np.zeros((N, 2), dtype=np.float64)
        y_pred[:lag] = y0[:lag]

        with torch.no_grad():
            for k in range(lag, N):
                feat: list[float] = []
                for j in range(1, lag + 1):
                    feat.append(float(y_pred[k - j, 0]))
                    feat.append(float(y_pred[k - j, 1]))
                    feat.append(float(u[k - j]))

                x_t = torch.tensor([feat], dtype=torch.float32, device=device)
                pred = self.net(x_t).squeeze(0).cpu().numpy()
                y_pred[k] = pred

        return y_pred
