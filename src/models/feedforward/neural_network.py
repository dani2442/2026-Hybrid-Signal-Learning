"""Feed-forward neural network for system identification."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.config import NeuralNetworkConfig
from src.models.base import BaseModel, PickleStateMixin
from src.models.predict_utils import autoregressive_free_run, one_step_ahead
from src.models.registry import register_model
from src.models.training import train_supervised_torch_model
from src.utils.regression import create_lagged_features


# ─────────────────────────────────────────────────────────────────────
# Network definition
# ─────────────────────────────────────────────────────────────────────

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
}


class _MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        activation: str = "selu",
    ) -> None:
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation, nn.SELU)
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────

@register_model("neural_network", NeuralNetworkConfig)
class NeuralNetworkModel(PickleStateMixin, BaseModel):
    """Feed-forward MLP trained on lagged features."""

    def __init__(self, config: NeuralNetworkConfig | None = None) -> None:
        super().__init__(config or NeuralNetworkConfig())
        self.config: NeuralNetworkConfig
        self.network: _MLP | None = None

    def _fit(self, u, y, *, val_data=None, logger=None) -> None:
        cfg = self.config
        if cfg.normalize:
            u_fit = self._normalize_u(u)
            y_fit = self._normalize_y(y)
        else:
            u_fit, y_fit = u, y

        X, y_target = create_lagged_features(y_fit, u_fit, cfg.ny, cfg.nu)
        if X.shape[0] == 0:
            raise ValueError("Not enough data for the given lag orders")

        input_dim = X.shape[1]
        self.network = _MLP(input_dim, cfg.hidden_layers, cfg.activation)

        # Build validation feature matrices using training normalisation
        val_feat = None
        if val_data is not None:
            u_v, y_v = val_data
            if cfg.normalize:
                u_v = self._normalize_u(np.asarray(u_v, dtype=np.float64).ravel())
                y_v = self._normalize_y(np.asarray(y_v, dtype=np.float64).ravel())
            X_val, y_val_t = create_lagged_features(y_v, u_v, cfg.ny, cfg.nu)
            if X_val.shape[0] > 0:
                val_feat = (X_val, y_val_t)

        train_supervised_torch_model(
            self.network, X, y_target,
            config=cfg, logger=logger, device=self.device,
            val_data=val_feat,
        )

    @property
    def max_lag(self) -> int:
        return max(self.config.ny, self.config.nu)

    def _make_predict_fn(self):
        """Return a single-step prediction callable."""
        self.network.eval()
        self.network.to(self.device)
        device = self.device
        net = self.network

        def _predict_one(features: np.ndarray) -> float:
            x_t = torch.tensor(
                features.reshape(1, -1), dtype=torch.float32, device=device
            )
            with torch.no_grad():
                return float(net(x_t).item())

        return _predict_one

    def _predict(self, u, *, y0=None) -> np.ndarray:
        cfg = self.config
        fn = self._make_predict_fn()

        if cfg.normalize:
            u_norm = self._normalize_u(u)
            y0_norm = self._normalize_y(y0) if y0 is not None else None
            y_pred_norm = autoregressive_free_run(
                fn, u_norm, cfg.ny, cfg.nu, y0=y0_norm
            )
            return self._denormalize_y(y_pred_norm)

        return autoregressive_free_run(fn, u, cfg.ny, cfg.nu, y0=y0)

    def _predict_osa(self, u, *, y=None, y0=None) -> np.ndarray:
        cfg = self.config
        if y is None:
            return self._predict(u, y0=y0)
        fn = self._make_predict_fn()

        if cfg.normalize:
            u_norm = self._normalize_u(u)
            y_norm = self._normalize_y(y)
            y_pred_norm = one_step_ahead(
                fn, u_norm, y_norm, cfg.ny, cfg.nu
            )
            return self._denormalize_y(y_pred_norm)

        return one_step_ahead(fn, u, y, cfg.ny, cfg.nu)
