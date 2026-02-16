"""Shared base for discrete sequence-to-one models (GRU, LSTM, TCN, Mamba).

All four models share identical:
  * z-score normalisation
  * ``_create_sequences`` windowing
  * training loop (Adam, grad-clip, val-loss early stopping, W&B logging)
  * ``predict_osa`` / ``predict_free_run`` logic

Subclasses only need to implement ``_build_model(input_size) → nn.Module``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import BaseModel, DEFAULT_GRAD_CLIP, NORM_EPS
from .training import train_supervised_torch_model


class SequenceModel(BaseModel):
    """Base class for recurrent / convolutional sequence models."""

    def __init__(self, config):
        super().__init__(config)
        self.model_ = None
        self._device = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._u_mean: float = 0.0
        self._u_std: float = 1.0

    # ── subclass hook ─────────────────────────────────────────────────

    @abstractmethod
    def _build_model(self, input_size: int):
        """Return an ``nn.Module`` that maps ``(batch, seq, input_size) → (batch, 1)``."""
        ...

    @property
    def _model_returns_hidden(self) -> bool:
        """Override to ``True`` in RNN-style models (GRU, LSTM) whose
        forward returns ``(output, hidden)``."""
        return False

    # ── data preparation ──────────────────────────────────────────────

    def _create_sequences(self, y: np.ndarray, u: np.ndarray):
        seq_len = self.max_lag
        n_samples = len(y) - seq_len
        X = np.zeros((n_samples, seq_len, 2))
        Y = np.zeros(n_samples)
        for i in range(n_samples):
            X[i, :, 0] = y[i : i + seq_len]
            X[i, :, 1] = u[i : i + seq_len]
            Y[i] = y[i + seq_len]
        return X, Y

    # ── training ──────────────────────────────────────────────────────

    def _fit(
        self,
        u: np.ndarray,
        y: np.ndarray,
        *,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        logger: Any = None,
    ) -> None:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        cfg = self.config

        # Normalise training data
        self._y_mean, self._y_std = float(y.mean()), float(y.std())
        self._u_mean, self._u_std = float(u.mean()), float(u.std())
        y_norm = (y - self._y_mean) / (self._y_std + NORM_EPS)
        u_norm = (u - self._u_mean) / (self._u_std + NORM_EPS)

        X, Y = self._create_sequences(y_norm, u_norm)
        if len(Y) == 0:
            raise ValueError("Not enough data for given lag orders")

        self._device = self._resolve_torch_device()

        self.model_ = self._build_model(input_size=2).to(self._device)

        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self._device)
        loader = DataLoader(
            TensorDataset(X_t, Y_t),
            batch_size=cfg.batch_size,
            shuffle=True,
        )

        # Validation data
        val_loader = None
        if val_data is not None:
            u_v, y_v = val_data
            yv_norm = (y_v - self._y_mean) / (self._y_std + NORM_EPS)
            uv_norm = (u_v - self._u_mean) / (self._u_std + NORM_EPS)
            Xv, Yv = self._create_sequences(yv_norm, uv_norm)
            if len(Yv) > 0:
                Xv_t = torch.tensor(Xv, dtype=torch.float32, device=self._device)
                Yv_t = torch.tensor(Yv, dtype=torch.float32, device=self._device)
                val_loader = DataLoader(
                    TensorDataset(Xv_t, Yv_t),
                    batch_size=cfg.batch_size,
                    shuffle=False,
                )

        optimizer = optim.Adam(self.model_.parameters(), lr=cfg.learning_rate)
        criterion = nn.MSELoss()

        self.training_loss_ = list(
            train_supervised_torch_model(
                model=self.model_,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=loader,
                epochs=cfg.epochs,
                verbose=cfg.verbose,
                progress_desc=f"Training {type(self).__name__}",
                forward_fn=self._forward,
                val_loader=val_loader,
                grad_clip_norm=DEFAULT_GRAD_CLIP,
                early_stopping_patience=cfg.early_stopping_patience,
                logger=logger,
                log_every=cfg.wandb_log_every,
            )
        )

    def _forward(self, x):
        """Run forward pass, handling models that return (output, hidden)."""
        out = self.model_(x)
        if self._model_returns_hidden:
            out = out[0]
        return out

    # ── prediction ────────────────────────────────────────────────────

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        import torch

        u = np.asarray(u, dtype=float)
        y = np.asarray(y, dtype=float)
        y_norm = (y - self._y_mean) / (self._y_std + NORM_EPS)
        u_norm = (u - self._u_mean) / (self._u_std + NORM_EPS)

        X, _ = self._create_sequences(y_norm, u_norm)
        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)

        self.model_.eval()
        with torch.no_grad():
            pred = self._forward(X_t).squeeze().cpu().numpy()
        return pred * self._y_std + self._y_mean

    def predict_free_run(self, u: np.ndarray, y_initial: np.ndarray) -> np.ndarray:
        import torch

        u = np.asarray(u, dtype=float)
        y_init = np.asarray(y_initial, dtype=float)

        if len(y_init) < self.max_lag:
            raise ValueError(f"Need {self.max_lag} initial conditions, got {len(y_init)}")

        u_norm = (u - self._u_mean) / (self._u_std + NORM_EPS)
        y_init_norm = (y_init - self._y_mean) / (self._y_std + NORM_EPS)

        n_total = len(u)
        y_hat_norm = np.zeros(n_total)
        y_hat_norm[: self.max_lag] = y_init_norm[: self.max_lag]

        self.model_.eval()

        with torch.no_grad():
            for k in range(self.max_lag, n_total):
                seq_y = y_hat_norm[k - self.max_lag : k]
                seq_u = u_norm[k - self.max_lag : k]
                x = np.stack([seq_y, seq_u], axis=1)
                x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self._device)
                pred = self._forward(x_t)
                y_hat_norm[k] = pred.squeeze().cpu().item()

        y_hat = y_hat_norm * self._y_std + self._y_mean
        return y_hat[self.max_lag :]

    # ── save / load hooks ─────────────────────────────────────────────

    def _collect_state(self) -> Dict[str, Any]:
        if self.model_ is not None:
            return {"model": self.model_.state_dict()}
        return {}

    def _restore_state(self, state: Dict[str, Any]) -> None:
        if "model" in state and self.model_ is not None:
            self.model_.load_state_dict(state["model"])

    def _collect_extra_state(self) -> Dict[str, Any]:
        return {
            "y_mean": self._y_mean,
            "y_std": self._y_std,
            "u_mean": self._u_mean,
            "u_std": self._u_std,
        }

    def _restore_extra_state(self, extra: Dict[str, Any]) -> None:
        self._y_mean = extra.get("y_mean", 0.0)
        self._y_std = extra.get("y_std", 1.0)
        self._u_mean = extra.get("u_mean", 0.0)
        self._u_std = extra.get("u_std", 1.0)

    def _build_for_load(self) -> None:
        self._device = self._resolve_torch_device("cpu")
        self.model_ = self._build_model(input_size=2).to(self._device)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(config={self.config!r})"
