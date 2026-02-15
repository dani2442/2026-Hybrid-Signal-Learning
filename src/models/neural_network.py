"""Feedforward Neural Network model for NARX-like system identification."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from ..config import NeuralNetworkConfig
from ..utils.regression import create_lagged_features
from .base import BaseModel, resolve_device


class NeuralNetwork(BaseModel):
    """Feedforward NN with lagged-feature inputs.

    Uses ``create_lagged_features`` to build a flat regression matrix
    rather than sequences, then trains a simple MLP.
    """

    def __init__(self, config: NeuralNetworkConfig | None = None, **kwargs):
        if config is None:
            config = NeuralNetworkConfig(**kwargs)
        super().__init__(config)
        self.model_ = None
        self._device = None

    # ── architecture ──────────────────────────────────────────────────

    def _build_model(self, n_inputs: int):
        import torch.nn as nn

        cfg = self.config
        act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU, "leaky_relu": nn.LeakyReLU}
        act_fn = act_map.get(cfg.activation, nn.SELU)

        layers = []
        prev = n_inputs
        for h in cfg.hidden_layers:
            layers += [nn.Linear(prev, h), act_fn()]
            prev = h
        layers.append(nn.Linear(prev, 1))

        model = nn.Sequential(*layers)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return model

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
        features, target = create_lagged_features(y, u, self.ny, self.nu)
        if len(target) == 0:
            raise ValueError("Not enough data for given lag orders")

        self._device = resolve_device(cfg.device)
        self.model_ = self._build_model(features.shape[1]).to(self._device)

        X = torch.tensor(features, dtype=torch.float32, device=self._device)
        Y = torch.tensor(target, dtype=torch.float32, device=self._device)
        loader = DataLoader(TensorDataset(X, Y), batch_size=cfg.batch_size, shuffle=True)

        # Validation
        val_loader = None
        if val_data is not None:
            vf, vt = create_lagged_features(val_data[1], val_data[0], self.ny, self.nu)
            if len(vt) > 0:
                Xv = torch.tensor(vf, dtype=torch.float32, device=self._device)
                Yv = torch.tensor(vt, dtype=torch.float32, device=self._device)
                val_loader = DataLoader(TensorDataset(Xv, Yv), batch_size=cfg.batch_size, shuffle=False)

        optimizer = optim.NAdam(self.model_.parameters(), lr=cfg.learning_rate)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        patience = cfg.early_stopping_patience or float("inf")

        epoch_iter = range(cfg.epochs)
        if cfg.verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training NeuralNetwork", unit="epoch")

        for epoch in epoch_iter:
            self.model_.train()
            epoch_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(bx).squeeze(), by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_train = epoch_loss / len(loader)
            self.training_loss_.append(avg_train)

            avg_val = None
            if val_loader is not None:
                self.model_.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        val_loss += criterion(self.model_(bx).squeeze(), by).item()
                avg_val = val_loss / len(val_loader)

            monitor = avg_val if avg_val is not None else avg_train
            if monitor < best_val_loss:
                best_val_loss = monitor
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                if cfg.verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            postfix = {"loss": avg_train}
            if avg_val is not None:
                postfix["val"] = avg_val
            if cfg.verbose and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(postfix)
            if logger and logger.active and (epoch + 1) % cfg.wandb_log_every == 0:
                metrics = {"train/loss": avg_train, "train/epoch": epoch + 1}
                if avg_val is not None:
                    metrics["val/loss"] = avg_val
                logger.log_metrics(metrics, step=epoch + 1)

        if best_state is not None:
            self.model_.load_state_dict(best_state)

    # ── prediction ────────────────────────────────────────────────────

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        import torch

        features, _ = create_lagged_features(y, u, self.ny, self.nu)
        X = torch.tensor(features, dtype=torch.float32, device=self._device)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(X).squeeze().cpu().numpy()

    def predict_free_run(self, u: np.ndarray, y_initial: np.ndarray) -> np.ndarray:
        import torch

        u = np.asarray(u, dtype=float)
        y_init = np.asarray(y_initial, dtype=float)
        if len(y_init) < self.max_lag:
            raise ValueError(f"Need {self.max_lag} initial conditions")

        n_total = len(u)
        y_hat = np.zeros(n_total)
        y_hat[: self.max_lag] = y_init[: self.max_lag]

        self.model_.eval()
        with torch.no_grad():
            for k in range(self.max_lag, n_total):
                feats = []
                for j in range(1, self.ny + 1):
                    feats.append(y_hat[k - j])
                for j in range(1, self.nu + 1):
                    feats.append(u[k - j])
                X = torch.tensor([feats], dtype=torch.float32, device=self._device)
                y_hat[k] = self.model_(X).squeeze().cpu().item()

        return y_hat[self.max_lag:]

    # ── save / load hooks ─────────────────────────────────────────────

    def _collect_state(self) -> Dict[str, Any]:
        if self.model_ is not None:
            return {"model": self.model_.state_dict()}
        return {}

    def _restore_state(self, state: Dict[str, Any]) -> None:
        if "model" in state and self.model_ is not None:
            self.model_.load_state_dict(state["model"])

    def _build_for_load(self) -> None:
        import torch
        self._device = torch.device("cpu")
        n_inputs = self.config.ny + self.config.nu
        self.model_ = self._build_model(n_inputs).to(self._device)
