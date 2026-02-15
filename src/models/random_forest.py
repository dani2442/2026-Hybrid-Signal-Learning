"""Random Forest model for time series forecasting."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from ..config import RandomForestConfig
from ..utils.regression import create_lagged_features
from .base import BaseModel, PickleStateMixin


class RandomForest(PickleStateMixin, BaseModel):
    """Random Forest Regressor with lagged features for system identification."""

    _pickle_attr = "model_"
    _pickle_key = "sklearn_model"

    def __init__(self, config: RandomForestConfig | None = None, **kwargs):
        if config is None:
            config = RandomForestConfig(**kwargs)
        super().__init__(config)
        self.model_ = None

    # ── training ──────────────────────────────────────────────────────

    def _fit(
        self,
        u: np.ndarray,
        y: np.ndarray,
        *,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        logger: Any = None,
    ) -> None:
        from sklearn.ensemble import RandomForestRegressor

        cfg = self.config
        features, target = create_lagged_features(y, u, self.ny, self.nu)
        if len(target) == 0:
            raise ValueError("Not enough data for given lag orders")

        self.model_ = RandomForestRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_split=cfg.min_samples_split,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.random_state,
            n_jobs=-1,
            verbose=1 if cfg.verbose else 0,
        )
        self.model_.fit(features, target)

        # Log to wandb if available
        if logger and logger.active:
            if val_data is not None:
                vf, vt = create_lagged_features(val_data[1], val_data[0], self.ny, self.nu)
                if len(vt) > 0:
                    val_pred = self.model_.predict(vf)
                    val_mse = float(np.mean((val_pred - vt) ** 2))
                    logger.log_metrics({"val/mse": val_mse})

    # ── prediction ────────────────────────────────────────────────────

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        features, _ = create_lagged_features(y, u, self.ny, self.nu)
        return self.model_.predict(features)

    def predict_free_run(self, u: np.ndarray, y_initial: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        y_init = np.asarray(y_initial, dtype=float)
        if len(y_init) < self.max_lag:
            raise ValueError(f"Need {self.max_lag} initial conditions, got {len(y_init)}")

        n_total = len(u)
        y_hat = np.zeros(n_total)
        y_hat[: self.max_lag] = y_init[: self.max_lag]

        for k in range(self.max_lag, n_total):
            y_lags = [y_hat[k - j] for j in range(1, self.ny + 1)] if self.ny > 0 else []
            u_lags = [u[k - j] for j in range(1, self.nu + 1)] if self.nu > 0 else []
            features = np.array(y_lags + u_lags).reshape(1, -1)
            y_hat[k] = self.model_.predict(features)[0]

        return y_hat[self.max_lag:]

