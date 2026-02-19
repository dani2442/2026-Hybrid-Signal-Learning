"""Base model class for all system identification models.

Provides multi-dataset support, normalisation helpers, serialisation
and a uniform ``fit`` / ``predict`` / ``save`` / ``load`` interface.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.config import BaseConfig
from src.models.constants import NORM_EPS
from src.utils.runtime import resolve_device


# ─────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────

DataPair = Tuple[np.ndarray, np.ndarray]  # (u, y)
DataInput = Union[DataPair, List[DataPair]]


# ─────────────────────────────────────────────────────────────────────
# Mixin for safe pickling of PyTorch models
# ─────────────────────────────────────────────────────────────────────

class PickleStateMixin:
    """Moves CUDA tensors to CPU before pickling."""

    def __getstate__(self) -> Dict[str, Any]:
        import torch
        state = self.__dict__.copy()
        for key, val in state.items():
            if isinstance(val, torch.Tensor):
                state[key] = val.detach().cpu()
            elif isinstance(val, torch.nn.Module):
                val = val.cpu()
                state[key] = val
                # Clear unpicklable closure attributes (e.g. _u_func set via
                # set_u_func()) from this module and all its submodules.
                for mod in val.modules():
                    if hasattr(mod, "_u_func"):
                        mod._u_func = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)


# ─────────────────────────────────────────────────────────────────────
# Base model
# ─────────────────────────────────────────────────────────────────────

class BaseModel:
    """Abstract base for every system identification model.

    Parameters
    ----------
    config : BaseConfig
        Model configuration dataclass.

    Subclass contract
    -----------------
    Implement ``_fit(u, y, *, val_data, logger)`` and
    ``_predict(u, *, y0)``.  Everything else (normalisation,
    multi-dataset routing, serialisation) is handled here.
    """

    name: str = "base"

    def __init__(self, config: BaseConfig | None = None) -> None:
        self.config = config or BaseConfig()
        self.device = resolve_device(getattr(self.config, "device", "auto"))
        self.is_trained: bool = False

        # Normalisation statistics (populated during fit)
        self._u_mean: float = 0.0
        self._u_std: float = 1.0
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

        # Multi-dataset storage (populated during fit)
        self._train_datasets: List[DataPair] = []
        self._val_datasets: Optional[List[DataPair]] = None

    # ── public API ────────────────────────────────────────────────────

    def fit(
        self,
        train_data: DataInput,
        *,
        val_data: DataInput | None = None,
        logger=None,
    ) -> None:
        """Train the model.

        Parameters
        ----------
        train_data
            Either a single ``(u, y)`` pair or a list of pairs
            for multi-dataset training.
        val_data
            Optional validation data (same format).
        logger
            :class:`~src.wandb_logger.WandbLogger` instance or *None*.
        """
        train_list = self._to_dataset_list(train_data)
        val_list = self._to_dataset_list(val_data) if val_data is not None else None

        # Store for models that need per-dataset access
        self._train_datasets = train_list
        self._val_datasets = val_list

        # Concatenate for the default _fit path
        u_cat = np.concatenate([d[0] for d in train_list])
        y_cat = np.concatenate([d[1] for d in train_list])

        val_cat: Optional[DataPair] = None
        if val_list is not None:
            val_cat = (
                np.concatenate([d[0] for d in val_list]),
                np.concatenate([d[1] for d in val_list]),
            )

        self._compute_normalization(u_cat, y_cat)
        self._fit(u_cat, y_cat, val_data=val_cat, logger=logger)
        self.is_trained = True

    @property
    def max_lag(self) -> int:
        """Number of initial samples that are not real predictions.

        Override in subclasses that use lagged features (NARX, RF, NN, …).
        """
        return 0

    def predict(
        self,
        u: np.ndarray,
        y: np.ndarray | None = None,
        *,
        y0: np.ndarray | None = None,
        mode: str = "FR",
    ) -> np.ndarray:
        """Generate predictions.

        Parameters
        ----------
        u : np.ndarray
            Input signal.
        y : np.ndarray | None
            True output — required for OSA mode, also used to seed
            ``y0`` when *y0* is not given.
        y0 : np.ndarray | None
            Initial output conditions (model-dependent).
        mode : str
            ``"FR"`` (free-run) or ``"OSA"`` (one-step-ahead).
            OSA feeds the *true* past outputs at each step.

        Returns
        -------
        np.ndarray
            Predicted output, same length as *u*.
        """
        if not self.is_trained:
            raise RuntimeError(f"{self.name}: model must be trained before prediction")

        u = np.asarray(u, dtype=np.float64).ravel()
        if y is not None:
            y = np.asarray(y, dtype=np.float64).ravel()

        mode = mode.upper()
        if mode not in ("FR", "OSA"):
            raise ValueError(f"mode must be 'FR' or 'OSA', got {mode!r}")

        # Auto-seed y0 from y when not explicitly given
        if y0 is None and y is not None:
            lag = self.max_lag
            if lag > 0:
                y0 = y[:lag]

        if mode == "OSA":
            return self._predict_osa(u, y=y, y0=y0)
        return self._predict(u, y0=y0)

    # ── abstract hooks ────────────────────────────────────────────────

    def _predict_osa(
        self,
        u: np.ndarray,
        *,
        y: np.ndarray | None = None,
        y0: np.ndarray | None = None,
    ) -> np.ndarray:
        """One-step-ahead prediction (override for AR-like models).

        Default implementation falls back to free-run.
        """
        return self._predict(u, y0=y0)

    def _fit(
        self,
        u: np.ndarray,
        y: np.ndarray,
        *,
        val_data: DataPair | None = None,
        logger=None,
    ) -> None:
        raise NotImplementedError

    def _predict(self, u: np.ndarray, *, y0: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    # ── normalisation helpers ─────────────────────────────────────────

    def _compute_normalization(self, u: np.ndarray, y: np.ndarray) -> None:
        self._u_mean = float(np.mean(u))
        self._u_std = float(np.std(u) + NORM_EPS)
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y) + NORM_EPS)

    def _normalize_u(self, u: np.ndarray) -> np.ndarray:
        return (u - self._u_mean) / self._u_std

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self._y_mean) / self._y_std

    def _denormalize_y(self, y: np.ndarray) -> np.ndarray:
        return y * self._y_std + self._y_mean

    # ── serialisation ─────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Pickle the model to *path*."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """Load a pickled model from *path*."""
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, BaseModel):
            raise TypeError(f"Loaded object is {type(model)}, not a BaseModel")
        return model

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _to_dataset_list(data: DataInput) -> List[DataPair]:
        """Normalise ``train_data`` / ``val_data`` to ``list[(u, y)]``."""
        if isinstance(data, tuple) and len(data) == 2:
            u, y = data
            if isinstance(u, np.ndarray):
                return [(np.asarray(u, dtype=np.float64).ravel(),
                         np.asarray(y, dtype=np.float64).ravel())]
        if isinstance(data, (list, tuple)):
            out = []
            for d in data:
                if not (isinstance(d, (list, tuple)) and len(d) == 2):
                    raise TypeError(f"Expected (u, y) pair, got {type(d)}")
                out.append((
                    np.asarray(d[0], dtype=np.float64).ravel(),
                    np.asarray(d[1], dtype=np.float64).ravel(),
                ))
            return out
        raise TypeError(f"Expected (u, y) or list[(u, y)], got {type(data)}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, trained={self.is_trained})"


# ─────────────────────────────────────────────────────────────────────
# Convenience loader
# ─────────────────────────────────────────────────────────────────────

def load_model(path: str) -> BaseModel:
    """Load any pickled model from disk."""
    return BaseModel.load(path)
