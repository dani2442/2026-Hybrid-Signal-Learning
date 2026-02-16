"""Abstract base class for system identification models."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..utils.runtime import resolve_device

# ─────────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────────

DEFAULT_GRAD_CLIP = 1.0
"""Gradient clipping norm used by most training loops."""

SHOOTING_GRAD_CLIP = 10.0
"""Gradient clipping norm used by multiple-shooting training loops."""

NORM_EPS = 1e-8
"""Epsilon guard for z-score normalisation denominators."""


# ─────────────────────────────────────────────────────────────────────
# Pickle save/load mixin for non-torch models
# ─────────────────────────────────────────────────────────────────────

class PickleStateMixin:
    """Mixin providing pickle-based ``_collect_extra_state`` / ``_restore_extra_state``.

    Subclasses set ``_pickle_attr`` to the name of the attribute holding
    the fitted model (e.g. ``"fitted_model_"`` or ``"model_"``), and
    ``_pickle_key`` to the checkpoint key (e.g. ``"statsmodel"``).
    """

    _pickle_attr: str
    _pickle_key: str

    def _collect_extra_state(self) -> Dict[str, Any]:
        obj = getattr(self, self._pickle_attr, None)
        if obj is not None:
            return {self._pickle_key: pickle.dumps(obj)}
        return {}

    def _restore_extra_state(self, extra: Dict[str, Any]) -> None:
        if self._pickle_key in extra:
            setattr(self, self._pickle_attr, pickle.loads(extra[self._pickle_key]))


class BaseModel(ABC):
    """Abstract base for every model in the library.

    Subclasses **must** implement ``_fit``, ``predict_osa``, ``predict_free_run``.

    They **may** override the save/load hooks:
        * ``_collect_state`` / ``_restore_state`` – torch state dicts
        * ``_collect_extra_state`` / ``_restore_extra_state`` – normalisation etc.
        * ``_build_for_load`` – rebuild network before restoring weights
    """

    config: Any  # typed per-model; always a BaseConfig subclass

    def __init__(self, config):
        from ..config import BaseConfig

        if not isinstance(config, BaseConfig):
            raise TypeError(f"config must be a BaseConfig subclass, got {type(config)}")
        self.config = config
        self.nu: int = config.nu
        self.ny: int = config.ny
        self.max_lag: int = max(config.nu, config.ny)
        self._is_fitted: bool = False
        self.training_loss_: list[float] = []

    def _resolve_torch_device(self, device: Optional[str] = None):
        """Resolve configured device string into a ``torch.device``."""
        import torch

        resolved = resolve_device(self.config.device if device is None else device)
        return torch.device(resolved)

    # ── public training entry point ───────────────────────────────────

    def fit(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "BaseModel":
        """Train model on ``(u, y)`` arrays.

        Args:
            train_data: ``(u_train, y_train)`` tuple.
            val_data:   Optional ``(u_val, y_val)`` for early stopping / metrics.

        Returns:
            ``self`` for method chaining.
        """
        from ..logging import WandbLogger

        u_train, y_train = train_data
        u_train = np.asarray(u_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        if val_data is not None:
            val_data = (
                np.asarray(val_data[0], dtype=float),
                np.asarray(val_data[1], dtype=float),
            )

        logger = WandbLogger(
            project=self.config.wandb_project,
            run_name=self.config.wandb_run_name,
            config=self.config.to_dict(),
        )

        try:
            self._fit(u_train, y_train, val_data=val_data, logger=logger)
        finally:
            logger.finish()

        self._is_fitted = True
        return self

    @abstractmethod
    def _fit(
        self,
        u: np.ndarray,
        y: np.ndarray,
        *,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        logger: Any = None,
    ) -> None:
        """Core training logic – implemented by each model family."""
        ...

    # ── prediction ────────────────────────────────────────────────────

    @abstractmethod
    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction using actual past outputs."""
        ...

    @abstractmethod
    def predict_free_run(self, u: np.ndarray, y_initial: np.ndarray) -> np.ndarray:
        """Free-run simulation from initial conditions.

        Returns an array of length ``len(u)`` with the first ``max_lag``
        entries copied from ``y_initial``.
        """
        ...

    def predict(
        self,
        u: np.ndarray,
        y: Optional[np.ndarray] = None,
        mode: str = "OSA",
    ) -> np.ndarray:
        """Unified prediction dispatch."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if mode == "OSA":
            if y is None:
                raise ValueError("y required for OSA prediction")
            return self.predict_osa(u, y)
        if mode == "FR":
            if y is None:
                raise ValueError("Initial conditions y required for free-run")
            return self.predict_free_run(u, y)
        raise ValueError(f"Unknown mode: {mode}. Use 'OSA' or 'FR'.")

    # ── save / load ───────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist model to *path* (PyTorch checkpoint format)."""
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: Dict[str, Any] = {
            "class_name": type(self).__name__,
            "module": type(self).__module__,
            "config": self.config.to_dict(),
            "config_class": type(self.config).__name__,
            "is_fitted": self._is_fitted,
            "training_loss": self.training_loss_,
        }

        state = self._collect_state()
        if state:
            checkpoint["state"] = state

        extra = self._collect_extra_state()
        if extra:
            checkpoint["extra"] = extra

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """Load model from *path*."""
        import torch

        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Resolve config
        from .. import config as cfg_module

        config_cls = getattr(cfg_module, checkpoint["config_class"])
        config = config_cls.from_dict(checkpoint["config"])

        model_cls = _resolve_model_class(checkpoint["class_name"])
        model = model_cls(config)

        if "extra" in checkpoint:
            model._restore_extra_state(checkpoint["extra"])

        model._build_for_load()

        if "state" in checkpoint:
            model._restore_state(checkpoint["state"])

        model._is_fitted = checkpoint.get("is_fitted", True)
        model.training_loss_ = checkpoint.get("training_loss", [])
        return model

    # ── hooks for subclasses ──────────────────────────────────────────

    def _collect_state(self) -> Dict[str, Any]:
        return {}

    def _restore_state(self, state: Dict[str, Any]) -> None:
        pass

    def _collect_extra_state(self) -> Dict[str, Any]:
        return {}

    def _restore_extra_state(self, extra: Dict[str, Any]) -> None:
        pass

    def _build_for_load(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────
# Convenience loader
# ─────────────────────────────────────────────────────────────────────

def load_model(path: str | Path) -> BaseModel:
    """Auto-detect class and load model from checkpoint."""
    import torch

    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_cls = _resolve_model_class(checkpoint["class_name"])
    return model_cls.load(path)


def _resolve_model_class(class_name: str) -> type:
    from .. import models as models_pkg

    cls = getattr(models_pkg, class_name, None)
    if cls is None:
        raise ValueError(
            f"Unknown model class '{class_name}'. "
            f"Ensure it is exported from src.models."
        )
    return cls
