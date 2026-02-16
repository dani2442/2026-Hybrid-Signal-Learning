"""Canonical model registry helpers.

This module centralizes:
  * model key -> class resolution
  * model key -> config class resolution
"""

from __future__ import annotations

from typing import Any, Type

from ..config import MODEL_CONFIGS


MODEL_CLASS_NAMES: dict[str, str] = {
    "narx": "NARX",
    "arima": "ARIMA",
    "exponential_smoothing": "ExponentialSmoothing",
    "random_forest": "RandomForest",
    "neural_network": "NeuralNetwork",
    "gru": "GRU",
    "lstm": "LSTM",
    "tcn": "TCN",
    "mamba": "Mamba",
    "neural_ode": "NeuralODE",
    "neural_sde": "NeuralSDE",
    "neural_cde": "NeuralCDE",
    "linear_physics": "LinearPhysics",
    "stribeck_physics": "StribeckPhysics",
    "hybrid_linear_beam": "HybridLinearBeam",
    "hybrid_nonlinear_cam": "HybridNonlinearCam",
    "ude": "UDE",
    "vanilla_node_2d": "VanillaNODE2D",
    "structured_node": "StructuredNODE",
    "adaptive_node": "AdaptiveNODE",
    "vanilla_ncde_2d": "VanillaNCDE2D",
    "structured_ncde": "StructuredNCDE",
    "adaptive_ncde": "AdaptiveNCDE",
    "vanilla_nsde_2d": "VanillaNSDE2D",
    "structured_nsde": "StructuredNSDE",
    "adaptive_nsde": "AdaptiveNSDE",
}

MODEL_KEYS: tuple[str, ...] = tuple(MODEL_CONFIGS.keys())


def list_model_keys() -> tuple[str, ...]:
    """Return canonical model keys."""
    return MODEL_KEYS


def get_model_config_class(key: str) -> Type:
    """Resolve model key to config class."""
    key_norm = key.strip().lower()
    try:
        return MODEL_CONFIGS[key_norm]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown model key '{key}'. Available: {available}") from exc


def get_model_class(key: str) -> Type:
    """Resolve model key to model class."""
    import src.models as models_pkg

    key_norm = key.strip().lower()
    class_name = MODEL_CLASS_NAMES.get(key_norm)
    if class_name is None:
        available = ", ".join(sorted(MODEL_CLASS_NAMES))
        raise ValueError(f"Unknown model key '{key}'. Available: {available}")

    model_cls = getattr(models_pkg, class_name, None)
    if model_cls is None:
        raise ValueError(
            f"Model class '{class_name}' for key '{key_norm}' is not exported from src.models."
        )
    return model_cls


def build_model(key: str, config: Any = None, **kwargs):
    """Instantiate a model from key + config/kwargs."""
    model_cls = get_model_class(key)
    if config is not None and kwargs:
        raise ValueError("Pass either config or kwargs, not both.")
    if config is not None:
        return model_cls(config=config)
    return model_cls(**kwargs)
