"""Model registry with decorator-based auto-registration.

Every model class decorates itself with ``@register_model`` which
populates the global registry.  No manual syncing of multiple files
is required.

Usage
-----
::

    from src.config import NARXConfig
    from src.models.registry import register_model

    @register_model("narx", NARXConfig)
    class NARXModel(BaseModel):
        ...

Then later::

    from src.models.registry import build_model, list_models

    model = build_model("narx", nu=5, ny=5)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Type

# Registry: name → (model_class, config_class)
_REGISTRY: Dict[str, Tuple[Type, Type]] = {}


def register_model(name: str, config_cls: Type):
    """Class decorator that registers a model under *name*.

    Parameters
    ----------
    name : str
        Unique model key (e.g. ``"narx"``, ``"gru"``).
    config_cls : type
        Corresponding config dataclass.
    """
    def decorator(cls):
        if name in _REGISTRY:
            existing = _REGISTRY[name][0].__name__
            raise ValueError(
                f"Duplicate model registration: '{name}' already maps to "
                f"{existing}, cannot register {cls.__name__}"
            )
        _REGISTRY[name] = (cls, config_cls)
        cls.name = name
        cls._config_cls = config_cls
        return cls
    return decorator


def get_model_class(name: str) -> Type:
    """Return the model class registered under *name*."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[name][0]


def get_config_class(name: str) -> Type:
    """Return the config class for model *name*."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[name][1]


def build_model(name: str, **config_kwargs: Any):
    """Instantiate a registered model with the given config overrides.

    Parameters
    ----------
    name : str
        Registered model key.
    **config_kwargs
        Passed to the model's config constructor.

    Returns
    -------
    BaseModel
        Fully constructed (untrained) model instance.
    """
    model_cls = get_model_class(name)
    cfg_cls = get_config_class(name)
    cfg = cfg_cls(**{k: v for k, v in config_kwargs.items()
                     if hasattr(cfg_cls, k) or k in {f.name for f in __import__('dataclasses').fields(cfg_cls)}})
    return model_cls(cfg)


def list_models() -> List[str]:
    """Return sorted list of registered model names."""
    return sorted(_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """Check if a model name is registered."""
    return name in _REGISTRY
