"""Model package — importing sub-packages triggers registration."""

# Import sub-packages so @register_model decorators execute
from src.models import classical  # noqa: F401
from src.models import feedforward  # noqa: F401
from src.models import sequence  # noqa: F401

# Continuous/hybrid/blackbox have optional deps — guard imports
try:
    from src.models import continuous  # noqa: F401
except ImportError:
    pass

try:
    from src.models import hybrid  # noqa: F401
except ImportError:
    pass

try:
    from src.models import blackbox  # noqa: F401
except ImportError:
    pass

# Re-export key infrastructure
from src.models.base import BaseModel, PickleStateMixin, load_model
from src.models.registry import (
    build_model,
    get_config_class,
    get_model_class,
    is_registered,
    list_models,
    register_model,
)

__all__ = [
    "BaseModel",
    "PickleStateMixin",
    "load_model",
    "build_model",
    "get_config_class",
    "get_model_class",
    "is_registered",
    "list_models",
    "register_model",
]
