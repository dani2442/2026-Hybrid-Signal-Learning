"""Hybrid Modeling — system identification with classical, neural, and
physics-informed models.

Public API
----------
::

    from src.data import Dataset, from_bab_experiment, DatasetCollection
    from src.models import build_model, list_models, load_model
    from src.validation import metrics
    from src.config import BaseConfig
"""

# Eager-import models so decorators register
import src.models  # noqa: F401

from src.config import BaseConfig
from src.data import Dataset, DatasetCollection, from_bab_experiment, from_mat
from src.models import build_model, list_models, load_model
from src.validation import metrics
from src.wandb_logger import WandbLogger

__all__ = [
    "BaseConfig",
    "Dataset",
    "DatasetCollection",
    "WandbLogger",
    "build_model",
    "from_bab_experiment",
    "from_mat",
    "list_models",
    "load_model",
    "metrics",
]
