from __future__ import annotations

from src.config import MODEL_CONFIGS
from src.models.base import BaseModel
from src.models.registry import (
    get_model_class,
    get_model_config_class,
    list_model_keys,
)


def test_registry_keys_match_model_configs():
    assert set(list_model_keys()) == set(MODEL_CONFIGS.keys())


def test_registry_resolves_model_and_config_classes():
    for key in list_model_keys():
        model_cls = get_model_class(key)
        config_cls = get_model_config_class(key)
        model = model_cls(config=config_cls())
        assert isinstance(model, BaseModel)
