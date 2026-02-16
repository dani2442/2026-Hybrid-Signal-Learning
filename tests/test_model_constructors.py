from __future__ import annotations

from src.models.base import BaseModel
from src.models.registry import (
    get_model_class,
    get_model_config_class,
    list_model_keys,
)


def test_models_accept_config_and_kwargs_forms():
    for key in list_model_keys():
        model_cls = get_model_class(key)
        config_cls = get_model_config_class(key)
        config = config_cls()

        model_from_config = model_cls(config=config)
        model_from_kwargs = model_cls(**config.to_dict())

        assert isinstance(model_from_config, BaseModel)
        assert isinstance(model_from_kwargs, BaseModel)
        assert type(model_from_config) is type(model_from_kwargs)
