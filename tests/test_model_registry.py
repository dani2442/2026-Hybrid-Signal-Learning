"""Tests for the model registry (decorator-based)."""

import pytest


def test_list_models_not_empty():
    """Registration decorators should have populated the registry."""
    from src.models import list_models
    names = list_models()
    assert len(names) > 0


def test_core_models_registered():
    """Check a selection of expected model names."""
    from src.models import list_models
    names = set(list_models())
    expected = {
        "narx", "arima", "exponential_smoothing", "random_forest",
        "neural_network",
        "gru", "lstm", "tcn", "mamba",
    }
    for m in expected:
        assert m in names, f"{m} not registered"


def test_get_model_class():
    from src.models import get_model_class
    cls = get_model_class("narx")
    assert cls is not None
    assert cls.name == "narx"


def test_get_config_class():
    from src.models import get_config_class
    from src.config import NARXConfig
    cfg_cls = get_config_class("narx")
    assert cfg_cls is NARXConfig


def test_build_model():
    from src.models import build_model
    model = build_model("narx", nu=3, ny=3)
    assert model.config.nu == 3
    assert model.config.ny == 3
    assert model.is_trained is False


def test_build_unknown_model_raises():
    from src.models import build_model
    with pytest.raises(KeyError):
        build_model("nonexistent_model")


def test_is_registered():
    from src.models import is_registered
    assert is_registered("narx")
    assert not is_registered("nonexistent")
