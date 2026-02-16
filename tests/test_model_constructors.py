"""Tests for model constructors — every registered model must
instantiate without errors.
"""

import pytest

from src.models import build_model, list_models


@pytest.fixture(params=list_models())
def model_name(request):
    return request.param


def test_model_constructor(model_name):
    """Each registered model must be instantiable with defaults."""
    try:
        model = build_model(model_name)
        assert model is not None
        assert model.is_trained is False
        assert model.name == model_name
    except ImportError:
        pytest.skip(f"Optional dependency missing for {model_name}")
