"""Smoke test: train_single example should work for NARX."""

import numpy as np
import pytest


def test_narx_train_predict_roundtrip():
    """Train NARX on synthetic data and verify predict returns array."""
    from src.models import build_model

    np.random.seed(42)
    N = 200
    u = np.random.randn(N)
    y = np.cumsum(u) * 0.1

    model = build_model("narx", nu=2, ny=2, verbose=False)
    model.fit((u[:160], y[:160]))
    y_pred = model.predict(u[160:], y0=y[158:160])

    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) == len(u[160:])
    # Should not be all-zero
    assert np.any(y_pred != 0)
