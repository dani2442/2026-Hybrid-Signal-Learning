"""Smoke test: NARX should support multi-dataset train/val/test usage."""

import numpy as np


def test_narx_train_predict_roundtrip_multi_dataset():
    """Train NARX on two synthetic datasets and verify predictions."""
    from src.models import build_model

    rng = np.random.default_rng(42)
    n = 160
    u1 = rng.standard_normal(n)
    y1 = np.cumsum(u1) * 0.1
    u2 = rng.standard_normal(n)
    y2 = np.cumsum(u2) * 0.1

    train_data = [(u1[:100], y1[:100]), (u2[:100], y2[:100])]
    val_data = [(u1[100:120], y1[100:120]), (u2[100:120], y2[100:120])]
    test_data = [(u1[120:], y1[120:]), (u2[120:], y2[120:])]

    model = build_model("narx", nu=2, ny=2, verbose=False)
    model.fit(train_data, val_data=val_data)

    for u_test, y_test in test_data:
        y_pred = model.predict(u_test, y_test, mode="FR")
        assert isinstance(y_pred, np.ndarray)
        assert len(y_pred) == len(u_test)
        assert np.any(y_pred != 0)
