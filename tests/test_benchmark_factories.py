"""Tests for the benchmarking runner."""

import numpy as np
import pytest


def test_run_single_benchmark_narx():
    """NARX benchmark should complete without error."""
    from src.benchmarking import run_single_benchmark

    np.random.seed(42)
    N = 200
    u = np.random.randn(N)
    y = np.cumsum(u) * 0.1

    result = run_single_benchmark(
        "narx",
        train_data=(u[:150], y[:150]),
        test_data=(u[150:], y[150:]),
        config_overrides={"nu": 2, "ny": 2, "verbose": False},
    )
    assert result["model_name"] == "narx"
    assert result["metrics"] is not None
    assert "RMSE" in result["metrics"]


def test_run_single_benchmark_bad_model():
    """Unknown model should return error in result."""
    from src.benchmarking import run_single_benchmark

    result = run_single_benchmark(
        "nonexistent_model",
        train_data=(np.zeros(10), np.zeros(10)),
        test_data=(np.zeros(10), np.zeros(10)),
    )
    assert result["error"] is not None
    assert result["metrics"] is None
