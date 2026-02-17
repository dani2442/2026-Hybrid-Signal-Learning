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


def test_run_all_benchmarks_dataset_collection_narx():
    """Benchmark runner should accept DatasetCollection input."""
    from src.benchmarking import run_all_benchmarks
    from src.data.dataset import Dataset, DatasetCollection

    np.random.seed(7)
    n = 180
    t = np.arange(n, dtype=np.float64)
    u1 = np.random.randn(n)
    y1 = np.cumsum(u1) * 0.1
    u2 = np.random.randn(n)
    y2 = np.cumsum(u2) * 0.1

    d1 = Dataset(t=t, u=u1, y=y1, name="d1")
    d2 = Dataset(t=t, u=u2, y=y2, name="d2")
    dc = DatasetCollection([d1, d2])

    results = run_all_benchmarks(
        dc,
        train_ratio=0.6,
        val_ratio=0.2,
        model_names=["narx"],
        config_overrides={"nu": 2, "ny": 2, "verbose": False},
        verbose=False,
    )
    assert len(results) == 1
    assert results[0]["model_name"] == "narx"
    assert results[0]["metrics"] is not None
