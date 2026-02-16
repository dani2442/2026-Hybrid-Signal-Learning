"""Tests for Dataset and DatasetCollection."""

import numpy as np
import pytest

from src.data.dataset import Dataset, DatasetCollection


def _make_dataset(n: int = 100, name: str = "test") -> Dataset:
    t = np.linspace(0, 1, n)
    u = np.sin(2 * np.pi * t)
    y = np.cos(2 * np.pi * t)
    return Dataset(t=t, u=u, y=y, name=name, sampling_rate=float(n))


class TestDataset:
    def test_n_samples(self):
        ds = _make_dataset(50)
        assert ds.n_samples == 50

    def test_dt(self):
        ds = _make_dataset(100)
        assert abs(ds.dt - 0.01010101) < 1e-4

    def test_split(self):
        ds = _make_dataset(100)
        train, test = ds.split(0.7)
        assert train.n_samples == 70
        assert test.n_samples == 30

    def test_train_val_test_split(self):
        ds = _make_dataset(100)
        train, val, test = ds.train_val_test_split(0.7, 0.15)
        assert train.n_samples == 70
        assert val.n_samples == 15
        assert test.n_samples == 15

    def test_arrays(self):
        ds = _make_dataset(10)
        t, u, y = ds.arrays
        assert len(t) == 10

    def test_repr(self):
        ds = _make_dataset(100, "my_experiment")
        r = repr(ds)
        assert "my_experiment" in r


class TestDatasetCollection:
    def test_len(self):
        dc = DatasetCollection([_make_dataset(50, "a"), _make_dataset(60, "b")])
        assert len(dc) == 2

    def test_names(self):
        dc = DatasetCollection([_make_dataset(50, "a"), _make_dataset(60, "b")])
        assert dc.names == ["a", "b"]

    def test_concatenated(self):
        dc = DatasetCollection([_make_dataset(50, "a"), _make_dataset(60, "b")])
        u, y = dc.concatenated()
        assert len(u) == 110

    def test_as_train_tuples(self):
        dc = DatasetCollection([_make_dataset(50, "a"), _make_dataset(60, "b")])
        pairs = dc.as_train_tuples()
        assert len(pairs) == 2
        assert len(pairs[0][0]) == 50

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            DatasetCollection([])

    def test_train_val_test_split(self):
        dc = DatasetCollection([_make_dataset(100, "a"), _make_dataset(100, "b")])
        trains, vals, tests = dc.train_val_test_split()
        assert len(trains) == 2
        assert len(vals) == 2
