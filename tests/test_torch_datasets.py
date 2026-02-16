"""Tests for WindowedTrainDataset and FullSequenceDataset."""

import numpy as np
import pytest
import torch

from src.data.torch_datasets import FullSequenceDataset, WindowedTrainDataset


class TestWindowedTrainDataset:
    def test_len(self):
        u = np.random.randn(100)
        y = np.random.randn(100)
        ds = WindowedTrainDataset(u, y, window_size=20)
        assert len(ds) == 5  # 100 // 20

    def test_item_shape(self):
        u = np.random.randn(100)
        y = np.random.randn(100)
        ds = WindowedTrainDataset(u, y, window_size=20)
        u_w, y_w = ds[0]
        assert u_w.shape == (20,)
        assert y_w.shape == (20,)
        assert isinstance(u_w, torch.Tensor)

    def test_random_offset(self):
        """Two calls should (usually) return different windows."""
        np.random.seed(None)  # Ensure non-deterministic
        u = np.arange(100, dtype=np.float32)
        y = np.arange(100, dtype=np.float32)
        ds = WindowedTrainDataset(u, y, window_size=10)
        starts = set()
        for _ in range(50):
            u_w, _ = ds[0]
            starts.add(int(u_w[0].item()))
        # Should have sampled multiple different start indices
        assert len(starts) > 1

    def test_custom_samples_per_epoch(self):
        u = np.random.randn(100)
        y = np.random.randn(100)
        ds = WindowedTrainDataset(u, y, window_size=20, samples_per_epoch=42)
        assert len(ds) == 42

    def test_short_signal(self):
        """When signal < window, use full signal."""
        u = np.random.randn(5)
        y = np.random.randn(5)
        ds = WindowedTrainDataset(u, y, window_size=20)
        u_w, y_w = ds[0]
        assert u_w.shape == (5,)


class TestFullSequenceDataset:
    def test_len(self):
        u = np.random.randn(100)
        y = np.random.randn(100)
        ds = FullSequenceDataset(u, y)
        assert len(ds) == 1

    def test_item_shape(self):
        u = np.random.randn(100)
        y = np.random.randn(100)
        ds = FullSequenceDataset(u, y)
        u_full, y_full = ds[0]
        assert u_full.shape == (100,)
        assert y_full.shape == (100,)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            FullSequenceDataset(np.zeros(10), np.zeros(20))
