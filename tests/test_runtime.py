from __future__ import annotations

import numpy as np
import torch

from src.utils.runtime import resolve_device, seed_all


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_resolve_device_explicit_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:0") == "cuda:0"


def test_seed_all_reproducible():
    seed_all(1234)
    torch_a = torch.rand(5)
    np_a = np.random.rand(5)

    seed_all(1234)
    torch_b = torch.rand(5)
    np_b = np.random.rand(5)

    assert torch.allclose(torch_a, torch_b)
    assert np.allclose(np_a, np_b)
