"""Tests for runtime utilities."""

import pytest


def test_resolve_device_cpu():
    from src.utils.runtime import resolve_device
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto():
    from src.utils.runtime import resolve_device
    device = resolve_device("auto")
    assert device in ("cpu", "cuda")


def test_seed_all_no_error():
    from src.utils.runtime import seed_all
    seed_all(42)  # Should not raise
