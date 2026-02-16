from __future__ import annotations

from src.benchmarking import build_benchmark_cases
from src.models.base import BaseModel


def test_benchmark_factories_instantiate_models():
    cases = build_benchmark_cases()
    assert len(cases) > 0

    for case in cases:
        model = case.factory(0.02)
        assert isinstance(model, BaseModel)
