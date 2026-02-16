"""Benchmarking runner — evaluate registered models on datasets."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.data import Dataset
from src.models import build_model, list_models
from src.validation.metrics import compute_all, summary


def run_single_benchmark(
    model_name: str,
    train_data: tuple[np.ndarray, np.ndarray],
    test_data: tuple[np.ndarray, np.ndarray],
    *,
    val_data: tuple[np.ndarray, np.ndarray] | None = None,
    config_overrides: Dict[str, Any] | None = None,
    logger=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train and evaluate a single model.

    Returns
    -------
    dict
        ``{"model_name": str, "metrics": {...}, "train_time": float,
          "model": BaseModel}``
    """
    overrides = config_overrides or {}
    if verbose:
        overrides.setdefault("verbose", True)

    try:
        model = build_model(model_name, **overrides)
    except KeyError as exc:
        return {
            "model_name": model_name,
            "metrics": None,
            "train_time": 0.0,
            "error": str(exc),
            "model": None,
        }

    t0 = time.time()
    try:
        model.fit(train_data, val_data=val_data, logger=logger)
    except Exception as exc:
        return {
            "model_name": model_name,
            "metrics": None,
            "train_time": time.time() - t0,
            "error": str(exc),
            "model": None,
        }
    train_time = time.time() - t0

    try:
        y_pred = model.predict(test_data[0], test_data[1], mode="FR")
        skip = getattr(model, "max_lag", 0)
        metrics = compute_all(test_data[1], y_pred, skip=skip)
    except Exception as exc:
        return {
            "model_name": model_name,
            "metrics": None,
            "train_time": train_time,
            "error": str(exc),
            "model": model,
        }

    return {
        "model_name": model_name,
        "metrics": metrics,
        "train_time": train_time,
        "model": model,
    }


def run_all_benchmarks(
    dataset: Dataset,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    model_names: List[str] | None = None,
    config_overrides: Dict[str, Any] | None = None,
    save_dir: Optional[str] = None,
    logger_factory=None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run benchmarks for all (or selected) registered models.

    Parameters
    ----------
    dataset : Dataset
        The full dataset.
    train_ratio, val_ratio : float
        Split ratios.
    model_names : list | None
        Subset of model names; defaults to all registered.
    config_overrides : dict | None
        Overrides applied to every model's config.
    save_dir : str | None
        Directory to persist trained models.
    logger_factory : callable | None
        ``logger_factory(model_name) → WandbLogger`` or similar.
    verbose : bool

    Returns
    -------
    list[dict]
        One result dict per model.
    """
    train_ds, val_ds, test_ds = dataset.train_val_test_split(
        train_ratio, val_ratio
    )

    names = model_names or list_models()
    overrides = config_overrides or {}
    results: List[Dict[str, Any]] = []

    for name in names:
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Benchmarking: {name}")
            print(f"{'=' * 50}")

        logger = logger_factory(name) if logger_factory else None

        result = run_single_benchmark(
            name,
            train_data=(train_ds.u, train_ds.y),
            test_data=(test_ds.u, test_ds.y),
            val_data=(val_ds.u, val_ds.y),
            config_overrides=overrides,
            logger=logger,
            verbose=verbose,
        )
        results.append(result)

        if result["metrics"]:
            if verbose:
                skip = getattr(result["model"], "max_lag", 0)
                y_pred = result["model"].predict(test_ds.u, test_ds.y, mode="FR")
                print(summary(test_ds.y, y_pred, name, skip=skip))
        elif verbose:
            print(f"  ERROR: {result.get('error', 'unknown')}")

        if save_dir and result["model"] is not None:
            os.makedirs(save_dir, exist_ok=True)
            result["model"].save(os.path.join(save_dir, f"{name}.pkl"))

        if logger is not None:
            try:
                logger.finish()
            except Exception:
                pass

    return results


def results_to_json(results: List[Dict[str, Any]], filepath: str) -> None:
    """Save benchmark results to JSON (excluding model objects)."""
    serialisable = []
    for r in results:
        entry = {
            "model_name": r["model_name"],
            "metrics": r.get("metrics"),
            "train_time": r.get("train_time"),
        }
        if "error" in r:
            entry["error"] = r["error"]
        serialisable.append(entry)
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(serialisable, f, indent=2)
