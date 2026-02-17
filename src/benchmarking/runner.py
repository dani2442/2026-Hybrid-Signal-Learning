"""Benchmarking runner — evaluate registered models on datasets."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.data import Dataset, DatasetCollection
from src.models import build_model, list_models
from src.validation.metrics import compute_all

DataPair = tuple[np.ndarray, np.ndarray]
DataInput = DataPair | List[DataPair]


def _to_pairs(data: DataInput) -> List[DataPair]:
    """Normalize ``(u, y)`` or ``[(u, y), ...]`` to a list of pairs."""
    if isinstance(data, tuple) and len(data) == 2:
        return [(np.asarray(data[0]).ravel(), np.asarray(data[1]).ravel())]
    return [
        (np.asarray(u).ravel(), np.asarray(y).ravel())
        for u, y in data
    ]


def _pack_pairs(pairs: List[DataPair]) -> DataInput:
    """Use a tuple for one dataset, list for many."""
    return pairs[0] if len(pairs) == 1 else pairs


def run_single_benchmark(
    model_name: str,
    train_data: DataInput,
    test_data: DataInput,
    *,
    val_data: DataInput | None = None,
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
        skip = getattr(model, "max_lag", 0)
        y_true_parts: List[np.ndarray] = []
        y_pred_parts: List[np.ndarray] = []

        for u_test, y_test in _to_pairs(test_data):
            y_pred = model.predict(u_test, y_test, mode="FR")
            n = min(len(y_test), len(y_pred))
            start = min(skip, n)
            y_true_parts.append(np.asarray(y_test)[start:n])
            y_pred_parts.append(np.asarray(y_pred)[start:n])

        y_true_all = np.concatenate(y_true_parts)
        y_pred_all = np.concatenate(y_pred_parts)
        metrics = compute_all(y_true_all, y_pred_all, skip=0)
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
    dataset: Dataset | DatasetCollection,
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
    dataset : Dataset | DatasetCollection
        The full dataset (single or multiple experiments).
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
    if isinstance(dataset, DatasetCollection):
        train_sets, val_sets, test_sets = dataset.train_val_test_split(
            train_ratio, val_ratio
        )
    else:
        train_ds, val_ds, test_ds = dataset.train_val_test_split(
            train_ratio, val_ratio
        )
        train_sets = [train_ds]
        val_sets = [val_ds]
        test_sets = [test_ds]

    train_data = _pack_pairs([(ds.u, ds.y) for ds in train_sets])
    val_data = _pack_pairs([(ds.u, ds.y) for ds in val_sets])
    test_data = _pack_pairs([(ds.u, ds.y) for ds in test_sets])

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
            train_data=train_data,
            test_data=test_data,
            val_data=val_data,
            config_overrides=overrides,
            logger=logger,
            verbose=verbose,
        )
        results.append(result)

        if result["metrics"] and verbose:
            m = result["metrics"]
            print(
                f"  RMSE={m['RMSE']:.6f}  FIT={m['FIT']:.4f}  "
                f"R2={m['R2']:.4f}  t={result['train_time']:.1f}s"
            )
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
