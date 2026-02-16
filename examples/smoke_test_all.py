#!/usr/bin/env python
"""Quick smoke test: train every model for a few epochs and report metrics.

Usage::
    python examples/smoke_test_all.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import Dataset
from src.models.registry import get_model_class, get_model_config_class, list_model_keys
from src.validation import Metrics
from src.utils import ensure_proxy_env

# How many epochs for each category (keep it tiny for smoke tests)
EPOCH_OVERRIDES: dict[str, int] = {
    # Statistical / non-iterative – epochs field often ignored, set low anyway
    "narx": 1,
    "arima": 1,
    "exponential_smoothing": 1,
    "random_forest": 1,
    # Simple NNs
    "neural_network": 5,
    "gru": 5,
    "lstm": 5,
    "tcn": 5,
    "mamba": 5,
    # Continuous-time
    "neural_ode": 5,
    "neural_sde": 5,
    "neural_cde": 5,
    # Physics / hybrid
    "linear_physics": 10,
    "stribeck_physics": 10,
    "hybrid_linear_beam": 10,
    "hybrid_nonlinear_cam": 10,
    "ude": 5,
    # Blackbox 2-D variants
    "vanilla_node_2d": 5,
    "structured_node": 5,
    "adaptive_node": 5,
    "vanilla_ncde_2d": 5,
    "structured_ncde": 5,
    "adaptive_ncde": 5,
    "vanilla_nsde_2d": 5,
    "structured_nsde": 5,
    "adaptive_nsde": 5,
}

DEFAULT_EPOCHS = 5


def run_one(model_key: str, train_ds, val_ds, test_ds) -> dict:
    """Train + evaluate one model; return a summary dict."""
    t0 = time.time()

    config_cls = get_model_config_class(model_key)
    config = config_cls()
    config.epochs = EPOCH_OVERRIDES.get(model_key, DEFAULT_EPOCHS)
    config.verbose = False          # keep output clean
    config.device = "cpu"           # fast for smoke tests

    model_cls = get_model_class(model_key)
    model = model_cls(config)
    model.fit(train_ds.arrays, val_data=val_ds.arrays)

    y_pred_osa = model.predict(test_ds.u, test_ds.y, mode="OSA")
    y_pred_fr  = model.predict(test_ds.u, test_ds.y, mode="FR")

    osa_metrics = Metrics.compute_all(test_ds.y, y_pred_osa)
    fr_metrics  = Metrics.compute_all(test_ds.y, y_pred_fr)
    elapsed = time.time() - t0

    return {
        "model": model_key,
        "time_s": round(elapsed, 1),
        "epochs": config.epochs,
        "osa_rmse": round(osa_metrics["RMSE"], 6),
        "fr_rmse": round(fr_metrics["RMSE"], 6),
        "osa_r2": round(osa_metrics.get("R2", float("nan")), 4),
        "fr_r2": round(fr_metrics.get("R2", float("nan")), 4),
        "status": "OK",
    }


def main():
    ensure_proxy_env()

    ds = Dataset.from_bab_experiment("multisine_05")
    train_ds, val_ds, test_ds = ds.train_val_test_split(train=0.7, val=0.15)
    print(f"Dataset: {ds.name}  ({len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test)")

    all_keys = sorted(list_model_keys())
    results = []

    print(f"\n{'Model':<25} {'Epochs':>6} {'Time':>7} {'OSA RMSE':>12} {'FR RMSE':>12} {'OSA R²':>9} {'FR R²':>9}  Status")
    print("─" * 100)

    for key in all_keys:
        try:
            r = run_one(key, train_ds, val_ds, test_ds)
        except Exception as exc:
            r = {"model": key, "time_s": 0, "epochs": "-", "osa_rmse": "-", "fr_rmse": "-",
                 "osa_r2": "-", "fr_r2": "-", "status": f"FAIL: {exc}"}
            traceback.print_exc()

        results.append(r)
        print(
            f"{r['model']:<25} {str(r['epochs']):>6} {str(r['time_s']):>6}s "
            f"{str(r['osa_rmse']):>12} {str(r['fr_rmse']):>12} "
            f"{str(r['osa_r2']):>9} {str(r['fr_r2']):>9}  {r['status']}"
        )

    # Summary
    ok = sum(1 for r in results if r["status"] == "OK")
    fail = len(results) - ok
    print(f"\n{'─' * 100}")
    print(f"Done: {ok} passed, {fail} failed out of {len(results)} models.")


if __name__ == "__main__":
    main()
