#!/usr/bin/env python
"""Quick smoke test: train every model for a few epochs and report metrics.

Usage::

    python examples/smoke_test_all.py
    python examples/smoke_test_all.py --include-slow
    python examples/smoke_test_all.py --models narx gru lstm
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

# Ensure src is on the path when running as a script
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import from_bab_experiment
from src.models.registry import get_config_class, get_model_class, list_models
from src.validation.metrics import compute_all
from src.utils.runtime import seed_all

# ── Per-model epoch overrides (tiny for smoke tests) ──────────────
EPOCH_OVERRIDES: dict[str, int] = {
    # Statistical / non-iterative
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
    "vanilla_ncde_2d": 2,
    "structured_ncde": 2,
    "adaptive_ncde": 2,
    "vanilla_nsde_2d": 3,
    "structured_nsde": 3,
    "adaptive_nsde": 3,
}

DEFAULT_EPOCHS = 5

# Overrides to keep heavyweight models fast during smoke test
CONFIG_OVERRIDES: dict[str, dict] = {
    # Blackbox 2-D: smaller networks, fewer k_steps
    "vanilla_node_2d":   {"k_steps": 5, "hidden_dim": 32, "batch_size": 32},
    "structured_node":   {"k_steps": 5, "hidden_dim": 32, "batch_size": 32},
    "adaptive_node":     {"k_steps": 5, "hidden_dim": 32, "batch_size": 32},
    "vanilla_ncde_2d":   {"k_steps": 5, "hidden_dim": 32, "batch_size": 4},
    "structured_ncde":   {"k_steps": 5, "hidden_dim": 32, "batch_size": 4},
    "adaptive_ncde":     {"k_steps": 5, "hidden_dim": 32, "batch_size": 4},
    "vanilla_nsde_2d":   {"k_steps": 5, "hidden_dim": 32, "batch_size": 32},
    "structured_nsde":   {"k_steps": 5, "hidden_dim": 32, "batch_size": 32},
    "adaptive_nsde":     {"k_steps": 5, "hidden_dim": 32, "batch_size": 32},
    # Continuous-time: fewer sequences to keep it fast
    "neural_ode":        {"sequences_per_epoch": 4},
    "neural_sde":        {"sequences_per_epoch": 4},
    "neural_cde":        {"sequences_per_epoch": 4},
}

# CDE models with adjoint backward are extremely slow even for 1 epoch
SLOW_MODELS = {"vanilla_ncde_2d", "structured_ncde", "adaptive_ncde"}


def run_one(model_key: str, train_ds, val_ds, test_ds) -> dict:
    """Train + evaluate one model; return a summary dict."""
    t0 = time.time()

    config_cls = get_config_class(model_key)
    config = config_cls()
    config.epochs = EPOCH_OVERRIDES.get(model_key, DEFAULT_EPOCHS)
    config.verbose = False
    config.device = "cpu"

    # Apply extra overrides for heavyweight models
    for k, v in CONFIG_OVERRIDES.get(model_key, {}).items():
        if hasattr(config, k):
            setattr(config, k, v)

    model_cls = get_model_class(model_key)
    model = model_cls(config)

    # fit expects (u, y) tuple
    model.fit((train_ds.u, train_ds.y), val_data=(val_ds.u, val_ds.y))

    # Predict with proper API — OSA uses true y, FR is free-run
    y_pred_osa = model.predict(test_ds.u, test_ds.y, mode="OSA")
    y_pred_fr  = model.predict(test_ds.u, test_ds.y, mode="FR")

    # Compute metrics with proper alignment (skip initial transient)
    skip = getattr(model, "max_lag", 0)
    osa_metrics = compute_all(test_ds.y, y_pred_osa, skip=skip)
    fr_metrics  = compute_all(test_ds.y, y_pred_fr, skip=skip)
    elapsed = time.time() - t0

    return {
        "model": model_key,
        "time_s": round(elapsed, 1),
        "epochs": config.epochs,
        "osa_rmse": round(osa_metrics["RMSE"], 6),
        "fr_rmse":  round(fr_metrics["RMSE"], 6),
        "osa_r2":   round(osa_metrics["R2"], 4),
        "fr_r2":    round(fr_metrics["R2"], 4),
        "skip":     skip,
        "status":   "OK",
    }


def main():
    parser = argparse.ArgumentParser(description="Smoke test all models")
    parser.add_argument("--include-slow", action="store_true",
                        help="Include very slow CDE models (skipped by default)")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Only test these models")
    args = parser.parse_args()

    seed_all(42)

    ds = from_bab_experiment("multisine_05")
    train_ds, val_ds, test_ds = ds.train_val_test_split(
        train_ratio=0.7, val_ratio=0.15
    )
    print(
        f"Dataset: {ds.name}  ({train_ds.n_samples} train / "
        f"{val_ds.n_samples} val / {test_ds.n_samples} test)"
    )

    if args.models:
        all_keys = args.models
    else:
        all_keys = sorted(list_models())

    if not args.include_slow:
        skipped = [k for k in all_keys if k in SLOW_MODELS]
        all_keys = [k for k in all_keys if k not in SLOW_MODELS]
        if skipped:
            print(f"  (Skipping slow: {skipped}. Use --include-slow.)")

    results: list[dict] = []

    hdr = (
        f"{'Model':<25} {'Ep':>4} {'Time':>7} {'Skip':>4} "
        f"{'OSA RMSE':>12} {'FR RMSE':>12} "
        f"{'OSA R²':>9} {'FR R²':>9}  Status"
    )
    print(f"\n{hdr}")
    print("─" * 110)

    for key in all_keys:
        try:
            r = run_one(key, train_ds, val_ds, test_ds)
        except Exception as exc:
            r = {
                "model": key, "time_s": 0, "epochs": "-",
                "osa_rmse": "-", "fr_rmse": "-",
                "osa_r2": "-", "fr_r2": "-",
                "skip": "-",
                "status": f"FAIL: {exc}",
            }
            traceback.print_exc()

        results.append(r)
        print(
            f"{r['model']:<25} {str(r['epochs']):>4} "
            f"{str(r['time_s']):>6}s {str(r['skip']):>4} "
            f"{str(r['osa_rmse']):>12} {str(r['fr_rmse']):>12} "
            f"{str(r['osa_r2']):>9} {str(r['fr_r2']):>9}  "
            f"{r['status']}"
        )
        sys.stdout.flush()

    # Summary
    ok = sum(1 for r in results if r["status"] == "OK")
    fail = len(results) - ok
    print(f"\n{'─' * 110}")
    print(f"Done: {ok} passed, {fail} failed out of {len(results)} models.")


if __name__ == "__main__":
    main()
