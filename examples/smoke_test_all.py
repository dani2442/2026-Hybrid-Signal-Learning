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

DEFAULT_EPOCHS = 5

# Per-model overrides tuned for convergence (R² > 0.9 target)
CONFIG_OVERRIDES: dict[str, dict] = {
    # Blackbox 2-D ODE: lower LR, tighter grad_clip, moderate architecture
    "vanilla_node_2d":   {"epochs": 800, "k_steps": 15, "hidden_dim": 64, "batch_size": 64, "learning_rate": 5e-4, "grad_clip": 1.0, "scheduler_patience": 30, "early_stopping_patience": 80},
    "structured_node":   {"epochs": 800, "k_steps": 15, "hidden_dim": 64, "batch_size": 64, "learning_rate": 5e-4, "grad_clip": 1.0, "scheduler_patience": 30, "early_stopping_patience": 80},
    "adaptive_node":     {"epochs": 800, "k_steps": 15, "hidden_dim": 64, "batch_size": 64, "learning_rate": 5e-4, "grad_clip": 1.0, "scheduler_patience": 30, "early_stopping_patience": 80},
    # Blackbox 2-D CDE
    "vanilla_ncde_2d":   {"epochs": 30, "k_steps": 10, "hidden_dim": 64, "batch_size": 8, "learning_rate": 1e-3, "grad_clip": 1.0},
    "structured_ncde":   {"epochs": 30, "k_steps": 10, "hidden_dim": 64, "batch_size": 8, "learning_rate": 1e-3, "grad_clip": 1.0},
    "adaptive_ncde":     {"epochs": 30, "k_steps": 10, "hidden_dim": 64, "batch_size": 8, "learning_rate": 1e-3, "grad_clip": 1.0},
    # Blackbox 2-D SDE: lower LR, tighter grad_clip, small diffusion network
    "vanilla_nsde_2d":   {"epochs": 800, "k_steps": 15, "hidden_dim": 64, "batch_size": 64, "learning_rate": 5e-4, "grad_clip": 1.0, "diffusion_hidden_dim": 16, "scheduler_patience": 40, "early_stopping_patience": 100},
    "structured_nsde":   {"epochs": 800, "k_steps": 15, "hidden_dim": 64, "batch_size": 64, "learning_rate": 5e-4, "grad_clip": 1.0, "diffusion_hidden_dim": 16, "scheduler_patience": 40, "early_stopping_patience": 100},
    "adaptive_nsde":     {"epochs": 800, "k_steps": 15, "hidden_dim": 64, "batch_size": 64, "learning_rate": 5e-4, "grad_clip": 1.0, "diffusion_hidden_dim": 16, "scheduler_patience": 40, "early_stopping_patience": 100},
    # Continuous-time: more sequences, bigger networks
    "neural_ode":        {"epochs": 300, "sequences_per_epoch": 16, "train_window_size": 50, "hidden_layers": [128, 128], "scheduler_patience": 20, "early_stopping_patience": 50},
    "neural_sde":        {"epochs": 300, "sequences_per_epoch": 16, "train_window_size": 50, "scheduler_patience": 20, "early_stopping_patience": 50},
    "neural_cde":        {"epochs": 200, "sequences_per_epoch": 16, "train_window_size": 50, "scheduler_patience": 20, "early_stopping_patience": 50, "learning_rate": 5e-3},
    # UDE: longer windows, bigger NN
    "ude":               {"epochs": 500, "train_window_size": 50, "hidden_layers": [128, 128], "scheduler_patience": 25, "early_stopping_patience": 60},
    # Physics: longer windows for better gradient signal
    "linear_physics":    {"epochs": 1000, "train_window_size": 100, "learning_rate": 1e-2, "scheduler_patience": 50},
    "stribeck_physics":  {"epochs": 2000, "solver": "rk4", "training_mode": "windowed", "train_window_size": 50, "learning_rate": 5e-3, "scheduler_patience": 30},
    # Hybrid: tuned LR + initial physics for stability
    "hybrid_linear_beam":    {"epochs": 600, "learning_rate": 2e-3, "integration_substeps": 4, "scheduler_patience": 20, "early_stopping_patience": 60},
    "hybrid_nonlinear_cam":  {"epochs": 80, "learning_rate": 5e-3, "integration_substeps": 20, "scheduler_patience": 15, "J": 0.1, "k": 10.0},
    # Sequence models: LR scheduler active, early stopping
    "gru":  {"epochs": 200, "scheduler_patience": 10, "early_stopping_patience": 40},
    "lstm": {"epochs": 200, "scheduler_patience": 10, "early_stopping_patience": 40},
    "tcn":  {"epochs": 200, "scheduler_patience": 10, "early_stopping_patience": 30},
    "mamba": {"epochs": 200, "scheduler_patience": 10, "early_stopping_patience": 40},
    # Classical: tune for FR stability
    "narx": {"epochs": 1, "nu": 8, "ny": 8},
    "random_forest": {"epochs": 1, "nu": 5, "ny": 5, "n_estimators": 200},
    "neural_network": {"epochs": 200, "nu": 10, "ny": 10, "hidden_layers": [128, 128, 128], "scheduler_patience": 15, "early_stopping_patience": 30},
    "arima": {"epochs": 1, "order": (1, 0, 1)},
    "exponential_smoothing": {"epochs": 1, "trend": None, "seasonal": "add", "seasonal_periods": 21},
}

# CDE models with adjoint backward are extremely slow even for 1 epoch
SLOW_MODELS = {"vanilla_ncde_2d", "structured_ncde", "adaptive_ncde"}


def run_one(model_key: str, train_ds, val_ds, test_ds) -> dict:
    """Train + evaluate one model; return a summary dict."""
    seed_all(42)  # Reset seed per model for reproducible results
    t0 = time.time()

    config_cls = get_config_class(model_key)
    config = config_cls()
    model_overrides = CONFIG_OVERRIDES.get(model_key, {})
    config.epochs = model_overrides.get("epochs", DEFAULT_EPOCHS)
    config.verbose = False
    config.device = "cpu"

    # Apply extra overrides for heavyweight models
    for k, v in model_overrides.items():
        if k == "epochs":
            continue
        if hasattr(config, k):
            setattr(config, k, v)

    model_cls = get_model_class(model_key)
    model = model_cls(config)

    # fit expects (u, y) tuple
    model.fit((train_ds.u, train_ds.y), val_data=(val_ds.u, val_ds.y))

    # Predict with proper API — OSA uses true y, FR is free-run
    y_val_osa = model.predict(val_ds.u, val_ds.y, mode="OSA")
    y_val_fr  = model.predict(val_ds.u, val_ds.y, mode="FR")
    y_pred_osa = model.predict(test_ds.u, test_ds.y, mode="OSA")
    y_pred_fr  = model.predict(test_ds.u, test_ds.y, mode="FR")

    # Compute metrics with proper alignment (skip initial transient)
    skip = getattr(model, "max_lag", 0)
    val_osa_metrics = compute_all(val_ds.y, y_val_osa, skip=skip)
    val_fr_metrics  = compute_all(val_ds.y, y_val_fr, skip=skip)
    osa_metrics = compute_all(test_ds.y, y_pred_osa, skip=skip)
    fr_metrics  = compute_all(test_ds.y, y_pred_fr, skip=skip)
    elapsed = time.time() - t0

    return {
        "model": model_key,
        "time_s": round(elapsed, 1),
        "epochs": config.epochs,
        "val_osa_r2": round(val_osa_metrics["R2"], 4),
        "val_fr_r2":  round(val_fr_metrics["R2"], 4),
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
        train_ratio=0.8, val_ratio=0.1
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
        f"{'Val OSA R²':>10} {'Val FR R²':>10} "
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
                "val_osa_r2": "-", "val_fr_r2": "-",
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
            f"{str(r['val_osa_r2']):>10} {str(r['val_fr_r2']):>10} "
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
