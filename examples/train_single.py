"""Train a single model on a BAB experiment dataset.

Usage
-----
::

    python -m examples.train_single --model narx --dataset multisine_05
    python -m examples.train_single --model narx --datasets multisine_05 swept_sine
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.data import from_bab_experiments, list_bab_experiments
from src.models import build_model, list_models
from src.utils.runtime import seed_all
from src.validation.metrics import compute_all, summary
from src.wandb_logger import WandbLogger


def _resolve_dataset_names(dataset: str, datasets: list[str] | None) -> list[str]:
    if datasets:
        return datasets
    return [name.strip() for name in dataset.split(",") if name.strip()]


def _pack_pairs(pairs):
    return pairs[0] if len(pairs) == 1 else pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single model")
    parser.add_argument(
        "--model", type=str, default="narx",
        help=f"Model name. Available: {', '.join(list_models())}",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="multisine_05",
        help="Single dataset key (or comma-separated keys).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more dataset keys. Overrides --dataset when provided. "
            f"Available: {', '.join(list_bab_experiments())}"
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs (default: use model config)")
    parser.add_argument("--save-dir", type=str, default="trained_models")
    parser.add_argument("--wandb-project", type=str, default="hybrid-learning")
    args = parser.parse_args()

    seed_all(args.seed)

    # Load and split datasets independently
    dataset_names = _resolve_dataset_names(args.dataset, args.datasets)
    datasets = from_bab_experiments(dataset_names)
    train_sets, val_sets, test_sets = datasets.train_val_test_split(
        args.train_ratio, args.val_ratio
    )
    print("Datasets:")
    for ds, tr, va, te in zip(datasets, train_sets, val_sets, test_sets):
        print(
            f"  {ds.name}: train={tr.n_samples}, val={va.n_samples}, "
            f"test={te.n_samples}"
        )

    # Build config overrides
    overrides = {"seed": args.seed}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs

    # Build model first so we can extract its config for wandb
    model = build_model(args.model, **overrides)

    # Assemble wandb run config: model hyperparameters + experiment settings
    wandb_config: dict = {}
    if hasattr(model, "config") and hasattr(model.config, "to_dict"):
        wandb_config.update(model.config.to_dict())
    wandb_config.update({
        "model_name": args.model,
        "datasets": dataset_names,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "n_train": sum(tr.n_samples for tr in train_sets),
        "n_val": sum(va.n_samples for va in val_sets),
        "n_test": sum(te.n_samples for te in test_sets),
    })

    logger = (
        WandbLogger(
            args.wandb_project,
            run_name=f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=wandb_config,
        )
        if args.wandb_project else None
    )

    print(f"\nTraining: {args.model}")
    train_data = [(ds.u, ds.y) for ds in train_sets]
    val_data = [(ds.u, ds.y) for ds in val_sets]
    t0 = time.time()
    model.fit(
        _pack_pairs(train_data),
        val_data=_pack_pairs(val_data),
        logger=logger,
    )
    train_time = time.time() - t0

    # Evaluate and log all metrics
    skip = getattr(model, "max_lag", 0)
    final_log: dict = {}

    # -- Validation: per-dataset + aggregated --
    val_y_true_all: list[np.ndarray] = []
    val_y_pred_all: list[np.ndarray] = []
    for val_ds in val_sets:
        y_val_fr = model.predict(val_ds.u, val_ds.y, mode="FR")
        val_fr_metrics = compute_all(val_ds.y, y_val_fr, skip=skip)
        if logger:
            prefixed = {f"val/{val_ds.name}/{k}": v for k, v in val_fr_metrics.items()}
            final_log.update(prefixed)
        yt, yp = np.asarray(val_ds.y).ravel(), np.asarray(y_val_fr).ravel()
        n = min(len(yt), len(yp))
        s = min(skip, n)
        val_y_true_all.append(yt[s:n])
        val_y_pred_all.append(yp[s:n])

    if val_y_true_all:
        agg_val = compute_all(
            np.concatenate(val_y_true_all),
            np.concatenate(val_y_pred_all),
            skip=0,
        )
        if logger:
            final_log.update({f"val/aggregated/{k}": v for k, v in agg_val.items()})

    # -- Test: per-dataset + aggregated --
    test_y_true_all: list[np.ndarray] = []
    test_y_pred_all: list[np.ndarray] = []
    for test_ds in test_sets:
        y_pred = model.predict(test_ds.u, test_ds.y, mode="FR")
        print(summary(test_ds.y, y_pred, f"{args.model}::{test_ds.name}", skip=skip))
        test_metrics = compute_all(test_ds.y, y_pred, skip=skip)
        if logger:
            prefixed = {f"test/{test_ds.name}/{k}": v for k, v in test_metrics.items()}
            final_log.update(prefixed)
        yt, yp = np.asarray(test_ds.y).ravel(), np.asarray(y_pred).ravel()
        n = min(len(yt), len(yp))
        s = min(skip, n)
        test_y_true_all.append(yt[s:n])
        test_y_pred_all.append(yp[s:n])

    if test_y_true_all:
        agg_test = compute_all(
            np.concatenate(test_y_true_all),
            np.concatenate(test_y_pred_all),
            skip=0,
        )
        print(f"\n{'=' * 40}")
        print(f"Aggregated test metrics ({len(test_sets)} datasets)")
        print(f"{'=' * 40}")
        for k, v in agg_test.items():
            print(f"  {k}: {v:.6f}")
        print(f"{'=' * 40}")
        if logger:
            final_log.update({f"test/aggregated/{k}": v for k, v in agg_test.items()})

    if logger and final_log:
        final_log["train_time"] = train_time
        # log_metrics → appears as W&B charts; log_summary → run summary table
        logger.log_metrics(final_log)
        logger.log_summary(final_log)

    # Save with timestamp to avoid overwriting previous runs
    os.makedirs(args.save_dir, exist_ok=True)
    dataset_tag = "__".join(dataset_names)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_dir, f"{args.model}_{dataset_tag}_{timestamp}.pkl")
    model.save(save_path)
    print(f"Model saved to {save_path}")

    if logger:
        logger.finish()


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────
# MODEL TRAINING SPEED  (fastest → slowest, dataset: multisine_05)
# Times marked * are measured in benchmark_results.json.
# Unmarked are estimated from epoch counts / solver cost in smoke_test_all.py.
# ─────────────────────────────────────────────────────────────────────
#
#  Rank  Model                  ~Time      Notes
#  ─────────────────────────────────────────────────────────────────────
#   1    arima                  < 0.1 s    analytical ARIMA fit (statsmodels)
#   2    exponential_smoothing  < 0.1 s    statsmodels ETS fit
#   3    narx                   0.14 s *   FROLS linear regression, epochs=1
#   4    random_forest          1.4 s  *   sklearn RF, epochs=1
#   5    linear_physics         ~ 5 s      analytical ODE, 1000 epochs, Euler
#   6    stribeck_physics       ~ 10 s     analytical ODE+Stribeck, 2000 epochs, RK4
#   7    neural_network         ~ 15 s     MLP supervised, 200 epochs
#   8    gru                    ~ 20 s     GRU sliding-window, 200 epochs
#   9    lstm                   ~ 25 s     LSTM sliding-window, 200 epochs
#  10    tcn                    ~ 25 s     TCN sliding-window, 200 epochs
#  11    mamba                  ~ 30 s     Mamba sliding-window, 200 epochs
#  12    neural_ode             101 s  *   continuous ODE integration, 300 epochs
#  13    neural_sde             118 s  *   stochastic integration, 300 epochs
#  14    neural_cde             ~ 150 s    CDE (heavier per-step than ODE), 200 epochs
#  15    ude                    ~ 200 s    UDE ODE shooting, 500 epochs
#  16    hybrid_linear_beam     284 s  *   physics+NN ODE, 600 epochs
#  17    vanilla_node_2d        ~ 400 s    2-D shooting ODE, 800 epochs
#  18    structured_node        ~ 400 s    2-D shooting ODE (structured), 800 epochs
#  19    adaptive_node          ~ 400 s    2-D shooting ODE (adaptive), 800 epochs
#  20    vanilla_nsde_2d        ~ 600 s    2-D shooting SDE, 800 epochs
#  21    structured_nsde        ~ 600 s    2-D shooting SDE (structured), 800 epochs
#  22    adaptive_nsde          ~ 600 s    2-D shooting SDE (adaptive), 800 epochs
#  23    hybrid_nonlinear_cam   716 s  *   physics+NN, 80 epochs but substeps=20
#  --- skipped by default (--include-slow) ---
#  24    vanilla_ncde_2d        >> hours   adjoint CDE backward, 30 epochs
#  25    structured_ncde        >> hours   adjoint CDE backward, 30 epochs
#  26    adaptive_ncde          >> hours   adjoint CDE backward, 30 epochs
#
# Usage examples:
#   python -m examples.train_single --model narx
#   python -m examples.train_single --model neural_ode --wandb-project hybrid-modeling-benchmark
#   python -m examples.train_single --model gru --dataset multisine_05 --epochs 100