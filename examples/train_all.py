#!/usr/bin/env python
"""Train all models on a dataset and compare results.

Usage::

    python examples/train_all.py
    python examples/train_all.py --dataset swept_sine --wandb my-project
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import Dataset
from src.config import MODEL_CONFIGS
from src.validation import Metrics
from src.visualization import plot_predictions

# Models that work out-of-the-box on the BAB (2-D rotary beam) data.
# Excludes hybrid_nonlinear_cam which requires cam-specific geometry.
DEFAULT_MODELS: list[str] = [
    "narx",
    "arima",
    "exponential_smoothing",
    "random_forest",
    "neural_network",
    "gru",
    "lstm",
    "tcn",
    # "mamba",            # uncomment if mamba-ssm is installed
    "neural_ode",
    # "neural_sde",
    # "neural_cde",
    "linear_physics",
    "stribeck_physics",
    "hybrid_linear_beam",
    "ude",
    "vanilla_node_2d",
    "structured_node",
    "adaptive_node",
    # "vanilla_ncde_2d",
    # "structured_ncde",
    # "adaptive_ncde",
    # "vanilla_nsde_2d",
    # "structured_nsde",
    # "adaptive_nsde",
]


def _get_model_class(name: str):
    import src.models as m

    from examples.train_single import _MODEL_REGISTRY

    return getattr(m, _MODEL_REGISTRY[name])


def main():
    parser = argparse.ArgumentParser(description="Train all models and compare.")
    parser.add_argument("--dataset", default="multisine_05")
    parser.add_argument("--wandb", default=None, help="W&B project name")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of model names to train (default: all in DEFAULT_MODELS)",
    )
    args = parser.parse_args()

    model_names = args.models or DEFAULT_MODELS

    # ── Data ──────────────────────────────────────────────────────────
    ds = Dataset.from_bab_experiment(args.dataset)
    train_ds, val_ds, test_ds = ds.train_val_test_split(train=0.7, val=0.15)
    print(
        f"Dataset: {ds.name}  "
        f"({len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test)\n"
    )

    out_dir = Path(args.out_dir)
    results: dict[str, dict[str, float]] = {}
    predictions_osa: dict[str, np.ndarray] = {}
    predictions_fr: dict[str, np.ndarray] = {}

    for name in model_names:
        if name not in MODEL_CONFIGS:
            print(f"⚠  Unknown model '{name}', skipping.")
            continue

        config = MODEL_CONFIGS[name]()
        if args.wandb:
            config.wandb_project = args.wandb
            config.wandb_run_name = f"{name}_{args.dataset}"

        model_cls = _get_model_class(name)
        model = model_cls(config)
        print(f"{'─' * 60}\nTraining {model!r}")

        try:
            model.fit(train_ds.arrays, val_data=val_ds.arrays)
        except Exception as exc:
            print(f"  ✗ {name} failed: {exc}\n")
            continue

        # Evaluate on test set
        y_osa = model.predict(test_ds.u, test_ds.y, mode="OSA")
        y_fr = model.predict(test_ds.u, test_ds.y, mode="FR")

        metrics_osa = Metrics.compute_all(test_ds.y, y_osa)
        metrics_fr = Metrics.compute_all(test_ds.y, y_fr)

        results[name] = {"osa": metrics_osa, "fr": metrics_fr}
        predictions_osa[name] = y_osa
        predictions_fr[name] = y_fr

        print(f"  OSA R²={metrics_osa['R2']:.4f}  FIT={metrics_osa['FIT%']:.2f}")
        print(f"  FR  R²={metrics_fr['R2']:.4f}  FIT={metrics_fr['FIT%']:.2f}\n")

        # Save checkpoint
        model.save(out_dir / f"{name}_{args.dataset}.pt")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"{'Model':<25} {'OSA R²':>8} {'FR R²':>8} {'FR FIT%':>9}")
    print(f"{'─' * 60}")
    for name, m in results.items():
        print(
            f"{name:<25} {m['osa']['R2']:>8.4f} {m['fr']['R2']:>8.4f} "
            f"{m['fr']['FIT%']:>8.2f}"
        )
    print(f"{'=' * 60}")

    # Save results JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"results_{args.dataset}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # ── Comparison plot ───────────────────────────────────────────────
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    if predictions_fr:
        plot_predictions(
            test_ds.t,
            test_ds.y,
            predictions_fr,
            title=f"Free-run comparison – {args.dataset}",
            save_path=str(plot_dir / f"comparison_fr_{args.dataset}.png"),
        )
    if predictions_osa:
        plot_predictions(
            test_ds.t,
            test_ds.y,
            predictions_osa,
            title=f"OSA comparison – {args.dataset}",
            save_path=str(plot_dir / f"comparison_osa_{args.dataset}.png"),
        )


if __name__ == "__main__":
    main()
