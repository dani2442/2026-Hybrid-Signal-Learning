#!/usr/bin/env python
"""Train a small default model set on one dataset and compare the plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._shared import (
    DEFAULT_MODEL_NAMES,
    describe_splits,
    load_train_val_test,
    save_prediction_plot,
    train_and_evaluate,
)


def main():
    parser = argparse.ArgumentParser(description="Train all models and compare.")
    parser.add_argument("--dataset", default="multisine_05")
    parser.add_argument("--wandb", default=None, help="W&B project name")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all models")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate when supported")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of model names to train (default: all in DEFAULT_MODEL_NAMES)",
    )
    args = parser.parse_args()

    model_names = args.models or DEFAULT_MODEL_NAMES
    splits = load_train_val_test(args.dataset)
    print(f"Dataset: {splits.dataset.name}  ({describe_splits(splits.train, splits.val, splits.test)})\n")

    out_dir = Path(args.out_dir)
    results: dict[str, dict[str, float]] = {}
    predictions_osa: dict[str, np.ndarray] = {}
    predictions_fr: dict[str, np.ndarray] = {}

    for name in model_names:
        try:
            artifacts = train_and_evaluate(
                name,
                splits,
                epochs=args.epochs,
                learning_rate=args.lr,
                wandb_project=args.wandb,
                checkpoint_dir=out_dir,
            )
        except Exception as exc:
            print(f"{'─' * 60}\n{name}: failed ({exc})\n")
            continue

        print(f"{'─' * 60}\nTraining {artifacts.model!r}")
        results[name] = {"osa": artifacts.metrics_osa, "fr": artifacts.metrics_fr}
        predictions_osa[name] = artifacts.y_osa
        predictions_fr[name] = artifacts.y_fr
        print(f"  OSA R²={artifacts.metrics_osa['R2']:.4f}  FIT={artifacts.metrics_osa['FIT%']:.2f}")
        print(f"  FR  R²={artifacts.metrics_fr['R2']:.4f}  FIT={artifacts.metrics_fr['FIT%']:.2f}\n")

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
    results_path = out_dir / f"results_{splits.dataset.name}.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # ── Comparison plot ───────────────────────────────────────────────
    if predictions_fr:
        save_prediction_plot(
            splits.test,
            predictions_fr,
            title=f"Free-run comparison – {args.dataset}",
            plot_dir="plots",
            filename=f"comparison_fr_{args.dataset}.png",
        )
    if predictions_osa:
        save_prediction_plot(
            splits.test,
            predictions_osa,
            title=f"OSA comparison – {args.dataset}",
            plot_dir="plots",
            filename=f"comparison_osa_{args.dataset}.png",
        )


if __name__ == "__main__":
    main()
