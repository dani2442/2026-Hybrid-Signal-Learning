#!/usr/bin/env python
"""Train one model, save its checkpoint, and keep the comparison plots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._shared import (
    MODEL_CLASS_NAMES,
    describe_splits,
    load_train_val_test,
    print_metric_summary,
    train_and_evaluate,
)


def main():
    parser = argparse.ArgumentParser(description="Train a single model.")
    parser.add_argument(
        "--model",
        default="gru",
        choices=sorted(MODEL_CLASS_NAMES),
        help="Model name (default: gru)",
    )
    parser.add_argument(
        "--dataset",
        default="multisine_05",
        help="BAB experiment key (default: multisine_05)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate when supported by the model config.",
    )
    parser.add_argument("--wandb", default=None, help="W&B project name")
    parser.add_argument(
        "--out-dir",
        default="checkpoints",
        help="Directory for saved models (default: checkpoints)",
    )
    args = parser.parse_args()

    splits = load_train_val_test(args.dataset)
    print(f"Dataset: {splits.dataset.name}  ({describe_splits(splits.train, splits.val, splits.test)})")

    artifacts = train_and_evaluate(
        args.model,
        splits,
        epochs=args.epochs,
        learning_rate=args.lr,
        wandb_project=args.wandb,
        checkpoint_dir=args.out_dir,
        plot_dir="plots",
    )

    print(f"\nTraining {artifacts.model!r}")
    print("\n── One-Step-Ahead ──")
    print_metric_summary(f"{args.model} (OSA)", splits.test.y, artifacts.y_osa)
    print("\n── Free-Run ──")
    print_metric_summary(f"{args.model} (FR)", splits.test.y, artifacts.y_fr)
    if artifacts.checkpoint_path is not None:
        print(f"\nModel saved → {artifacts.checkpoint_path}")
    if artifacts.plot_path is not None:
        print(f"Plot saved → {artifacts.plot_path}")


if __name__ == "__main__":
    main()
