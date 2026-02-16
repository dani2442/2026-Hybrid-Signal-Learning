"""Train a single model on a BAB experiment dataset.

Usage
-----
::

    python -m examples.train_single --model narx --dataset multisine_05
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import from_bab_experiment
from src.models import build_model, list_models
from src.utils.runtime import seed_all
from src.validation.metrics import summary
from src.wandb_logger import WandbLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single model")
    parser.add_argument(
        "--model", type=str, default="narx",
        help=f"Model name. Available: {', '.join(list_models())}",
    )
    parser.add_argument("--dataset", type=str, default="multisine_05")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="trained_models")
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()

    seed_all(args.seed)

    # Load and split dataset
    ds = from_bab_experiment(args.dataset)
    train_ds, val_ds, test_ds = ds.train_val_test_split(
        args.train_ratio, args.val_ratio
    )
    print(f"Dataset: {ds.name}  (train={train_ds.n_samples}, "
          f"val={val_ds.n_samples}, test={test_ds.n_samples})")

    # Build config overrides
    overrides = {"seed": args.seed}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs

    # Build and train model
    model = build_model(args.model, **overrides)
    logger = WandbLogger(args.wandb_project, run_name=args.model) if args.wandb_project else None

    print(f"\nTraining: {args.model}")
    model.fit(
        (train_ds.u, train_ds.y),
        val_data=(val_ds.u, val_ds.y),
        logger=logger,
    )

    # Evaluate
    y_pred = model.predict(test_ds.u, y0=test_ds.y[:1])
    print(summary(test_ds.y, y_pred, args.model))

    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.model}_{args.dataset}.pkl")
    model.save(save_path)
    print(f"Model saved to {save_path}")

    if logger:
        logger.finish()


if __name__ == "__main__":
    main()
