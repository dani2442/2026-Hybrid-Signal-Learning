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

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import from_bab_experiments, list_bab_experiments
from src.models import build_model, list_models
from src.utils.runtime import seed_all
from src.validation.metrics import summary
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="trained_models")
    parser.add_argument("--wandb-project", type=str, default=None)
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

    # Build and train model
    model = build_model(args.model, **overrides)
    logger = WandbLogger(args.wandb_project, run_name=args.model) if args.wandb_project else None

    print(f"\nTraining: {args.model}")
    train_data = [(ds.u, ds.y) for ds in train_sets]
    val_data = [(ds.u, ds.y) for ds in val_sets]
    model.fit(
        _pack_pairs(train_data),
        val_data=_pack_pairs(val_data),
        logger=logger,
    )

    # Evaluate
    skip = getattr(model, "max_lag", 0)
    for test_ds in test_sets:
        y_pred = model.predict(test_ds.u, test_ds.y, mode="FR")
        print(summary(test_ds.y, y_pred, f"{args.model}::{test_ds.name}", skip=skip))

    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    dataset_tag = "__".join(dataset_names)
    save_path = os.path.join(args.save_dir, f"{args.model}_{dataset_tag}.pkl")
    model.save(save_path)
    print(f"Model saved to {save_path}")

    if logger:
        logger.finish()


if __name__ == "__main__":
    main()
