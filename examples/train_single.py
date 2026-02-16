#!/usr/bin/env python
"""Train a single model and save the checkpoint.

Usage::

    python examples/train_single.py                    # defaults: GRU on multisine_05
    python examples/train_single.py --model lstm       # pick a different model
    python examples/train_single.py --wandb my-project # enable W&B logging
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import Dataset
from src.models.registry import get_model_class, get_model_config_class, list_model_keys
from src.validation import Metrics
from src.visualization import plot_predictions
from src.utils import ensure_proxy_env

def main():
    model_keys = sorted(list_model_keys())

    parser = argparse.ArgumentParser(description="Train a single model.")
    parser.add_argument(
        "--model",
        default="gru",
        choices=model_keys,
        help="Model name (default: gru)",
    )
    parser.add_argument(
        "--dataset",
        default="multisine_05",
        help="BAB experiment key (default: multisine_05)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--wandb", default=None, help="W&B project name")
    parser.add_argument(
        "--out-dir",
        default="checkpoints",
        help="Directory for saved models (default: checkpoints)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Runtime device: auto | cpu | cuda | cuda:N (default: auto)",
    )
    args = parser.parse_args()

    ensure_proxy_env()

    # ── Data ──────────────────────────────────────────────────────────
    ds = Dataset.from_bab_experiment(args.dataset)
    train_ds, val_ds, test_ds = ds.train_val_test_split(train=0.7, val=0.15)
    print(f"Dataset: {ds.name}  ({len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test)")

    # ── Config ────────────────────────────────────────────────────────
    config_cls = get_model_config_class(args.model)
    config = config_cls()
    if args.epochs is not None:
        config.epochs = args.epochs
    config.device = args.device
    if args.wandb:
        config.wandb_project = args.wandb
        config.wandb_run_name = f"{args.model}_{args.dataset}"

    # ── Train ─────────────────────────────────────────────────────────
    model_cls = get_model_class(args.model)
    model = model_cls(config)
    print(f"\nTraining {model!r} …")
    model.fit(train_ds.arrays, val_data=val_ds.arrays)

    # ── Evaluate ──────────────────────────────────────────────────────
    y_pred_osa = model.predict(test_ds.u, test_ds.y, mode="OSA")
    y_pred_fr = model.predict(test_ds.u, test_ds.y, mode="FR")

    print("\n── One-Step-Ahead ──")
    Metrics.summary(test_ds.y, y_pred_osa, name=f"{args.model} (OSA)")
    print("\n── Free-Run ──")
    Metrics.summary(test_ds.y, y_pred_fr, name=f"{args.model} (FR)")

    # ── Save ──────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    save_path = out_dir / f"{args.model}_{args.dataset}.pt"
    model.save(save_path)
    print(f"\nModel saved → {save_path}")

    # ── Plot ──────────────────────────────────────────────────────────
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    plot_predictions(
        test_ds.t,
        test_ds.y,
        {f"{args.model} OSA": y_pred_osa, f"{args.model} FR": y_pred_fr},
        title=f"{args.model} on {args.dataset}",
        save_path=str(plot_dir / f"{args.model}_{args.dataset}.png"),
    )


if __name__ == "__main__":
    main()
