#!/usr/bin/env python
"""Load a saved model checkpoint and evaluate on test data.

Usage::

    python examples/load_and_test.py checkpoints/gru_multisine_05.pt
    python examples/load_and_test.py checkpoints/gru_multisine_05.pt --dataset swept_sine
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.base import load_model
from src.data import Dataset
from src.validation import Metrics
from src.visualization import plot_predictions


def main():
    parser = argparse.ArgumentParser(description="Load a model and evaluate.")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument(
        "--dataset",
        default="multisine_05",
        help="BAB experiment key for test data (default: multisine_05)",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.85,
        help="Train/test split ratio; test = 1-split (default: 0.85)",
    )
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────
    model = load_model(args.checkpoint)
    print(f"Loaded: {model!r}")
    print(f"  config: {model.config}")
    print(f"  fitted: {model._is_fitted}")

    # ── Data ──────────────────────────────────────────────────────────
    ds = Dataset.from_bab_experiment(args.dataset)
    _, test_ds = ds.split(ratio=args.split)
    print(f"\nTest set: {test_ds.name}  ({len(test_ds)} samples)")

    # ── Predict ───────────────────────────────────────────────────────
    y_osa = model.predict(test_ds.u, test_ds.y, mode="OSA")
    y_fr = model.predict(test_ds.u, test_ds.y, mode="FR")

    # ── Metrics ───────────────────────────────────────────────────────
    ckpt_name = Path(args.checkpoint).stem
    print("\n── One-Step-Ahead ──")
    Metrics.summary(test_ds.y, y_osa, name=f"{ckpt_name} (OSA)")
    print("\n── Free-Run ──")
    Metrics.summary(test_ds.y, y_fr, name=f"{ckpt_name} (FR)")

    # ── Plot ──────────────────────────────────────────────────────────
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    plot_predictions(
        test_ds.t,
        test_ds.y,
        {f"{ckpt_name} OSA": y_osa, f"{ckpt_name} FR": y_fr},
        title=f"Loaded model – {ckpt_name}",
        save_path=str(plot_dir / f"test_{ckpt_name}.png"),
    )
    print(f"\nPlot saved → plots/test_{ckpt_name}.png")


if __name__ == "__main__":
    main()
