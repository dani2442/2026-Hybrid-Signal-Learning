#!/usr/bin/env python
"""Load a saved model checkpoint and evaluate on test data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._shared import print_metric_summary, predict_model, resolve_model_key, save_prediction_plot


def main():
    parser = argparse.ArgumentParser(description="Load a model and evaluate.")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument(
        "--dataset",
        default="multisine_05",
        help="BAB experiment key for test data (default: multisine_05)",
    )
    args = parser.parse_args()

    from src.data import Dataset
    from src.models.base import load_model

    # ── Load model ────────────────────────────────────────────────────
    model = load_model(args.checkpoint)
    print(f"Loaded: {model!r}")
    print(f"  config: {model.config}")
    print(f"  fitted: {model._is_fitted}")

    # ── Data ──────────────────────────────────────────────────────────
    ds = Dataset.from_bab_experiment(args.dataset)
    _, test_ds = ds.split(ratio=0.85)
    print(f"\nTest set: {test_ds.name}  ({len(test_ds)} samples)")

    # ── Predict ───────────────────────────────────────────────────────
    model_key = resolve_model_key(model)
    y_osa, y_fr = predict_model(model, test_ds, model_key=model_key)

    # ── Metrics ───────────────────────────────────────────────────────
    ckpt_name = Path(args.checkpoint).stem
    print("\n── One-Step-Ahead ──")
    print_metric_summary(f"{ckpt_name} (OSA)", test_ds.y, y_osa)
    print("\n── Free-Run ──")
    print_metric_summary(f"{ckpt_name} (FR)", test_ds.y, y_fr)

    # ── Plot ──────────────────────────────────────────────────────────
    plot_path = save_prediction_plot(
        test_ds,
        {f"{ckpt_name} OSA": y_osa, f"{ckpt_name} FR": y_fr},
        title=f"Loaded model – {ckpt_name}",
        plot_dir="plots",
        filename=f"test_{ckpt_name}.png",
    )
    print(f"\nPlot saved → {plot_path}")


if __name__ == "__main__":
    main()
