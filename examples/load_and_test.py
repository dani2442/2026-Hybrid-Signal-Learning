"""Load a saved model and test it.

Usage
-----
::

    python -m examples.load_and_test --model-path trained_models/narx_multisine_05.pkl \\
        --dataset multisine_05
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import from_bab_experiment
from src.models import load_model
from src.validation.metrics import summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and test a saved model")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="multisine_05")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    ds = from_bab_experiment(args.dataset)
    _, _, test_ds = ds.train_val_test_split(args.train_ratio, args.val_ratio)

    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path)
    print(f"Model: {model}")

    y_pred = model.predict(test_ds.u, y0=test_ds.y[:1])
    print(summary(test_ds.y, y_pred, model.name))


if __name__ == "__main__":
    main()
