#!/usr/bin/env python
"""Train a single model and save the checkpoint.

Usage::

    python examples/train_single.py                    # defaults: GRU on multisine_05
    python examples/train_single.py --model lstm       # pick a different model
    python examples/train_single.py --wandb my-project # enable W&B logging
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import Dataset
from src.config import MODEL_CONFIGS
from src.validation import Metrics
from src.visualization import plot_predictions

# Map of friendly name → (model class import path, config class key)
_MODEL_REGISTRY: dict[str, str] = {
    "narx": "NARX",
    "arima": "ARIMA",
    "exponential_smoothing": "ExponentialSmoothing",
    "random_forest": "RandomForest",
    "neural_network": "NeuralNetwork",
    "gru": "GRU",
    "lstm": "LSTM",
    "tcn": "TCN",
    "mamba": "Mamba",
    "neural_ode": "NeuralODE",
    "neural_sde": "NeuralSDE",
    "neural_cde": "NeuralCDE",
    "linear_physics": "LinearPhysics",
    "stribeck_physics": "StribeckPhysics",
    "hybrid_linear_beam": "HybridLinearBeam",
    "hybrid_nonlinear_cam": "HybridNonlinearCam",
    "ude": "UDE",
    "vanilla_node_2d": "VanillaNODE2D",
    "structured_node": "StructuredNODE",
    "adaptive_node": "AdaptiveNODE",
    "vanilla_ncde_2d": "VanillaNCDE2D",
    "structured_ncde": "StructuredNCDE",
    "adaptive_ncde": "AdaptiveNCDE",
    "vanilla_nsde_2d": "VanillaNSDE2D",
    "structured_nsde": "StructuredNSDE",
    "adaptive_nsde": "AdaptiveNSDE",
}


def _get_model_class(name: str):
    """Import and return the model class by friendly name."""
    import src.models as m

    class_name = _MODEL_REGISTRY[name]
    return getattr(m, class_name)


def main():
    parser = argparse.ArgumentParser(description="Train a single model.")
    parser.add_argument(
        "--model",
        default="gru",
        choices=sorted(_MODEL_REGISTRY),
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
    args = parser.parse_args()

    # ── Data ──────────────────────────────────────────────────────────
    ds = Dataset.from_bab_experiment(args.dataset)
    train_ds, val_ds, test_ds = ds.train_val_test_split(train=0.7, val=0.15)
    print(f"Dataset: {ds.name}  ({len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test)")

    # ── Config ────────────────────────────────────────────────────────
    config_cls = MODEL_CONFIGS[args.model]
    config = config_cls()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.wandb:
        config.wandb_project = args.wandb
        config.wandb_run_name = f"{args.model}_{args.dataset}"

    # ── Train ─────────────────────────────────────────────────────────
    model_cls = _get_model_class(args.model)
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
