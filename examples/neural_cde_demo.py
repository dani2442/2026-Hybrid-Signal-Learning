#!/usr/bin/env python3
"""Demo script for Neural CDE model with multisine_05 dataset."""

import os
import sys
 
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src import Dataset, NeuralCDE
from src.validation.metrics import Metrics
from src.visualization.plots import plot_predictions
import matplotlib.pyplot as plt


def main():
    print("=" * 70)
    print("Neural CDE Demo - multisine_05 dataset")
    print("=" * 70)

    # Load and preprocess dataset
    print("\n1. Loading dataset...")
    dataset = Dataset.from_bab_experiment(
        "multisine_05",
        preprocess=True,
        resample_factor=50,
    )
    print(f"   {dataset}")
    print(f"   Sampling rate: {dataset.sampling_rate:.2f} Hz")
    print(f"   dt = {1.0 / dataset.sampling_rate:.4f} s")

    # Split into train/test
    print("\n2. Splitting dataset (80% train / 20% test)...")
    train_data, test_data = dataset.split(ratio=0.8)
    print(f"   Train: {len(train_data)} samples")
    print(f"   Test:  {len(test_data)} samples")

    # Initialize Neural CDE model
    print("\n3. Initializing Neural CDE model...")
    dt = 1.0 / dataset.sampling_rate
    model = NeuralCDE(
        hidden_dim=32,
        input_dim=2,
        hidden_layers=[64, 64],
        interpolation="cubic",
        solver="rk4",
        learning_rate=1e-3,
        epochs=200,
    )
    print(f"   {model}")

    # Train
    print("\n4. Training Neural CDE...")
    model.fit(train_data.u, train_data.y, verbose=True)
    print(f"   Final training loss: {model.training_loss_[-1]:.6f}")

    # Evaluate on test set
    print("\n5. Evaluating on test set...")

    # One-step-ahead prediction
    print("\n   5a. One-step-ahead (OSA) prediction...")
    y_pred_osa = model.predict_osa(test_data.u, test_data.y)
    metrics_osa = Metrics.compute_all(
        y_true=test_data.y[1:],  # Skip first point (no prediction)
        y_pred=y_pred_osa,
    )
    print(f"      OSA Metrics:")
    for key, val in metrics_osa.items():
        print(f"        {key:8s}: {val:.6f}")

    # Free-run simulation
    print("\n   5b. Free-run (FR) simulation...")
    y_initial = test_data.y[:max(model.max_lag, 1)]
    y_pred_fr = model.predict_free_run(test_data.u, y_initial, show_progress=True)
    metrics_fr = Metrics.compute_all(
        y_true=test_data.y,
        y_pred=y_pred_fr,
    )
    print(f"      FR Metrics:")
    for key, val in metrics_fr.items():
        print(f"        {key:8s}: {val:.6f}")

    # Visualize results
    print("\n6. Generating plots...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    osa_r2 = metrics_osa.get("R2", 0.0)
    fr_r2 = metrics_fr.get("R2", 0.0)

    # OSA prediction
    axes[0].plot(test_data.t[1:], test_data.y[1:], "b-", label="True", linewidth=1.5)
    axes[0].plot(test_data.t[1:], y_pred_osa, "r--", label="Predicted (OSA)", linewidth=1.5)
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Output")
    axes[0].set_title(f"One-Step-Ahead Prediction (R²={osa_r2:.4f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Free-run simulation
    axes[1].plot(test_data.t, test_data.y, "b-", label="True", linewidth=1.5)
    axes[1].plot(test_data.t, y_pred_fr, "r--", label="Predicted (FR)", linewidth=1.5)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Output")
    axes[1].set_title(f"Free-Run Simulation (R²={fr_r2:.4f})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("neural_cde_demo.png", dpi=150, bbox_inches="tight")
    print("   Saved plot to: neural_cde_demo.png")

    plt.show()

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
