#!/usr/bin/env python
"""
System Identification Model Comparison

This script demonstrates how to use the control_helon library to:
1. Load and preprocess data
2. Train multiple models (NARX, ARIMA, Neural Network, Neural ODE, ExpSmoothing, RandomForest, GRU)
3. Compare their performance using various metrics
4. Save results to images/ directory
"""

import os
import numpy as np

from control_helon import (
    NARX, ARIMA, NeuralNetwork, NeuralODE, ExponentialSmoothing, RandomForest, GRU,
    Dataset, Metrics,
    plot_predictions, plot_residuals, plot_signals, plot_model_comparison,
)
from control_helon.validation.metrics import compare_models

# Output directory for images
IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")


def main():
    """Main function demonstrating model comparison."""
    
    # Create images directory if it doesn't exist
    os.makedirs(IMAGES_DIR, exist_ok=True)
    print(f"Images will be saved to: {IMAGES_DIR}")
    
    # =========================================================================
    # 1. LOAD AND PREPROCESS DATA
    # =========================================================================
    print("=" * 60)
    print("1. Loading and preprocessing data...")
    print("=" * 60)
    
    # Load from Helon's GitHub repository
    dataset = Dataset.from_helon_github("05_multisine_01.mat")
    print(f"Loaded: {dataset}")
    
    # Preprocess: slice data and resample
    # Remove initial trigger period and resample by factor of 50
    dataset = dataset.preprocess(
        start_idx=400,  # Skip initial transient
        end_idx=57400,
        resample_factor=50,
    )
    print(f"Preprocessed: {dataset}")
    
    # Plot raw signals
    plot_signals(
        dataset.t, dataset.u, dataset.y, 
        title="Preprocessed Signals",
        save_path=os.path.join(IMAGES_DIR, "01_preprocessed_signals.png")
    )
    
    # =========================================================================
    # 2. DEFINE MODEL PARAMETERS
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. Setting up models...")
    print("=" * 60)
    
    # Common parameters
    ny = 10  # Output lags
    nu = 10  # Input lags
    
    # Initialize models
    models = {
        "NARX": NARX(nu=nu, ny=ny, poly_order=2, selection_criteria=10),
        "ARIMA": ARIMA(order=(5, 0, 2), nu=1),
        "RandomForest": RandomForest(nu=nu, ny=ny, n_estimators=50, max_depth=10),
        "NeuralNet": NeuralNetwork(
            nu=nu, ny=ny, hidden_layers=[64, 64, 64],
            epochs=100, learning_rate=1e-3
        ),
        "GRU": GRU(
            nu=nu, ny=ny, hidden_size=32, num_layers=2,
            epochs=100, learning_rate=1e-3
        ),
        "NeuralODE": NeuralODE(
            state_dim=1, input_dim=1, hidden_layers=[32, 32],
            epochs=100, dt=0.05, solver="rk4"
        ),
    }
    
    for name, model in models.items():
        print(f"  - {name}: {model}")
    
    # =========================================================================
    # 3. TRAIN MODELS
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. Training models...")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(dataset.u, dataset.y)
        print(f"  {name} fitted successfully!")
    
    # Print NARX summary
    print("\nNARX Model Summary:")
    models["NARX"].summary()
    
    # =========================================================================
    # 4. GENERATE PREDICTIONS
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. Generating predictions...")
    print("=" * 60)
    
    max_lag = max(ny, nu)
    y_initial = dataset.y[:max_lag]
    
    # One-Step-Ahead predictions
    osa_predictions = {}
    for name, model in models.items():
        print(f"  OSA prediction: {name}")
        osa_predictions[name] = model.predict(dataset.u, dataset.y, mode="OSA")
    
    # Free-Run predictions
    fr_predictions = {}
    for name, model in models.items():
        print(f"  Free-run simulation: {name}")
        fr_predictions[name] = model.predict(dataset.u, y_initial, mode="FR")
    
    # =========================================================================
    # 5. EVALUATE AND COMPARE MODELS
    # =========================================================================
    print("\n" + "=" * 60)
    print("5. Model Evaluation - One-Step-Ahead")
    print("=" * 60)
    
    y_true_aligned = dataset.y[max_lag:]
    t_aligned = dataset.t[max_lag:]
    
    osa_metrics = compare_models(y_true_aligned, osa_predictions)
    
    print("\n" + "=" * 60)
    print("5. Model Evaluation - Free-Run")
    print("=" * 60)
    
    fr_metrics = compare_models(y_true_aligned, fr_predictions)
    
    # =========================================================================
    # 6. VISUALIZE RESULTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("6. Generating plots...")
    print("=" * 60)
    
    # OSA Predictions
    plot_predictions(
        t_aligned, y_true_aligned, osa_predictions,
        title="One-Step-Ahead Predictions",
        save_path=os.path.join(IMAGES_DIR, "02_osa_predictions.png")
    )
    
    # Free-Run Predictions
    plot_predictions(
        t_aligned, y_true_aligned, fr_predictions,
        title="Free-Run Simulation",
        save_path=os.path.join(IMAGES_DIR, "03_free_run_predictions.png")
    )
    
    # Residuals
    plot_residuals(
        t_aligned, y_true_aligned, fr_predictions,
        title="Free-Run Prediction Residuals",
        save_path=os.path.join(IMAGES_DIR, "04_residuals.png")
    )
    
    # Model comparison bar chart
    plot_model_comparison(
        fr_metrics, metric_names=["R2", "FIT%", "NRMSE"],
        save_path=os.path.join(IMAGES_DIR, "05_model_comparison.png")
    )
    
    # OSA comparison bar chart
    plot_model_comparison(
        osa_metrics, metric_names=["R2", "FIT%", "NRMSE"],
        save_path=os.path.join(IMAGES_DIR, "06_osa_model_comparison.png")
    )
    
    print("\n" + "=" * 60)
    print("Model comparison complete!")
    print("=" * 60)
    
    return models, osa_metrics, fr_metrics


if __name__ == "__main__":
    main()
