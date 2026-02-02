# Control Helon - System Identification Library

A modular Python library for system identification with multiple model types.

## Project Structure

```
control_helon/
├── __init__.py              # Main exports
├── data/
│   ├── __init__.py
│   └── dataset.py           # Dataset loading and preprocessing
├── models/
│   ├── __init__.py
│   ├── base.py              # Abstract base class
│   ├── narx.py              # NARX with FROLS
│   ├── arima.py             # ARIMA/ARIMAX wrapper
│   ├── neural_network.py    # Feedforward NN
│   └── neural_ode.py        # Neural ODE
├── utils/
│   ├── __init__.py
│   ├── regression.py        # Regression matrix utilities
│   └── frols.py             # FROLS algorithm
├── validation/
│   ├── __init__.py
│   └── metrics.py           # Evaluation metrics
├── visualization/
│   ├── __init__.py
│   └── plots.py             # Plotting utilities
└── examples/
    ├── __init__.py
    └── model_comparison.py  # Full example
```

## Installation

```bash
pip install numpy scipy matplotlib torch tqdm statsmodels
```

## Quick Start

```python
from control_helon import (
    NARX, ARIMA, NeuralNetwork, NeuralODE,
    Dataset, Metrics,
    plot_predictions, plot_residuals,
)

# Load data
dataset = Dataset.from_helon_github("05_multisine_01.mat")
dataset = dataset.preprocess(start_idx=400, end_idx=57400, resample_factor=50)

# Train models
narx = NARX(nu=10, ny=10, poly_order=2, selection_criteria=10)
narx.fit(dataset.u, dataset.y)

nn = NeuralNetwork(nu=5, ny=5, hidden_layers=[64, 64], epochs=50)
nn.fit(dataset.u, dataset.y)

# Predict
y_osa = narx.predict(dataset.u, dataset.y, mode="OSA")
y_fr = narx.predict(dataset.u, dataset.y[:10], mode="FR")

# Evaluate
Metrics.summary(dataset.y[10:], y_fr, name="NARX Free-Run")
```

## Models

### NARX (Nonlinear ARX)
Polynomial NARX model with FROLS term selection.

```python
model = NARX(nu=10, ny=10, poly_order=2, selection_criteria=0.01)
```

### ARIMA
Wrapper around statsmodels ARIMA for time series with exogenous inputs.

```python
model = ARIMA(order=(5, 0, 2), nu=1)
```

### NeuralNetwork
Feedforward neural network for NARX-style modeling.

```python
model = NeuralNetwork(nu=5, ny=5, hidden_layers=[64, 64, 64], epochs=100)
```

### NeuralODE
Continuous-time neural ODE for dynamical systems.

```python
model = NeuralODE(state_dim=1, input_dim=1, hidden_layers=[32, 32], solver="rk4")
```

## Prediction Modes

All models support two prediction modes:
- **OSA (One-Step-Ahead)**: Uses actual past outputs for each prediction
- **FR (Free-Run)**: Uses only initial conditions, simulating forward

```python
y_osa = model.predict(u, y, mode="OSA")
y_fr = model.predict(u, y_initial, mode="FR")
```

## Metrics

```python
from control_helon.validation.metrics import Metrics, compare_models

# Single model
metrics = Metrics.compute_all(y_true, y_pred)
Metrics.summary(y_true, y_pred, name="Model Name")

# Compare multiple
results = compare_models(y_true, {"NARX": y_narx, "NN": y_nn})
```

Available metrics: MSE, RMSE, MAE, R², NRMSE, FIT%
