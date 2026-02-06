# Hybrid Modeling - System Identification Library

A modular Python library for system identification with classical, neural, and
physics-guided hybrid models.

## Project Structure

```text
hybrid_modeling/
├── src/
│   ├── data/
│   │   └── dataset.py              # Dataset loading and preprocessing
│   ├── models/
│   │   ├── base.py                 # Abstract base class
│   │   ├── narx.py                 # NARX with FROLS
│   │   ├── arima.py                # ARIMA/ARIMAX wrapper
│   │   ├── neural_network.py       # Feedforward NN
│   │   ├── neural_ode.py           # Neural ODE
│   │   ├── neural_sde.py           # Neural SDE
│   │   ├── hybrid_linear_beam.py   # J-R-K beam model
│   │   └── hybrid_nonlinear_cam.py # Nonlinear cam-bar-motor model
│   ├── validation/
│   │   └── metrics.py              # Evaluation metrics
│   └── visualization/
│       └── plots.py                # Plotting utilities
└── examples/
    └── model_comparison.py         # Full example
```

## Installation

```bash
pip install numpy scipy matplotlib torch torchsde tqdm statsmodels scikit-learn
```

## Datasets

### bab_datasets-compatible loader

`Dataset.from_bab_experiment(...)` follows the behavior described in
`https://github.com/helonayala/bab_datasets`:

- Supports experiment keys: `multisine_05`, `random_steps_01`, etc.
- Supports aliases: `05_multisine_01`, `03_random_steps_01`, etc.
- Uses local files under `data/` when available.
- Downloads missing files from the configured GitHub URL.
- Supports trigger-based start detection, reference-based end detection, and resampling.

```python
from src import Dataset

dataset = Dataset.from_bab_experiment(
    "multisine_05",
    preprocess=True,
    resample_factor=50,
    end_ref_tolerance=1e-8,
)

print(dataset)
print(Dataset.list_bab_experiments())
```

## Quick Start

```python
from src import (
    NARX, NeuralNetwork,
    HybridLinearBeam, HybridNonlinearCam,
    Dataset, Metrics,
)

# Load and preprocess dataset
train_data = Dataset.from_bab_experiment("multisine_05", preprocess=True, resample_factor=50)
dt = 1.0 / train_data.sampling_rate

# Baseline model
narx = NARX(nu=10, ny=10, poly_order=2, selection_criteria=10)
narx.fit(train_data.u, train_data.y)

# Hybrid model 1: linear beam dynamics
beam = HybridLinearBeam(sampling_time=dt, tau=1.0, estimate_delta=True)
beam.fit(train_data.u, train_data.y)

# Hybrid model 2: nonlinear cam-bar-motor dynamics
cam = HybridNonlinearCam(
    sampling_time=dt,
    R=0.02,
    r=0.005,
    e=0.002,
    L=0.12,
    I=5e-4,
    J=1e-4,
    k=25.0,
    delta=0.01,
    k_t=0.08,
    k_b=0.08,
    R_M=2.0,
    L_M=0.01,
    trainable_params=("J", "k", "delta", "k_t"),
    epochs=400,
)
cam.fit(train_data.u, train_data.y)

# Prediction
y_fr_beam = beam.predict(train_data.u, train_data.y[:2], mode="FR")
y_fr_cam = cam.predict(train_data.u, train_data.y[:2], mode="FR")

# Evaluation
Metrics.summary(train_data.y[2:], y_fr_beam, name="HybridLinearBeam")
Metrics.summary(train_data.y[2:], y_fr_cam, name="HybridNonlinearCam")
```

## Hybrid Models

### 1) `HybridLinearBeam`

Model equation:

\[
J\ddot{\theta} + R\dot{\theta} + K(\theta + \delta) = \tau V
\]

- Identifies normalized coefficients from derivative regression.
- Recovers physical parameters `J`, `R`, `K` (and derived `delta`).
- Supports `OSA` and `FR` prediction modes.

### 2) `HybridNonlinearCam`

Implements the nonlinear model combining:

- Eccentric cam geometry (`y(\theta)`, `\phi(\theta)`, `A(\theta)`, `B(\theta)`)
- Coupled cam-bar dynamics
- DC motor electrical subsystem

Detailed derivation and notation: `docs/hybrid_models.md`.

## Prediction Modes

All models support two prediction modes:

- `OSA`: One-step-ahead prediction with measured past outputs.
- `FR`: Free-run simulation from initial conditions.

```python
y_osa = model.predict(u, y, mode="OSA")
y_fr = model.predict(u, y_initial, mode="FR")
```

## Metrics

```python
from src.validation.metrics import Metrics, compare_models

metrics = Metrics.compute_all(y_true, y_pred)
Metrics.summary(y_true, y_pred, name="Model")

results = compare_models(y_true, {"NARX": y_narx, "Hybrid": y_hybrid})
```

Available metrics: `MSE`, `RMSE`, `MAE`, `R2`, `NRMSE`, `FIT%`.
