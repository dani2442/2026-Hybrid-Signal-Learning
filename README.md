# Hybrid Signal Learning

> **System identification library**: classical physics, machine learning, Neural ODE/SDE, and physics-informed hybrid models — all under one unified framework.

This repository provides a comprehensive benchmarking suite for **nonlinear system identification** combining first-principles physics models with modern deep learning. Models range from classical linear ODEs to Neural SDEs and Mamba sequence models, all trained and evaluated through a single, consistent pipeline on the **BAB (Ball-and-Beam)** experimental datasets.

---

## Table of Contents

- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Models](#models)
  - [Physics-Based Models](#1-physics-based-models-no-neural-network)
  - [Black-Box Neural ODE Models](#2-black-box-neural-ode-models)
  - [Hybrid Models (Physics + NN)](#3-hybrid-models-physics--neural-network)
  - [Reservoir Computing](#4-reservoir-computing)
  - [Stochastic Models](#5-stochastic-models)
  - [Discrete-Time Sequence Models](#6-discrete-time-sequence-models)
  - [Feedforward Model](#7-feedforward-model)
- [NN Variants](#nn-variants)
- [Training Pipeline](#training-pipeline)
- [Usage](#usage)
  - [Quick Start (Smoke Test)](#quick-start-smoke-test)
  - [Full Benchmark Run](#full-benchmark-run)
  - [CLI Reference](#cli-reference)
- [Output Structure](#output-structure)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualization](#visualization)
- [Notebooks](#notebooks)
- [License](#license)

---

## Key Features

- **17 model families** spanning physics-only, black-box, hybrid, reservoir, stochastic, sequence, and feedforward paradigms.
- **4 NN capacity variants** (`compact`, `base`, `wide`, `deep`) applicable to all neural models.
- **Unified training loop** — a single `train_model()` function handles ODE, SDE, sequence, and feedforward models transparently.
- **Protocol-2 evaluation** — deterministic 50/50 temporal train/test split for core datasets; non-core datasets are test-only.
- **Multi-run statistical analysis** — train *N* independent seeds per model spec; aggregate mean/std metrics automatically.
- **Rich visualization** — prediction overlays, residual plots, raincloud distributions, ACF, spectra, and more.
- **Checkpoint & artifact system** — save/load model weights, export predictions as `.npz`, and persist full run metadata as JSON.

---

## Project Structure

```
hybrid_modeling/
├── pyproject.toml                    # Project metadata & dependencies
├── README.md                         # This file
│
├── bab_datasets/                     # Dataset loaders for BAB experiments
│   ├── __init__.py
│   ├── core.py                       # Dataset registry, loading, preprocessing, velocity estimation
│   └── video.py                      # Optional video synchronization utilities
│
├── data/                             # Raw .mat experiment files (auto-downloaded if missing)
│   ├── 01_rampa_positiva.mat
│   ├── 02_rampa_negativa.mat
│   ├── 03_random_steps_01..04.mat
│   ├── 04_swept_sine.mat
│   ├── 05_multisine_01.mat
│   └── 06_multisine_02.mat
│
├── hybrid_signal_learning/           # Core library
│   ├── __init__.py                   # Public API re-exports
│   ├── data.py                       # ExperimentData, Protocol-2 splits, rollout builder
│   ├── io.py                         # CSV/JSON/NPZ I/O, metric aggregation, run directory management
│   ├── plots.py                      # All visualization functions
│   ├── train.py                      # Unified training loop, ODE/SDE/sequence simulation, metrics
│   └── models/                       # Model definitions
│       ├── __init__.py               # Consolidated public API
│       ├── base.py                   # InterpNeuralODEBase, NN variants, MLP builder
│       ├── physics.py                # Linear ODE, Stribeck ODE
│       ├── blackbox.py               # BlackBox, Structured, Adaptive Neural ODEs
│       ├── hybrid.py                 # Joint & Frozen hybrid models (Linear + Stribeck)
│       ├── esn.py                    # Continuous-Time Echo State Network
│       ├── ude.py                    # Universal Differential Equation
│       ├── neural_sde.py             # Neural Stochastic Differential Equation
│       ├── sequence.py               # GRU, LSTM, TCN, Mamba sequence models
│       ├── feedforward.py            # Feedforward NN with lagged I/O features
│       └── factory.py                # build_model(), checkpoint save/load, iteration helpers
│
├── scripts/
│   └── run_bab_models.py            # Main CLI entry point for training & evaluation
│
└── notebooks/                        # Jupyter notebooks for exploration & analysis
    ├── All_blackboxes.ipynb
    ├── BAB_models_analysis.ipynb
    ├── BAB_NODE_models.ipynb
    ├── HYCO_BAB.ipynb
    ├── Notebook_with_NODE.ipynb
    └── Pedagogical_example_NODE.ipynb
```

---

## Installation

### Prerequisites

- **Python ≥ 3.13**
- (Optional) CUDA-enabled GPU for faster training

### Install

```bash
# Clone the repository
git clone <repository-url>
cd hybrid_modeling

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

# Install the package with all dependencies
pip install -e .

# (Optional) Install development dependencies
pip install -e ".[dev]"
```

### Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.10.0 | Neural network framework |
| `torchdiffeq` | — | ODE integration (`odeint`) |
| `torchsde` | ≥ 0.2.6 | SDE integration (`sdeint`) |
| `numpy` | ≥ 2.4.2 | Numerical computing |
| `scipy` | ≥ 1.16.1 | Signal processing, `.mat` loading |
| `matplotlib` | ≥ 3.10.8 | Visualization |
| `scikit-learn` | ≥ 1.8.0 | Metrics and preprocessing |
| `statsmodels` | ≥ 0.14.6 | Statistical analysis (ACF, etc.) |
| `tqdm` | ≥ 4.67.2 | Progress bars |
| `wandb` | ≥ 0.24.2 | Experiment tracking |
| `transformers` | ≥ 5.1.0 | Transformer utilities |
| `optuna` | ≥ 4.7.0 | Hyperparameter optimization |

---

## Datasets

The datasets come from a **Ball-and-Beam (BAB)** experimental testbed for nonlinear system identification. Each `.mat` file contains time-series recordings of input voltage (`u`), output position (`y`), reference signal (`y_ref`), and trigger information.

| Dataset Key | File | Description | Protocol-2 Role |
|---|---|---|---|
| `rampa_positiva` | `01_rampa_positiva.mat` | Positive ramp excitation | Test only |
| `rampa_negativa` | `02_rampa_negativa.mat` | Negative ramp excitation | Test only |
| `random_steps_01` | `03_random_steps_01.mat` | Random step sequence (variant 1) | **Train + Test** (50/50) |
| `random_steps_02` | `03_random_steps_02.mat` | Random step sequence (variant 2) | **Train + Test** (50/50) |
| `random_steps_03` | `03_random_steps_03.mat` | Random step sequence (variant 3) | **Train + Test** (50/50) |
| `random_steps_04` | `03_random_steps_04.mat` | Random step sequence (variant 4) | **Train + Test** (50/50) |
| `swept_sine` | `04_swept_sine.mat` | Frequency-swept sine wave | Test only |
| `multisine_05` | `05_multisine_01.mat` | Multi-sine excitation (variant 1) | **Train + Test** (50/50) |
| `multisine_06` | `06_multisine_02.mat` | Multi-sine excitation (variant 2) | **Train + Test** (50/50) |

### Data Preprocessing

- **Trigger-based cropping** — signals are trimmed using the trigger channel to remove idle regions.
- **Resampling** — default decimation factor of 50 reduces sampling rate for efficiency.
- **Velocity estimation** — $\dot{y}$ is computed via configurable methods: `central` differences (default), Savitzky–Golay filter, spline derivative, Butterworth filter, or total-variation regularization.
- **Auto-download** — missing `.mat` files are automatically fetched from the [sysid repository](https://github.com/helonayala/sysid).

---

## Models

All models predict a 2-D state vector $\mathbf{x} = [\theta, \dot{\theta}]^\top$ (position and velocity) from input $u$ (voltage). The table below summarizes all 17 model families:

### Complete Model Reference

| # | Model Key | Class | Type | Physics Prior | Neural Component | Description |
|---|---|---|---|---|---|---|
| 1 | `linear` | `LinearPhysODE` | ODE | Full | None | $J\ddot{\theta} + R\dot{\theta} + K(\theta + \delta) = \tau u$ |
| 2 | `stribeck` | `StribeckPhysODE` | ODE | Full | None | Linear + Stribeck friction: $F_c + (F_s - F_c)e^{-(\dot{\theta}/v_s)^2}$ |
| 3 | `blackbox` | `BlackBoxODE` | ODE | None | Full | $\dot{\mathbf{x}} = \text{NN}(\theta, \dot{\theta}, u)$ |
| 4 | `structured_blackbox` | `StructuredBlackBoxODE` | ODE | Kinematic | Partial | $\dot{\theta}_0 = \theta_1$; $\ddot{\theta} = \text{NN}(\theta, \dot{\theta}, u)$ |
| 5 | `adaptive_blackbox` | `AdaptiveBlackBoxODE` | ODE | Kinematic | Dual NN | Base dynamics NN + near-zero residual NN |
| 6 | `ct_esn` | `ContinuousTimeESN` | ODE | Kinematic | Reservoir | Echo State Network with augmented reservoir state |
| 7 | `hybrid_joint` | `HybridJointODE` | ODE | Linear (learnable) | Residual NN | $\ddot{\theta} = \text{physics}(\theta, \dot{\theta}, u) + \text{NN}(\theta, \dot{\theta}, u)$ |
| 8 | `hybrid_joint_stribeck` | `HybridJointStribeckODE` | ODE | Stribeck (learnable) | Residual NN | Stribeck physics + NN residual, jointly trained |
| 9 | `hybrid_frozen` | `HybridFrozenPhysODE` | ODE | Linear (frozen) | Residual NN | Pre-trained linear params frozen; only NN trains |
| 10 | `hybrid_frozen_stribeck` | `HybridFrozenStribeckPhysODE` | ODE | Stribeck (frozen) | Residual NN | Pre-trained Stribeck params frozen; only NN trains |
| 11 | `ude` | `UDEODE` | ODE | Linear SS (learnable) | Residual NN | $\ddot{\theta} = A\mathbf{x} + Bu + \text{NN}(\theta, \dot{\theta}, u)$ |
| 12 | `neural_sde` | `BlackBoxSDE` | SDE | Kinematic | Drift + Diffusion | $d\mathbf{x} = f(\mathbf{x}, u)\,dt + g(\mathbf{x}, u)\,dW_t$ |
| 13 | `gru` | `GRUSeqModel` | Sequence | None | Full | GRU encoder → linear decoder |
| 14 | `lstm` | `LSTMSeqModel` | Sequence | None | Full | LSTM encoder → linear decoder |
| 15 | `tcn` | `TCNSeqModel` | Sequence | None | Full | Temporal Convolutional Network with causal convolutions |
| 16 | `mamba` | `MambaSeqModel` | Sequence | None | Full | Selective State Space Model (Mamba architecture) |
| 17 | `feedforward_nn` | `FeedForwardNN` | Discrete | None | Full | MLP with lagged I/O features, autoregressive rollout |

---

### 1. Physics-Based Models (No Neural Network)

These models have **no learnable neural network component** — only physical parameters are optimized.

#### Linear ODE (`linear`)

Classical second-order system:

$$J\ddot{\theta} + R\dot{\theta} + K(\theta + \delta) = \tau \cdot u$$

Parameters: inertia $J$, damping $R$, stiffness $K$, offset $\delta$, gain $\tau$ (all learned in log-space for positivity).

#### Stribeck ODE (`stribeck`)

Extends the linear model with **Stribeck friction**:

$$J\ddot{\theta} + R\dot{\theta} + K(\theta + \delta) + F_{\text{str}}(\dot{\theta}) = \tau \cdot u$$

where $F_{\text{str}} = \left[F_c + (F_s - F_c)\exp\!\left(-\left(\frac{\dot{\theta}}{v_s}\right)^2\right)\right]\text{sgn}(\dot{\theta}) + b\dot{\theta}$

---

### 2. Black-Box Neural ODE Models

These learn dynamics entirely from data using neural networks integrated via `torchdiffeq.odeint`.

| Model | Kinematic Constraint | Architecture | Notes |
|---|---|---|---|
| `blackbox` | None | $\dot{\mathbf{x}} = \text{MLP}(\theta, \dot{\theta}, u) \to \mathbb{R}^2$ | Fully unconstrained |
| `structured_blackbox` | $\dot{\theta}_0 = \theta_1$ | $\ddot{\theta} = \text{MLP}(\theta, \dot{\theta}, u) \to \mathbb{R}^1$ | Improved sample efficiency |
| `adaptive_blackbox` | $\dot{\theta}_0 = \theta_1$ | Main MLP + zero-init residual MLP | Two-stream acceleration prediction |

---

### 3. Hybrid Models (Physics + Neural Network)

These combine a **physics backbone** with a **neural network residual correction**:

$$\ddot{\theta} = \underbrace{\ddot{\theta}_{\text{physics}}}_{\text{first-principles}} + \underbrace{\text{NN}(\theta, \dot{\theta}, u)}_{\text{learned residual}}$$

| Model | Physics Backbone | Params Trainable? | Notes |
|---|---|---|---|
| `hybrid_joint` | Linear ODE | Yes (jointly) | Physics + NN trained together |
| `hybrid_joint_stribeck` | Stribeck ODE | Yes (jointly) | Stribeck physics + NN residual |
| `hybrid_frozen` | Linear ODE | No (frozen) | Pre-trained linear params; only NN trains |
| `hybrid_frozen_stribeck` | Stribeck ODE | No (frozen) | Pre-trained Stribeck params; only NN trains |

#### Universal Differential Equation (`ude`)

A **state-space formulation** with learnable linear matrices:

$$\ddot{\theta} = A\mathbf{x} + Bu + \text{NN}(\theta, \dot{\theta}, u)$$

where $A \in \mathbb{R}^{1 \times 2}$ and $B \in \mathbb{R}^{1 \times 1}$ are learnable.

---

### 4. Reservoir Computing

#### Continuous-Time Echo State Network (`ct_esn`)

Augmented ODE state $\mathbf{z} = [\theta, \dot{\theta}, \mathbf{r}]$ where $\mathbf{r} \in \mathbb{R}^D$ is the reservoir hidden state:

$$\dot{\mathbf{r}} = \lambda\left(-\mathbf{r} + \tanh(W_{\text{res}}\mathbf{r} + W_{\text{in}}[\mathbf{x}, u])\right)$$

$$\ddot{\theta} = W_{\text{out}}[\mathbf{r}, \mathbf{x}, u]$$

The reservoir matrix $W_{\text{res}}$ is **fixed** (sparse, scaled to a target spectral radius); only $W_{\text{in}}$, $W_{\text{out}}$, and $\lambda$ are learned.

---

### 5. Stochastic Models

#### Neural SDE (`neural_sde`)

Models dynamics with **stochastic noise** via Ito SDE:

$$d\mathbf{x} = f(\mathbf{x}, u)\,dt + g(\mathbf{x}, u)\,dW_t$$

- **Drift** $f$: structured (kinematic constraint) MLP predicting acceleration.
- **Diffusion** $g$: small MLP producing state-dependent diagonal noise $[\sigma_{\text{pos}}, \sigma_{\text{vel}}]$.
- Integrated with `torchsde.sdeint` (Euler–Maruyama method).

---

### 6. Discrete-Time Sequence Models

These map the full input sequence $u(t)$ directly to the state trajectory $[\theta(t), \dot{\theta}(t)]$ in a **sequence-to-sequence** fashion.

| Model | Architecture | Key Feature |
|---|---|---|
| `gru` | Multi-layer GRU → Linear | Gated recurrent units |
| `lstm` | Multi-layer LSTM → Linear | Long short-term memory cells |
| `tcn` | Causal Conv1D residual blocks → Linear | Exponentially increasing dilation for large receptive field |
| `mamba` | Selective SSM blocks → Linear | Input-dependent state transitions (Mamba architecture) |

---

### 7. Feedforward Model

#### Feedforward NN (`feedforward_nn`)

A standard MLP operating in **discrete time** with lagged I/O features:

$$[\theta(k), \dot{\theta}(k)] = \text{MLP}\!\big(\theta(k\!-\!1), \dot{\theta}(k\!-\!1), u(k\!-\!1), \ldots, \theta(k\!-\!L), \dot{\theta}(k\!-\!L), u(k\!-\!L)\big)$$

- **Lag** $L = 10$ (default) — the feature vector has dimension $L \times 3$.
- At inference, predicted outputs are **fed back** as lagged features for autoregressive free-run simulation.

---

## NN Variants

All neural-network-based models support configurable capacity through **NN variants**:

| Variant | Hidden Dim | Depth (layers) | Dropout | Typical Use Case |
|---|---|---|---|---|
| `compact` | 64 | 2 | 0.05 | Fast prototyping, limited data |
| `base` | 128 | 3 | 0.05 | Default — balanced performance |
| `wide` | 256 | 3 | 0.05 | Higher capacity, same depth |
| `deep` | 128 | 5 | 0.05 | Deeper representations |

All MLPs use **SELU activation** with **AlphaDropout** for self-normalizing behavior.

---

## Training Pipeline

The training pipeline follows these steps:

```
┌──────────────────┐    ┌────────────────────┐    ┌──────────────────┐
│  Load Datasets   │───▶│  Protocol-2 Split   │───▶│  Build Rollout   │
│  (BAB .mat)      │    │  50/50 temporal      │    │  (concatenate)   │
└──────────────────┘    └────────────────────┘    └──────────────────┘
                                                          │
                        ┌────────────────────┐            ▼
                        │  For each model     │    ┌──────────────────┐
                        │  spec × N runs:     │◀───│  Compute Valid   │
                        │                     │    │  Start Indices   │
                        │  1. Build model     │    └──────────────────┘
                        │  2. Train (k-step)  │
                        │  3. Evaluate all DS  │
                        │  4. Save checkpoint  │
                        │  5. Save predictions │
                        └─────────┬───────────┘
                                  ▼
                        ┌────────────────────┐
                        │  Aggregate metrics  │
                        │  Save CSV / JSON    │
                        │  Plot convergence   │
                        └────────────────────┘
```

### Training Strategy

- **K-step prediction loss**: random windows of `k_steps=20` are sampled; models predict forward and MSE is computed against ground truth.
- **Position-only loss** (default): loss targets position channel only, which empirically improves generalization.
- **Optimizer**: Adam with `lr=0.01`.
- **Batch size**: 128 random start indices per epoch.
- **Epochs**: 1000 (default).

---

## Usage

### Quick Start (Smoke Test)

```bash
python scripts/run_bab_models.py --quick
```

This runs a minimal configuration:
- 1 training run (instead of 10)
- 60 epochs (instead of 1000)
- Only the `base` NN variant
- Subset of datasets: `multisine_05`, `multisine_06`, `random_steps_01`, `swept_sine`

### Full Benchmark Run

```bash
python scripts/run_bab_models.py \
    --models linear,stribeck,blackbox,structured_blackbox,hybrid_joint,ude,gru,mamba \
    --nn-variants base,wide,deep \
    --n-runs 10 \
    --epochs 1000 \
    --output-root results/bab_runs
```

### Train Specific Models Only

```bash
# Physics-only models
python scripts/run_bab_models.py --models linear,stribeck --n-runs 5

# Only hybrid models
python scripts/run_bab_models.py --models hybrid_joint,hybrid_frozen,ude --nn-variants base,wide

# Only sequence models
python scripts/run_bab_models.py --models gru,lstm,tcn,mamba --nn-variants compact,base
```

### CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--output-root` | `results/bab_runs` | Base folder for run outputs |
| `--run-name` | Auto (timestamp) | Custom name for the run folder |
| `--models` | All 17 models | Comma-separated model keys |
| `--nn-variants` | `base,wide,deep` | Comma-separated NN variant names |
| `--n-runs` | `10` | Independent training runs per model specification |
| `--epochs` | `1000` | Number of training epochs |
| `--lr` | `0.01` | Learning rate (Adam) |
| `--batch-size` | `128` | Batch size (random start indices per epoch) |
| `--k-steps` | `20` | Prediction horizon for training loss |
| `--obs-dim` | `2` | Observation dimension (position + velocity) |
| `--resample-factor` | `50` | Decimation factor for raw data |
| `--y-dot-method` | `central` | Velocity estimation method (`central`, `savgol`, `spline`, `butter`) |
| `--seed` | `1234` | Base random seed |
| `--device` | `auto` | Compute device (`auto`, `cpu`, `cuda`) |
| `--quick` | `false` | Enable smoke-test mode |
| `--include-datasets` | All | Comma-separated subset of dataset keys |
| `--save-predictions` | `true` | Save full rollout predictions as `.npz` |
| `--progress` / `--no-progress` | `true` | Enable/disable tqdm progress bars |

---

## Output Structure

Each run produces a structured output directory:

```
results/bab_runs/<run_name>/
├── metadata/
│   ├── config.json                    # Full run configuration
│   ├── best_model_ids_test_r2_pos.json  # Best model ID per family (by test R²)
│   └── model_registry.json            # Registry of all trained models
├── models/
│   ├── linear__physics__run00.pt      # Model checkpoints
│   ├── blackbox__base__run00.pt
│   ├── hybrid_joint__wide__run03.pt
│   └── ...
├── predictions/
│   ├── linear__physics__run00.npz     # Full rollout predictions per dataset
│   └── ...
├── tables/
│   ├── training_history.csv           # Epoch-by-epoch loss for all models
│   ├── metrics_long.csv               # Per-model, per-dataset, per-split metrics
│   └── metrics_aggregate.csv          # Mean ± std grouped by model family
└── plots/
    └── training_loss_curves.png       # Convergence plot for all models
```

---

## Evaluation Metrics

All models are evaluated on both train and test splits with the following metrics:

| Metric | Formula | Description |
|---|---|---|
| **RMSE (pos)** | $\sqrt{\frac{1}{N}\sum_i (y_i - \hat{y}_i)^2}$ | Root mean squared error for position |
| **RMSE (vel)** | Same, for velocity channel | Root mean squared error for velocity |
| **R² (pos)** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Coefficient of determination (position) |
| **R² (vel)** | Same, for velocity channel | Coefficient of determination (velocity) |
| **FIT% (pos)** | $100\left(1 - \frac{\lVert y - \hat{y}\rVert}{\lVert y - \bar{y}\rVert}\right)$ | NRMSE-based fit percentage (position) |
| **FIT% (vel)** | Same, for velocity channel | NRMSE-based fit percentage (velocity) |

---

## Visualization

The library provides a rich set of plotting functions in `hybrid_signal_learning/plots.py`:

| Function | Description |
|---|---|
| `plot_training_curves()` | Epoch-vs-loss convergence for all models |
| `plot_predictions()` | Measured vs. predicted (position + velocity) with train/test split marker |
| `plot_zoom_position()` | Three zoomed windows (start, middle, end) of position predictions |
| `plot_residuals()` | Time-series residual plots for position and velocity |
| `plot_raincloud_models()` | Raincloud (violin + box + scatter) of position residuals per model |
| `plot_y_vs_yhat()` | Scatter plot of measured vs. predicted position |
| `plot_acf()` | Autocorrelation function of residuals |
| `plot_spectra()` | Frequency-domain spectra of residuals |

Each plot function accepts an optional `save_path` argument to export figures as PNG files at 150 DPI.

---

## Notebooks

| Notebook | Description |
|---|---|
| `All_blackboxes.ipynb` | Comparison of all black-box model variants |
| `BAB_models_analysis.ipynb` | Post-hoc analysis of trained models with metrics tables and plots |
| `BAB_NODE_models.ipynb` | Neural ODE model exploration on BAB data |
| `HYCO_BAB.ipynb` | Hybrid continuous-time model experiments |
| `Notebook_with_NODE.ipynb` | Step-by-step Neural ODE workflow |
| `Pedagogical_example_NODE.ipynb` | Educational introduction to Neural ODEs |

---

## Programmatic API

```python
import hybrid_signal_learning as hsl

# Load datasets with Protocol-2 splits
data_map = hsl.load_protocol2_datasets(y_dot_method="central")

# Build a model
model = hsl.build_model("hybrid_joint", nn_variant="wide")

# Build training data
train_data = hsl.build_train_rollout_data(data_map)
tensors = hsl.to_tensor_bundle(t=train_data.t, u=train_data.u, y_sim=train_data.y_sim, device="cpu")
valid_starts = hsl.compute_valid_start_indices(train_data.segments, k_steps=20)

# Train
cfg = hsl.TrainConfig(epochs=500, lr=0.01, k_steps=20)
model, history = hsl.train_model(model=model, tensors=tensors, valid_start_indices=valid_starts, cfg=cfg)

# Evaluate on a dataset
results = hsl.evaluate_model_on_dataset(model=model, ds=data_map["multisine_05"], device="cpu")
print(results["metrics"]["test"])

# Save & reload checkpoint
hsl.save_model_checkpoint("model.pt", model=model, model_key="hybrid_joint", nn_variant="wide", run_idx=0, seed=42)
loaded_model, meta = hsl.load_model_checkpoint("model.pt")
```

---

## License

See the repository for license details.
