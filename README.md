# Hybrid Modeling

System identification library combining classical models, discrete-time ML,
neural ODE/SDE/CDE, and physics-informed hybrids under a single API.

## Features

- **Unified interface** — every model exposes `fit()` → `predict(mode="OSA"|"FR")`.
- **27 models** across 4 families: classical, discrete-time ML, neural continuous-time, and physics-guided hybrids.
- **Config dataclasses** — typed, serialisable hyperparameters per model with registry lookup.
- **Save / Load** — PyTorch-standard checkpoints with automatic class resolution.
- **W&B logging** — opt-in via `config.wandb_project`; no-op when omitted.
- **Benchmarking** — built-in runner for reproducible multi-model comparisons.
- **Single SDE backend** — all continuous-time integration uses `torchsde` (ODE models use zero diffusion).

## Requirements

- Python **≥ 3.13**
- CUDA-capable GPU recommended for neural ODE/SDE/CDE models (CPU works but is significantly slower)

## Installation

```bash
# recommended (uses uv lockfile)
uv sync

# or with pip
pip install -e .
```

> **GPU/CUDA:** If `torch` was installed without CUDA support, reinstall it
> following the [PyTorch guide](https://pytorch.org/get-started/locally/).
> The `device` config field defaults to `"auto"`, which selects CUDA when
> available and falls back to CPU.

## Quick Start

```python
from src import Dataset, GRU, Metrics
from src.config import GRUConfig

# 1. Load data
ds = Dataset.from_bab_experiment("multisine_05")
train, val, test = ds.train_val_test_split(train=0.7, val=0.15)

# 2. Configure & train
cfg = GRUConfig(epochs=200, hidden_size=64)
model = GRU(cfg)
model.fit(train.arrays, val_data=val.arrays)

# 3. Evaluate
y_osa = model.predict(test.u, test.y, mode="OSA")
y_fr  = model.predict(test.u, test.y, mode="FR")
print(Metrics.compute_all(test.y, y_fr))

# 4. Save / load
model.save("checkpoints/gru.pt")

from src.models.base import load_model
model = load_model("checkpoints/gru.pt")
```

## Example Scripts

| Script | Purpose |
|--------|---------|
| `examples/train_single.py` | Train one model, evaluate, save checkpoint |
| `examples/train_all.py` | Train multiple models and compare metrics |
| `examples/load_and_test.py` | Load a checkpoint and evaluate on test data |

```bash
# Train a GRU with defaults
python examples/train_single.py --model gru

# Train all default models with W&B logging
python examples/train_all.py --wandb my-project

# Load and test a saved model
python examples/load_and_test.py checkpoints/gru_multisine_05.pt
```

## Unified Model API

Every model inherits from `BaseModel`:

```python
class BaseModel:
    def __init__(self, config: BaseConfig): ...
    def fit(self, train_data, val_data=None) -> self: ...
    def predict(self, u, y, mode="OSA"|"FR") -> np.ndarray: ...
    def save(self, path): ...

    @classmethod
    def load(cls, path) -> BaseModel: ...
```

- `train_data` / `val_data` are `(u, y)` tuples (use `dataset.arrays`).
- Configs are Python dataclasses inheriting `BaseConfig`.

## Configuration

```python
from src.config import GRUConfig, MODEL_CONFIGS

# Create with defaults
cfg = GRUConfig()

# Override fields
cfg = GRUConfig(epochs=500, hidden_size=128, wandb_project="my-project")

# Serialise / deserialise
d = cfg.to_dict()
cfg2 = GRUConfig.from_dict(d)

# Registry lookup
cfg_cls = MODEL_CONFIGS["gru"]  # → GRUConfig
```

Shared fields on every config: `nu`, `ny`, `learning_rate`, `epochs`,
`batch_size`, `verbose`, `device`, `seed`, `wandb_project`,
`wandb_run_name`, `wandb_log_every`, `early_stopping_patience`.

## Save & Load

```python
model.save("checkpoints/model.pt")

from src.models.base import load_model
model = load_model("checkpoints/model.pt")
```

Checkpoints store: class name, config dict, model state, extra state
(e.g. physics parameters), and training loss history.

## W&B Logging

Set `config.wandb_project` to enable. Metrics are logged per-epoch
automatically:

```python
cfg = GRUConfig(wandb_project="sysid", wandb_run_name="gru-run-1")
model = GRU(cfg)
model.fit(train.arrays, val_data=val.arrays)  # logs to W&B
```

## Dataset

```python
from src import Dataset

# List available experiments
Dataset.list_bab_experiments()

# Load with preprocessing
ds = Dataset.from_bab_experiment("multisine_05", preprocess=True, resample_factor=50)

# Split options
train, test = ds.split(0.8)
train, val, test = ds.train_val_test_split(train=0.7, val=0.15)

# Access (u, y) tuples
u, y = ds.arrays
```

Available experiments: `rampa_positiva`, `rampa_negativa`,
`random_steps_01`–`04`, `swept_sine`, `multisine_05`, `multisine_06`.

## Model Catalogue

### Model Summary

| Model | Family | Time | Stateful | Physics |
|-------|--------|------|----------|---------|
| NARX | Classical | Discrete | No | No |
| ARIMA(X) | Classical | Discrete | No | No |
| Exponential Smoothing | Classical | Discrete | No | No |
| Random Forest | ML | Discrete | No | No |
| Neural Network | ML | Discrete | No | No |
| GRU | ML | Discrete | Yes | No |
| LSTM | ML | Discrete | Yes | No |
| TCN | ML | Discrete | No | No |
| Mamba | ML | Discrete | Yes | No |
| Neural ODE | Neural CT | Continuous | Yes | No |
| Neural SDE | Neural CT | Continuous | Yes | No |
| Neural CDE | Neural CT | Continuous | Yes | No |
| Hybrid Linear Beam | Hybrid | Continuous | Yes | Yes |
| Hybrid Nonlinear Cam | Hybrid | Continuous | Yes | Yes |
| UDE | Hybrid | Continuous | Yes | Yes |

**Additional 2-D black-box variants** (all share a unified `_BlackboxODE2D` base):

| NODE | NSDE | NCDE |
|------|------|------|
| `VanillaNODE2D` | `VanillaNSDE2D` | `VanillaNCDE2D` |
| `StructuredNODE` | `StructuredNSDE` | `StructuredNCDE` |
| `AdaptiveNODE` | `AdaptiveNSDE` | `AdaptiveNCDE` |

**Physics ODE wrappers:** `LinearPhysics`, `StribeckPhysics`

All 27 models follow the same `fit` / `predict` / `save` / `load` interface.

### Mathematical Notation

- Discrete index: $k$, continuous time: $t$
- Input: $u_k$ or $u(t)$, output: $y_k$ or $y(t)$
- State: $x_k$ or $x(t)$, position/velocity: $\theta$, $\omega$

### 1) Classical / Statistical

#### NARX

$$
y_k = \sum_{i=1}^{M} \theta_i \phi_i(y_{k-1}, \ldots, y_{k-n_y}, u_{k-1}, \ldots, u_{k-n_u}) + e_k
$$

Polynomial terms selected with FROLS.

#### ARIMA(X)

$$
\Phi(B)(1 - B)^d y_k = \Theta(B) e_k + \beta u_k
$$

#### Exponential Smoothing (Holt-Winters)

$$
\hat{y}_{k+1|k} = \alpha y_k + (1 - \alpha)\hat{y}_{k|k-1}
$$

### 2) Machine Learning (Discrete-Time)

#### Random Forest

$$
y_k = \frac{1}{T}\sum_{t=1}^{T} f_t(x_k)
$$

#### Neural Network (MLP)

$$
y_k = W_L \sigma(\cdots \sigma(W_1 x_k + b_1)\cdots) + b_L
$$

#### GRU

$$
z_k = \sigma(W_z[h_{k-1}, x_k]),\quad r_k = \sigma(W_r[h_{k-1}, x_k])
$$

$$
\tilde{h}_k = \tanh(W_h[r_k \odot h_{k-1}, x_k]),\quad h_k = (1-z_k)\odot h_{k-1} + z_k \odot \tilde{h}_k
$$

#### LSTM

$$
f_k = \sigma(W_f[h_{k-1}, x_k]),\quad i_k = \sigma(W_i[h_{k-1}, x_k])
$$

$$
c_k = f_k \odot c_{k-1} + i_k \odot \tanh(W_c[h_{k-1}, x_k]),\quad h_k = \sigma(W_o[h_{k-1}, x_k]) \odot \tanh(c_k)
$$

#### TCN

Causal dilated 1-D convolutions with residual connections.

#### Mamba (Selective SSM)

$$
\dot{x}(t) = Ax(t) + B(u(t))u(t),\quad y(t) = C(u(t))x(t) + Du(t)
$$

Input-dependent discretisation: $\bar{A}_k = \exp(\Delta_k A)$.

### 3) Neural Continuous-Time

All continuous-time models use **torchsde** as the integration backend.
ODE models are expressed as SDEs with zero diffusion, giving a single
consistent solver interface across the library.

#### Neural ODE

$$
\dot{x}(t) = f_\theta(x(t), u(t)),\quad x(0) = x_0
$$

#### Neural SDE

$$
dx(t) = f_\theta(x(t), u(t))\,dt + g_\phi(x(t), u(t))\,dW_t
$$

#### Neural CDE

$$
\dot{z}(t) = f_\theta(z(t))\,\dot{X}(t),\quad z(t_0) = z_0
$$

### 4) Physics-Guided Hybrids

#### Hybrid Linear Beam

$$
J\ddot{\theta} + R\dot{\theta} + K(\theta + \delta) = \tau V
$$

#### Hybrid Nonlinear Cam

$$
J_{\mathrm{eff}}(\theta)\ddot{\theta} = \tau_{\mathrm{motor}}(V, \dot{\theta}) - k(y(\theta) - \delta)A(\theta) - B(\theta)\dot{\theta}^2
$$

#### UDE (Universal Differential Equation)

$$
\dot{\omega} = \frac{\tau V - R\omega - K(\theta + \delta)}{J} + r_\phi(\omega)
$$

## Architecture

```
src/
├── __init__.py            # Public re-exports
├── config.py              # Dataclass configs + MODEL_CONFIGS registry
├── logging.py             # WandbLogger wrapper
├── data/
│   └── dataset.py         # Dataset loaders & preprocessing
├── models/
│   ├── base.py            # BaseModel ABC, resolve_device(), PickleStateMixin
│   ├── sequence_base.py   # Shared base for GRU / LSTM / TCN / Mamba
│   ├── torchsde_utils.py  # SDE integration helpers, interp_u()
│   ├── blackbox_ode.py    # Unified ODE+SDE 2-D base (_BlackboxODE2D)
│   ├── blackbox_sde.py    # Re-exports NSDE classes (backward compat)
│   ├── blackbox_cde.py    # CDE-inspired 2-D variants
│   └── ...                # One file per model family
├── benchmarking/
│   └── runner.py          # BenchmarkRunner + helpers
├── utils/
│   ├── frols.py           # FROLS term selection for NARX
│   └── regression.py      # Regression utilities
├── validation/
│   └── metrics.py         # MSE, RMSE, R², NRMSE, FIT%
└── visualization/
    └── plots.py           # Plotting helpers
```

```
examples/
├── train_single.py        # Train one model
├── train_all.py           # Train & compare multiple models
└── load_and_test.py       # Load checkpoint & evaluate
```

## Documentation

- [Benchmarking guide](docs/benchmarking.md) — protocol, runner API, W&B integration
- [Hybrid models](docs/hybrid_models.md) — equations and notation for physics-guided models
- [Neural CDE](docs/neural_cde.md) — mathematical formulation and API reference

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Install PyTorch: `pip install torch` or follow the [official guide](https://pytorch.org/get-started/locally/). |
| Training is slow on CPU | Set `device="cuda"` in the config (or leave `"auto"` with a CUDA-capable GPU). |
| `RuntimeError: CUDA out of memory` | Reduce `batch_size` or `hidden_size` in the model config. |
| `ImportError: torchcde` | Run `pip install torchcde>=0.2.5`. |
| Checkpoint fails to load | Ensure the library version matches the one used to save. Use `load_model()` for automatic class resolution. |
