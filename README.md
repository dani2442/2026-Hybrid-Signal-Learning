# Hybrid Modeling

Hybrid Modeling is a system identification library that combines classical models, discrete-time machine learning models, neural continuous-time models, and physics-guided hybrids under one API.

## What You Get

- A shared interface for model fitting and prediction.
- One-step-ahead (OSA) and free-run (FR) prediction modes.
- Core families for interpretability, speed, and nonlinear expressiveness.
- Continuous-time training with `torchdiffeq` / `torchsde`.
- Reproducible benchmarking utilities.
- Built-in loaders for BAB datasets, including `random_steps_03` and `random_steps_04`.

## Unified API

Every model follows `BaseModel`:

- `fit(u, y)`
- `predict_osa(u, y)`
- `predict_free_run(u, y_initial)`
- `predict(u, y, mode="OSA" | "FR")`

```python
from src import NARX

model = NARX(max_lag=10)
model.fit(u_train, y_train)

y_osa = model.predict_osa(u_test, y_test)
y_fr = model.predict_free_run(u_test, y_test[: model.max_lag])
```

## Installation

Python requirement: `>=3.13`

Option 1 (recommended with `uv`):

```bash
uv sync
```

Option 2 (pip):

```bash
pip install -e .
```

Main dependencies:

- `numpy`, `scipy`, `matplotlib`
- `scikit-learn`, `statsmodels`
- `torch`, `torchdiffeq`, `torchsde`, `torchcde`
- `tqdm`, `wandb`

## Dataset Loading

Use the built-in dataset helper:

```python
from src import Dataset

print(Dataset.list_bab_experiments())
ds = Dataset.from_bab_experiment("multisine_05", preprocess=True, resample_factor=50)
print(ds)
```

Available experiment keys:

- `rampa_positiva`
- `rampa_negativa`
- `random_steps_01`
- `random_steps_02`
- `random_steps_03`
- `random_steps_04`
- `swept_sine`
- `multisine_05`
- `multisine_06`

## Mathematical Notation

The equations below use one consistent style:

- Discrete index: `k`
- Continuous time: `t`
- Input: `u_k` or `u(t)`
- Output: `y_k` or `y(t)`
- State: `x_k` or `x(t)`
- Position and velocity states: `theta`, `omega`

## Core Models (15)

The library includes 15 core models across four families.

### 1) Classical / Statistical Models

#### 1.1 NARX

$$
y_k = \sum_{i=1}^{M} \theta_i \phi_i\left(
y_{k-1}, \ldots, y_{k-n_y},
u_{k-1}, \ldots, u_{k-n_u}
\right) + e_k
$$

Polynomial terms are selected with FROLS.

#### 1.2 ARIMA(X)

$$
\Phi(B) (1 - B)^d y_k = \Theta(B) e_k + \beta u_k
$$

`B` is the backshift operator.

#### 1.3 Exponential Smoothing (Holt-Winters)

$$
\hat{y}_{k+1|k} = \alpha y_k + (1-\alpha)\hat{y}_{k|k-1}
$$

Extended with optional trend and seasonality terms.

### 2) Machine Learning Models (Discrete-Time)

#### 2.1 Random Forest

$$
y_k = \frac{1}{T}\sum_{t=1}^{T} f_t(x_k), \quad
x_k = [y_{k-1}, \ldots, y_{k-n_y}, u_{k-1}, \ldots, u_{k-n_u}]
$$

#### 2.2 Neural Network (MLP)

$$
y_k = W_L \sigma\left(\cdots \sigma(W_1 x_k + b_1)\cdots\right) + b_L
$$

#### 2.3 GRU

$$
z_k = \sigma(W_z[h_{k-1}, x_k]), \quad
r_k = \sigma(W_r[h_{k-1}, x_k])
$$

$$
\tilde{h}_k = \tanh(W_h[r_k \odot h_{k-1}, x_k]), \quad
h_k = (1-z_k)\odot h_{k-1} + z_k \odot \tilde{h}_k
$$

$$
y_k = W_o h_k + b_o
$$

#### 2.4 LSTM

$$
f_k = \sigma(W_f[h_{k-1}, x_k] + b_f), \quad
i_k = \sigma(W_i[h_{k-1}, x_k] + b_i)
$$

$$
\tilde{c}_k = \tanh(W_c[h_{k-1}, x_k] + b_c), \quad
c_k = f_k \odot c_{k-1} + i_k \odot \tilde{c}_k
$$

$$
o_k = \sigma(W_o[h_{k-1}, x_k] + b_o), \quad
h_k = o_k \odot \tanh(c_k), \quad
y_k = W_y h_k + b_y
$$

#### 2.5 TCN

$$
y_k = W_o \mathrm{ResBlock}_L\left(\cdots \mathrm{ResBlock}_1(X)\cdots\right)\Big|_{t=\mathrm{last}}
$$

Each residual block uses causal dilated convolutions.

#### 2.6 Mamba (Selective State Space Model)

Continuous form:

$$
\dot{x}(t) = A x(t) + B(u(t))u(t), \quad
y(t) = C(u(t))x(t) + D u(t)
$$

Input-dependent discretization:

$$
\bar{A}_k = \exp(\Delta_k A), \quad
\bar{B}_k = \Delta_k B_k, \quad
x_k = \bar{A}_k x_{k-1} + \bar{B}_k u_k
$$

$$
y_k = \bar{C}_k x_k + D u_k
$$

### 3) Neural Continuous-Time Models

#### 3.1 Neural ODE

$$
\dot{x}(t) = f_{\theta}(x(t), u(t)), \quad x(0) = x_0
$$

#### 3.2 Neural SDE

$$
dx(t) = f_{\theta}(x(t), u(t))dt + g_{\phi}(x(t), u(t))dW_t
$$

#### 3.3 Neural CDE

$$
\dot{z}(t) = f_{\theta}(z(t))\dot{X}(t), \quad z(t_0) = z_0
$$

`X(t)` is a continuous interpolation of observed signals.

### 4) Physics-Guided Hybrid Models

#### 4.1 Hybrid Linear Beam

$$
J\ddot{\theta} + R\dot{\theta} + K(\theta + \delta) = \tau V
$$

#### 4.2 Hybrid Nonlinear Cam

$$
J_{\mathrm{eff}}(\theta)\ddot{\theta} =
\tau_{\mathrm{motor}}(V, \dot{\theta})
- k\big(y(\theta)-\delta\big)A(\theta)
- B(\theta)\dot{\theta}^{2}
$$

#### 4.3 UDE (Universal Differential Equation)

$$
\dot{\theta} = \omega
$$

$$
\dot{\omega} = \frac{\tau V - R\omega - K(\theta+\delta)}{J} + r_{\phi}(\omega)
$$

In this implementation, the residual network is applied to `omega`.

## Core Model Summary

| Model | Family | Time Domain | Stateful | Physics Prior |
| --- | --- | --- | --- | --- |
| NARX | Classical | Discrete | No | No |
| ARIMA(X) | Classical | Discrete | No | No |
| Exponential Smoothing | Classical | Discrete | No | No |
| Random Forest | ML | Discrete | No | No |
| Neural Network (MLP) | ML | Discrete | No | No |
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

## Additional Exported Models

In addition to the 15 core models, the package also exports:

- Physics ODE wrappers: `LinearPhysics`, `StribeckPhysics`
- 2D black-box NODE variants: `VanillaNODE2D`, `StructuredNODE`, `AdaptiveNODE`
- 2D black-box NCDE variants: `VanillaNCDE2D`, `StructuredNCDE`, `AdaptiveNCDE`
- 2D black-box NSDE variants: `VanillaNSDE2D`, `StructuredNSDE`, `AdaptiveNSDE`

These models follow the same `fit / predict_osa / predict_free_run` pattern.

## Quick Start

### 1) Load and split one dataset

```python
from src import Dataset

ds = Dataset.from_bab_experiment("multisine_05", preprocess=True, resample_factor=50)
train_ds, test_ds = ds.split(0.8)
```

### 2) Fit one model and evaluate OSA/FR

```python
from src import UDE

dt = 1.0 / ds.sampling_rate
model = UDE(
    sampling_time=dt,
    hidden_layers=[64, 64],
    learning_rate=1e-3,
    epochs=500,
    sequence_length=50,
    training_mode="subsequence",
)

model.fit(train_ds.u, train_ds.y, verbose=True)

y_osa = model.predict_osa(test_ds.u, test_ds.y)
y_fr = model.predict_free_run(test_ds.u, test_ds.y[: model.max_lag])
```

### 3) Run benchmark

Default benchmark:

```bash
python3 examples/benchmark.py
```

Custom benchmark:

```bash
python3 examples/benchmark.py \
  --datasets multisine_05,multisine_06 \
  --models narx,random_forest,neural_network,gru,lstm,tcn,mamba,neural_ode,neural_sde,neural_cde,linear_physics,stribeck_physics,ude \
  --resample-factor 50 \
  --train-ratio 0.8 \
  --output-json results/benchmark.json
```

Disable Weights and Biases logging:

```bash
python3 examples/benchmark.py --disable-wandb
```

## Weights and Biases Logging

When supported by a model, `fit(..., wandb_run=..., wandb_log_every=...)` logs:

- training loss
- gradient norm
- model-specific scalar diagnostics

## Repository Layout

```text
hybrid-modeling/
├── src/
│   ├── data/             # Dataset loaders and preprocessing
│   ├── models/           # Classical, ML, CT, and hybrid models
│   ├── benchmarking/     # Benchmark runner and case presets
│   ├── validation/       # Metrics
│   └── visualization/    # Plot helpers
├── examples/             # End-to-end scripts
├── docs/                 # Extra documentation
└── full_comparison.ipynb # Full comparison notebook
```

## Documentation

- `docs/benchmarking.md`
- `docs/hybrid_models.md`
- `docs/neural_cde.md`
- `docs/README.md`
