# Hybrid Modeling

System identification library with classical models, neural continuous-time models, and physics-guided hybrid models.

## Models

The library includes **15 models** organised in four families. All share the same `BaseModel` interface: `fit(u, y)`, `predict_osa(u, y)`, `predict_free_run(u, y_initial)`.

---

### 1 · Classical / Statistical Models

#### 1.1 NARX — Nonlinear AutoRegressive with eXogenous inputs

$$y(k) = \sum_{i} \theta_i \, \phi_i\!\bigl(y(k\!-\!1),\dots,y(k\!-\!n_y),\;u(k\!-\!1),\dots,u(k\!-\!n_u)\bigr) + e(k)$$

where the $\phi_i$ are polynomial basis functions up to order $p$ and the terms are selected automatically via the **FROLS** (Forward Regression with Orthogonal Least Squares) algorithm.

| | |
|---|---|
| **Ventajas** | Interpretable, fast to fit, built-in term selection, no hyperparameter tuning needed |
| **Inconvenientes** | Limited by polynomial expressiveness; struggles with highly nonlinear or chaotic dynamics |

#### 1.2 ARIMA(X) — AutoRegressive Integrated Moving Average

$$\Phi(B)\,(1-B)^d\, y(k) = \Theta(B)\, e(k) + \beta\, u(k)$$

where $B$ is the backshift operator, $\Phi$ the AR polynomial of order $p$, $\Theta$ the MA polynomial of order $q$, $d$ the differencing order, and $u$ the optional exogenous regressor.

| | |
|---|---|
| **Ventajas** | Solid statistical foundation; handles trends and non-stationarity; interpretable coefficients |
| **Inconvenientes** | Linear model; cannot capture nonlinear dynamics; no native multi-step exogenous support |

#### 1.3 Exponential Smoothing (Holt-Winters)

$$\hat{y}(k+1) = \alpha\, y(k) + (1-\alpha)\,\hat{y}(k)$$

Extended with additive/multiplicative trend ($\beta$) and seasonal ($\gamma$) components.

| | |
|---|---|
| **Ventajas** | Very fast; good for smooth, trending signals; minimal tuning |
| **Inconvenientes** | Purely univariate — ignores input $u$; linear; no dynamic modelling |

---

### 2 · Machine Learning Models (Discrete-Time)

#### 2.1 Random Forest

$$y(k) = \frac{1}{T}\sum_{t=1}^{T} f_t\!\bigl(\mathbf{x}(k)\bigr), \quad \mathbf{x}(k) = \bigl[y(k\!-\!1),\dots,y(k\!-\!n_y),\;u(k\!-\!1),\dots,u(k\!-\!n_u)\bigr]$$

Ensemble of $T$ decision trees, each trained on a bootstrap sample with random feature subsets.

| | |
|---|---|
| **Ventajas** | Non-parametric; captures nonlinearities without feature engineering; resistant to overfitting |
| **Inconvenientes** | Stateless (no hidden memory); cannot extrapolate beyond training range; opaque |

#### 2.2 Neural Network (MLP)

$$y(k) = W_L\,\sigma\!\bigl(\cdots\sigma(W_1\,\mathbf{x}(k) + b_1)\cdots\bigr) + b_L$$

Feedforward network with $L$ hidden layers and activation $\sigma$ (ReLU), operating on the same flat lag vector $\mathbf{x}(k)$ as the Random Forest.

| | |
|---|---|
| **Ventajas** | Universal approximation; differentiable; can learn complex input–output maps |
| **Inconvenientes** | Stateless; requires normalisation and careful hyperparameter tuning; prone to overfitting on small data |

#### 2.3 GRU — Gated Recurrent Unit

$$z_k = \sigma(W_z [\mathbf{h}_{k-1}, \mathbf{x}_k]),\quad r_k = \sigma(W_r [\mathbf{h}_{k-1}, \mathbf{x}_k])$$
$$\tilde{\mathbf{h}}_k = \tanh(W_h [r_k \odot \mathbf{h}_{k-1}, \mathbf{x}_k]),\quad \mathbf{h}_k = (1-z_k)\odot \mathbf{h}_{k-1} + z_k \odot \tilde{\mathbf{h}}_k$$
$$y(k) = W_o\,\mathbf{h}_k + b_o$$

Two gates (update $z$, reset $r$) modulate the hidden state $\mathbf{h}$. Input is a sequence of $[y, u]$ pairs of length `max_lag`.

| | |
|---|---|
| **Ventajas** | Captures temporal dependencies; fewer parameters than LSTM; fast training |
| **Inconvenientes** | Can struggle with very long-range dependencies; sequential processing limits parallelism |

#### 2.4 LSTM — Long Short-Term Memory

$$f_k = \sigma(W_f [\mathbf{h}_{k-1}, \mathbf{x}_k] + b_f) \quad\text{(forget gate)}$$
$$i_k = \sigma(W_i [\mathbf{h}_{k-1}, \mathbf{x}_k] + b_i) \quad\text{(input gate)}$$
$$\tilde{\mathbf{c}}_k = \tanh(W_c [\mathbf{h}_{k-1}, \mathbf{x}_k] + b_c)$$
$$\mathbf{c}_k = f_k \odot \mathbf{c}_{k-1} + i_k \odot \tilde{\mathbf{c}}_k \quad\text{(cell state)}$$
$$o_k = \sigma(W_o [\mathbf{h}_{k-1}, \mathbf{x}_k] + b_o) \quad\text{(output gate)}$$
$$\mathbf{h}_k = o_k \odot \tanh(\mathbf{c}_k), \qquad y(k) = W_y\,\mathbf{h}_k + b_y$$

Three gates (forget, input, output) plus a dedicated cell state $\mathbf{c}$ that can carry information across many time steps.

| | |
|---|---|
| **Ventajas** | Better long-range memory than GRU thanks to the cell state; well-established architecture |
| **Inconvenientes** | ~33% more parameters than GRU; sequential; slower to train |

#### 2.5 TCN — Temporal Convolutional Network

$$y(k) = W_o\;\text{ResBlock}_L\!\bigl(\cdots\text{ResBlock}_1(\mathbf{X})\cdots\bigr)\Big|_{t=\text{last}}$$

Each residual block applies two causal dilated 1-D convolutions:

$$\text{ResBlock}_l: \quad h = \text{ReLU}\!\bigl(\text{CausalConv}_{d=2^l}(x)\bigr), \quad \text{out} = \text{ReLU}(h + \text{skip}(x))$$

Dilation grows exponentially ($1, 2, 4, 8, \dots$) so the receptive field covers the full sequence with few layers.

| | |
|---|---|
| **Ventajas** | Fully parallelisable (no sequential recurrence); stable gradients; large receptive field with few parameters |
| **Inconvenientes** | Fixed receptive field (set by architecture); no hidden state to carry across free-run steps; memory grows with sequence length |

#### 2.6 Mamba — Selective State Space Model (S6)

Continuous-time SSM with input-dependent discretisation:

$$\mathbf{x}'(t) = \mathbf{A}\,\mathbf{x}(t) + \mathbf{B}(u)\,u(t), \qquad y(t) = \mathbf{C}(u)\,\mathbf{x}(t) + D\,u(t)$$

Discretised with a learned step size $\Delta(u)$:

$$\bar{\mathbf{A}} = e^{\Delta \cdot \mathbf{A}}, \quad \bar{\mathbf{B}} = \Delta \cdot \mathbf{B}, \quad \mathbf{x}_k = \bar{\mathbf{A}}\,\mathbf{x}_{k-1} + \bar{\mathbf{B}}\,u_k$$

$\mathbf{A}$ is diagonal (log-parameterised), and $\Delta$, $\mathbf{B}$, $\mathbf{C}$ are **input-dependent projections** (selective scan). The architecture includes a gated SiLU path and causal depth-wise convolution.

| | |
|---|---|
| **Ventajas** | Linear-time sequence processing; captures long-range dependencies; theoretically grounded in control theory |
| **Inconvenientes** | Pure-Python sequential scan is slow (no CUDA kernel); more complex to tune than RNNs; relatively new architecture |

---

### 3 · Neural Continuous-Time Models

#### 3.1 Neural ODE

$$\frac{d\mathbf{x}}{dt} = f_\theta(\mathbf{x}, u), \qquad \mathbf{x}(0) = \mathbf{x}_0$$

The dynamics $f_\theta$ are a multi-layer perceptron. The trajectory is obtained by numerical integration (Euler / RK4) via `torchsde`. Gradients flow through the solver for end-to-end training.

| | |
|---|---|
| **Ventajas** | Continuous-time; constant memory regardless of sequence length (adjoint); naturally handles irregular sampling |
| **Inconvenientes** | Slow training (ODE solves per batch); stiff dynamics can cause numerical issues; no stochasticity |

#### 3.2 Neural SDE

$$d\mathbf{x} = f_\theta(\mathbf{x}, u)\,dt + g_\phi(\mathbf{x}, u)\,dW_t$$

Extends Neural ODE with a learned diffusion term $g_\phi$ that models aleatoric uncertainty. Uses Itô integration via `torchsde`.

| | |
|---|---|
| **Ventajas** | Captures both deterministic dynamics and stochastic variability; principled uncertainty quantification |
| **Inconvenientes** | Harder to train (noisy gradients); diffusion can collapse to zero; slower than Neural ODE |

#### 3.3 Neural CDE — Neural Controlled Differential Equation

$$\frac{d\mathbf{z}}{dt} = f_\theta(\mathbf{z})\,\frac{dX}{dt}, \qquad \mathbf{z}(0) = \zeta_\psi(X(t_0))$$

where $X(t)$ is a continuous interpolation (cubic spline) of the observed input path $[t, u, y]$, and $f_\theta(\mathbf{z})$ outputs a matrix of shape $(d_z \times d_X)$ that multiplies the path derivative. The initial state $\mathbf{z}(0)$ is a learned projection of $X(t_0)$.

| | |
|---|---|
| **Ventajas** | Theoretically optimal for irregular time series; input-driven dynamics; continuous-time |
| **Inconvenientes** | Requires path interpolation (cubic splines); slowest to train; most complex implementation |

---

### 4 · Physics-Guided Hybrid Models

#### 4.1 Hybrid Linear Beam

Known second-order beam dynamics with trainable physical parameters:

$$J\,\ddot{\theta} + R\,\dot{\theta} + K\,(\theta + \delta) = \tau\, V$$

The parameters $(J, R, K, \delta)$ are initialised via least-squares and refined through gradient descent through the ODE integrator. Positivity is enforced via softplus reparameterisation: $J = \text{softplus}(J_\text{raw})$.

| | |
|---|---|
| **Ventajas** | Physically interpretable parameters; data-efficient; extrapolates well within the physics regime |
| **Inconvenientes** | Assumes the physics model is correct; cannot capture unmodelled nonlinearities |

#### 4.2 Hybrid Nonlinear Cam

Full nonlinear cam-bar-motor mechanism dynamics:

$$J_\text{eff}(\theta)\,\ddot{\theta} = \tau_\text{motor}(V, \dot{\theta}) - k\,(y(\theta) - \delta) \cdot A(\theta) - B(\theta)\,\dot{\theta}^2$$

where $y(\theta)$, $A(\theta)$, $B(\theta)$ are nonlinear geometry functions derived from the cam profile. Selected parameters (e.g., $J$, $k$, $\delta$, $k_t$) are trainable while the rest are fixed from CAD/measurements.

| | |
|---|---|
| **Ventajas** | Highest physical fidelity; few trainable DOFs; robust to distribution shift |
| **Inconvenientes** | Requires detailed physics knowledge; stiff ODE (needs sub-stepping); model-specific |

#### 4.3 UDE — Universal Differential Equation

Combines the known physics prior with a neural network residual:

$$\frac{d\theta}{dt} = \omega, \qquad \frac{d\omega}{dt} = \underbrace{\frac{\tau V - R\omega - K(\theta + \delta)}{J}}_{\text{physics}} + \underbrace{f_\text{nn}(\theta,\, \omega,\, V)}_{\text{neural residual}}$$

Both the physical parameters $(J, R, K, \delta)$ and the neural network weights are trained **jointly** through the ODE integrator. The residual is initialised with small weights (gain = 0.1) so that physics dominates at the start.

| | |
|---|---|
| **Ventajas** | Best of both worlds: physics provides inductive bias, NN captures model mismatch; interpretable physics params |
| **Inconvenientes** | Risk of the NN "absorbing" the physics if not regularised; training is as slow as Neural ODE; requires a physics prior |

---

### Model Summary Table

| Model | Family | Time | Stateful | Physics | Params (typical) |
|-------|--------|------|----------|---------|-------------------|
| NARX | Classical | Discrete | ✗ | ✗ | ~50 |
| ARIMA(X) | Classical | Discrete | ✗ | ✗ | ~5 |
| Exp. Smoothing | Classical | Discrete | ✗ | ✗ | ~3 |
| Random Forest | ML | Discrete | ✗ | ✗ | ~100K trees |
| Neural Network | ML | Discrete | ✗ | ✗ | ~10K |
| GRU | ML | Discrete | ✔ | ✗ | ~50K |
| LSTM | ML | Discrete | ✔ | ✗ | ~50K |
| TCN | ML | Discrete | ✗ | ✗ | ~87K |
| Mamba | ML | Discrete | ✔ | ✗ | ~50K |
| Neural ODE | Neural CT | Continuous | ✔ | ✗ | ~10K |
| Neural SDE | Neural CT | Continuous | ✔ | ✗ | ~20K |
| Neural CDE | Neural CT | Continuous | ✔ | ✗ | ~15K |
| Hybrid Linear | Hybrid | Continuous | ✔ | ✔ | 4 |
| Hybrid Nonlinear | Hybrid | Continuous | ✔ | ✔ | 4 |
| UDE | Hybrid | Continuous | ✔ | ✔ | ~10K + 4 |

## Installation

```bash
pip install numpy scipy matplotlib torch torchsde tqdm statsmodels scikit-learn wandb
```

## Repository Layout

```text
hybrid_modeling/
├── src/
│   ├── data/                  # Dataset loading and preprocessing
│   ├── models/                # Classical + neural + hybrid models
│   ├── benchmarking/          # Benchmark runner and presets
│   ├── validation/            # Metrics
│   └── visualization/         # Plots
├── examples/
│   ├── hybrid_models_demo.py  # Hybrid model demo
│   ├── model_comparison.py    # Legacy comparison example
│   └── benchmark.py           # Reproducible benchmark entrypoint
└── docs/
    ├── hybrid_models.md
    └── benchmarking.md
```

## Dataset Usage

```python
from src import Dataset

dataset = Dataset.from_bab_experiment(
    "multisine_05",
    preprocess=True,
    resample_factor=50,
)
print(dataset)
print(Dataset.list_bab_experiments())
```

## Hybrid Model Quick Start

```python
from src import Dataset, HybridLinearBeam, HybridNonlinearCam

dataset = Dataset.from_bab_experiment("multisine_05", preprocess=True, resample_factor=50)
dt = 1.0 / dataset.sampling_rate

beam = HybridLinearBeam(sampling_time=dt, tau=1.0, estimate_delta=True)
beam.fit(dataset.u, dataset.y)

cam = HybridNonlinearCam(
    sampling_time=dt,
    R=0.02, r=0.005, e=0.002, L=0.12, I=5e-4,
    J=1e-4, k=25.0, delta=0.01, k_t=0.08, k_b=0.08, R_M=2.0, L_M=0.01,
    trainable_params=("J", "k", "delta", "k_t"),
)
cam.fit(dataset.u, dataset.y)
```

## W&B Logging

TorchSDE-based models (NeuralODE, NeuralSDE, NeuralCDE) accept `wandb_run` in `fit(...)` and log per-epoch training signals:

- `train/loss`
- `train/grad_norm`
- decoded physical parameters (`params/*`) for hybrid models

Example:

```python
import wandb
from src import NeuralODE

run = wandb.init(project="hybrid-modeling")
model = NeuralODE(state_dim=1, input_dim=1, dt=0.05)
model.fit(u, y, wandb_run=run, wandb_log_every=1)
run.finish()
```

## Neural CDE Quick Start

Neural Controlled Differential Equations (CDEs) are a powerful continuous-time model for irregular time series:

```python
from src import Dataset, NeuralCDE

dataset = Dataset.from_bab_experiment("multisine_05", preprocess=True, resample_factor=50)

model = NeuralCDE(
    hidden_dim=32,
    input_dim=2,
    hidden_layers=[64, 64],
    interpolation="cubic",
    solver="dopri5",
    learning_rate=1e-3,
    epochs=100,
)
model.fit(dataset.u, dataset.y, verbose=True)

# Predict
y_pred_osa = model.predict_osa(dataset.u, dataset.y)
y_pred_fr = model.predict_free_run(dataset.u, dataset.y[:1])
```

Run the demo:
```bash
python3 examples/neural_cde_demo.py
```

## Benchmark (Recommended Workflow)

Run benchmark with default model set:

```bash
python3 examples/benchmark.py
```

Run custom datasets/models (including neural_cde):

```bash
python3 examples/benchmark.py \
  --datasets multisine_05,multisine_06 \
  --models narx,neural_ode,neural_cde,hybrid_linear_beam,hybrid_nonlinear_cam \
  --resample-factor 50 \
  --train-ratio 0.8 \
  --output-json results/benchmark.json
```

Disable W&B:

```bash
python3 examples/benchmark.py --disable-wandb
```

Benchmark output:

- JSON file with config + full result rows
- console leaderboard sorted by free-run `FIT%`
- optional W&B table (`benchmark/results_table`) and scalar logs

Details: `docs/benchmarking.md`.

