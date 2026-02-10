# Hybrid Modeling

System identification library that brings together classical models, neural continuous-time models, and physics-guided hybrid models under a single, consistent API.

All models implement the same `BaseModel` interface:

- `fit(u, y)`
- `predict_osa(u, y)` (one-step-ahead)
- `predict_free_run(u, y_initial)` (free-run / rollout)

---

## Features

- 15 models in 4 families (classical, ML discrete-time, neural continuous-time, physics-guided hybrids)
- Shared training / inference interface across families
- OSA and free-run prediction modes
- TorchSDE support for Neural ODE/SDE/CDE
- Optional Weights & Biases logging for CT and hybrid models
- Benchmark runner with reproducible presets

---

## Models

The library includes **15 models** organised in four families.

### 1) Classical / Statistical Models

#### 1.1 NARX — Nonlinear AutoRegressive with eXogenous inputs

\[
y(k) = \sum_{i} \theta_i \, \phi_i\!\bigl(y(k\!-\!1),\dots,y(k\!-\!n_y),\;u(k\!-\!1),\dots,u(k\!-\!n_u)\bigr) + e(k)
\]

where the \(\phi_i\) are polynomial basis functions up to order \(p\), and the terms are selected automatically via **FROLS** (Forward Regression with Orthogonal Least Squares).

| | |
|---|---|
| **Ventajas** | Interpretable, fast to fit, built-in term selection, no hyperparameter tuning needed |
| **Inconvenientes** | Limited by polynomial expressiveness; struggles with highly nonlinear or chaotic dynamics |

#### 1.2 ARIMA(X) — AutoRegressive Integrated Moving Average

\[
\Phi(B)\,(1-B)^d\, y(k) = \Theta(B)\, e(k) + \beta\, u(k)
\]

where \(B\) is the backshift operator, \(\Phi\) the AR polynomial of order \(p\), \(\Theta\) the MA polynomial of order \(q\), \(d\) the differencing order, and \(u\) the optional exogenous regressor.

| | |
|---|---|
| **Ventajas** | Solid statistical foundation; handles trends and non-stationarity; interpretable coefficients |
| **Inconvenientes** | Linear model; cannot capture nonlinear dynamics; no native multi-step exogenous support |

#### 1.3 Exponential Smoothing (Holt-Winters)

\[
\hat{y}(k+1) = \alpha\, y(k) + (1-\alpha)\,\hat{y}(k)
\]

Extended with additive/multiplicative trend (\(\beta\)) and seasonal (\(\gamma\)) components.

| | |
|---|---|
| **Ventajas** | Very fast; good for smooth, trending signals; minimal tuning |
| **Inconvenientes** | Purely univariate (ignores input \(u\)); linear; not a dynamical model |

---

### 2) Machine Learning Models (Discrete-Time)

All discrete-time ML models work with lagged regressors and/or sequences derived from \([y, u]\).

#### 2.1 Random Forest

\[
y(k) = \frac{1}{T}\sum_{t=1}^{T} f_t\!\bigl(\mathbf{x}(k)\bigr), \quad \mathbf{x}(k) = \bigl[y(k\!-\!1),\dots,y(k\!-\!n_y),\;u(k\!-\!1),\dots,u(k\!-\!n_u)\bigr]
\]

| | |
|---|---|
| **Ventajas** | Non-parametric; captures nonlinearities without feature engineering; resistant to overfitting |
| **Inconvenientes** | Stateless (no hidden memory); cannot extrapolate beyond training range; opaque |

#### 2.2 Neural Network (MLP)

\[
y(k) = W_L\,\sigma\!\bigl(\cdots\sigma(W_1\,\mathbf{x}(k) + b_1)\cdots\bigr) + b_L
\]

| | |
|---|---|
| **Ventajas** | Universal approximation; differentiable; can learn complex input–output maps |
| **Inconvenientes** | Stateless; needs normalisation and tuning; can overfit on small data |

#### 2.3 GRU — Gated Recurrent Unit

\[
z_k = \sigma(W_z [\mathbf{h}_{k-1}, \mathbf{x}_k]),\quad r_k = \sigma(W_r [\mathbf{h}_{k-1}, \mathbf{x}_k])
\]
\[
\tilde{\mathbf{h}}_k = \tanh(W_h [r_k \odot \mathbf{h}_{k-1}, \mathbf{x}_k]),\quad \mathbf{h}_k = (1-z_k)\odot \mathbf{h}_{k-1} + z_k \odot \tilde{\mathbf{h}}_k
\]
\[
y(k) = W_o\,\mathbf{h}_k + b_o
\]

Input is a sequence of \([y, u]\) pairs of length `max_lag`.

| | |
|---|---|
| **Ventajas** | Captures temporal dependencies; fewer parameters than LSTM; fast training |
| **Inconvenientes** | Can struggle with very long memory; sequential computation limits parallelism |

#### 2.4 LSTM — Long Short-Term Memory

\[
f_k = \sigma(W_f [\mathbf{h}_{k-1}, \mathbf{x}_k] + b_f)
\quad
i_k = \sigma(W_i [\mathbf{h}_{k-1}, \mathbf{x}_k] + b_i)
\]
\[
\tilde{\mathbf{c}}_k = \tanh(W_c [\mathbf{h}_{k-1}, \mathbf{x}_k] + b_c)
\quad
\mathbf{c}_k = f_k \odot \mathbf{c}_{k-1} + i_k \odot \tilde{\mathbf{c}}_k
\]
\[
o_k = \sigma(W_o [\mathbf{h}_{k-1}, \mathbf{x}_k] + b_o)
\quad
\mathbf{h}_k = o_k \odot \tanh(\mathbf{c}_k),
\qquad
y(k) = W_y\,\mathbf{h}_k + b_y
\]

| | |
|---|---|
| **Ventajas** | Strong long-range memory via cell state; well-established baseline |
| **Inconvenientes** | More parameters than GRU; sequential; slower to train |

#### 2.5 TCN — Temporal Convolutional Network

\[
y(k) = W_o\;\text{ResBlock}_L\!\bigl(\cdots\text{ResBlock}_1(\mathbf{X})\cdots\bigr)\Big|_{t=\text{last}}
\]

Each residual block applies two causal dilated 1-D convolutions:

\[
\text{ResBlock}_l:\quad
h = \text{ReLU}\!\bigl(\text{CausalConv}_{d=2^l}(x)\bigr), \quad
\text{out} = \text{ReLU}(h + \text{skip}(x))
\]

| | |
|---|---|
| **Ventajas** | Parallelisable (no recurrence); stable gradients; large receptive field with few layers |
| **Inconvenientes** | Fixed receptive field; no persistent hidden state across free-run steps; memory grows with sequence length |

#### 2.6 Mamba — Selective State Space Model (S6)

Continuous-time SSM with input-dependent discretisation:

\[
\mathbf{x}'(t) = \mathbf{A}\,\mathbf{x}(t) + \mathbf{B}(u)\,u(t), \qquad
y(t) = \mathbf{C}(u)\,\mathbf{x}(t) + D\,u(t)
\]

Discretised with a learned step size \(\Delta(u)\):

\[
\bar{\mathbf{A}} = e^{\Delta \cdot \mathbf{A}}, \quad
\bar{\mathbf{B}} = \Delta \cdot \mathbf{B}, \quad
\mathbf{x}_k = \bar{\mathbf{A}}\,\mathbf{x}_{k-1} + \bar{\mathbf{B}}\,u_k
\]

| | |
|---|---|
| **Ventajas** | Linear-time sequence processing; strong long-range dependencies; control-theoretic grounding |
| **Inconvenientes** | Pure-Python scan can be slow without fused kernels; more complex tuning; newer architecture |

---

### 3) Neural Continuous-Time Models

#### 3.1 Neural ODE

\[
\frac{d\mathbf{x}}{dt} = f_\theta(\mathbf{x}, u), \qquad \mathbf{x}(0) = \mathbf{x}_0
\]

The dynamics \(f_\theta\) are an MLP. Trajectories are obtained by numerical integration (Euler / RK4) via `torchsde`.

| | |
|---|---|
| **Ventajas** | Continuous-time; handles irregular sampling; adjoint gives constant memory in sequence length |
| **Inconvenientes** | Slow training (ODE solves per batch); stiff dynamics can be numerically challenging; deterministic |

#### 3.2 Neural SDE

\[
d\mathbf{x} = f_\theta(\mathbf{x}, u)\,dt + g_\phi(\mathbf{x}, u)\,dW_t
\]

Adds a learned diffusion term \(g_\phi\) (Itô integration via `torchsde`).

| | |
|---|---|
| **Ventajas** | Models stochasticity + uncertainty; principled aleatoric modelling |
| **Inconvenientes** | Harder optimisation (noisy gradients); diffusion can collapse; slower than Neural ODE |

#### 3.3 Neural CDE — Neural Controlled Differential Equation

\[
\frac{d\mathbf{z}}{dt} = f_\theta(\mathbf{z})\,\frac{dX}{dt}, \qquad
\mathbf{z}(0) = \zeta_\psi(X(t_0))
\]

where \(X(t)\) is a continuous interpolation (e.g. cubic spline) of the observed path \([t, u, y]\), and \(f_\theta(\mathbf{z})\) outputs a \((d_z \times d_X)\) matrix that multiplies the path derivative.

| | |
|---|---|
| **Ventajas** | Excellent for irregular time series; input-driven continuous dynamics; strong theory |
| **Inconvenientes** | Requires interpolation; typically slowest to train; most complex implementation |

---

### 4) Physics-Guided Hybrid Models

#### 4.1 Hybrid Linear Beam

Second-order beam dynamics with trainable physical parameters:

\[
J\,\ddot{\theta} + R\,\dot{\theta} + K\,(\theta + \delta) = \tau\, V
\]

Parameters \((J, R, K, \delta)\) are initialised (least-squares) and refined by gradient descent through the ODE integrator. Positivity is enforced via softplus reparameterisation, e.g. \(J = \text{softplus}(J_\text{raw})\).

| | |
|---|---|
| **Ventajas** | Interpretable parameters; data-efficient; good extrapolation within the physics regime |
| **Inconvenientes** | Assumes correct physics; limited ability to capture unmodelled nonlinearities |

#### 4.2 Hybrid Nonlinear Cam

Full nonlinear cam-bar-motor mechanism:

\[
J_\text{eff}(\theta)\,\ddot{\theta} =
\tau_\text{motor}(V, \dot{\theta})
- k\,(y(\theta) - \delta)\cdot A(\theta)
- B(\theta)\,\dot{\theta}^2
\]

with geometry terms \(y(\theta), A(\theta), B(\theta)\) derived from the cam profile. Selected parameters (e.g. \(J, k, \delta, k_t\)) are trainable; the rest may come from CAD/measurements.

| | |
|---|---|
| **Ventajas** | High physical fidelity; few trainable DOFs; robust under distribution shift |
| **Inconvenientes** | Requires detailed physics; can be stiff (needs sub-stepping); model-specific |

#### 4.3 UDE — Universal Differential Equation

Known physics prior + neural residual:

\[
\frac{d\theta}{dt} = \omega, \qquad
\frac{d\omega}{dt} =
\underbrace{\frac{\tau V - R\omega - K(\theta + \delta)}{J}}_{\text{physics}}
+
\underbrace{f_\text{nn}(\theta, \omega, V)}_{\text{neural residual}}
\]

Both the physical parameters \((J, R, K, \delta)\) and NN weights are trained jointly. The residual is typically initialised with small weights so physics dominates early.

| | |
|---|---|
| **Ventajas** | Physics inductive bias + NN flexibility; interpretable physics parameters |
| **Inconvenientes** | NN can “absorb” physics without regularisation; training cost similar to Neural ODE; needs a physics prior |

---

## Model Summary

| Model | Family | Time | Stateful | Physics | Params (typical) |
|-------|--------|------|----------|---------|------------------|
| NARX | Classical | Discrete | ✗ | ✗ | ~50 |
| ARIMA(X) | Classical | Discrete | ✗ | ✗ | ~5 |
| Exp. Smoothing | Classical | Discrete | ✗ | ✗ | ~3 |
| Random Forest | ML | Discrete | ✗ | ✗ | depends (trees/depth) |
| Neural Network (MLP) | ML | Discrete | ✗ | ✗ | ~10K |
| GRU | ML | Discrete | ✔ | ✗ | ~50K |
| LSTM | ML | Discrete | ✔ | ✗ | ~50K |
| TCN | ML | Discrete | ✗ | ✗ | ~87K |
| Mamba | ML | Discrete | ✔ | ✗ | ~50K |
| Neural ODE | Neural CT | Continuous | ✔ | ✗ | ~10K |
| Neural SDE | Neural CT | Continuous | ✔ | ✗ | ~20K |
| Neural CDE | Neural CT | Continuous | ✔ | ✗ | ~15K |
| Hybrid Linear Beam | Hybrid | Continuous | ✔ | ✔ | 4 |
| Hybrid Nonlinear Cam | Hybrid | Continuous | ✔ | ✔ | ~4 (trainable subset) |
| UDE | Hybrid | Continuous | ✔ | ✔ | ~10K + 4 |

---

## Installation

This repository currently assumes a source checkout (no published wheel). Install dependencies:

```bash
pip install numpy scipy matplotlib torch torchsde tqdm statsmodels scikit-learn wandb
