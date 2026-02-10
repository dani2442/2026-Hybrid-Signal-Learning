# Hybrid Modeling

System identification library that brings together classical models, neural continuous-time models, and physics-guided hybrid models under a single, consistent API.

All models implement the same `BaseModel` interface:

- `fit(u, y)`
- `predict_osa(u, y)` (one-step-ahead)
- `predict_free_run(u, y_initial)` (free-run / rollout)

---

## Features

- 15 models across 4 families (classical, discrete-time ML, neural continuous-time, physics-guided hybrids)
- Shared training and inference interface across families
- One-step-ahead (OSA) and free-run prediction modes
- `torchsde` support for continuous-time stochastic models
- Optional Weights & Biases logging for continuous-time and hybrid models
- Benchmark runner with reproducible presets

---

## Models

The library includes **15 models** organized into four families.

## 1) Classical / Statistical Models

### 1.1 NARX — Nonlinear AutoRegressive with eXogenous Inputs

$$
y(k)=\sum_i \theta_i\,\phi_i\!\big(y(k-1),\dots,y(k-n_y),\,u(k-1),\dots,u(k-n_u)\big)+e(k)
$$

where $\phi_i$ are polynomial basis functions up to order $p$, and terms are selected automatically via **FROLS** (Forward Regression with Orthogonal Least Squares).

| Pros | Cons |
|---|---|
| Interpretable, fast to fit, built-in term selection, little hyperparameter tuning | Limited polynomial expressiveness; struggles with highly nonlinear or chaotic dynamics |

### 1.2 ARIMA(X) — AutoRegressive Integrated Moving Average

$$
\Phi(B)(1-B)^d y(k)=\Theta(B)e(k)+\beta\,u(k)
$$

where $B$ is the backshift operator, $\Phi$ is the AR polynomial of order $p$, $\Theta$ is the MA polynomial of order $q$, $d$ is differencing order, and $u$ is an optional exogenous regressor.

| Pros | Cons |
|---|---|
| Strong statistical foundation, handles trends/non-stationarity, interpretable coefficients | Linear model, cannot capture nonlinear dynamics, limited native multi-step exogenous support |

### 1.3 Exponential Smoothing (Holt-Winters)

$$
\hat y(k+1)=\alpha\,y(k)+(1-\alpha)\,\hat y(k)
$$

Extended with additive/multiplicative trend ($\beta$) and seasonal ($\gamma$) components.

| Pros | Cons |
|---|---|
| Very fast, good for smooth/trending signals, minimal tuning | Univariate (ignores input $u$), linear, not a dynamical model |

---

## 2) Machine Learning Models (Discrete-Time)

All discrete-time ML models use lagged regressors and/or sequences derived from $[y,u]$.

### 2.1 Random Forest

$$
y(k)=\frac{1}{T}\sum_{t=1}^{T}f_t\!\big(\mathbf{x}(k)\big),\quad
\mathbf{x}(k)=\big[y(k-1),\dots,y(k-n_y),\,u(k-1),\dots,u(k-n_u)\big]
$$

| Pros | Cons |
|---|---|
| Non-parametric, captures nonlinearities without manual feature engineering, robust to overfitting | Stateless, weak extrapolation outside training range, low interpretability |

### 2.2 Neural Network (MLP)

$$
y(k)=W_L\,\sigma\!\big(\cdots \sigma(W_1\mathbf{x}(k)+b_1)\cdots\big)+b_L
$$

| Pros | Cons |
|---|---|
| Universal approximation, differentiable, learns complex input-output maps | Stateless, needs normalization and tuning, can overfit small datasets |

### 2.3 GRU — Gated Recurrent Unit

$$
z_k=\sigma\!\left(W_z[\mathbf{h}_{k-1},\mathbf{x}_k]\right),\quad
r_k=\sigma\!\left(W_r[\mathbf{h}_{k-1},\mathbf{x}_k]\right)
$$

$$
\tilde{\mathbf{h}}_k=\tanh\!\left(W_h[r_k\odot \mathbf{h}_{k-1},\mathbf{x}_k]\right),\quad
\mathbf{h}_k=(1-z_k)\odot \mathbf{h}_{k-1}+z_k\odot \tilde{\mathbf{h}}_k
$$

$$
y(k)=W_o\mathbf{h}_k+b_o
$$

Input is a sequence of $[y,u]$ pairs of length `max_lag`.

| Pros | Cons |
|---|---|
| Captures temporal dependencies, fewer parameters than LSTM, fast training | Can struggle with very long memory, sequential computation limits parallelism |

### 2.4 LSTM — Long Short-Term Memory

$$
f_k=\sigma(W_f[\mathbf{h}_{k-1},\mathbf{x}_k]+b_f),\quad
i_k=\sigma(W_i[\mathbf{h}_{k-1},\mathbf{x}_k]+b_i)
$$

$$
\tilde{\mathbf{c}}_k=\tanh(W_c[\mathbf{h}_{k-1},\mathbf{x}_k]+b_c),\quad
\mathbf{c}_k=f_k\odot \mathbf{c}_{k-1}+i_k\odot \tilde{\mathbf{c}}_k
$$

$$
o_k=\sigma(W_o[\mathbf{h}_{k-1},\mathbf{x}_k]+b_o),\quad
\mathbf{h}_k=o_k\odot \tanh(\mathbf{c}_k),\quad
y(k)=W_y\mathbf{h}_k+b_y
$$

| Pros | Cons |
|---|---|
| Strong long-range memory via cell state, widely used baseline | More parameters than GRU, sequential, slower training |

### 2.5 TCN — Temporal Convolutional Network

$$
y(k)=W_o\;\mathrm{ResBlock}_L\!\big(\cdots\mathrm{ResBlock}_1(\mathbf{X})\cdots\big)\Big|_{t=\text{last}}
$$

Each residual block applies two causal dilated 1D convolutions:

$$
\mathrm{ResBlock}_l:\;
h=\mathrm{ReLU}\!\left(\mathrm{CausalConv}_{d=2^l}(x)\right),\quad
\mathrm{out}=\mathrm{ReLU}(h+\mathrm{skip}(x))
$$

| Pros | Cons |
|---|---|
| Parallelizable (no recurrence), stable gradients, large receptive field with few layers | Fixed receptive field, no persistent hidden state across free-run steps, memory grows with sequence length |

### 2.6 Mamba — Selective State Space Model (S6)

Continuous-time SSM with input-dependent discretization:

$$
\mathbf{x}'(t)=\mathbf{A}\mathbf{x}(t)+\mathbf{B}(u)\,u(t),\qquad
y(t)=\mathbf{C}(u)\mathbf{x}(t)+D\,u(t)
$$

Discretized with learned step size $\Delta(u)$:

$$
\bar{\mathbf{A}}=e^{\Delta\mathbf{A}},\quad
\bar{\mathbf{B}}=\Delta\mathbf{B},\quad
\mathbf{x}_k=\bar{\mathbf{A}}\mathbf{x}_{k-1}+\bar{\mathbf{B}}u_k
$$

| Pros | Cons |
|---|---|
| Linear-time sequence processing, strong long-range dependencies, control-theoretic grounding | Slower without fused kernels, more complex tuning, newer architecture |

---

## 3) Neural Continuous-Time Models

### 3.1 Neural ODE

$$
\frac{d\mathbf{x}}{dt}=f_\theta(\mathbf{x},u),\qquad \mathbf{x}(0)=\mathbf{x}_0
$$

The dynamics $f_\theta$ are parameterized by an MLP. Trajectories are obtained by numerical integration.

| Pros | Cons |
|---|---|
| Continuous-time formulation, supports irregular sampling, principled dynamics modeling | Slower training due to ODE solves, stiff systems can be challenging, deterministic |

### 3.2 Neural SDE

$$
d\mathbf{x}=f_\theta(\mathbf{x},u)\,dt+g_\phi(\mathbf{x},u)\,dW_t
$$

Adds a learned diffusion term $g_\phi$ (It\^o integration via `torchsde`).

| Pros | Cons |
|---|---|
| Captures stochasticity and aleatoric uncertainty | Harder optimization (noisy gradients), possible diffusion collapse, slower than Neural ODE |

### 3.3 Neural CDE — Neural Controlled Differential Equation

$$
\frac{d\mathbf{z}}{dt}=f_\theta(\mathbf{z})\,\frac{dX}{dt},\qquad
\mathbf{z}(0)=\zeta_\psi\!\big(X(t_0)\big)
$$

where $X(t)$ is a continuous interpolation (e.g., cubic spline) of the observed path $[t,u,y]$, and $f_\theta(\mathbf{z})$ outputs a $(d_z\times d_X)$ matrix multiplied by path derivative.

| Pros | Cons |
|---|---|
| Strong for irregularly sampled time series, input-driven continuous dynamics, strong theory | Requires interpolation step, usually slowest to train, highest implementation complexity |

---

## 4) Physics-Guided Hybrid Models

### 4.1 Hybrid Linear Beam

Second-order beam dynamics with trainable physical parameters:

$$
J\ddot{\theta}+R\dot{\theta}+K(\theta+\delta)=\tau V
$$

Parameters $(J,R,K,\delta)$ are initialized (least squares) and refined with gradient descent through the ODE integrator. Positivity is enforced with softplus reparameterization (e.g., $J=\mathrm{softplus}(J_{\mathrm{raw}})$).

| Pros | Cons |
|---|---|
| Interpretable parameters, data-efficient, stronger extrapolation inside the physics regime | Depends on model correctness, limited flexibility for strong unmodeled nonlinearities |

### 4.2 Hybrid Nonlinear Cam

Full nonlinear cam-bar-motor mechanism:

$$
J_{\mathrm{eff}}(\theta)\ddot{\theta}=
\tau_{\mathrm{motor}}(V,\dot{\theta})
-k\,(y(\theta)-\delta)\,A(\theta)
-B(\theta)\,\dot{\theta}^2
$$

with geometry terms $y(\theta),A(\theta),B(\theta)$ derived from the cam profile. Selected parameters (e.g., $J,k,\delta,k_t$) are trainable; others may come from CAD or measurements.

| Pros | Cons |
|---|---|
| High physical fidelity, few trainable degrees of freedom, robust under distribution shift | Requires detailed physics, can be stiff (needs sub-stepping), model-specific |

### 4.3 UDE — Universal Differential Equation

Known physics prior plus neural residual:

$$
\frac{d\theta}{dt}=\omega,\qquad
\frac{d\omega}{dt}=
\underbrace{\frac{\tau V-R\omega-K(\theta+\delta)}{J}}_{\text{physics}}
+
\underbrace{f_{\mathrm{nn}}(\theta,\omega,V)}_{\text{neural residual}}
$$

Physical parameters $(J,R,K,\delta)$ and NN weights are trained jointly. Residual weights are often initialized small so physics dominates early training.

| Pros | Cons |
|---|---|
| Physics inductive bias plus neural flexibility, interpretable physical parameters | Without regularization, NN may absorb physics, training cost near Neural ODE, requires a useful prior |

---

## Model Summary

| Model | Family | Time | Stateful | Physics | Params (typical) |
|---|---|---|---|---|---|
| NARX | Classical | Discrete | No | No | ~50 |
| ARIMA(X) | Classical | Discrete | No | No | ~5 |
| Exponential Smoothing | Classical | Discrete | No | No | ~3 |
| Random Forest | ML | Discrete | No | No | Depends on trees/depth |
| Neural Network (MLP) | ML | Discrete | No | No | ~10K |
| GRU | ML | Discrete | Yes | No | ~50K |
| LSTM | ML | Discrete | Yes | No | ~50K |
| TCN | ML | Discrete | No | No | ~87K |
| Mamba | ML | Discrete | Yes | No | ~50K |
| Neural ODE | Neural CT | Continuous | Yes | No | ~10K |
| Neural SDE | Neural CT | Continuous | Yes | No | ~20K |
| Neural CDE | Neural CT | Continuous | Yes | No | ~15K |
| Hybrid Linear Beam | Hybrid | Continuous | Yes | Yes | 4 |
| Hybrid Nonlinear Cam | Hybrid | Continuous | Yes | Yes | ~4 (trainable subset) |
| UDE | Hybrid | Continuous | Yes | Yes | ~10K + 4 |

---

## Installation

This repository currently assumes a source checkout (no published wheel yet). Install dependencies with:

```bash
pip install numpy scipy matplotlib torch torchsde tqdm statsmodels scikit-learn wandb
```

(Optional, for editable local development)

```bash
pip install -e .
```
