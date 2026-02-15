# Neural CDE

Neural Controlled Differential Equation model for continuous-time system identification.  
Reference: Kidger et al., *Neural Controlled Differential Equations for Irregular Time Series*, NeurIPS 2020. ([arXiv:2005.08926](https://arxiv.org/abs/2005.08926))

## Mathematical Formulation

### Governing Equation

$$
\mathrm{d}z(t) = f_\theta\!\bigl(z(t)\bigr)\;\mathrm{d}X(t), \qquad z(t_0) = z_0
$$

Equivalently, since $X$ is a differentiable spline:

$$
\frac{\mathrm{d}z}{\mathrm{d}t} = f_\theta\!\bigl(z(t)\bigr)\;\frac{\mathrm{d}X}{\mathrm{d}t}(t)
$$

| Symbol | Shape | Meaning |
|--------|-------|---------|
| $z(t)$ | $\mathbb{R}^{d_h}$ | Hidden (latent) state |
| $X(t)$ | $\mathbb{R}^{d_x}$ | Continuous control path from data |
| $f_\theta$ | $\mathbb{R}^{d_h} \to \mathbb{R}^{d_h \times d_x}$ | Learnable vector field (MLP) |
| $z_0$ | $\mathbb{R}^{d_h}$ | Learnable initial condition |

### Control Path

Discrete observations $\{(t_i, u_i, y_i)\}_{i=0}^{N}$ are interpolated into a continuous path:

$$
X(t) = \operatorname{CubicHermiteSpline}\!\bigl(\{(t_i,\; u_i,\; y_i)\}\bigr) \in \mathbb{R}^3
$$

### Readout

$$
\hat{y}(t) = W\,z(t) + b, \qquad W \in \mathbb{R}^{1 \times d_h}
$$

### Loss

$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}\bigl(y_i - \hat{y}(t_i)\bigr)^2
$$

### Comparison with Other Neural DE Models

| Model | Equation |
|-------|----------|
| Neural ODE | $\dfrac{\mathrm{d}z}{\mathrm{d}t} = f_\theta(z,\, u)$ |
| Neural SDE | $\mathrm{d}z = f_\theta(z,\, u)\,\mathrm{d}t + g_\theta(z)\,\mathrm{d}W_t$ |
| Neural CDE | $\mathrm{d}z = f_\theta(z)\,\mathrm{d}X$ |

The CDE couples input through $\mathrm{d}X$ rather than concatenating $u$ into the drift, making it naturally invariant to time reparameterisation and suited for irregular / missing data.

## Installation

`torchcde` is included in the project dependencies. No extra install needed
when using `uv sync` or `pip install -e .`.

## API

```python
from src import NeuralCDE
from src.config import NeuralCDEConfig

cfg = NeuralCDEConfig(
    hidden_dim=32,          # latent state dimension
    input_dim=2,            # non-time input channels
    hidden_layers=[64, 64], # vector-field MLP widths
    interpolation="cubic",  # "cubic" | "linear"
    solver="rk4",           # "rk4" | "euler" | "midpoint"
    learning_rate=1e-3,
    epochs=100,
    rtol=1e-4,              # adaptive solver tolerance
    atol=1e-5,
)
model = NeuralCDE(cfg)
```

| Method | Description |
|--------|-------------|
| `fit(train_data, val_data)` | Train on `(u, y)` tuple |
| `predict(u, y, mode="OSA"\|"FR")` | Unified prediction interface |
| `save(path)` / `load(path)` | Checkpoint persistence |

## Quick Start

```python
from src import Dataset, NeuralCDE
from src.config import NeuralCDEConfig

dataset = Dataset.from_bab_experiment("multisine_05", preprocess=True, resample_factor=50)
train, test = dataset.split(0.8)

cfg = NeuralCDEConfig(hidden_dim=32, hidden_layers=[64, 64], epochs=100)
model = NeuralCDE(cfg)
model.fit(train.arrays)

y_osa = model.predict(test.u, test.y, mode="OSA")
y_fr  = model.predict(test.u, test.y, mode="FR")
```

## Benchmarking

See the [benchmarking guide](benchmarking.md) for running reproducible
comparisons across model families.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Solver not converging | `rtol=1e-3, atol=1e-4` or `solver="rk4"` |
| Slow training | `solver="euler"`, `interpolation="linear"`, smaller `hidden_dim` |
| `torchcde` import error | `pip install torchcde` |

## References

- Kidger et al. (2020). [arXiv:2005.08926](https://arxiv.org/abs/2005.08926)
- `torchcde`: <https://github.com/patrick-kidger/torchcde>
