# Hybrid Modeling

System identification library with classical models, neural continuous-time models, and physics-guided hybrid models.

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

TorchSDE-based models accept `wandb_run` in `fit(...)` and log per-epoch training signals:

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

## Benchmark (Recommended Workflow)

Run benchmark with default model set:

```bash
python3 examples/benchmark.py
```

Run custom datasets/models:

```bash
python3 examples/benchmark.py \
  --datasets multisine_05,multisine_06 \
  --models narx,neural_ode,hybrid_linear_beam,hybrid_nonlinear_cam \
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

