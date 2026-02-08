# Benchmarking Guide

## Goal

Build a reproducible leaderboard across model families with identical data handling, split policy, metrics, and reporting.

## Recommended Protocol

1. Use `Dataset.from_bab_experiment(..., preprocess=True, resample_factor=...)` for all runs.
2. Use one fixed train/test split ratio across all models (default `0.8`).
3. Evaluate both:
   - `OSA` (one-step-ahead)
   - `FR` (free-run)
4. Report the same metrics for every run:
   - `MSE`, `RMSE`, `MAE`, `R2`, `NRMSE`, `FIT%`
5. Persist all rows as JSON and log the same rows to W&B.

## Built-in Runner

`src/benchmarking/runner.py` provides:

- `BenchmarkConfig`: dataset/split/output settings
- `build_benchmark_cases(...)`: standard model presets
- `BenchmarkRunner`: benchmark execution + JSON export
- `summarize_results(...)`: leaderboard sorting helper

## CLI Example

```bash
python3 examples/benchmark.py \
  --datasets multisine_05,multisine_06 \
  --models narx,random_forest,neural_ode,neural_sde,hybrid_linear_beam,hybrid_nonlinear_cam \
  --resample-factor 50 \
  --train-ratio 0.8 \
  --output-json results/benchmark.json
```

## W&B Logging

By default, `examples/benchmark.py` initializes a W&B run and logs:

- dataset metadata (`dataset/*`)
- per-run metrics (`benchmark/*`)
- full tabular artifact (`benchmark/results_table`)

Disable logging with:

```bash
python3 examples/benchmark.py --disable-wandb
```

## Scaling Strategy

For a larger benchmark campaign:

1. Start with one dataset and all models to validate configuration.
2. Expand to multiple datasets with the same case list.
3. Add repeated runs (different seeds) and aggregate mean/std in post-processing.
4. Promote the benchmark script to CI for regression tracking of new model changes.

