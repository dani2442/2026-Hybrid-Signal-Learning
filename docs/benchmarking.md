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

## Programmatic Usage

```python
from src import BenchmarkConfig, BenchmarkRunner, build_benchmark_cases

cfg = BenchmarkConfig(
    datasets=["multisine_05", "multisine_06"],
    train_ratio=0.8,
    resample_factor=50,
    output_json="results/benchmark.json",
)
cases = build_benchmark_cases(cfg)
runner = BenchmarkRunner(cfg)
results = runner.run(cases)
```

## W&B Logging

Enable logging by setting `wandb_project` in the model configs passed
to each benchmark case. Metrics logged per-run include:

- dataset metadata (`dataset/*`)
- per-run metrics (`benchmark/*`)
- full tabular artifact (`benchmark/results_table`)

## Scaling Strategy

For a larger benchmark campaign:

1. Start with one dataset and all models to validate configuration.
2. Expand to multiple datasets with the same case list.
3. Add repeated runs (different seeds) and aggregate mean/std in post-processing.
4. Promote the benchmark script to CI for regression tracking of new model changes.

