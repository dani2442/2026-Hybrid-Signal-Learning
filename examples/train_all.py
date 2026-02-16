"""Train all registered models and save benchmark results.

Usage
-----
::

    python -m examples.train_all --dataset multisine_05
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmarking import results_to_json, run_all_benchmarks
from src.data import from_bab_experiment
from src.models import list_models
from src.utils.runtime import seed_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark all models")
    parser.add_argument("--dataset", type=str, default="multisine_05")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Subset of model names (default: all)")
    parser.add_argument("--save-dir", type=str, default="trained_models")
    parser.add_argument("--results-file", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    seed_all(args.seed)
    ds = from_bab_experiment(args.dataset)

    print(f"Dataset: {ds.name}  ({ds.n_samples} samples)")
    print(f"Available models: {list_models()}")

    overrides = {"seed": args.seed}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs

    results = run_all_benchmarks(
        ds,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        model_names=args.models,
        config_overrides=overrides,
        save_dir=args.save_dir,
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        name = r["model_name"]
        if r["metrics"]:
            print(f"  {name:30s}  RMSE={r['metrics']['RMSE']:.6f}  "
                  f"FIT={r['metrics']['FIT']:.4f}  "
                  f"t={r['train_time']:.1f}s")
        else:
            print(f"  {name:30s}  ERROR: {r.get('error', 'unknown')}")

    results_to_json(results, args.results_file)
    print(f"\nResults saved to {args.results_file}")


if __name__ == "__main__":
    main()
