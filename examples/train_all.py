"""Train all registered models and save benchmark results.

Usage
-----
::

    python -m examples.train_all
    python -m examples.train_all --dataset multisine_05
    python -m examples.train_all --datasets multisine_05 swept_sine
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from examples.smoke_test_all import CONFIG_OVERRIDES as SMOKE_CONFIG_OVERRIDES
from examples.smoke_test_all import SLOW_MODELS as SMOKE_SLOW_MODELS
from src.benchmarking import results_to_json, run_all_benchmarks
from src.data import (
    DatasetCollection,
    from_bab_experiment,
    from_bab_experiments,
    list_bab_experiments,
)
from src.models import list_models
from src.utils.runtime import seed_all
from src.wandb_logger import WandbLogger


def _resolve_dataset_names(dataset: str | None, datasets: list[str] | None) -> list[str]:
    if datasets:
        return datasets
    if dataset is None or dataset.strip().lower() == "all":
        return list_bab_experiments()
    return [name.strip() for name in dataset.split(",") if name.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark all models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Single dataset key, comma-separated keys, or 'all' (default).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more dataset keys. Overrides --dataset when provided. "
            f"Available: {', '.join(list_bab_experiments())}"
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Subset of model names (default: all)")
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include very slow CDE models (skipped by default).",
    )
    parser.add_argument("--save-dir", type=str, default="trained_models")
    parser.add_argument("--results-file", type=str, default="benchmark_results.json")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="hybrid-learning",
        help="W&B project name. If set, creates one run per benchmarked model.",
    )
    parser.add_argument(
        "--wandb-run-prefix",
        type=str,
        default="train_all",
        help="Prefix for per-model W&B run names.",
    )
    args = parser.parse_args()

    seed_all(args.seed)
    dataset_names = _resolve_dataset_names(args.dataset, args.datasets)
    ds = (
        from_bab_experiment(dataset_names[0])
        if len(dataset_names) == 1
        else from_bab_experiments(dataset_names)
    )

    if isinstance(ds, DatasetCollection):
        print(f"Datasets: {', '.join(ds.names)}  ({len(ds)} experiments)")
    else:
        print(f"Dataset: {ds.name}  ({ds.n_samples} samples)")
    print(f"Available models: {list_models()}")

    selected_models = args.models if args.models else list_models()
    if not args.include_slow:
        skipped = [name for name in selected_models if name in SMOKE_SLOW_MODELS]
        selected_models = [
            name for name in selected_models if name not in SMOKE_SLOW_MODELS
        ]
        if skipped:
            print(f"Skipping slow models: {skipped}. Use --include-slow to include.")

    if not selected_models:
        raise SystemExit(
            "No models selected after filtering. Pass --include-slow to include "
            "slow CDE models."
        )

    overrides = {"seed": args.seed}
    per_model_overrides = SMOKE_CONFIG_OVERRIDES
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
        # Preserve smoke-test tuning while allowing explicit global epoch override.
        per_model_overrides = {
            name: {k: v for k, v in cfg.items() if k != "epochs"}
            for name, cfg in SMOKE_CONFIG_OVERRIDES.items()
        }

    logger_factory = None
    if args.wandb_project:
        dataset_tag = dataset_names[0] if len(dataset_names) == 1 else f"{len(dataset_names)}ds"

        # Pre-compute a human-readable dataset description for the config
        if isinstance(ds, DatasetCollection):
            _ds_info = {"dataset_names": list(ds.names), "n_datasets": len(ds)}
        else:
            _ds_info = {"dataset_names": [ds.name], "n_datasets": 1}

        def logger_factory(model_name: str):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{args.wandb_run_prefix}_{model_name}_{dataset_tag}_{timestamp}"
            return WandbLogger(
                args.wandb_project,
                run_name=run_name,
                config={
                    "script": "examples.train_all",
                    "model_name": model_name,
                    "train_ratio": args.train_ratio,
                    "val_ratio": args.val_ratio,
                    "seed": args.seed,
                    "epochs_override": args.epochs,
                    **_ds_info,
                },
            )

    results = run_all_benchmarks(
        ds,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        model_names=selected_models,
        config_overrides=overrides,
        per_model_config_overrides=per_model_overrides,
        save_dir=args.save_dir,
        logger_factory=logger_factory,
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        name = r["model_name"]
        if r["metrics"]:
            print(f"  {name:30s}  MSE={r['metrics']['MSE']:.6f}  "
                  f"FIT={r['metrics']['FIT']:.4f}  "
                  f"R2={r['metrics']['R2']:.4f}  "
                  f"t={r['train_time']:.1f}s")
        else:
            print(f"  {name:30s}  ERROR: {r.get('error', 'unknown')}")

    results_to_json(results, args.results_file)
    print(f"\nResults saved to {args.results_file}")


if __name__ == "__main__":
    main()
