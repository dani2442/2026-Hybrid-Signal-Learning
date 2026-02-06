#!/usr/bin/env python3
"""Run a reproducible benchmark across multiple models and datasets."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.benchmarking import (
    BenchmarkConfig,
    BenchmarkRunner,
    build_benchmark_cases,
    summarize_results,
)


def _parse_csv(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="multisine_05",
        help="Comma-separated bab_datasets keys.",
    )
    parser.add_argument(
        "--models",
        default="narx,random_forest,neural_ode,neural_sde,hybrid_linear_beam,hybrid_nonlinear_cam",
        help="Comma-separated model case keys.",
    )
    parser.add_argument("--resample-factor", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--output-json", default="benchmark_results.json")

    parser.add_argument("--wandb-project", default="hybrid-modeling-benchmark")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()

    config = BenchmarkConfig(
        datasets=tuple(_parse_csv(args.datasets)),
        preprocess=True,
        resample_factor=args.resample_factor,
        train_ratio=args.train_ratio,
        output_json=args.output_json,
    )
    cases = build_benchmark_cases(_parse_csv(args.models))
    runner = BenchmarkRunner(cases=cases, config=config)

    wandb_run = None
    if not args.disable_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config={
                    "benchmark": asdict(config),
                    "cases": [case.key for case in cases],
                },
            )
        except Exception as exc:
            print(f"W&B disabled: {exc}")
            wandb_run = None

    rows = runner.run_and_save(wandb_run=wandb_run)
    leaderboard = summarize_results(rows, mode="FR", metric="FIT%")

    print(f"\nSaved benchmark results to: {args.output_json}")
    print("\nTop free-run models by FIT%:")
    for rank, row in enumerate(leaderboard[:10], start=1):
        print(
            f"{rank:2d}. {row['dataset']} | {row['model']} | "
            f"FIT%={row['FIT%']:.4f} | R2={row['R2']:.4f} | "
            f"NRMSE={row['NRMSE']:.4f} | fit_s={row['fit_seconds']:.2f}"
        )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
