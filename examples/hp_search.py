"""Optuna hyperparameter search for any registered model.

Mirrors ``train_single.py`` but wraps the train/eval loop inside an
Optuna study.  Results are persisted in an SQLite database so studies
can be resumed across sessions.

Usage
-----
::

    # 50 trials for GRU on multisine_05, minimising validation MSE
    python -m examples.hp_search --model gru --dataset multisine_05 --n-trials 50

    # Resume a previous study (same --study-name + --storage)
    python -m examples.hp_search --model gru --dataset multisine_05 --n-trials 20 \\
        --study-name gru_multisine_05 --storage sqlite:///optuna.db

    # Search with W&B logging enabled
    python -m examples.hp_search --model neural_ode --dataset multisine_05 \\
        --n-trials 30 --wandb-project hp-search

    # Override fixed params (not searched)
    python -m examples.hp_search --model lstm --dataset multisine_05 \\
        --n-trials 40 --fixed epochs=200 device=cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import optuna
from src.data import from_bab_experiments, list_bab_experiments
from src.hp_search import (
    create_objective,
    list_searchable_models,
)
from src.models import list_models
from src.wandb_logger import WandbLogger


def _resolve_dataset_names(dataset: str, datasets: list[str] | None) -> list[str]:
    if datasets:
        return datasets
    return [name.strip() for name in dataset.split(",") if name.strip()]


def _parse_fixed_overrides(raw: list[str] | None) -> Dict[str, Any]:
    """Parse ``key=value`` pairs from the CLI into a dict.

    Attempts to cast values to int / float / bool automatically.
    """
    if not raw:
        return {}
    overrides: Dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got: {item!r}")
        k, v = item.split("=", 1)
        # Auto-cast
        if v.lower() in ("true", "false"):
            overrides[k] = v.lower() == "true"
        else:
            for caster in (int, float):
                try:
                    overrides[k] = caster(v)
                    break
                except ValueError:
                    continue
            else:
                overrides[k] = v
    return overrides


def _make_wandb_factory(project: str | None, model_name: str, dataset_tag: str):
    """Return a logger factory for per-trial W&B runs, or None."""
    if not project:
        return None

    def factory(trial: optuna.Trial, config: dict) -> WandbLogger:
        return WandbLogger(
            project,
            run_name=f"{model_name}_t{trial.number}_{datetime.now():%H%M%S}",
            config={**config, "trial_number": trial.number, "datasets": dataset_tag},
        )

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for system-id models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"Model to optimise. Searchable: {', '.join(list_searchable_models())}",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="multisine_05",
        help="Dataset key (or comma-separated keys).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help=f"One or more dataset keys. Available: {', '.join(list_bab_experiments())}",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="R2",
        choices=["MSE", "R2", "FIT"],
        help="Validation metric to optimise (default: MSE).",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default=None,
        help="Optimisation direction. Auto-detected from --metric if omitted.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name (default: <model>_<dataset>).",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help=(
            "Optuna storage URL for persistence / resumption. "
            "E.g. sqlite:///optuna.db  (default: in-memory)."
        ),
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="none",
        choices=["none", "median", "hyperband"],
        help="Optuna pruner (default: none).",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="tpe",
        choices=["tpe", "random", "cmaes"],
        help="Optuna sampler (default: tpe).",
    )
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument(
        "--fixed",
        type=str,
        nargs="*",
        default=None,
        help="Fixed config overrides as key=value pairs (not searched).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="hp_results",
        help="Directory to save best config + summary JSON.",
    )
    args = parser.parse_args()

    # ── Data ──────────────────────────────────────────────────────────
    dataset_names = _resolve_dataset_names(args.dataset, args.datasets)
    datasets = from_bab_experiments(dataset_names)
    train_sets, val_sets, test_sets = datasets.train_val_test_split(
        args.train_ratio, args.val_ratio
    )
    print("Datasets:")
    for ds, tr, va, te in zip(datasets, train_sets, val_sets, test_sets):
        print(
            f"  {ds.name}: train={tr.n_samples}, val={va.n_samples}, "
            f"test={te.n_samples}"
        )

    # ── Direction ─────────────────────────────────────────────────────
    direction = args.direction
    if direction is None:
        # MSE → minimise; R2, FIT → maximise
        direction = "maximize" if args.metric in ("R2", "FIT") else "minimize"

    # ── Sampler ───────────────────────────────────────────────────────
    sampler: optuna.samplers.BaseSampler
    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=args.seed)
    elif args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    elif args.sampler == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(seed=args.seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=args.seed)

    # ── Pruner ────────────────────────────────────────────────────────
    pruner: optuna.pruners.BasePruner
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner()
    elif args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    # ── Study ─────────────────────────────────────────────────────────
    study_name = args.study_name or f"{args.model}_{'__'.join(dataset_names)}"

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    fixed_overrides = _parse_fixed_overrides(args.fixed)

    dataset_tag = "__".join(dataset_names)
    wandb_factory = _make_wandb_factory(args.wandb_project, args.model, dataset_tag)

    objective = create_objective(
        args.model,
        train_sets,
        val_sets,
        metric=args.metric,
        seed=args.seed,
        fixed_overrides=fixed_overrides,
        logger_factory=wandb_factory,
    )

    # ── Optimise ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Optuna HP search: {args.model}  |  metric={args.metric} ({direction})")
    print(f"Study: {study_name}  |  n_trials={args.n_trials}")
    print(f"{'=' * 60}\n")

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # ── Report ────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'=' * 60}")
    print(f"Best trial #{best.number}  |  {args.metric} = {best.value:.6f}")
    print(f"{'=' * 60}")
    print("Best hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    if best.user_attrs:
        print("Metrics:")
        for k, v in best.user_attrs.items():
            print(f"  {k}: {v}")

    # ── Save results ──────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "model": args.model,
        "datasets": dataset_names,
        "metric": args.metric,
        "direction": direction,
        "n_trials": len(study.trials),
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "fixed_overrides": fixed_overrides,
        "seed": args.seed,
        "timestamp": timestamp,
    }
    out_path = os.path.join(
        args.save_dir, f"{args.model}_{dataset_tag}_{timestamp}.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
