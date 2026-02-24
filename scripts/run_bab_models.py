from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_signal_learning import (
    MODEL_KEYS,
    TrainConfig,
    aggregate_metric_rows,
    build_model,
    build_train_rollout_data,
    compute_valid_start_indices,
    evaluate_model_on_dataset,
    extract_linear_params,
    extract_stribeck_params,
    iter_model_specs,
    load_protocol2_datasets,
    make_run_dir,
    plot_training_curves,
    save_metrics_csv,
    save_model_checkpoint,
    save_model_prediction_npz,
    save_training_history_csv,
    select_best_model_ids,
    set_global_seed,
    summarize_protocol2,
    to_tensor_bundle,
    train_model,
    write_json,
)


def _csv_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate BAB Neural ODE models with protocol-2 split (50/50 temporal).")

    parser.add_argument("--output-root", type=str, default="results/bab_runs", help="Base folder for run outputs")
    parser.add_argument("--run-name", type=str, default=None, help="Run folder name. Default uses timestamp")

    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODEL_KEYS),
        help="Comma-separated model keys: linear,stribeck,blackbox,hybrid_joint,hybrid_joint_stribeck,hybrid_frozen,hybrid_frozen_stribeck",
    )
    parser.add_argument(
        "--nn-variants",
        type=str,
        default="base,wide,deep",
        help="Comma-separated NN variants for blackbox/hybrid models",
    )

    parser.add_argument("--n-runs", type=int, default=10, help="Independent training runs per model specification")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--k-steps", type=int, default=20)
    parser.add_argument("--obs-dim", type=int, default=2)

    parser.add_argument("--resample-factor", type=int, default=50)
    parser.add_argument("--zoom-last-n", type=int, default=200)
    parser.add_argument("--y-dot-method", type=str, default="central")

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--quick", action="store_true", help="Quick local smoke run")
    parser.add_argument(
        "--include-datasets",
        type=str,
        default=None,
        help="Optional comma-separated subset of datasets",
    )

    parser.add_argument("--save-predictions", action="store_true", default=True)
    parser.add_argument("--no-save-predictions", action="store_false", dest="save_predictions")
    parser.add_argument("--progress", action="store_true", default=True, help="Show tqdm progress bars when available")
    parser.add_argument("--no-progress", action="store_false", dest="progress", help="Disable progress bars")

    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()

    model_keys = _csv_list(args.models)
    nn_variants = _csv_list(args.nn_variants)
    include_datasets = _csv_list(args.include_datasets) if args.include_datasets else None

    if args.quick:
        # Practical defaults for local sanity checks.
        args.n_runs = min(args.n_runs, 1)
        args.epochs = min(args.epochs, 60)
        nn_variants = ["base"]
        if include_datasets is None:
            include_datasets = ["multisine_05", "multisine_06", "random_steps_01", "swept_sine"]

    device = select_device(args.device)
    print(f"Using device: {device}")
    print("Protocol: fixed protocol-2 (temporal 50/50 for core datasets)")
    if args.progress and tqdm is None:
        print("Progress bars requested but tqdm is not installed; falling back to plain logs.")

    data_map = load_protocol2_datasets(
        y_dot_method=args.y_dot_method,
        resample_factor=args.resample_factor,
        zoom_last_n=args.zoom_last_n,
        include=include_datasets,
    )
    print(summarize_protocol2(data_map))

    train_data = build_train_rollout_data(data_map)
    valid_starts = compute_valid_start_indices(train_data.segments, k_steps=args.k_steps)

    tensors = to_tensor_bundle(
        t=train_data.t,
        u=train_data.u,
        y_sim=train_data.y_sim,
        device=device,
    )

    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        k_steps=args.k_steps,
        obs_dim=args.obs_dim,
        position_only_loss=True,
    )

    run_dir = make_run_dir(args.output_root, run_name=args.run_name)
    print(f"Output dir: {run_dir}")

    config_dump = {
        "protocol": "protocol_2",
        "model_keys": model_keys,
        "nn_variants": nn_variants,
        "n_runs": args.n_runs,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "k_steps": args.k_steps,
        "obs_dim": args.obs_dim,
        "seed": args.seed,
        "device": str(device),
        "resample_factor": args.resample_factor,
        "y_dot_method": args.y_dot_method,
        "zoom_last_n": args.zoom_last_n,
        "include_datasets": include_datasets,
        "quick": bool(args.quick),
        "train_segments": train_data.segments,
    }
    write_json(run_dir / "metadata" / "config.json", config_dump)

    history_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    history_by_model_id: dict[str, list[dict[str, float]]] = {}
    registry_rows: list[dict[str, Any]] = []

    requested_specs = iter_model_specs(model_keys, nn_variants)

    def train_and_evaluate(
        *,
        model,
        model_key: str,
        nn_variant: str,
        run_idx: int,
        seed_value: int,
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        model_id = f"{model_key}__{nn_variant}__run{run_idx:02d}"
        print(f"\n=== Training {model_id} ===")

        trained_model, hist = train_model(
            model=model,
            tensors=tensors,
            valid_start_indices=valid_starts,
            cfg=train_cfg,
            show_progress=args.progress,
            progress_desc=model_id,
        )

        history_by_model_id[model_id] = hist
        for item in hist:
            history_rows.append(
                {
                    "model_id": model_id,
                    "model_key": model_key,
                    "nn_variant": nn_variant,
                    "run_idx": run_idx,
                    "seed": seed_value,
                    "epoch": int(item["epoch"]),
                    "loss": float(item["loss"]),
                }
            )

        ckpt_path = run_dir / "models" / f"{model_id}.pt"
        save_model_checkpoint(
            ckpt_path,
            model=trained_model,
            model_key=model_key,
            nn_variant=nn_variant,
            run_idx=run_idx,
            seed=seed_value,
            extra=extra_meta,
        )

        eval_results: dict[str, dict[str, Any]] = {}
        for ds_name, ds in data_map.items():
            out = evaluate_model_on_dataset(model=trained_model, ds=ds, device=device)
            eval_results[ds_name] = out

            for split in ("train", "test"):
                mm = out["metrics"][split]
                if mm is None:
                    continue
                metrics_rows.append(
                    {
                        "model_id": model_id,
                        "model_key": model_key,
                        "nn_variant": nn_variant,
                        "run_idx": run_idx,
                        "seed": seed_value,
                        "dataset": ds_name,
                        "split": split,
                        **mm,
                    }
                )

        pred_path = run_dir / "predictions" / f"{model_id}.npz"
        if args.save_predictions:
            save_model_prediction_npz(pred_path, data_map=data_map, eval_results=eval_results)

        registry_rows.append(
            {
                "model_id": model_id,
                "model_key": model_key,
                "nn_variant": nn_variant,
                "run_idx": run_idx,
                "seed": seed_value,
                "checkpoint_path": str(ckpt_path),
                "prediction_path": str(pred_path) if args.save_predictions else None,
                "extra": extra_meta or {},
            }
        )

    for run_idx in range(args.n_runs):
        run_seed = args.seed + run_idx
        set_global_seed(run_seed)
        print(f"\n\n########## RUN {run_idx:02d} | seed={run_seed} ##########")

        # Linear can be both a target model and a dependency for hybrid_frozen.
        linear_ref_model = None
        stribeck_ref_model = None
        if ("linear" in model_keys) or ("hybrid_frozen" in model_keys):
            linear_ref_model = build_model("linear")
            train_and_evaluate(
                model=linear_ref_model,
                model_key="linear",
                nn_variant="physics",
                run_idx=run_idx,
                seed_value=run_seed,
            )

        if ("stribeck" in model_keys) or ("hybrid_frozen_stribeck" in model_keys):
            stribeck_ref_model = build_model("stribeck")
            train_and_evaluate(
                model=stribeck_ref_model,
                model_key="stribeck",
                nn_variant="physics",
                run_idx=run_idx,
                seed_value=run_seed,
            )

        for model_key, nn_variant in requested_specs:
            if model_key in {"linear", "stribeck"}:
                continue

            if model_key == "hybrid_frozen":
                if linear_ref_model is None:
                    raise RuntimeError("Hybrid frozen requested but no linear reference model available")
                frozen = extract_linear_params(linear_ref_model)
                model = build_model("hybrid_frozen", nn_variant=nn_variant, frozen_phys_params=frozen)
                extra = {"frozen_from": f"linear__physics__run{run_idx:02d}", "frozen_phys": copy.deepcopy(frozen)}
            elif model_key == "hybrid_frozen_stribeck":
                if stribeck_ref_model is None:
                    raise RuntimeError("Hybrid frozen stribeck requested but no stribeck reference model available")
                frozen = extract_stribeck_params(stribeck_ref_model)
                model = build_model("hybrid_frozen_stribeck", nn_variant=nn_variant, frozen_phys_params=frozen)
                extra = {"frozen_from": f"stribeck__physics__run{run_idx:02d}", "frozen_phys": copy.deepcopy(frozen)}
            else:
                model = build_model(model_key, nn_variant=nn_variant)
                extra = None

            train_and_evaluate(
                model=model,
                model_key=model_key,
                nn_variant=nn_variant,
                run_idx=run_idx,
                seed_value=run_seed,
                extra_meta=extra,
            )

    # Persist tables and metadata.
    save_training_history_csv(run_dir / "tables" / "training_history.csv", history_rows)
    save_metrics_csv(run_dir / "tables" / "metrics_long.csv", metrics_rows)

    agg_rows = aggregate_metric_rows(metrics_rows)
    save_metrics_csv(run_dir / "tables" / "metrics_aggregate.csv", agg_rows)

    best_ids = select_best_model_ids(metrics_rows, split="test")
    write_json(run_dir / "metadata" / "best_model_ids_test_r2_pos.json", best_ids)
    write_json(run_dir / "metadata" / "model_registry.json", {"models": registry_rows})

    # Training convergence figure.
    plot_training_curves(history_by_model_id, save_path=run_dir / "plots" / "training_loss_curves.png")

    print("\nDone.")
    print(f"Run folder: {run_dir}")
    print(f"Saved {len(registry_rows)} model checkpoints")
    print(f"Saved {len(metrics_rows)} metric rows")
    print(f"Best model IDs (test r2_pos): {best_ids}")


if __name__ == "__main__":
    main()
