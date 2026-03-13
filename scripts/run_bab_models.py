from __future__ import annotations

import argparse
import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    parser.add_argument("--n-runs", type=int, default=5, help="Independent training runs per model specification")
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

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Max parallel training threads (0 = auto: #GPUs if CUDA else 1). "
             "Each worker gets its own GPU via round-robin when multiple GPUs are available.",
    )

    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _available_devices(device_arg: str) -> list[torch.device]:
    """Return an ordered list of devices to round-robin across workers."""
    if device_arg == "cpu":
        return [torch.device("cpu")]
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        return [torch.device(f"cuda:{i}") for i in range(n)]
    return [torch.device("cpu")]


def _resolve_workers(workers_arg: int, n_devices: int) -> int:
    """Return the actual worker count."""
    if workers_arg > 0:
        return workers_arg
    # auto: one worker per GPU, minimum 1
    return max(n_devices, 1)


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

    # ── Device & worker setup ─────────────────────────────────────────
    devices = _available_devices(args.device)
    n_workers = _resolve_workers(args.workers, len(devices))
    primary_device = devices[0]

    print(f"Available devices: {[str(d) for d in devices]}")
    print(f"Parallel workers: {n_workers}")
    for d in devices:
        if d.type == "cuda":
            print(f"  {d}: {torch.cuda.get_device_name(d)}")
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

    # Build one TensorBundle per device so workers sharing a GPU reuse
    # the same tensors (read-only during training after set_series).
    tensor_bundles: dict[torch.device, Any] = {}
    for dev in devices:
        tensor_bundles[dev] = to_tensor_bundle(
            t=train_data.t,
            u=train_data.u,
            y_sim=train_data.y_sim,
            device=dev,
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
        "devices": [str(d) for d in devices],
        "n_workers": n_workers,
        "resample_factor": args.resample_factor,
        "y_dot_method": args.y_dot_method,
        "zoom_last_n": args.zoom_last_n,
        "include_datasets": include_datasets,
        "quick": bool(args.quick),
        "train_segments": train_data.segments,
    }
    write_json(run_dir / "metadata" / "config.json", config_dump)

    # ── Thread-safe accumulators ──────────────────────────────────────
    _lock = threading.Lock()
    history_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    history_by_model_id: dict[str, list[dict[str, float]]] = {}
    registry_rows: list[dict[str, Any]] = []

    # Monotonic counter for round-robin device assignment.
    _task_counter = 0
    _counter_lock = threading.Lock()

    def _next_device() -> torch.device:
        nonlocal _task_counter
        with _counter_lock:
            dev = devices[_task_counter % len(devices)]
            _task_counter += 1
        return dev

    requested_specs = iter_model_specs(model_keys, nn_variants)

    # ── Core worker function (runs in a thread) ──────────────────────
    def train_and_evaluate(
        *,
        model,
        model_key: str,
        nn_variant: str,
        run_idx: int,
        seed_value: int,
        device: torch.device,
        extra_meta: dict[str, Any] | None = None,
        signal_event: threading.Event | None = None,
        ref_holder: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Train one model, evaluate, persist artifacts, return result dict.

        Thread-safe: uses per-task RNG and only appends to shared lists
        under a lock.

        Parameters
        ----------
        signal_event : threading.Event | None
            Set after training finishes so dependent workers can proceed.
        ref_holder : dict | None
            Mutable dict where the trained model is stored under key
            ``"model"`` so dependents can read it after *signal_event*.
        """
        model_id = f"{model_key}__{nn_variant}__run{run_idx:02d}"
        print(f"\n=== Training {model_id} on {device} ===")

        # Per-task reproducible RNG (avoids contention on global state).
        task_rng = np.random.default_rng(seed_value + hash(model_id) % (2**31))

        # Per-task torch seed (thread-local generator).
        torch.manual_seed(seed_value + hash(model_id) % (2**31))

        tensors = tensor_bundles[device]
        model = model.to(device)

        trained_model, hist = train_model(
            model=model,
            tensors=tensors,
            valid_start_indices=valid_starts,
            cfg=train_cfg,
            show_progress=args.progress,
            progress_desc=f"{model_id}@{device}",
            rng=task_rng,
        )

        # Signal dependents that this physics model is ready.
        if signal_event is not None and ref_holder is not None:
            ref_holder["model"] = trained_model
            signal_event.set()

        # Build local rows first, then bulk-append under lock.
        local_history: list[dict[str, Any]] = []
        for item in hist:
            local_history.append(
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

        local_metrics: list[dict[str, Any]] = []
        eval_results: dict[str, dict[str, Any]] = {}
        for ds_name, ds in data_map.items():
            out = evaluate_model_on_dataset(model=trained_model, ds=ds, device=device)
            eval_results[ds_name] = out

            for split in ("train", "test"):
                mm = out["metrics"][split]
                if mm is None:
                    continue
                local_metrics.append(
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

        reg_entry = {
            "model_id": model_id,
            "model_key": model_key,
            "nn_variant": nn_variant,
            "run_idx": run_idx,
            "seed": seed_value,
            "checkpoint_path": str(ckpt_path),
            "prediction_path": str(pred_path) if args.save_predictions else None,
            "extra": extra_meta or {},
        }

        # Append to shared accumulators under lock.
        with _lock:
            history_rows.extend(local_history)
            history_by_model_id[model_id] = hist
            metrics_rows.extend(local_metrics)
            registry_rows.append(reg_entry)

        print(f"  ✓ {model_id} done on {device}")
        return {"model_id": model_id, "model": trained_model}

    # ── Deferred worker for hybrid_frozen* models ─────────────────────
    # These closures wait for the physics dependency, extract frozen
    # params, build the model, and then train — all inside the thread.
    def _make_frozen_worker(
        *,
        model_key: str,
        nn_variant: str,
        run_idx: int,
        run_seed: int,
        device: torch.device,
        wait_event: threading.Event,
        ref_holder: dict[str, Any],
        physics_label: str,
        extract_fn,
    ):
        """Return a zero-arg callable suitable for ``pool.submit``."""
        def _worker():
            print(f"  ⏳ {model_key}__{nn_variant}__run{run_idx:02d} waiting for {physics_label} …")
            wait_event.wait()
            print(f"  ▶ {model_key}__{nn_variant}__run{run_idx:02d} dependency ready, building model")
            ref_model = ref_holder["model"]
            frozen = extract_fn(ref_model)
            model = build_model(model_key, nn_variant=nn_variant, frozen_phys_params=frozen)
            extra = {
                "frozen_from": f"{physics_label}__physics__run{run_idx:02d}",
                "frozen_phys": copy.deepcopy(frozen),
            }
            return train_and_evaluate(
                model=model,
                model_key=model_key,
                nn_variant=nn_variant,
                run_idx=run_idx,
                seed_value=run_seed,
                device=device,
                extra_meta=extra,
            )
        return _worker

    # ── Main training loop ────────────────────────────────────────────
    for run_idx in range(args.n_runs):
        run_seed = args.seed + run_idx
        set_global_seed(run_seed)
        print(f"\n\n########## RUN {run_idx:02d} | seed={run_seed} ##########")

        # Synchronisation: physics models signal these events once trained
        # so hybrid_frozen* workers (waiting in the same pool) can proceed.
        need_linear = ("linear" in model_keys) or ("hybrid_frozen" in model_keys)
        need_stribeck = ("stribeck" in model_keys) or ("hybrid_frozen_stribeck" in model_keys)

        linear_ready = threading.Event()
        stribeck_ready = threading.Event()
        linear_holder: dict[str, Any] = {}
        stribeck_holder: dict[str, Any] = {}

        # Collect tasks.  Each entry is either:
        #   - dict  → kwargs for train_and_evaluate  (regular models)
        #   - callable → deferred worker             (hybrid_frozen*)
        pool_items: list[Any] = []

        # Physics models (submitted to pool; signal event on completion).
        if need_linear:
            pool_items.append(
                dict(
                    model=build_model("linear"),
                    model_key="linear",
                    nn_variant="physics",
                    run_idx=run_idx,
                    seed_value=run_seed,
                    device=_next_device(),
                    signal_event=linear_ready,
                    ref_holder=linear_holder,
                )
            )

        if need_stribeck:
            pool_items.append(
                dict(
                    model=build_model("stribeck"),
                    model_key="stribeck",
                    nn_variant="physics",
                    run_idx=run_idx,
                    seed_value=run_seed,
                    device=_next_device(),
                    signal_event=stribeck_ready,
                    ref_holder=stribeck_holder,
                )
            )

        # All remaining models.
        for model_key, nn_variant in requested_specs:
            if model_key in {"linear", "stribeck"}:
                continue

            dev = _next_device()

            if model_key == "hybrid_frozen":
                pool_items.append(
                    _make_frozen_worker(
                        model_key=model_key,
                        nn_variant=nn_variant,
                        run_idx=run_idx,
                        run_seed=run_seed,
                        device=dev,
                        wait_event=linear_ready,
                        ref_holder=linear_holder,
                        physics_label="linear",
                        extract_fn=extract_linear_params,
                    )
                )
            elif model_key == "hybrid_frozen_stribeck":
                pool_items.append(
                    _make_frozen_worker(
                        model_key=model_key,
                        nn_variant=nn_variant,
                        run_idx=run_idx,
                        run_seed=run_seed,
                        device=dev,
                        wait_event=stribeck_ready,
                        ref_holder=stribeck_holder,
                        physics_label="stribeck",
                        extract_fn=extract_stribeck_params,
                    )
                )
            else:
                pool_items.append(
                    dict(
                        model=build_model(model_key, nn_variant=nn_variant),
                        model_key=model_key,
                        nn_variant=nn_variant,
                        run_idx=run_idx,
                        seed_value=run_seed,
                        device=dev,
                    )
                )

        if n_workers <= 1:
            # Sequential fallback.
            for item in pool_items:
                if callable(item):
                    item()
                else:
                    train_and_evaluate(**item)
        else:
            print(f"\nDispatching {len(pool_items)} models across {n_workers} workers …")
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {}
                for item in pool_items:
                    if callable(item):
                        futures[pool.submit(item)] = "frozen_deferred"
                    else:
                        futures[pool.submit(train_and_evaluate, **item)] = item.get("model_key", "?")
                for fut in as_completed(futures):
                    fut.result()  # propagate exceptions

    # ── Persist tables and metadata ───────────────────────────────────
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
