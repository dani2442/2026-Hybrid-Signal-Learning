from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .bab_data import ExperimentData


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(output_root: str | Path, run_name: str | None = None) -> Path:
    root = Path(output_root)
    name = run_name if run_name is not None else f"run_{timestamp_tag()}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("models", "predictions", "tables", "metadata", "plots"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        p.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_training_history_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(path, rows)


def save_metrics_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(path, rows)


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean/std statistics grouped by (model_key, nn_variant, dataset, split)."""
    if not rows:
        return []

    by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        k = (str(row["model_key"]), str(row["nn_variant"]), str(row["dataset"]), str(row["split"]))
        by_key.setdefault(k, []).append(row)

    out: list[dict[str, Any]] = []
    metrics = ["rmse_pos", "rmse_vel", "r2_pos", "r2_vel", "fit_pos", "fit_vel"]

    for (model_key, nn_variant, dataset, split), group in sorted(by_key.items()):
        agg: dict[str, Any] = {
            "model_key": model_key,
            "nn_variant": nn_variant,
            "dataset": dataset,
            "split": split,
            "n_models": len(group),
        }
        for m in metrics:
            vals = np.asarray([float(g[m]) for g in group], dtype=float)
            agg[f"{m}_mean"] = float(np.nanmean(vals))
            agg[f"{m}_std"] = float(np.nanstd(vals))
        out.append(agg)

    return out


def select_best_model_ids(rows: list[dict[str, Any]], *, split: str = "test") -> dict[str, str]:
    """
    Select one best model_id per high-level family (model_key) using mean r2_pos on selected split.
    This naturally handles multiple runs and NN variants.
    """

    if not rows:
        return {}

    by_model_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row["split"]) != split:
            continue
        by_model_id.setdefault(str(row["model_id"]), []).append(row)

    model_scores: dict[str, tuple[str, float]] = {}
    # model_key -> (best_model_id, score)

    for model_id, group in by_model_id.items():
        if not group:
            continue
        model_key = str(group[0]["model_key"])
        vals = np.asarray([float(g["r2_pos"]) for g in group], dtype=float)
        score = float(np.nanmean(vals))

        if model_key not in model_scores or score > model_scores[model_key][1]:
            model_scores[model_key] = (model_id, score)

    return {k: v[0] for k, v in model_scores.items()}


def save_model_prediction_npz(
    path: str | Path,
    *,
    data_map: dict[str, ExperimentData],
    eval_results: dict[str, dict[str, Any]],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}

    for ds_name, ds in data_map.items():
        res = eval_results[ds_name]
        payload[f"{ds_name}__t"] = ds.t
        payload[f"{ds_name}__u"] = ds.u
        payload[f"{ds_name}__y_true"] = ds.y_sim
        payload[f"{ds_name}__y_pred"] = np.asarray(res["y_pred"], dtype=float)
        payload[f"{ds_name}__train_idx"] = ds.train_idx
        payload[f"{ds_name}__test_idx"] = ds.test_idx
        payload[f"{ds_name}__Ts"] = np.asarray([ds.Ts], dtype=float)

    np.savez_compressed(out, **payload)


def load_model_prediction_npz(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=False)
    keys = sorted(data.files)

    dataset_names = sorted({k.split("__", 1)[0] for k in keys if "__" in k})
    out: dict[str, dict[str, np.ndarray]] = {}

    for ds in dataset_names:
        out[ds] = {
            "t": data[f"{ds}__t"],
            "u": data[f"{ds}__u"],
            "y_true": data[f"{ds}__y_true"],
            "y_pred": data[f"{ds}__y_pred"],
            "train_idx": data[f"{ds}__train_idx"],
            "test_idx": data[f"{ds}__test_idx"],
            "Ts": float(data[f"{ds}__Ts"][0]),
        }

    return out
