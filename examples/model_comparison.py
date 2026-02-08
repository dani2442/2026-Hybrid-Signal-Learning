#!/usr/bin/env python3
"""Full model comparison: train every model, evaluate OSA + Free-Run,
generate publication-quality tables and figures.

Usage
-----
  # All 13 models (≈ 30-40 min on CPU)
  python examples/model_comparison.py

  # Quick test with 3 models
  python examples/model_comparison.py --models narx,random_forest,neural_ode

  # Change output directory
  python examples/model_comparison.py --output-dir results
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.benchmarking import build_benchmark_cases
from src.data import Dataset
from src.validation.metrics import Metrics

# ── colour palette ─────────────────────────────────────────────────────
MODEL_COLOURS: Dict[str, str] = {
    "NARX":               "#1f77b4",
    "RandomForest":       "#ff7f0e",
    "NeuralODE":          "#2ca02c",
    "NeuralSDE":          "#d62728",
    "NeuralCDE":          "#9467bd",
    "LSTM":               "#8c564b",
    "TCN":                "#e377c2",
    "Mamba":              "#7f7f7f",
    "UDE":                "#bcbd22",
    "HybridLinearBeam":   "#17becf",
    "HybridNonlinearCam": "#000000",
    "GRU":                "#FFD700",
    "NeuralNetwork":      "#FF4500",
}
_FALLBACK = list(MODEL_COLOURS.values())


def _colour(name: str, idx: int) -> str:
    return MODEL_COLOURS.get(name, _FALLBACK[idx % len(_FALLBACK)])


# ── matplotlib configuration ──────────────────────────────────────────
def _configure_mpl():
    import matplotlib
    matplotlib.rcParams.update({
        "font.family":     "serif",
        "font.size":       11,
        "axes.labelsize":  13,
        "axes.titlesize":  15,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi":      150,
        "savefig.dpi":     300,
        "savefig.bbox":    "tight",
        "axes.grid":       True,
        "grid.alpha":      0.30,
        "grid.linewidth":  0.5,
    })


# ── helpers ────────────────────────────────────────────────────────────
def _fit_model(model, u, y):
    sig = inspect.signature(model.fit)
    kw = {}
    if "verbose" in sig.parameters:
        kw["verbose"] = True
    model.fit(u, y, **kw)


def _align(arr: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).flatten()
    if len(arr) >= n:
        return arr[:n]
    return np.pad(arr, (0, n - len(arr)), constant_values=np.nan)


# ── core training loop ────────────────────────────────────────────────
def train_and_evaluate(
    model_keys: List[str],
    dataset_name: str = "multisine_05",
    resample_factor: int = 50,
    train_ratio: float = 0.8,
) -> dict:
    """Train all models and return structured results."""
    ds = Dataset.from_bab_experiment(
        dataset_name, preprocess=True, resample_factor=resample_factor,
    )
    train_ds, test_ds = ds.split(train_ratio)
    dt = 1.0 / ds.sampling_rate

    cases = build_benchmark_cases(model_keys)
    max_lag = max(c.factory(dt).max_lag for c in cases)

    y_true = test_ds.y[max_lag:]
    t = np.arange(len(y_true)) * dt
    n = len(y_true)

    results = {
        "t": t,
        "y_true": y_true,
        "dt": dt,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "models": {},
    }

    for i, case in enumerate(cases, 1):
        print(
            f"\n{'='*60}\n"
            f"[{i}/{len(cases)}] Training {case.name} …\n"
            f"{'='*60}"
        )
        model = case.factory(dt)
        t0 = time.perf_counter()

        try:
            _fit_model(model, train_ds.u, train_ds.y)
            fit_s = time.perf_counter() - t0
            print(f"  Training done in {fit_s:.1f}s")
        except Exception as exc:
            print(f"  ✗ {case.name} training failed: {exc}")
            results["models"][case.name] = {"error": str(exc)}
            continue

        entry: Dict[str, object] = {"fit_seconds": fit_s}

        # ── One-Step-Ahead ──
        try:
            y_osa = model.predict(test_ds.u, test_ds.y, mode="OSA")
            y_osa = _align(y_osa, n)
            entry["osa_pred"] = y_osa
            entry["osa_metrics"] = Metrics.compute_all(y_true, y_osa)
            print(f"  ✓ OSA  R²={entry['osa_metrics']['R2']:.4f}  "
                  f"FIT%={entry['osa_metrics']['FIT%']:.4f}")
        except Exception as exc:
            print(f"  ✗ OSA failed: {exc}")

        # ── Free-Run ──
        try:
            y_init = test_ds.y[: model.max_lag]
            y_fr = model.predict(test_ds.u, y_init, mode="FR")
            y_fr = _align(y_fr, n)
            entry["fr_pred"] = y_fr
            entry["fr_metrics"] = Metrics.compute_all(y_true, y_fr)
            print(f"  ✓ FR   R²={entry['fr_metrics']['R2']:.4f}  "
                  f"FIT%={entry['fr_metrics']['FIT%']:.4f}")
        except Exception as exc:
            print(f"  ✗ Free-run failed: {exc}")

        results["models"][case.name] = entry

    return results


# ── plotting functions ─────────────────────────────────────────────────
def plot_predictions_overlay(results: dict, mode: str, out_dir: str):
    """Overlay all model predictions vs ground truth."""
    import matplotlib.pyplot as plt
    _configure_mpl()

    t = results["t"]
    y_true = results["y_true"]
    pred_key = "fr_pred" if mode == "FR" else "osa_pred"
    title = "Free-Run Simulation" if mode == "FR" else "One-Step-Ahead Predictions"

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, y_true, "k-", lw=1.4, alpha=0.7, label="Ground Truth", zorder=10)

    for i, (name, entry) in enumerate(results["models"].items()):
        if pred_key not in entry:
            continue
        c = _colour(name, i)
        ax.plot(t, entry[pred_key], color=c, lw=0.9, alpha=0.85, label=name)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Output (y)")
    ax.set_title(title)
    ax.set_xlim(t[0], t[-1])

    handles, labels = ax.get_legend_handles_labels()
    ncol = min(len(handles), 7)
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=ncol,
              framealpha=0.9, columnspacing=1.0)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)

    fname = f"{mode.lower()}_predictions.png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_residuals_overlay(results: dict, mode: str, out_dir: str):
    """Overlay residuals for all models."""
    import matplotlib.pyplot as plt
    _configure_mpl()

    t = results["t"]
    y_true = results["y_true"]
    pred_key = "fr_pred" if mode == "FR" else "osa_pred"
    title = "Free-Run Residuals" if mode == "FR" else "OSA Residuals"

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)

    for i, (name, entry) in enumerate(results["models"].items()):
        if pred_key not in entry:
            continue
        residual = y_true - entry[pred_key]
        c = _colour(name, i)
        ax.plot(t, residual, color=c, lw=0.8, alpha=0.8, label=name)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual (y − ŷ)")
    ax.set_title(title)
    ax.set_xlim(t[0], t[-1])

    handles, labels = ax.get_legend_handles_labels()
    ncol = min(len(handles), 7)
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=ncol,
              framealpha=0.9, columnspacing=1.0)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)

    fname = f"{mode.lower()}_residuals.png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_metric_bars(results: dict, mode: str, out_dir: str):
    """Grouped bar chart of key metrics for every model."""
    import matplotlib.pyplot as plt
    _configure_mpl()

    metrics_key = "fr_metrics" if mode == "FR" else "osa_metrics"
    bar_metrics = ["R2", "FIT%", "NRMSE"]

    names, values_per_metric = [], {m: [] for m in bar_metrics}
    colours = []
    for i, (name, entry) in enumerate(results["models"].items()):
        if metrics_key not in entry:
            continue
        names.append(name)
        colours.append(_colour(name, i))
        for m in bar_metrics:
            values_per_metric[m].append(entry[metrics_key].get(m, 0.0))

    if not names:
        return

    n_models = len(names)
    n_metrics = len(bar_metrics)
    x = np.arange(n_models)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, bar_metrics):
        vals = values_per_metric[metric]
        bars = ax.bar(x, vals, color=colours, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        # Add value labels on top of bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    title = "Free-Run Metrics" if mode == "FR" else "OSA Metrics"
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()

    fname = f"{mode.lower()}_metrics.png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_training_time(results: dict, out_dir: str):
    """Horizontal bar chart of training time per model."""
    import matplotlib.pyplot as plt
    _configure_mpl()

    names, times, colours = [], [], []
    for i, (name, entry) in enumerate(results["models"].items()):
        if "fit_seconds" not in entry:
            continue
        names.append(name)
        times.append(entry["fit_seconds"])
        colours.append(_colour(name, i))

    if not names:
        return

    # Sort by time descending
    order = np.argsort(times)[::-1]
    names = [names[i] for i in order]
    times = [times[i] for i in order]
    colours = [colours[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(names))))
    y = np.arange(len(names))
    bars = ax.barh(y, times, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Training Time (s)")
    ax.set_title("Training Time Comparison")
    ax.invert_yaxis()

    for bar, t_val in zip(bars, times):
        ax.text(bar.get_width() + max(times) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{t_val:.1f}s", va="center", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "training_time.png")
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_individual_model(results: dict, name: str, entry: dict, out_dir: str):
    """Save a prediction + residual figure for a single model (OSA & FR)."""
    import matplotlib.pyplot as plt
    _configure_mpl()

    t = results["t"]
    y_true = results["y_true"]
    safe_name = name.lower().replace(" ", "_")
    idx = list(results["models"].keys()).index(name)
    c = _colour(name, idx)

    for mode in ("OSA", "FR"):
        pred_key = "fr_pred" if mode == "FR" else "osa_pred"
        metrics_key = "fr_metrics" if mode == "FR" else "osa_metrics"
        if pred_key not in entry:
            continue

        y_pred = entry[pred_key]
        residual = y_true - y_pred
        m = entry.get(metrics_key, {})
        r2 = m.get("R2", float("nan"))
        fit_pct = m.get("FIT%", float("nan"))

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})

        # ── top: prediction ──
        ax = axes[0]
        ax.plot(t, y_true, "k-", lw=1.2, alpha=0.7, label="Ground Truth")
        ax.plot(t, y_pred, color=c, lw=1.0, alpha=0.85, label=f"{name}")
        mode_label = "Free-Run" if mode == "FR" else "One-Step-Ahead"
        ax.set_title(f"{name} — {mode_label}  (R²={r2:.4f}, FIT%={fit_pct:.4f})")
        ax.set_ylabel("Output (y)")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlim(t[0], t[-1])

        # ── bottom: residual ──
        ax = axes[1]
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.plot(t, residual, color=c, lw=0.7, alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Residual")
        ax.set_xlim(t[0], t[-1])

        fig.tight_layout()
        fname = f"{safe_name}_{mode.lower()}.png"
        path = os.path.join(out_dir, fname)
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_fr_zoom(results: dict, out_dir: str, t_start: float = 2.0, t_end: float = 5.0):
    """Zoomed-in free-run predictions for a time window."""
    import matplotlib.pyplot as plt
    _configure_mpl()

    t = results["t"]
    y_true = results["y_true"]
    mask = (t >= t_start) & (t <= t_end)
    if mask.sum() < 5:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t[mask], y_true[mask], "k-", lw=1.8, label="Ground Truth", zorder=10)

    for i, (name, entry) in enumerate(results["models"].items()):
        if "fr_pred" not in entry:
            continue
        c = _colour(name, i)
        ax.plot(t[mask], entry["fr_pred"][mask], color=c, lw=1.1,
                alpha=0.9, label=name)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Output (y)")
    ax.set_title(f"Free-Run Detail ({t_start:.1f}–{t_end:.1f} s)")
    ax.set_xlim(t_start, t_end)

    handles, labels = ax.get_legend_handles_labels()
    ncol = min(len(handles), 7)
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=ncol,
              framealpha=0.9, columnspacing=1.0)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)

    path = os.path.join(out_dir, "fr_predictions_zoom.png")
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


def print_leaderboard(results: dict, mode: str):
    """Print a sorted leaderboard table to the terminal."""
    metrics_key = "fr_metrics" if mode == "FR" else "osa_metrics"
    rows = []
    for name, entry in results["models"].items():
        if metrics_key not in entry:
            continue
        m = entry[metrics_key]
        rows.append({
            "Model": name,
            "R²": m["R2"],
            "FIT%": m["FIT%"],
            "NRMSE": m["NRMSE"],
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
            "Time(s)": entry.get("fit_seconds", 0.0),
        })

    rows.sort(key=lambda r: r["FIT%"], reverse=True)

    title = "FREE-RUN LEADERBOARD" if mode == "FR" else "OSA LEADERBOARD"
    header = f"{'Rank':>4}  {'Model':<22} {'R²':>8} {'FIT%':>8} {'NRMSE':>8} {'RMSE':>10} {'MAE':>10} {'Time(s)':>8}"
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(header)
    print(sep)
    for rank, r in enumerate(rows, 1):
        print(
            f"{rank:4d}  {r['Model']:<22} {r['R²']:8.4f} {r['FIT%']:8.4f} "
            f"{r['NRMSE']:8.4f} {r['RMSE']:10.6f} {r['MAE']:10.6f} {r['Time(s)']:8.1f}"
        )
    print(sep)
    return rows


def save_results_json(results: dict, out_dir: str):
    """Persist numeric results to JSON (no numpy arrays)."""
    payload = {
        "n_train": results["n_train"],
        "n_test": results["n_test"],
        "dt": results["dt"],
        "models": {},
    }
    for name, entry in results["models"].items():
        d = {}
        if "fit_seconds" in entry:
            d["fit_seconds"] = entry["fit_seconds"]
        if "osa_metrics" in entry:
            d["osa_metrics"] = entry["osa_metrics"]
        if "fr_metrics" in entry:
            d["fr_metrics"] = entry["fr_metrics"]
        if "error" in entry:
            d["error"] = entry["error"]
        payload["models"][name] = d

    path = os.path.join(out_dir, "comparison_results.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {path}")


# ── CLI ────────────────────────────────────────────────────────────────
ALL_MODELS = (
    "narx,random_forest,neural_ode,neural_sde,neural_cde,"
    "lstm,tcn,mamba,ude,"
    "hybrid_linear_beam,hybrid_nonlinear_cam,gru,neural_network"
)


def main():
    parser = argparse.ArgumentParser(
        description="Full model comparison with publication-quality outputs.",
    )
    parser.add_argument("--models", default=ALL_MODELS,
                        help="Comma-separated model keys.")
    parser.add_argument("--dataset", default="multisine_05")
    parser.add_argument("--resample-factor", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--output-dir", default="comparison")
    args = parser.parse_args()

    model_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # ── 1. Train & evaluate ──
    results = train_and_evaluate(
        model_keys=model_keys,
        dataset_name=args.dataset,
        resample_factor=args.resample_factor,
        train_ratio=args.train_ratio,
    )

    if not results["models"]:
        print("No models completed — nothing to do.")
        return

    # ── 2. Leaderboards ──
    print_leaderboard(results, mode="OSA")
    fr_board = print_leaderboard(results, mode="FR")

    # ── 3. Figures ──
    fig_dir = os.path.join(ROOT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"\nGenerating overlay figures → {out}/")
    for mode in ("OSA", "FR"):
        plot_predictions_overlay(results, mode, out)
        plot_residuals_overlay(results, mode, out)
        plot_metric_bars(results, mode, out)

    plot_training_time(results, out)
    plot_fr_zoom(results, out, t_start=2.0, t_end=5.0)

    print(f"\nGenerating per-model figures → {fig_dir}/")
    for name, entry in results["models"].items():
        if "error" in entry and len(entry) == 1:
            continue
        plot_individual_model(results, name, entry, fig_dir)

    # ── 4. JSON dump ──
    save_results_json(results, out)

    print(f"\n{'='*60}")
    print("Model comparison complete!")
    print(f"  Figures + JSON → {out}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
