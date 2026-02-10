#!/usr/bin/env python3
"""Full model comparison: multi-dataset training and cross-dataset evaluation.

Prediction Modes
----------------
**OSA (One-Step-Ahead):** At each step the model receives the TRUE measured
output y[k] and predicts only y[k+1].  Errors never accumulate because the
model always resets from measured data.  This tests local dynamics accuracy.

**FR (Free-Run / Simulation):** The model receives only the initial condition
and then simulates the entire trajectory feeding its OWN predictions back.
Errors accumulate over time.  This is the gold-standard metric for system
identification — if a model has good FR performance it has truly learned the
system dynamics.

Multi-Dataset Training
----------------------
Models are trained on concatenated data from multiple excitation signals
(e.g., swept sines + multisines) and evaluated on unseen excitation types
(e.g., random steps) to test generalisation.

Usage
-----
  # Default: train on swept+multisine, test on random steps
  python examples/model_comparison.py

  # Single-dataset 80/20 split (legacy mode)
  python examples/model_comparison.py --single-dataset multisine_05

  # Quick test with specific models
  python examples/model_comparison.py --models narx,neural_ode,ude
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
    "LinearPhysics":     "#17becf",
    "StribeckPhysics":    "#000000",
    "VanillaNODE2D":      "#aec7e8",
    "StructuredNODE":     "#98df8a",
    "AdaptiveNODE":       "#c5b0d5",
    "GRU":                "#FFD700",
    "Neural Network":     "#FF4500",
}
_FALLBACK = list(MODEL_COLOURS.values())


def _colour(name: str, idx: int = 0) -> str:
    return MODEL_COLOURS.get(name, _FALLBACK[idx % len(_FALLBACK)])


def _configure_mpl():
    import matplotlib
    matplotlib.rcParams.update({
        "font.family":     "serif",
        "font.size":       11,
        "axes.labelsize":  13,
        "axes.titlesize":  14,
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


def _fit_model(model, u, y):
    sig = inspect.signature(model.fit)
    kw = {}
    if "verbose" in sig.parameters:
        kw["verbose"] = True
    model.fit(u, y, **kw)


# ── evaluation helpers (correct per-model alignment) ──────────────────

def _evaluate_osa(model, u, y, dt):
    """One-step-ahead prediction with correct per-model alignment.

    Different models return different-length predictions:
      NARX (max_lag=10)     -> len(y)-10 predictions starting at y[10]
      NeuralODE (max_lag=1) -> len(y)-1  predictions starting at y[1]

    The offset = len(y) - len(pred) tells us where predictions begin.
    """
    y_pred = model.predict(u, y, mode="OSA")
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    offset = len(y) - len(y_pred)
    y_true = y[offset : offset + len(y_pred)]
    t_axis = np.arange(len(y_pred)) * dt + offset * dt
    metrics = Metrics.compute_all(y_true, y_pred)
    return y_pred, y_true, t_axis, metrics


def _evaluate_fr(model, u, y, dt):
    """Free-run prediction with correct per-model alignment.

    Some models return the full trajectory (length = len(u)), including the
    initial condition.  Others already skip max_lag elements.  We normalise
    by always comparing from y[max_lag:] onward.
    """
    lag = model.max_lag
    y_init = y[:lag]
    y_pred_raw = model.predict(u, y_init, mode="FR")
    y_pred_raw = np.asarray(y_pred_raw, dtype=float).flatten()

    # offset = how many leading samples the model already removed
    offset = len(u) - len(y_pred_raw)

    # Ensure comparison starts at y[lag] (skip initial-condition region)
    if offset < lag:
        skip = lag - offset
        y_pred = y_pred_raw[skip:]
    else:
        y_pred = y_pred_raw

    y_true = y[lag : lag + len(y_pred)]
    t_axis = np.arange(len(y_pred)) * dt + lag * dt
    metrics = Metrics.compute_all(y_true, y_pred)
    return y_pred, y_true, t_axis, metrics


# ── training-loss summary ─────────────────────────────────────────────

# Loss function used by each model family (all are MSE variants):
_LOSS_DESCRIPTIONS: Dict[str, str] = {
    "NeuralODE":          "MSE  = mean((ŷ − y)²)  over full trajectory",
    "NeuralSDE":          "MSE  = mean((ŷ − y)²)  over random subsequences",
    "NeuralCDE":          "MSE  = mean((ŷ − y)²)  over random subsequences",
    "UDE":                "MSE  = mean((θ̂ − θ)²)  over full trajectory (θ only)",
    "LinearPhysics":     "MSE  = mean((θ̂ − y)²)  over full trajectory",
    "StribeckPhysics":    "MSE  = mean((θ̂ − y)²)  over full trajectory",
    "VanillaNODE2D":      "MSE  = mean((x̂ − x)²)  multiple-shooting (K windows)",
    "StructuredNODE":     "MSE  = mean((x̂ − x)²)  multiple-shooting (K windows)",
    "AdaptiveNODE":       "MSE  = mean((x̂ − x)²)  multiple-shooting (K windows)",
    "LSTM":               "MSE  = mean((ŷ − y)²)  over mini-batches",
    "GRU":                "MSE  = mean((ŷ − y)²)  over mini-batches",
    "TCN":                "MSE  = mean((ŷ − y)²)  over mini-batches",
    "Neural Network":     "MSE  = mean((ŷ − y)²)  over mini-batches",
    "Mamba":              "MSE  = mean((ŷ − y)²)  over mini-batches",
}


def _print_training_summary(model, name: str, fit_s: float):
    """Print training loss info right after fitting."""
    loss_desc = _LOSS_DESCRIPTIONS.get(name, "(analytical / non-iterative)")
    print(f"  Training done in {fit_s:.1f}s")
    print(f"  Loss metric: {loss_desc}")

    loss_hist = getattr(model, "training_loss_", None)
    if loss_hist and len(loss_hist) > 0:
        final_loss = loss_hist[-1]
        best_loss = min(loss_hist)
        print(f"  Final loss = {final_loss:.6f}   "
              f"Best loss = {best_loss:.6f}   "
              f"({len(loss_hist)} epochs)")


# ── dataset loading ───────────────────────────────────────────────────

def load_datasets(names: List[str], resample_factor: int = 50) -> List[Dataset]:
    datasets = []
    for name in names:
        ds = Dataset.from_bab_experiment(
            name, preprocess=True, resample_factor=resample_factor,
        )
        datasets.append(ds)
    return datasets


# ── core loop ─────────────────────────────────────────────────────────

def train_and_evaluate(
    model_keys: List[str],
    train_datasets: List[Dataset],
    test_datasets: List[Dataset],
    dt: float,
) -> dict:
    """Train on concatenated training data, evaluate on every dataset.

    Multi-dataset protocol
    ----------------------
    The input/output arrays from all training datasets are **concatenated**
    into a single long signal (u_train, y_train).  This works directly for
    lag-based models (NARX, LSTM, etc.) because each prediction only uses
    a short local window of lagged values.

    For differential-equation models (NeuralODE, NeuralSDE, UDE, …) that
    integrate over long horizons, concatenation creates artificial
    discontinuities at dataset boundaries.  To handle this, the code
    automatically switches to **subsequence training** when there are
    multiple training datasets: random short subsequences (e.g. 50 steps)
    are sampled from the concatenated data.  As long as the subsequence
    length is much shorter than each individual dataset (~1100 samples),
    the probability of sampling across a boundary is negligible.
    """
    u_train = np.concatenate([ds.u for ds in train_datasets])
    y_train = np.concatenate([ds.y for ds in train_datasets])

    cases = build_benchmark_cases(model_keys)
    results: dict = {
        "dt": dt,
        "train_names": [ds.name for ds in train_datasets],
        "test_names": [ds.name for ds in test_datasets],
        "n_train_total": len(u_train),
        "models": {},
    }

    for i, case in enumerate(cases, 1):
        tag = f"[{i}/{len(cases)}] {case.name}"
        print(f"\n{'='*60}\n{tag}\n{'='*60}")
        model = case.factory(dt)

        # Multi-dataset: force subsequence training for DE models so that
        # boundary discontinuities between concatenated datasets don't
        # destabilise full-trajectory integration.
        if len(train_datasets) > 1 and hasattr(model, "training_mode"):
            model.training_mode = "subsequence"

        t0 = time.perf_counter()
        try:
            _fit_model(model, u_train, y_train)
            fit_s = time.perf_counter() - t0
            _print_training_summary(model, case.name, fit_s)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results["models"][case.name] = {"error": str(exc)}
            continue

        # Collect training loss history
        loss_hist = getattr(model, "training_loss_", None)
        best_loss = None
        final_loss = None
        if loss_hist and len(loss_hist) > 0:
            final_loss = float(loss_hist[-1])
            best_loss = float(min(loss_hist))

        entry: dict = {
            "fit_seconds": fit_s,
            "max_lag": model.max_lag,
            "loss_metric": _LOSS_DESCRIPTIONS.get(case.name,
                                                   "analytical / non-iterative"),
            "best_loss": best_loss,
            "final_loss": final_loss,
            "train_fr": {},
            "test_osa": {},
            "test_fr": {},
        }

        # ── Training-set fit (FR on each training dataset) ──
        for ds in train_datasets:
            try:
                pred, true, t_ax, m = _evaluate_fr(model, ds.u, ds.y, dt)
                entry["train_fr"][ds.name] = {
                    "pred": pred, "true": true, "t": t_ax, "metrics": m,
                }
                print(f"    Train-FR {ds.name}: "
                      f"R²={m['R2']:.4f}  FIT%={m['FIT%']:.4f}")
            except Exception as exc:
                print(f"    Train-FR {ds.name} failed: {exc}")

        # ── Test predictions (OSA + FR on each test dataset) ──
        for ds in test_datasets:
            try:
                pred, true, t_ax, m = _evaluate_osa(model, ds.u, ds.y, dt)
                entry["test_osa"][ds.name] = {
                    "pred": pred, "true": true, "t": t_ax, "metrics": m,
                }
                print(f"    Test-OSA {ds.name}: R²={m['R2']:.4f}")
            except Exception as exc:
                print(f"    Test-OSA {ds.name} failed: {exc}")

            try:
                pred, true, t_ax, m = _evaluate_fr(model, ds.u, ds.y, dt)
                entry["test_fr"][ds.name] = {
                    "pred": pred, "true": true, "t": t_ax, "metrics": m,
                }
                print(f"    Test-FR  {ds.name}: R²={m['R2']:.4f}")
            except Exception as exc:
                print(f"    Test-FR  {ds.name} failed: {exc}")

        results["models"][case.name] = entry

    return results


# ── plotting ──────────────────────────────────────────────────────────

def plot_model_overview(
    name: str,
    entry: dict,
    train_datasets: List[Dataset],
    test_datasets: List[Dataset],
    dt: float,
    out_dir: str,
):
    """One summary figure per model: training fit + test FR predictions."""
    import matplotlib.pyplot as plt
    _configure_mpl()

    c = _colour(name)
    n_train = len(train_datasets)
    n_test = len(test_datasets)
    n_rows = n_train + n_test
    if n_rows == 0:
        return

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(14, 3.2 * n_rows), sharex=False,
    )
    if n_rows == 1:
        axes = [axes]

    row = 0
    for ds in train_datasets:
        ax = axes[row]
        d = entry.get("train_fr", {}).get(ds.name)
        if d:
            ax.plot(d["t"], d["true"], "k-", lw=1.2, alpha=0.7,
                    label="Ground Truth")
            r2 = d["metrics"]["R2"]
            ax.plot(d["t"], d["pred"], color=c, lw=1.0, alpha=0.85,
                    label=f"FR (R²={r2:.4f})")
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=8,
            )
        ax.set_title(f"TRAIN: {ds.name}  —  Free-Run Simulation", fontsize=11)
        ax.set_ylabel("y")
        row += 1

    for ds in test_datasets:
        ax = axes[row]
        d = entry.get("test_fr", {}).get(ds.name)
        if d:
            ax.plot(d["t"], d["true"], "k-", lw=1.2, alpha=0.7,
                    label="Ground Truth")
            r2 = d["metrics"]["R2"]
            ax.plot(d["t"], d["pred"], color=c, lw=1.0, alpha=0.85,
                    label=f"FR (R²={r2:.4f})")
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=8,
            )
        ax.set_title(f"TEST: {ds.name}  —  Free-Run Simulation", fontsize=11)
        ax.set_ylabel("y")
        row += 1

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(name, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.82, 0.97])

    safe = name.lower().replace(" ", "_")
    path = os.path.join(out_dir, f"{safe}_overview.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_model_osa_vs_fr(
    name: str,
    entry: dict,
    test_datasets: List[Dataset],
    dt: float,
    out_dir: str,
):
    """Side-by-side OSA vs FR figure — shows the difference visually."""
    import matplotlib.pyplot as plt
    _configure_mpl()

    c = _colour(name)
    n_test = len(test_datasets)
    if n_test == 0:
        return

    fig, axes = plt.subplots(
        n_test, 2, figsize=(14, 3.0 * n_test), sharex=False, squeeze=False,
    )

    for row, ds in enumerate(test_datasets):
        # ── OSA column ──
        ax = axes[row, 0]
        d_osa = entry.get("test_osa", {}).get(ds.name)
        if d_osa:
            ax.plot(d_osa["t"], d_osa["true"], "k-", lw=1.0, alpha=0.7,
                    label="Ground Truth")
            r2 = d_osa["metrics"]["R2"]
            ax.plot(d_osa["t"], d_osa["pred"], color=c, lw=0.9, alpha=0.85,
                    label=f"R²={r2:.4f}")
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=7,
            )
        ax.set_title(
            f"{ds.name} — OSA\n(model gets true y[k], predicts y[k+1])",
            fontsize=9,
        )
        ax.set_ylabel("y")

        # ── FR column ──
        ax = axes[row, 1]
        d_fr = entry.get("test_fr", {}).get(ds.name)
        if d_fr:
            ax.plot(d_fr["t"], d_fr["true"], "k-", lw=1.0, alpha=0.7,
                    label="Ground Truth")
            r2 = d_fr["metrics"]["R2"]
            ax.plot(d_fr["t"], d_fr["pred"], color=c, lw=0.9, alpha=0.85,
                    label=f"R²={r2:.4f}")
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=7,
            )
        ax.set_title(
            f"{ds.name} — FR\n(model uses its own predictions)",
            fontsize=9,
        )
        ax.set_ylabel("y")

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")

    fig.suptitle(f"{name}:  OSA vs Free-Run", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.82, 0.97])

    safe = name.lower().replace(" ", "_")
    path = os.path.join(out_dir, f"{safe}_osa_vs_fr.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fr_overlay(results: dict, ds_name: str, ds_kind: str, out_dir: str):
    """FR overlay of all models on one dataset."""
    import matplotlib.pyplot as plt
    _configure_mpl()

    fig, ax = plt.subplots(figsize=(14, 5))
    truth_plotted = False

    for idx, (name, entry) in enumerate(results["models"].items()):
        if "error" in entry:
            continue
        sub = entry.get(f"{ds_kind}_fr", {}).get(ds_name)
        if not sub:
            continue
        if not truth_plotted:
            ax.plot(sub["t"], sub["true"], "k-", lw=1.4, alpha=0.7,
                    label="Ground Truth", zorder=10)
            truth_plotted = True
        c = _colour(name, idx)
        r2 = sub["metrics"]["R2"]
        ax.plot(sub["t"], sub["pred"], color=c, lw=0.9, alpha=0.85,
                label=f"{name} (R²={r2:.3f})")

    kind_label = "TRAIN" if ds_kind == "train" else "TEST"
    ax.set_title(f"Free-Run Simulation  —  {kind_label}: {ds_name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("y")

    handles, labels = ax.get_legend_handles_labels()
    ncol = min(len(handles), 5)
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), ncol=ncol, framealpha=0.9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26)

    safe = ds_name.lower().replace(" ", "_")
    path = os.path.join(out_dir, f"fr_{ds_kind}_{safe}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_osa_overlay(results: dict, ds_name: str, out_dir: str):
    """OSA overlay on one test dataset."""
    import matplotlib.pyplot as plt
    _configure_mpl()

    fig, ax = plt.subplots(figsize=(14, 5))
    truth_plotted = False

    for idx, (name, entry) in enumerate(results["models"].items()):
        if "error" in entry:
            continue
        sub = entry.get("test_osa", {}).get(ds_name)
        if not sub:
            continue
        if not truth_plotted:
            ax.plot(sub["t"], sub["true"], "k-", lw=1.4, alpha=0.7,
                    label="Ground Truth", zorder=10)
            truth_plotted = True
        c = _colour(name, idx)
        r2 = sub["metrics"]["R2"]
        ax.plot(sub["t"], sub["pred"], color=c, lw=0.9, alpha=0.85,
                label=f"{name} (R²={r2:.3f})")

    ax.set_title(
        f"One-Step-Ahead  —  TEST: {ds_name}\n"
        "(model receives true y[k] at each step, predicts y[k+1])"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("y")

    handles, labels = ax.get_legend_handles_labels()
    ncol = min(len(handles), 5)
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), ncol=ncol, framealpha=0.9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26)

    safe = ds_name.lower().replace(" ", "_")
    path = os.path.join(out_dir, f"osa_test_{safe}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_metrics_summary(results: dict, out_dir: str):
    """Horizontal bar chart: average FR R² across test datasets."""
    import matplotlib.pyplot as plt
    _configure_mpl()

    names, avg_r2, colours = [], [], []
    for idx, (name, entry) in enumerate(results["models"].items()):
        if "error" in entry:
            continue
        r2_vals = [s["metrics"]["R2"]
                   for s in entry.get("test_fr", {}).values()]
        if r2_vals:
            names.append(name)
            avg_r2.append(float(np.mean(r2_vals)))
            colours.append(_colour(name, idx))

    if not names:
        return

    order = np.argsort(avg_r2)[::-1]
    names = [names[i] for i in order]
    avg_r2 = [avg_r2[i] for i in order]
    colours = [colours[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(names))))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, avg_r2, color=colours, edgecolor="white",
                   linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Average R² (Free-Run, across test datasets)")
    ax.set_title("Model Generalisation — Free-Run R²")
    ax.invert_yaxis()
    ax.axvline(0, color="gray", lw=0.7, ls="--")

    for bar, v in zip(bars, avg_r2):
        ax.text(bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "fr_r2_summary.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_training_time(results: dict, out_dir: str):
    """Horizontal bar chart of training time per model."""
    import matplotlib.pyplot as plt
    _configure_mpl()

    names, times, colours = [], [], []
    for idx, (name, entry) in enumerate(results["models"].items()):
        if "fit_seconds" not in entry:
            continue
        names.append(name)
        times.append(entry["fit_seconds"])
        colours.append(_colour(name, idx))

    if not names:
        return

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
        ax.text(bar.get_width() + max(times) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{t_val:.1f}s", va="center", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "training_time.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── leaderboard and JSON ─────────────────────────────────────────────

def print_leaderboard(
    results: dict, ds_name: str | None = None, mode: str = "FR",
):
    key = "test_fr" if mode == "FR" else "test_osa"
    rows = []
    for name, entry in results["models"].items():
        if "error" in entry:
            continue
        if ds_name:
            sub = entry.get(key, {}).get(ds_name)
            if sub:
                m = sub["metrics"]
                rows.append({
                    "Model": name, "R2": m["R2"], "FIT%": m["FIT%"],
                    "NRMSE": m["NRMSE"], "RMSE": m["RMSE"], "MAE": m["MAE"],
                    "Time": entry.get("fit_seconds", 0.0),
                })
        else:
            r2s, fits, nrmses = [], [], []
            for _, sub in entry.get(key, {}).items():
                m = sub["metrics"]
                r2s.append(m["R2"])
                fits.append(m["FIT%"])
                nrmses.append(m["NRMSE"])
            if r2s:
                rows.append({
                    "Model": name, "R2": np.mean(r2s), "FIT%": np.mean(fits),
                    "NRMSE": np.mean(nrmses),
                    "Time": entry.get("fit_seconds", 0.0),
                })

    rows.sort(key=lambda r: r.get("R2", -999), reverse=True)

    scope = f" — {ds_name}" if ds_name else " (avg across test sets)"
    title = f"{mode} LEADERBOARD{scope}"
    header = (
        f"{'#':>3}  {'Model':<22} {'R²':>8} {'FIT%':>8} "
        f"{'NRMSE':>8} {'Time':>8}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}\n  {title}\n{sep}\n{header}\n{sep}")
    for rank, r in enumerate(rows, 1):
        print(
            f"{rank:3d}  {r['Model']:<22} {r['R2']:8.4f} "
            f"{r.get('FIT%', 0):8.4f} {r.get('NRMSE', 0):8.4f} "
            f"{r.get('Time', 0):8.1f}"
        )
    print(sep)


def save_results_json(results: dict, out_dir: str):
    """Persist numeric metrics (without numpy arrays)."""
    payload = {
        "train_datasets": results.get("train_names", []),
        "test_datasets": results.get("test_names", []),
        "n_train_total": results.get("n_train_total", 0),
        "dt": results["dt"],
        "models": {},
    }
    for name, entry in results["models"].items():
        if "error" in entry:
            payload["models"][name] = {"error": entry["error"]}
            continue
        d: dict = {
            "fit_seconds": entry["fit_seconds"],
            "max_lag": entry["max_lag"],
            "loss_metric": entry.get("loss_metric"),
            "best_loss": entry.get("best_loss"),
            "final_loss": entry.get("final_loss"),
        }
        for section in ("train_fr", "test_osa", "test_fr"):
            d[section] = {}
            for ds_name, sub in entry.get(section, {}).items():
                d[section][ds_name] = sub["metrics"]
        payload["models"][name] = d

    path = os.path.join(out_dir, "comparison_results.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {path}")


# ── CLI & main ────────────────────────────────────────────────────────

ALL_MODELS = (
    "narx,random_forest,neural_ode,neural_sde,neural_cde,"
    "lstm,tcn,mamba,ude,"
    "linear_physics,stribeck_physics,"
    "vanilla_node_2d,structured_node,adaptive_node,"
    "gru,neural_network"
)

DEFAULT_TRAIN = "swept_sine,multisine_05,multisine_06"
DEFAULT_TEST = (
    "random_steps_01,random_steps_02,random_steps_03,random_steps_04"
)


def main():
    parser = argparse.ArgumentParser(
        description="Model comparison with multi-dataset support.",
    )
    parser.add_argument("--models", default=ALL_MODELS,
                        help="Comma-separated model keys.")
    parser.add_argument("--train-datasets", default=DEFAULT_TRAIN,
                        help="Comma-separated training dataset keys.")
    parser.add_argument("--test-datasets", default=DEFAULT_TEST,
                        help="Comma-separated test dataset keys.")
    parser.add_argument("--single-dataset", default=None,
                        help="Use a single dataset with 80/20 split instead "
                             "of the multi-dataset protocol.")
    parser.add_argument("--resample-factor", type=int, default=50)
    parser.add_argument("--output-dir", default="comparison")
    args = parser.parse_args()

    model_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    fig_dir = os.path.join(ROOT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── 1. Load datasets ──
    if args.single_dataset:
        ds = Dataset.from_bab_experiment(
            args.single_dataset, preprocess=True,
            resample_factor=args.resample_factor,
        )
        train_ds, test_ds = ds.split(0.8)
        train_datasets = [train_ds]
        test_datasets = [test_ds]
    else:
        train_names = [n.strip() for n in args.train_datasets.split(",")
                       if n.strip()]
        test_names = [n.strip() for n in args.test_datasets.split(",")
                      if n.strip()]
        train_datasets = load_datasets(train_names, args.resample_factor)
        test_datasets = load_datasets(test_names, args.resample_factor)

    dt = 1.0 / train_datasets[0].sampling_rate
    n_train = sum(len(ds) for ds in train_datasets)
    n_test = sum(len(ds) for ds in test_datasets)

    print(f"\n{'─'*60}")
    print("  TRAINING PROTOCOL")
    print(f"{'─'*60}")
    if args.single_dataset:
        print("  Mode: single-dataset 80/20 split")
    else:
        print("  Mode: multi-dataset (concatenated training signals)")
        print("  Note: Lag-based models see the concatenated array directly.")
        print("        DE models use subsequence training to avoid boundary")
        print("        discontinuities between datasets.")
    print(f"  Training datasets: {[ds.name for ds in train_datasets]}")
    print(f"    Total: {n_train} samples")
    print(f"  Test datasets:     {[ds.name for ds in test_datasets]}")
    print(f"    Total: {n_test} samples")
    print(f"  dt = {dt:.4f}s,  fs = {1/dt:.1f} Hz")
    print(f"{'─'*60}\n")

    # ── 2. Train & evaluate ──
    results = train_and_evaluate(
        model_keys, train_datasets, test_datasets, dt,
    )

    if not results["models"]:
        print("No models completed — nothing to do.")
        return

    # ── 3. Leaderboards ──
    print_leaderboard(results, mode="FR")
    for ds in test_datasets:
        print_leaderboard(results, ds_name=ds.name, mode="FR")

    # ── 4. Overlay figures ──
    print(f"\nGenerating overlay figures → {out}/")
    for ds in train_datasets:
        plot_fr_overlay(results, ds.name, "train", out)
    for ds in test_datasets:
        plot_fr_overlay(results, ds.name, "test", out)
        plot_osa_overlay(results, ds.name, out)

    plot_metrics_summary(results, out)
    plot_training_time(results, out)

    # ── 5. Per-model figures ──
    print(f"\nGenerating per-model figures → {fig_dir}/")
    for name, entry in results["models"].items():
        if "error" in entry:
            continue
        plot_model_overview(
            name, entry, train_datasets, test_datasets, dt, fig_dir,
        )
        plot_model_osa_vs_fr(name, entry, test_datasets, dt, fig_dir)

    # ── 6. JSON dump ──
    save_results_json(results, out)

    print(f"\n{'='*60}")
    print("Model comparison complete!")
    print(f"  Per-model figures → {fig_dir}/")
    print(f"  Overlays + JSON   → {out}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
