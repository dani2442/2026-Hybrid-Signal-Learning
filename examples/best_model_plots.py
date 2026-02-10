#!/usr/bin/env python3
"""Generate best-model-per-timestep plots for free-run predictions.

For every time step the script identifies which model achieves the
smallest absolute residual and colours that point accordingly.

Outputs
-------
- ``best_residual.png``  – minimum |residual| at each time step
- ``best_prediction.png`` – prediction from the locally-best model
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from typing import Dict, List

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.benchmarking import build_benchmark_cases
from src.data import Dataset

# ── colour palette (user-specified) ────────────────────────────────────
MODEL_COLOURS: Dict[str, str] = {
    "NARX":              "#1f77b4",  # Blue
    "RandomForest":      "#ff7f0e",  # Orange
    "NeuralODE":         "#2ca02c",  # Green
    "NeuralSDE":         "#d62728",  # Red
    "NeuralCDE":         "#9467bd",  # Purple
    "LSTM":              "#8c564b",  # Brown
    "TCN":               "#e377c2",  # Pink
    "Mamba":             "#7f7f7f",  # Grey
    "UDE":               "#bcbd22",  # Olive
    "HybridLinearBeam":  "#17becf",  # Cyan
    "HybridNonlinearCam":"#000000",  # Black
    "GRU":               "#FFD700",  # Yellow
    "NeuralNetwork":     "#FF4500",  # OrangeRed (spare)
}
_FALLBACK_COLOURS = list(MODEL_COLOURS.values())


def _colour_for(name: str, idx: int) -> str:
    return MODEL_COLOURS.get(name, _FALLBACK_COLOURS[idx % len(_FALLBACK_COLOURS)])


# ── model training & prediction ───────────────────────────────────────
def _fit_model(model, u, y):
    sig = inspect.signature(model.fit)
    kw = {}
    if "verbose" in sig.parameters:
        kw["verbose"] = True
    model.fit(u, y, **kw)


def collect_free_run_predictions(
    model_keys: List[str],
    dataset_name: str = "multisine_05",
    resample_factor: int = 50,
    train_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Train each model and return (time, y_true, {name: y_pred_fr}).

    All prediction arrays are aligned to the same time grid
    (test set after ``max_lag`` initial conditions are consumed).
    """
    ds = Dataset.from_bab_experiment(
        dataset_name, preprocess=True, resample_factor=resample_factor,
    )
    train_ds, test_ds = ds.split(train_ratio)
    dt = 1.0 / ds.sampling_rate

    cases = build_benchmark_cases(model_keys)
    predictions: Dict[str, np.ndarray] = {}

    for i, case in enumerate(cases, 1):
        print(
            f"\n{'='*60}\n"
            f"[{i}/{len(cases)}] Training {case.name} …\n"
            f"{'='*60}"
        )
        model = case.factory(dt)
        try:
            _fit_model(model, train_ds.u, train_ds.y)
        except Exception as exc:
            print(f"  ✗ {case.name} failed during training: {exc}")
            continue

        try:
            y_init = test_ds.y[: model.max_lag]
            y_fr = model.predict(test_ds.u, y_init, mode="FR")
            predictions[case.name] = np.asarray(y_fr, dtype=float).flatten()
            print(f"  ✓ {case.name} — {len(y_fr)} predictions collected")
        except Exception as exc:
            print(f"  ✗ {case.name} free-run prediction failed: {exc}")

    # Align to the shortest common length
    max_lag = max(
        (build_benchmark_cases([k])[0].factory(dt).max_lag for k in model_keys),
        default=1,
    )
    y_true = test_ds.y[max_lag:]
    t = np.arange(len(y_true)) * dt

    # Trim all predictions to the same length
    n = len(y_true)
    aligned: Dict[str, np.ndarray] = {}
    for name, pred in predictions.items():
        aligned[name] = pred[:n] if len(pred) >= n else np.pad(
            pred, (0, n - len(pred)), constant_values=np.nan,
        )

    return t, y_true, aligned


# ── plotting ───────────────────────────────────────────────────────────
def _configure_matplotlib():
    import matplotlib
    matplotlib.rcParams.update({
        "font.family":       "serif",
        "font.size":         11,
        "axes.labelsize":    13,
        "axes.titlesize":    15,
        "legend.fontsize":   10,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "axes.grid":         True,
        "grid.alpha":        0.30,
        "grid.linewidth":    0.5,
    })


def plot_best_residual(
    t: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: str = "best_residual.png",
):
    """Plot the minimum |residual| at each time step, coloured by model."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    _configure_matplotlib()

    names = list(predictions.keys())
    n = len(t)

    # Stack residuals → (n_models, n_time)
    residuals = np.array([np.abs(predictions[nm] - y_true) for nm in names])
    best_idx = np.argmin(residuals, axis=0)           # winning model index
    best_res = residuals[best_idx, np.arange(n)]      # min residual value

    # Assign colours
    colours = [_colour_for(nm, i) for i, nm in enumerate(names)]
    point_colours = [colours[idx] for idx in best_idx]

    # Build coloured line segments
    points = np.column_stack([t, best_res]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_colours = point_colours[:-1]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    lc = LineCollection(segments, colors=seg_colours, linewidths=1.2)
    ax.add_collection(lc)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, np.nanpercentile(best_res, 99.5) * 1.15)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Minimum |Residual|")
    ax.set_title("Best Model Residual at Each Time Step (Free-Run)")

    # Legend — only models that win at least once
    winners = sorted(set(best_idx))
    handles = [
        Line2D([0], [0], color=colours[w], lw=2.5, label=names[w])
        for w in winners
    ]
    ncol = min(len(winners), 6)
    ax.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, -0.18), ncol=ncol,
        framealpha=0.9, columnspacing=1.2,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    print(f"\n  Saved: {save_path}")
    plt.close(fig)


def plot_best_prediction(
    t: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: str = "best_prediction.png",
):
    """Plot the best prediction at each time step, coloured by model."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    _configure_matplotlib()

    names = list(predictions.keys())
    n = len(t)

    # Find locally-best model (min |residual|)
    residuals = np.array([np.abs(predictions[nm] - y_true) for nm in names])
    best_idx = np.argmin(residuals, axis=0)

    # Assemble the best prediction signal
    pred_stack = np.array([predictions[nm] for nm in names])  # (n_models, n)
    best_pred = pred_stack[best_idx, np.arange(n)]

    colours = [_colour_for(nm, i) for i, nm in enumerate(names)]
    point_colours = [colours[idx] for idx in best_idx]

    # Coloured line segments for the best prediction
    points_pred = np.column_stack([t, best_pred]).reshape(-1, 1, 2)
    seg_pred = np.concatenate([points_pred[:-1], points_pred[1:]], axis=1)
    seg_colours = point_colours[:-1]

    fig, ax = plt.subplots(figsize=(14, 5))

    # True signal
    ax.plot(t, y_true, color="#333333", linewidth=1.4, alpha=0.55,
            linestyle="--", label="Ground Truth", zorder=1)

    # Best prediction (coloured)
    lc = LineCollection(seg_pred, colors=seg_colours, linewidths=1.4, zorder=2)
    ax.add_collection(lc)
    ax.set_xlim(t[0], t[-1])
    y_margin = (y_true.max() - y_true.min()) * 0.08
    ax.set_ylim(y_true.min() - y_margin, y_true.max() + y_margin)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Output (y)")
    ax.set_title("Best Model Prediction at Each Time Step (Free-Run)")

    # Legend — ground truth + winning models
    winners = sorted(set(best_idx))
    handles = [
        Line2D([0], [0], color="#333333", lw=1.4, ls="--", label="Ground Truth"),
    ] + [
        Line2D([0], [0], color=colours[w], lw=2.5, label=names[w])
        for w in winners
    ]
    ncol = min(len(handles), 6)
    ax.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, -0.16), ncol=ncol,
        framealpha=0.9, columnspacing=1.2,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────
def _parse_csv(v: str) -> list[str]:
    return [s.strip() for s in v.split(",") if s.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="narx,random_forest,neural_ode,neural_sde,neural_cde,"
                "lstm,tcn,mamba,ude,hybrid_linear_beam,hybrid_nonlinear_cam",
        help="Comma-separated model keys.",
    )
    parser.add_argument("--dataset", default="multisine_05")
    parser.add_argument("--resample-factor", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    model_keys = _parse_csv(args.models)
    out = args.output_dir

    # ── train ALL requested models once ──
    t, y_true, preds = collect_free_run_predictions(
        model_keys=model_keys,
        dataset_name=args.dataset,
        resample_factor=args.resample_factor,
        train_ratio=args.train_ratio,
    )

    if not preds:
        print("No predictions collected — nothing to plot.")
        return

    # ── all-models plots ──
    plot_best_residual(t, y_true, preds,
                       save_path=os.path.join(out, "best_residual.png"))
    plot_best_prediction(t, y_true, preds,
                         save_path=os.path.join(out, "best_prediction.png"))

    # ── continuous-time-only plots (reuse already-trained predictions) ──
    ct_preds = {k: v for k, v in preds.items()
                if any(ct in k.lower().replace(" ", "")
                       for ct in ["neuralode", "neuralsde", "neuralcde",
                                  "ude", "hybridlinear", "hybridnonlinear"])}
    if len(ct_preds) >= 2:
        print("\n" + "="*60)
        print("Generating continuous-time-only plots …")
        print("="*60)
        plot_best_residual(
            t, y_true, ct_preds,
            save_path=os.path.join(out, "best_residual_ct.png"),
        )
        plot_best_prediction(
            t, y_true, ct_preds,
            save_path=os.path.join(out, "best_prediction_ct.png"),
        )
    else:
        print("\n  ⚠ Fewer than 2 continuous-time models available; "
              "skipping CT-only plots.")

    print(f"\nDone — figures saved to {out}/")


if __name__ == "__main__":
    main()
