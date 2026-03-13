#!/usr/bin/env python
"""Compare all BAB benchmark results from a run folder.

Loads ``metrics_aggregate.csv`` and ``metrics_long.csv``, filters out
diverged / NaN models, and produces a battery of comparison plots saved
into the ``plots/`` sub-folder of the run directory.

Usage
-----
    python examples/compare_bab_results.py [RUN_DIR]

If *RUN_DIR* is omitted the most recent run under ``results/bab_runs/``
is used automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Reuse colour / style conventions from the main package
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid_signal_learning.plots import (
    DEFAULT_COLORS,
    DEFAULT_STYLES,
    model_display_name,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Thresholds used to discard diverged models from the visualisations
RMSE_CLIP = 50.0        # any RMSE_mean above this is considered diverged
R2_FLOOR = -0.5          # any R² mean below this is clipped / excluded
FIT_FLOOR = -0.5         # any FIT% mean below this is clipped

METRIC_LABELS = {
    "rmse_pos_mean": "RMSE Position",
    "rmse_vel_mean": "RMSE Velocity",
    "r2_pos_mean":   "R² Position",
    "r2_vel_mean":   "R² Velocity",
    "fit_pos_mean":  "FIT% Position",
    "fit_vel_mean":  "FIT% Velocity",
}

# Metrics where *lower* is better
LOWER_IS_BETTER = {"rmse_pos_mean", "rmse_vel_mean"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_run_dir(argv: list[str]) -> Path:
    """Return the run directory from CLI arg or pick the latest."""
    if len(argv) > 1:
        p = Path(argv[1])
        if p.is_dir():
            return p
    runs_root = ROOT / "results" / "bab_runs"
    candidates = sorted(runs_root.glob("run_*"))
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {runs_root}")
    return candidates[-1]


def _model_label(row) -> str:
    """Combine model_key + nn_variant into a readable label."""
    return f"{model_display_name(row['model_key'])} ({row['nn_variant']})"


def _load_and_filter(run_dir: Path):
    """Load aggregate CSV and apply sanity filters."""
    csv_path = run_dir / "tables" / "metrics_aggregate.csv"
    df = pd.read_csv(csv_path)

    # Build a combined label
    df["model_label"] = df.apply(_model_label, axis=1)

    # Replace inf with NaN
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return df


def _filter_diverged(df: pd.DataFrame) -> pd.DataFrame:
    """Remove model+variant combos that diverge on *all* test datasets."""
    test = df[df["split"] == "test"].copy()
    # A model-variant is diverged if RMSE is huge or NaN everywhere
    grp = test.groupby(["model_key", "nn_variant"])
    keep = []
    for (mk, nv), gdf in grp:
        rmse_ok = (gdf["rmse_pos_mean"] < RMSE_CLIP) & gdf["rmse_pos_mean"].notna()
        if rmse_ok.any():
            keep.append((mk, nv))
    idx = pd.MultiIndex.from_tuples(keep, names=["model_key", "nn_variant"])
    return df.set_index(["model_key", "nn_variant"]).loc[idx].reset_index()


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def _color_for(model_key: str) -> str:
    return DEFAULT_COLORS.get(model_key, "gray")


def plot_test_metric_heatmap(
    df: pd.DataFrame,
    metric: str,
    *,
    save_path: Path | None = None,
) -> plt.Figure:
    """Heatmap of one metric across models × datasets (test split only)."""
    test = df[df["split"] == "test"].copy()

    # Clip extreme values for display
    if metric in LOWER_IS_BETTER:
        test[metric] = test[metric].clip(upper=RMSE_CLIP)
    elif "r2" in metric:
        test[metric] = test[metric].clip(lower=R2_FLOOR)
    elif "fit" in metric:
        test[metric] = test[metric].clip(lower=FIT_FLOOR)

    pivot = test.pivot_table(
        index="model_label", columns="dataset", values=metric, aggfunc="mean"
    )
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.2), max(6, len(pivot) * 0.4)))

    if metric in LOWER_IS_BETTER:
        cmap = "RdYlGn_r"
    else:
        cmap = "RdYlGn"

    # Set sensible colour-scale bounds so extreme negatives don't wash out
    if metric in LOWER_IS_BETTER:
        vmin, vmax = 0, min(RMSE_CLIP, np.nanmax(pivot.values))
    elif "r2" in metric:
        vmin, vmax = R2_FLOOR, 1.0
    elif "fit" in metric:
        vmin, vmax = FIT_FLOOR, 100.0
    else:
        vmin, vmax = None, None

    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, interpolation="nearest",
                   vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.8, label=METRIC_LABELS.get(metric, metric))

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(f"Test {METRIC_LABELS.get(metric, metric)} by model × dataset")
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_bar_comparison(
    df: pd.DataFrame,
    metric: str,
    *,
    datasets: list[str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Grouped bar chart of one metric for each dataset (test split)."""
    test = df[df["split"] == "test"].copy()
    if datasets:
        test = test[test["dataset"].isin(datasets)]

    # Clip
    if metric in LOWER_IS_BETTER:
        test[metric] = test[metric].clip(upper=RMSE_CLIP)
    elif "r2" in metric:
        test[metric] = test[metric].clip(lower=R2_FLOOR)
    elif "fit" in metric:
        test[metric] = test[metric].clip(lower=FIT_FLOOR)

    ds_list = sorted(test["dataset"].unique())
    models = sorted(test["model_label"].unique())

    n_ds = len(ds_list)
    n_mod = len(models)
    x = np.arange(n_ds)
    width = 0.8 / max(n_mod, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_ds * 1.5), 7))

    model_keys_seen = {}
    for i, ml in enumerate(models):
        vals = []
        for ds in ds_list:
            sub = test[(test["model_label"] == ml) & (test["dataset"] == ds)]
            vals.append(sub[metric].mean() if len(sub) else np.nan)

        mk = test[test["model_label"] == ml]["model_key"].iloc[0]
        color = _color_for(mk)
        # Slightly vary shade for different variants of the same model
        if mk in model_keys_seen:
            import colorsys
            rgb = matplotlib.colors.to_rgb(color)
            h, l, s = colorsys.rgb_to_hls(*rgb)
            l = min(1.0, l + 0.12 * model_keys_seen[mk])
            color = colorsys.hls_to_rgb(h, l, s)
            model_keys_seen[mk] += 1
        else:
            model_keys_seen[mk] = 1

        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=ml, color=color, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_list, rotation=45, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"Test {METRIC_LABELS.get(metric, metric)} per dataset")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=6, ncol=3, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metric_boxplots(
    df: pd.DataFrame,
    metric: str,
    *,
    save_path: Path | None = None,
) -> plt.Figure:
    """Box plot of one metric across datasets for each model (test split)."""
    test = df[df["split"] == "test"].copy()

    if metric in LOWER_IS_BETTER:
        test[metric] = test[metric].clip(upper=RMSE_CLIP)
    elif "r2" in metric:
        test[metric] = test[metric].clip(lower=R2_FLOOR)
    elif "fit" in metric:
        test[metric] = test[metric].clip(lower=FIT_FLOOR)

    models = sorted(test["model_label"].unique())
    data_per_model = []
    labels = []
    colors = []
    for ml in models:
        vals = test[test["model_label"] == ml][metric].dropna().values
        if len(vals) == 0:
            continue
        data_per_model.append(vals)
        labels.append(ml)
        mk = test[test["model_label"] == ml]["model_key"].iloc[0]
        colors.append(_color_for(mk))

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.6), 6))
    bp = ax.boxplot(data_per_model, patch_artist=True, vert=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"Test {METRIC_LABELS.get(metric, metric)} across datasets")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_train_vs_test(
    df: pd.DataFrame,
    metric: str,
    *,
    save_path: Path | None = None,
) -> plt.Figure:
    """Scatter: train metric vs test metric for each model × dataset."""
    both = df[df["split"].isin(["train", "test"])].copy()

    if metric in LOWER_IS_BETTER:
        both[metric] = both[metric].clip(upper=RMSE_CLIP)
    elif "r2" in metric:
        both[metric] = both[metric].clip(lower=R2_FLOOR)
    elif "fit" in metric:
        both[metric] = both[metric].clip(lower=FIT_FLOOR)

    pivot = both.pivot_table(
        index=["model_label", "model_key", "dataset"],
        columns="split",
        values=metric,
    ).dropna()

    if pivot.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No paired train/test data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(7, 7))
    lo = min(pivot["train"].min(), pivot["test"].min())
    hi = max(pivot["train"].max(), pivot["test"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="y = x")

    for (ml, mk, ds), row in pivot.iterrows():
        ax.scatter(row["train"], row["test"], color=_color_for(mk), s=30, alpha=0.6)

    # Legend: one entry per model_key
    handles = []
    for mk in sorted(pivot.index.get_level_values("model_key").unique()):
        h = matplotlib.lines.Line2D(
            [], [], marker="o", linestyle="", color=_color_for(mk),
            label=model_display_name(mk), markersize=6,
        )
        handles.append(h)
    ax.legend(handles=handles, fontsize=7, loc="upper left")

    ax.set_xlabel(f"Train {METRIC_LABELS.get(metric, metric)}")
    ax.set_ylabel(f"Test {METRIC_LABELS.get(metric, metric)}")
    ax.set_title(f"Train vs Test {METRIC_LABELS.get(metric, metric)}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_summary_radar(
    df: pd.DataFrame,
    *,
    save_path: Path | None = None,
) -> plt.Figure:
    """Radar / spider chart showing average (normalised) test metrics per model.

    Only models with *all* metrics finite (after filtering) are shown.
    """
    test = df[df["split"] == "test"].copy()

    radar_metrics = ["r2_pos_mean", "r2_vel_mean", "fit_pos_mean", "fit_vel_mean"]
    for m in radar_metrics:
        if "r2" in m:
            test[m] = test[m].clip(lower=R2_FLOOR)
        elif "fit" in m:
            test[m] = test[m].clip(lower=FIT_FLOOR)

    avg = test.groupby(["model_label", "model_key"])[radar_metrics].mean().dropna()
    if avg.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data for radar", ha="center", va="center")
        return fig

    # Normalise each metric to [0, 1]
    normed = avg.copy()
    for m in radar_metrics:
        lo, hi = normed[m].min(), normed[m].max()
        if hi - lo > 1e-9:
            normed[m] = (normed[m] - lo) / (hi - lo)
        else:
            normed[m] = 0.5

    N = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for (ml, mk), row in normed.iterrows():
        vals = row.tolist() + [row.iloc[0]]
        ax.plot(angles, vals, linewidth=1.4, label=ml, color=_color_for(mk), alpha=0.7)
        ax.fill(angles, vals, color=_color_for(mk), alpha=0.05)

    ax.set_thetagrids(
        np.degrees(angles[:-1]),
        [METRIC_LABELS.get(m, m) for m in radar_metrics],
        fontsize=8,
    )
    ax.set_title("Normalised test metrics (higher = better)", pad=20)
    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_per_dataset_ranking(
    df: pd.DataFrame,
    metric: str = "fit_pos_mean",
    *,
    save_path: Path | None = None,
) -> plt.Figure:
    """Horizontal bar charts: rank models per dataset (test split)."""
    test = df[df["split"] == "test"].copy()
    if "r2" in metric:
        test[metric] = test[metric].clip(lower=R2_FLOOR)
    elif "fit" in metric:
        test[metric] = test[metric].clip(lower=FIT_FLOOR)
    elif metric in LOWER_IS_BETTER:
        test[metric] = test[metric].clip(upper=RMSE_CLIP)

    ds_list = sorted(test["dataset"].unique())
    n_ds = len(ds_list)
    ncols = min(3, n_ds)
    nrows = int(np.ceil(n_ds / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, max(4, nrows * 3.5)))
    if n_ds == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    ascending = metric in LOWER_IS_BETTER
    for idx, ds in enumerate(ds_list):
        ax = axes[idx // ncols, idx % ncols]
        sub = test[test["dataset"] == ds][["model_label", "model_key", metric]].dropna()
        sub = sub.sort_values(metric, ascending=ascending)

        colors = [_color_for(mk) for mk in sub["model_key"]]
        ax.barh(sub["model_label"], sub[metric], color=colors, alpha=0.7, edgecolor="white")
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel(METRIC_LABELS.get(metric, metric), fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, axis="x", alpha=0.3)

    # Hide empty subplots
    for idx in range(n_ds, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f"Per-dataset ranking — Test {METRIC_LABELS.get(metric, metric)}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_long_violin(
    long_df: pd.DataFrame,
    metric: str = "fit_pos",
    *,
    save_path: Path | None = None,
) -> plt.Figure:
    """Violin plot from the *long* (per-run) CSV to show run variability."""
    test = long_df[long_df["split"] == "test"].copy()

    if "r2" in metric:
        test[metric] = test[metric].clip(lower=R2_FLOOR)
    elif "fit" in metric:
        test[metric] = test[metric].clip(lower=FIT_FLOOR)
    elif "rmse" in metric:
        test[metric] = test[metric].clip(upper=RMSE_CLIP)

    test["model_label"] = test.apply(_model_label, axis=1)
    models = sorted(test["model_label"].unique())

    data, labels, colors = [], [], []
    for ml in models:
        vals = test[test["model_label"] == ml][metric].dropna().values
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(ml)
        mk = test[test["model_label"] == ml]["model_key"].iloc[0]
        colors.append(_color_for(mk))

    if not data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.6), 6))
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.5)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric.replace("_", " ").title()))
    ax.set_title(f"Test {metric.replace('_', ' ').title()} — per run (violin)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    run_dir = _resolve_run_dir(sys.argv)
    print(f"Run directory: {run_dir}")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data --------------------------------------------------------
    df_agg = _load_and_filter(run_dir)
    print(f"Loaded aggregate CSV: {len(df_agg)} rows")

    long_csv = run_dir / "tables" / "metrics_long.csv"
    df_long = pd.read_csv(long_csv) if long_csv.exists() else None
    if df_long is not None:
        numeric_cols = df_long.select_dtypes(include="number").columns
        df_long[numeric_cols] = df_long[numeric_cols].replace([np.inf, -np.inf], np.nan)
        print(f"Loaded long CSV: {len(df_long)} rows")

    # --- Filter diverged models ------------------------------------------
    df_good = _filter_diverged(df_agg)
    n_orig = df_agg[["model_key", "nn_variant"]].drop_duplicates().shape[0]
    n_good = df_good[["model_key", "nn_variant"]].drop_duplicates().shape[0]
    print(f"Kept {n_good}/{n_orig} model-variants after divergence filter")

    # --- Generate plots ---------------------------------------------------
    metrics_agg = [
        "rmse_pos_mean", "rmse_vel_mean",
        "r2_pos_mean", "r2_vel_mean",
        "fit_pos_mean", "fit_vel_mean",
    ]

    # 1) Heatmaps
    print("Generating heatmaps …")
    for m in metrics_agg:
        plot_test_metric_heatmap(df_good, m, save_path=plots_dir / f"heatmap_{m}.png")
        plt.close("all")

    # 2) Grouped bar charts
    print("Generating bar charts …")
    for m in metrics_agg:
        plot_bar_comparison(df_good, m, save_path=plots_dir / f"bar_{m}.png")
        plt.close("all")

    # 3) Box plots
    print("Generating box plots …")
    for m in metrics_agg:
        plot_metric_boxplots(df_good, m, save_path=plots_dir / f"boxplot_{m}.png")
        plt.close("all")

    # 4) Train vs Test scatter
    print("Generating train-vs-test scatter …")
    for m in metrics_agg:
        plot_train_vs_test(df_good, m, save_path=plots_dir / f"train_vs_test_{m}.png")
        plt.close("all")

    # 5) Radar chart
    print("Generating radar chart …")
    plot_summary_radar(df_good, save_path=plots_dir / "radar_summary.png")
    plt.close("all")

    # 6) Per-dataset ranking bars
    print("Generating per-dataset rankings …")
    for m in ["fit_pos_mean", "fit_vel_mean", "r2_pos_mean", "r2_vel_mean"]:
        plot_per_dataset_ranking(df_good, m, save_path=plots_dir / f"ranking_{m}.png")
        plt.close("all")

    # 7) Violin plots from long CSV (per-run variability)
    if df_long is not None:
        print("Generating violin plots …")
        for m in ["fit_pos", "fit_vel", "r2_pos", "r2_vel", "rmse_pos", "rmse_vel"]:
            if m in df_long.columns:
                plot_long_violin(df_long, m, save_path=plots_dir / f"violin_{m}.png")
                plt.close("all")

    print(f"\nAll plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
