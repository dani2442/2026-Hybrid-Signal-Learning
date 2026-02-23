from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


DEFAULT_COLORS = {
    "linear": "tab:red",
    "stribeck": "tab:blue",
    "blackbox": "tab:green",
    "structured_blackbox": "tab:purple",
    "adaptive_blackbox": "tab:olive",
    "ct_esn": "tab:cyan",
    "hybrid_joint": "tab:orange",
    "hybrid_joint_stribeck": "tab:brown",
    "hybrid_frozen": "darkviolet",
    "hybrid_frozen_stribeck": "tab:pink",
}

DEFAULT_STYLES = {
    "linear": "--",
    "stribeck": "-.",
    "blackbox": ":",
    "structured_blackbox": (0, (5, 1)),
    "adaptive_blackbox": (0, (3, 1, 1, 1)),
    "ct_esn": (0, (4, 1, 1, 1, 1, 1)),
    "hybrid_joint": "-",
    "hybrid_joint_stribeck": (0, (1, 1)),
    "hybrid_frozen": (0, (3, 1, 1, 1)),
    "hybrid_frozen_stribeck": (0, (5, 2, 1, 2)),
}


def model_display_name(model_key: str) -> str:
    mapping = {
        "linear": "Linear",
        "stribeck": "Stribeck",
        "blackbox": "Black-box",
        "structured_blackbox": "Structured-BB",
        "adaptive_blackbox": "Adaptive-BB",
        "ct_esn": "CT-ESN",
        "hybrid_joint": "Hybrid-Joint",
        "hybrid_joint_stribeck": "Hybrid-Joint-Stribeck",
        "hybrid_frozen": "Hybrid-Frozen",
        "hybrid_frozen_stribeck": "Hybrid-Frozen-Stribeck",
    }
    return mapping.get(model_key, model_key)


def _savefig(fig: plt.Figure, save_path: str | Path | None = None) -> None:
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")


def plot_training_curves(history_by_model_id: dict[str, list[dict[str, float]]], save_path: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_id, hist in history_by_model_id.items():
        if not hist:
            continue
        epochs = [h["epoch"] for h in hist]
        losses = [h["loss"] for h in hist]
        ax.plot(epochs, losses, linewidth=1.2, alpha=0.8, label=model_id)

    ax.set_title("Training loss convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.grid(True, alpha=0.3)
    if len(history_by_model_id) <= 12:
        ax.legend(fontsize=8)

    _savefig(fig, save_path)
    return fig


def plot_predictions(
    *,
    t: np.ndarray,
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    split_idx: int | None = None,
    title_suffix: str = "",
    colors: dict[str, str] | None = None,
    styles: dict[str, str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    colors = colors or DEFAULT_COLORS
    styles = styles or DEFAULT_STYLES

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t, y_true[:, 0], color="k", linewidth=2, alpha=0.6, label="Measured")
    axes[1].plot(t, y_true[:, 1], color="k", linewidth=2, alpha=0.6, label="Measured")

    for model_key, y_hat in preds.items():
        lbl = model_display_name(model_key)
        axes[0].plot(t, y_hat[:, 0], color=colors.get(model_key, None), linestyle=styles.get(model_key, "-"), linewidth=1.4, label=lbl)
        axes[1].plot(t, y_hat[:, 1], color=colors.get(model_key, None), linestyle=styles.get(model_key, "-"), linewidth=1.4, label=lbl)

    if split_idx is not None and 0 < split_idx < len(t):
        t_split = t[split_idx]
        for ax in axes:
            ax.axvline(t_split, color="gray", linestyle="--", linewidth=1)

    axes[0].set_ylabel("Position")
    axes[1].set_ylabel("Velocity")
    axes[1].set_xlabel("Time (s)")
    axes[0].set_title(f"Measured vs predicted {title_suffix}".strip())

    for ax in axes:
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", ncol=2)
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


def plot_zoom_position(
    *,
    t: np.ndarray,
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    Ts: float,
    window_sec: float = 5.0,
    colors: dict[str, str] | None = None,
    styles: dict[str, str] | None = None,
    title_suffix: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    colors = colors or DEFAULT_COLORS
    styles = styles or DEFAULT_STYLES

    win_n = int(max(2, window_sec / Ts))
    starts = [0, max(0, (len(t) - win_n) // 2), max(0, len(t) - win_n)]
    labels = ["Start", "Middle", "End"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    for i, s in enumerate(starts):
        e = min(len(t), s + win_n)
        ax = axes[i]
        ax.plot(t[s:e], y_true[s:e, 0], color="k", linewidth=2, alpha=0.6, label="Measured")
        for model_key, y_hat in preds.items():
            ax.plot(
                t[s:e],
                y_hat[s:e, 0],
                color=colors.get(model_key, None),
                linestyle=styles.get(model_key, "-"),
                linewidth=1.3,
                label=model_display_name(model_key),
            )
        ax.set_title(f"Zoom {labels[i]} {title_suffix}".strip())
        ax.set_ylabel("Position")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", ncol=2)
        if i == 2:
            ax.set_xlabel("Time (s)")

    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


def plot_residuals(
    *,
    t: np.ndarray,
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    split_idx: int | None = None,
    colors: dict[str, str] | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, dict[str, dict[str, np.ndarray]]]:
    colors = colors or DEFAULT_COLORS

    residuals: dict[str, dict[str, np.ndarray]] = {}
    for mk, y_hat in preds.items():
        residuals[mk] = {
            "pos": y_true[:, 0] - y_hat[:, 0],
            "vel": y_true[:, 1] - y_hat[:, 1],
        }

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for mk, rr in residuals.items():
        axes[0].plot(t, rr["pos"], color=colors.get(mk, None), linewidth=1.2, label=model_display_name(mk))
        axes[1].plot(t, rr["vel"], color=colors.get(mk, None), linewidth=1.2, label=model_display_name(mk))

    if split_idx is not None and 0 < split_idx < len(t):
        t_split = t[split_idx]
        for ax in axes:
            ax.axvline(t_split, color="gray", linestyle="--", linewidth=1)

    axes[0].axhline(0.0, color="gray", linewidth=1)
    axes[1].axhline(0.0, color="gray", linewidth=1)

    axes[0].set_title("Residuals")
    axes[0].set_ylabel("Position residual")
    axes[1].set_ylabel("Velocity residual")
    axes[1].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    fig.tight_layout()
    _savefig(fig, save_path)
    return fig, residuals


def plot_raincloud_models(
    *,
    residuals_pos: dict[str, np.ndarray],
    colors: dict[str, str] | None = None,
    title: str = "Residual raincloud (position)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    colors = colors or DEFAULT_COLORS

    labels = list(residuals_pos.keys())
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, mk in enumerate(labels):
        res = np.asarray(residuals_pos[mk])
        res = res[np.isfinite(res)]
        if len(res) < 5:
            continue

        xs = np.linspace(np.min(res), np.max(res), 200)
        kde = gaussian_kde(res)
        ys = kde(xs)
        ys = ys / max(ys.max(), 1e-12) * 0.30

        ax.fill_between(xs, i + ys, i - ys, color=colors.get(mk, "gray"), alpha=0.3)
        q1, q2, q3 = np.percentile(res, [25, 50, 75])
        ax.plot([q1, q3], [i, i], color=colors.get(mk, "gray"), linewidth=6)
        ax.plot([q2, q2], [i - 0.10, i + 0.10], color="k", linewidth=1)

        jitter = (np.random.rand(len(res)) - 0.5) * 0.20
        ax.scatter(res, i + jitter, s=3, color=colors.get(mk, "gray"), alpha=0.3)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([model_display_name(mk) for mk in labels])
    ax.set_xlabel("Position residual")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


def plot_y_vs_yhat(
    *,
    y_true_pos: np.ndarray,
    preds: dict[str, np.ndarray],
    colors: dict[str, str] | None = None,
    title: str = "y vs y_hat (position)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    colors = colors or DEFAULT_COLORS
    fig, ax = plt.subplots(figsize=(6, 6))

    y_min = float(np.min(y_true_pos))
    y_max = float(np.max(y_true_pos))
    ax.plot([y_min, y_max], [y_min, y_max], "k--", linewidth=1, label="Ideal")

    for mk, y_hat in preds.items():
        ax.scatter(y_true_pos, y_hat[:, 0], s=8, alpha=0.35, color=colors.get(mk, None), label=model_display_name(mk))

    ax.set_xlabel("Measured y")
    ax.set_ylabel("Predicted y")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


def plot_acf(
    *,
    residuals_pos: dict[str, np.ndarray],
    colors: dict[str, str] | None = None,
    max_lag: int = 2000,
    title: str = "Residual ACF (position)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    colors = colors or DEFAULT_COLORS

    any_res = next(iter(residuals_pos.values()))
    n = len(any_res)
    mlag = min(max_lag, n - 1)
    conf = 1.96 / np.sqrt(max(n, 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    for mk, rr in residuals_pos.items():
        res = rr - np.mean(rr)
        acf = np.correlate(res, res, mode="full")
        acf = acf[n - 1 : n + mlag] / max(acf[n - 1], 1e-12)
        ax.plot(np.arange(0, mlag + 1), acf, color=colors.get(mk, None), linewidth=1.2, label=model_display_name(mk))

    ax.axhline(conf, color="red", linestyle="--", linewidth=1)
    ax.axhline(-conf, color="red", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


def plot_spectra(
    *,
    t: np.ndarray,
    Ts: float,
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    colors: dict[str, str] | None = None,
    title_suffix: str = "",
    save_path_data: str | Path | None = None,
    save_path_res: str | Path | None = None,
) -> tuple[plt.Figure, plt.Figure]:
    colors = colors or DEFAULT_COLORS

    freqs = np.fft.rfftfreq(len(t), d=Ts)
    Y_meas = np.fft.rfft(y_true[:, 0])

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.semilogy(freqs, np.abs(Y_meas), color="k", linewidth=1.3, label="Measured")

    for mk, y_hat in preds.items():
        Y_pred = np.fft.rfft(y_hat[:, 0])
        ax1.semilogy(freqs, np.abs(Y_pred), color=colors.get(mk, None), linewidth=1.2, label=model_display_name(mk))

    ax1.set_title(f"Spectrum (position) {title_suffix}".strip())
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)
    fig1.tight_layout()
    _savefig(fig1, save_path_data)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.semilogy(freqs, np.abs(Y_meas), color="k", linewidth=1.3, label="Measured")

    for mk, y_hat in preds.items():
        res_fft = np.fft.rfft(y_true[:, 0] - y_hat[:, 0])
        ax2.semilogy(freqs, np.abs(res_fft), color=colors.get(mk, None), linewidth=1.2, label=model_display_name(mk))

    ax2.set_title(f"Residual spectrum (position) {title_suffix}".strip())
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)
    fig2.tight_layout()
    _savefig(fig2, save_path_res)

    return fig1, fig2
