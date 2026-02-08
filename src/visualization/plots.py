"""Visualization utilities for system identification."""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _check_matplotlib():
    """Check matplotlib is available."""
    if plt is None:
        raise ImportError("matplotlib required. Install with: pip install matplotlib")


def _save_or_show(save_path: Optional[str] = None, dpi: int = 150):
    """Save figure to file or show interactively."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_signals(
    t: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    title: str = "Input-Output Signals",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot input and output signals.
    
    Args:
        t: Time vector
        u: Input signal
        y: Output signal
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    _check_matplotlib()

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(t, u, "b-", linewidth=0.8)
    axes[0].set_ylabel("Input (u)")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(["u"], loc="upper right")

    axes[1].plot(t, y, "r-", linewidth=0.8)
    axes[1].set_ylabel("Output (y)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(["y"], loc="upper right")

    plt.tight_layout()
    _save_or_show(save_path)


def plot_predictions(
    t: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    title: str = "Model Predictions",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot actual vs predicted outputs for multiple models.
    
    Args:
        t: Time vector
        y_true: Actual output
        predictions: Dict of {model_name: predicted_output}
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    _check_matplotlib()

    plt.figure(figsize=figsize)
    plt.plot(t, y_true, "k-", linewidth=1.0, label="Actual")

    colors = plt.cm.tab10.colors
    for i, (name, y_pred) in enumerate(predictions.items()):
        # Handle length mismatch
        n = min(len(t), len(y_pred))
        plt.plot(t[:n], y_pred[:n], "--", color=colors[i % 10], 
                 linewidth=0.8, label=name)

    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_residuals(
    t: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    title: str = "Prediction Residuals",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot prediction residuals for multiple models.
    
    Args:
        t: Time vector
        y_true: Actual output
        predictions: Dict of {model_name: predicted_output}
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    _check_matplotlib()

    plt.figure(figsize=figsize)

    colors = plt.cm.tab10.colors
    for i, (name, y_pred) in enumerate(predictions.items()):
        n = min(len(y_true), len(y_pred))
        residual = y_true[:n] - y_pred[:n]
        plt.plot(t[:n], residual, color=colors[i % 10], 
                 linewidth=0.8, label=f"{name} residual")

    plt.xlabel("Time (s)")
    plt.ylabel("Residual (y - ŷ)")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_spectrograms(
    u: np.ndarray,
    y: np.ndarray,
    fs: float = 1000,
    nfft: int = 256,
    noverlap: int = 128,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot spectrograms of input and output signals.
    
    Args:
        u: Input signal
        y: Output signal
        fs: Sampling frequency
        nfft: FFT size
        noverlap: Overlap between segments
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    _check_matplotlib()

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Output spectrogram
    Pyy, freqs, _, im = axes[0].specgram(y, NFFT=nfft, Fs=fs, noverlap=noverlap)
    axes[0].set_title("Output Signal Spectrogram")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_yscale("log")
    axes[0].set_ylim([freqs[1], freqs[-1]])
    plt.colorbar(im, ax=axes[0], label="Intensity (dB)")

    # Input spectrogram
    Puu, freqs, _, im = axes[1].specgram(u, NFFT=nfft, Fs=fs, noverlap=noverlap)
    axes[1].set_title("Input Signal Spectrogram")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_yscale("log")
    axes[1].set_ylim([freqs[1], freqs[-1]])
    plt.colorbar(im, ax=axes[1], label="Intensity (dB)")

    plt.tight_layout()
    _save_or_show(save_path)


def plot_model_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Bar plot comparing multiple models across metrics.
    
    Args:
        metrics: Dict of {model_name: {metric_name: value}}
        metric_names: List of metrics to plot
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    _check_matplotlib()

    model_names = list(metrics.keys())
    
    if metric_names is None:
        metric_names = list(metrics[model_names[0]].keys())

    n_models = len(model_names)
    n_metrics = len(metric_names)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=figsize)

    for i, model in enumerate(model_names):
        values = [metrics[model].get(m, 0) for m in metric_names]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _save_or_show(save_path)
