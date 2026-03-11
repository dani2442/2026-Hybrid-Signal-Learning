"""Control System Identification Library.

Heavy submodules (models, vision, visualization) are loaded lazily so that
lightweight imports like ``from src.data.registry import …`` don't pull in
torch / matplotlib on *every* invocation.
"""

import importlib as _importlib

__version__ = "0.1.0"

# ── lazy-import map: attribute → (module, name) ────────────────────────
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # models
    "ARIMA": (".models", "ARIMA"),
    "GRU": (".models", "GRU"),
    "HybridLinearBeam": (".models", "HybridLinearBeam"),
    "HybridNonlinearCam": (".models", "HybridNonlinearCam"),
    "LSTM": (".models", "LSTM"),
    "LinearPhysics": (".models", "LinearPhysics"),
    "NARX": (".models", "NARX"),
    "NeuralCDE": (".models", "NeuralCDE"),
    "NeuralNetwork": (".models", "NeuralNetwork"),
    "NeuralODE": (".models", "NeuralODE"),
    "NeuralSDE": (".models", "NeuralSDE"),
    "RandomForest": (".models", "RandomForest"),
    "StribeckPhysics": (".models", "StribeckPhysics"),
    "VanillaNODE2D": (".models", "VanillaNODE2D"),
    "StructuredNODE": (".models", "StructuredNODE"),
    "AdaptiveNODE": (".models", "AdaptiveNODE"),
    "VanillaNCDE2D": (".models", "VanillaNCDE2D"),
    "StructuredNCDE": (".models", "StructuredNCDE"),
    "AdaptiveNCDE": (".models", "AdaptiveNCDE"),
    "VanillaNSDE2D": (".models", "VanillaNSDE2D"),
    "StructuredNSDE": (".models", "StructuredNSDE"),
    "AdaptiveNSDE": (".models", "AdaptiveNSDE"),
    "TCN": (".models", "TCN"),
    "UDE": (".models", "UDE"),
    "ExponentialSmoothing": (".models", "ExponentialSmoothing"),
    "Mamba": (".models", "Mamba"),
    # data
    "Dataset": (".data", "Dataset"),
    # benchmarking
    "BenchmarkCase": (".benchmarking", "BenchmarkCase"),
    "BenchmarkConfig": (".benchmarking", "BenchmarkConfig"),
    "BenchmarkRunner": (".benchmarking", "BenchmarkRunner"),
    "build_benchmark_cases": (".benchmarking", "build_benchmark_cases"),
    "summarize_results": (".benchmarking", "summarize_results"),
    # visualization
    "plot_predictions": (".visualization", "plot_predictions"),
    "plot_spectrograms": (".visualization", "plot_spectrograms"),
    "plot_residuals": (".visualization", "plot_residuals"),
    "plot_signals": (".visualization", "plot_signals"),
    "plot_model_comparison": (".visualization", "plot_model_comparison"),
    # validation
    "Metrics": (".validation", "Metrics"),
}

# Submodules that should be importable as ``src.<name>``
_LAZY_SUBMODULES = {"vision", "models", "data", "benchmarking", "visualization", "validation", "utils"}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        mod_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(mod_path, __package__)
        return getattr(mod, attr)
    if name in _LAZY_SUBMODULES:
        return _importlib.import_module(f".{name}", __package__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "NARX",
    "ARIMA",
    "NeuralNetwork",
    "NeuralCDE",
    "NeuralODE",
    "NeuralSDE",
    "ExponentialSmoothing",
    "RandomForest",
    "GRU",
    "LSTM",
    "TCN",
    "UDE",
    "Mamba",
    "HybridLinearBeam",
    "HybridNonlinearCam",
    "LinearPhysics",
    "StribeckPhysics",
    "VanillaNODE2D",
    "StructuredNODE",
    "AdaptiveNODE",
    "VanillaNCDE2D",
    "StructuredNCDE",
    "AdaptiveNCDE",
    "VanillaNSDE2D",
    "StructuredNSDE",
    "AdaptiveNSDE",
    "BenchmarkCase",
    "BenchmarkConfig",
    "BenchmarkRunner",
    "build_benchmark_cases",
    "summarize_results",
    "Dataset",
    "Metrics",
    "plot_predictions",
    "plot_spectrograms",
    "plot_residuals",
    "plot_signals",
    "plot_model_comparison",
    "vision",
]
