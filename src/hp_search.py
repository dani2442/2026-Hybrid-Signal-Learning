"""Optuna-based hyperparameter search for all registered models.

Provides per-model search spaces and a reusable ``create_objective``
factory that wraps the train → validate cycle into an Optuna objective.

Usage (programmatic)::

    import optuna
    from src.hp_search import create_objective, get_search_space

    objective = create_objective("gru", train_sets, val_sets, seed=42)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
"""

from __future__ import annotations

import time
from dataclasses import fields
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np
import optuna

from src.models.registry import build_model, get_config_class


# ─────────────────────────────────────────────────────────────────────
# Search-space protocol
# ─────────────────────────────────────────────────────────────────────

class SearchSpaceFn(Protocol):
    """Callable that samples hyperparameters from an Optuna trial."""

    def __call__(self, trial: optuna.Trial) -> Dict[str, Any]: ...


# ─────────────────────────────────────────────────────────────────────
# Per-model search spaces
# ─────────────────────────────────────────────────────────────────────
# Each function receives an optuna.Trial and returns a dict of
# config overrides.  Keep search ranges reasonable — these are
# starting points; users can register custom spaces via
# ``register_search_space``.
# ─────────────────────────────────────────────────────────────────────

def _common_lr(trial: optuna.Trial, *, low: float = 1e-4, high: float = 1e-1) -> float:
    return trial.suggest_float("learning_rate", low, high, log=True)


def _common_window(trial: optuna.Trial, *, low: int = 10, high: int = 100) -> int:
    return trial.suggest_int("train_window_size", low, high, step=10)


# -- Classical / statistical ------------------------------------------------

def _narx_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "nu": trial.suggest_int("nu", 1, 20),
        "ny": trial.suggest_int("ny", 1, 20),
        "poly_order": trial.suggest_int("poly_order", 1, 4),
        "selection_criteria": trial.suggest_float(
            "selection_criteria", 1e-4, 0.1, log=True
        ),
    }


def _random_forest_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "nu": trial.suggest_int("nu", 2, 20),
        "ny": trial.suggest_int("ny", 2, 20),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
    }


def _arima_space(trial: optuna.Trial) -> Dict[str, Any]:
    p = trial.suggest_int("arima_p", 0, 5)
    d = trial.suggest_int("arima_d", 0, 2)
    q = trial.suggest_int("arima_q", 0, 5)
    return {"order": (p, d, q)}


def _exp_smoothing_space(trial: optuna.Trial) -> Dict[str, Any]:
    trend = trial.suggest_categorical("trend", ["add", "mul", "None"])
    return {"trend": None if trend == "None" else trend}


# -- Feed-forward -----------------------------------------------------------

def _neural_network_space(trial: optuna.Trial) -> Dict[str, Any]:
    nu = trial.suggest_int("nu", 2, 20)
    ny = trial.suggest_int("ny", 2, 20)
    n_layers = trial.suggest_int("n_hidden_layers", 1, 5)
    width = trial.suggest_int("hidden_width", 32, 256, step=16)
    return {
        "nu": nu,
        "ny": ny,
        "hidden_layers": [width] * n_layers,
        "activation": trial.suggest_categorical("activation", ["relu", "selu", "tanh"]),
        "learning_rate": _common_lr(trial),
        "epochs": trial.suggest_int("epochs", 100, 500, step=50),
    }


# -- Sequence ---------------------------------------------------------------

def _rnn_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Shared space for GRU / LSTM."""
    return {
        "nu": trial.suggest_int("nu", 5, 30),
        "ny": trial.suggest_int("ny", 5, 30),
        "hidden_size": trial.suggest_int("hidden_size", 32, 256, step=16),
        "num_layers": trial.suggest_int("num_layers", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.05),
        "learning_rate": _common_lr(trial),
        "epochs": trial.suggest_int("epochs", 100, 500, step=50),
    }


def _tcn_space(trial: optuna.Trial) -> Dict[str, Any]:
    nu = trial.suggest_int("nu", 5, 30)
    ny = trial.suggest_int("ny", 5, 30)
    n_layers = trial.suggest_int("tcn_layers", 2, 6)
    n_channels = trial.suggest_int("tcn_channels", 32, 128, step=16)
    return {
        "nu": nu,
        "ny": ny,
        "num_channels": [n_channels] * n_layers,
        "kernel_size": trial.suggest_int("kernel_size", 2, 7),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.05),
        "learning_rate": _common_lr(trial),
        "epochs": trial.suggest_int("epochs", 100, 500, step=50),
    }


def _mamba_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "nu": trial.suggest_int("nu", 5, 30),
        "ny": trial.suggest_int("ny", 5, 30),
        "d_model": trial.suggest_int("d_model", 32, 128, step=16),
        "d_state": trial.suggest_int("d_state", 8, 32, step=4),
        "d_conv": trial.suggest_int("d_conv", 2, 8),
        "n_layers": trial.suggest_int("n_layers", 1, 4),
        "expand_factor": trial.suggest_int("expand_factor", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.05),
        "learning_rate": _common_lr(trial),
        "epochs": trial.suggest_int("epochs", 100, 500, step=50),
    }


# -- Continuous-time --------------------------------------------------------

def _neural_ode_space(trial: optuna.Trial) -> Dict[str, Any]:
    n_layers = trial.suggest_int("n_hidden_layers", 1, 4)
    width = trial.suggest_int("hidden_width", 32, 128, step=16)
    return {
        "hidden_layers": [width] * n_layers,
        "solver": trial.suggest_categorical("solver", ["euler", "rk4"]),
        "dt": trial.suggest_float("dt", 0.01, 0.1, step=0.01),
        "train_window_size": _common_window(trial),
        "activation": trial.suggest_categorical("activation", ["relu", "selu", "tanh"]),
        "learning_rate": _common_lr(trial),
        "epochs": trial.suggest_int("epochs", 100, 600, step=50),
    }


def _neural_sde_space(trial: optuna.Trial) -> Dict[str, Any]:
    n_layers = trial.suggest_int("n_hidden_layers", 1, 4)
    width = trial.suggest_int("hidden_width", 32, 128, step=16)
    diff_layers = trial.suggest_int("n_diff_layers", 1, 3)
    diff_width = trial.suggest_int("diff_width", 32, 128, step=16)
    return {
        "hidden_layers": [width] * n_layers,
        "diffusion_hidden_layers": [diff_width] * diff_layers,
        "solver": trial.suggest_categorical("solver", ["euler", "rk4"]),
        "dt": trial.suggest_float("dt", 0.01, 0.1, step=0.01),
        "train_window_size": _common_window(trial),
        "learning_rate": _common_lr(trial),
        "epochs": trial.suggest_int("epochs", 100, 600, step=50),
    }


def _neural_cde_space(trial: optuna.Trial) -> Dict[str, Any]:
    n_layers = trial.suggest_int("n_hidden_layers", 1, 4)
    width = trial.suggest_int("hidden_width", 32, 128, step=16)
    return {
        "hidden_dim": trial.suggest_int("hidden_dim", 32, 128, step=16),
        "hidden_layers": [width] * n_layers,
        "interpolation": trial.suggest_categorical(
            "interpolation", ["cubic", "linear"]
        ),
        "solver": trial.suggest_categorical("solver", ["rk4", "euler"]),
        "train_window_size": _common_window(trial),
        "learning_rate": _common_lr(trial),
        "epochs": trial.suggest_int("epochs", 50, 300, step=50),
    }


# -- Physics / hybrid -------------------------------------------------------

def _linear_physics_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "dt": trial.suggest_float("dt", 0.01, 0.1, step=0.01),
        "solver": trial.suggest_categorical("solver", ["euler", "rk4"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "epochs": trial.suggest_int("epochs", 500, 5000, step=500),
        "train_window_size": _common_window(trial),
    }


def _stribeck_physics_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "dt": trial.suggest_float("dt", 0.01, 0.1, step=0.01),
        "solver": trial.suggest_categorical("solver", ["euler", "rk4"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "epochs": trial.suggest_int("epochs", 500, 5000, step=500),
        "train_window_size": _common_window(trial),
    }


def _hybrid_linear_beam_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "dt": trial.suggest_float("dt", 0.01, 0.1, step=0.01),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "epochs": trial.suggest_int("epochs", 200, 1000, step=100),
        "integration_substeps": trial.suggest_int("integration_substeps", 1, 5),
    }


def _hybrid_nonlinear_cam_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "dt": trial.suggest_float("dt", 0.01, 0.1, step=0.01),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "epochs": trial.suggest_int("epochs", 40, 200, step=20),
        "integration_substeps": trial.suggest_int("integration_substeps", 5, 30, step=5),
    }


def _ude_space(trial: optuna.Trial) -> Dict[str, Any]:
    n_layers = trial.suggest_int("n_hidden_layers", 1, 4)
    width = trial.suggest_int("hidden_width", 32, 128, step=16)
    return {
        "hidden_layers": [width] * n_layers,
        "solver": trial.suggest_categorical("solver", ["euler", "rk4"]),
        "dt": trial.suggest_float("dt", 0.01, 0.1, step=0.01),
        "train_window_size": _common_window(trial),
        "activation": trial.suggest_categorical("activation", ["relu", "selu", "tanh"]),
        "learning_rate": _common_lr(trial),
        "epochs": trial.suggest_int("epochs", 200, 2000, step=200),
    }


# -- Blackbox 2-D variants --------------------------------------------------

def _blackbox_ode_2d_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "hidden_dim": trial.suggest_int("hidden_dim", 64, 256, step=32),
        "solver": trial.suggest_categorical("solver", ["euler", "rk4"]),
        "dt": trial.suggest_float("dt", 0.01, 0.1, step=0.01),
        "k_steps": trial.suggest_int("k_steps", 5, 40, step=5),
        "learning_rate": _common_lr(trial),
        "epochs": trial.suggest_int("epochs", 1000, 8000, step=1000),
        "batch_size": trial.suggest_int("batch_size", 32, 256, step=32),
        "grad_clip": trial.suggest_float("grad_clip", 1.0, 20.0, step=1.0),
    }


def _blackbox_sde_2d_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "hidden_dim": trial.suggest_int("hidden_dim", 64, 256, step=32),
        "diffusion_hidden_dim": trial.suggest_int(
            "diffusion_hidden_dim", 32, 128, step=16
        ),
        "solver": trial.suggest_categorical("solver", ["euler", "rk4"]),
        "dt": trial.suggest_float("dt", 0.01, 0.1, step=0.01),
        "k_steps": trial.suggest_int("k_steps", 5, 40, step=5),
        "learning_rate": _common_lr(trial),
        "epochs": trial.suggest_int("epochs", 1000, 8000, step=1000),
        "batch_size": trial.suggest_int("batch_size", 32, 256, step=32),
        "grad_clip": trial.suggest_float("grad_clip", 1.0, 20.0, step=1.0),
    }


def _blackbox_cde_2d_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "hidden_dim": trial.suggest_int("hidden_dim", 64, 256, step=32),
        "solver": trial.suggest_categorical("solver", ["euler", "rk4"]),
        "dt": trial.suggest_float("dt", 0.01, 0.1, step=0.01),
        "k_steps": trial.suggest_int("k_steps", 5, 40, step=5),
        "learning_rate": _common_lr(trial),
        "epochs": trial.suggest_int("epochs", 1000, 8000, step=1000),
        "batch_size": trial.suggest_int("batch_size", 32, 256, step=32),
        "grad_clip": trial.suggest_float("grad_clip", 1.0, 20.0, step=1.0),
    }


# ─────────────────────────────────────────────────────────────────────
# Search-space registry
# ─────────────────────────────────────────────────────────────────────

_SEARCH_SPACES: Dict[str, SearchSpaceFn] = {
    # Classical
    "narx": _narx_space,
    "arima": _arima_space,
    "exponential_smoothing": _exp_smoothing_space,
    "random_forest": _random_forest_space,
    # Feed-forward
    "neural_network": _neural_network_space,
    # Sequence
    "gru": _rnn_space,
    "lstm": _rnn_space,
    "tcn": _tcn_space,
    "mamba": _mamba_space,
    # Continuous
    "neural_ode": _neural_ode_space,
    "neural_sde": _neural_sde_space,
    "neural_cde": _neural_cde_space,
    "linear_physics": _linear_physics_space,
    "stribeck_physics": _stribeck_physics_space,
    # Hybrid
    "hybrid_linear_beam": _hybrid_linear_beam_space,
    "hybrid_nonlinear_cam": _hybrid_nonlinear_cam_space,
    "ude": _ude_space,
    # Blackbox 2-D
    "vanilla_node_2d": _blackbox_ode_2d_space,
    "structured_node": _blackbox_ode_2d_space,
    "adaptive_node": _blackbox_ode_2d_space,
    "vanilla_nsde_2d": _blackbox_sde_2d_space,
    "structured_nsde": _blackbox_sde_2d_space,
    "adaptive_nsde": _blackbox_sde_2d_space,
    "vanilla_ncde_2d": _blackbox_cde_2d_space,
    "structured_ncde": _blackbox_cde_2d_space,
    "adaptive_ncde": _blackbox_cde_2d_space,
}


def register_search_space(model_name: str, space_fn: SearchSpaceFn) -> None:
    """Register (or override) a custom search space for *model_name*."""
    _SEARCH_SPACES[model_name] = space_fn


def get_search_space(model_name: str) -> SearchSpaceFn:
    """Return the search-space function for *model_name*.

    Raises ``KeyError`` if no space is registered.
    """
    if model_name not in _SEARCH_SPACES:
        raise KeyError(
            f"No search space for '{model_name}'. "
            f"Available: {', '.join(sorted(_SEARCH_SPACES))}"
        )
    return _SEARCH_SPACES[model_name]


def list_searchable_models() -> List[str]:
    """Return sorted list of models with a registered search space."""
    return sorted(_SEARCH_SPACES.keys())


# ─────────────────────────────────────────────────────────────────────
# Objective factory
# ─────────────────────────────────────────────────────────────────────

def create_objective(
    model_name: str,
    train_sets: list,
    val_sets: list,
    *,
    metric: str = "MSE",
    predict_mode: str = "FR",
    seed: int = 42,
    fixed_overrides: Optional[Dict[str, Any]] = None,
    logger_factory: Optional[Callable] = None,
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective that trains and evaluates *model_name*.

    Parameters
    ----------
    model_name : str
        Registered model key.
    train_sets : list[Dataset]
        Training dataset(s) (already split).
    val_sets : list[Dataset]
        Validation dataset(s) used as the optimisation target.
    metric : str
        Metric to minimise (key from ``compute_all``, default ``"MSE"``).
    predict_mode : str
        Prediction mode passed to ``model.predict`` (``"FR"`` or ``"OSA"``).
    seed : int
        Base random seed (each trial: ``seed + trial.number``).
    fixed_overrides : dict | None
        Config fields that should NOT be searched (e.g. ``{"device": "cuda"}``).
    logger_factory : callable | None
        ``(trial, config_dict) -> WandbLogger | None`` for per-trial logging.

    Returns
    -------
    callable
        ``objective(trial) -> float`` ready for ``study.optimize``.
    """
    from src.validation.metrics import compute_all
    from src.utils.runtime import seed_all

    space_fn = get_search_space(model_name)

    def _pack_pairs(pairs):
        return pairs[0] if len(pairs) == 1 else pairs

    def objective(trial: optuna.Trial) -> float:
        # 1. Sample hyperparameters
        hp = space_fn(trial)
        hp["seed"] = seed + trial.number
        if fixed_overrides:
            hp.update(fixed_overrides)

        # 2. Build model
        seed_all(hp["seed"])
        model = build_model(model_name, **hp)

        # 3. Optional W&B logging
        wlogger = None
        if logger_factory is not None:
            config_dict = model.config.to_dict() if hasattr(model.config, "to_dict") else hp
            wlogger = logger_factory(trial, config_dict)

        # 4. Train
        train_data = [(ds.u, ds.y) for ds in train_sets]
        val_data = [(ds.u, ds.y) for ds in val_sets]
        t0 = time.time()
        try:
            model.fit(
                _pack_pairs(train_data),
                val_data=_pack_pairs(val_data),
                logger=wlogger,
            )
        except Exception as exc:
            if wlogger is not None:
                wlogger.finish()
            # Report failure as pruned so Optuna keeps going
            raise optuna.TrialPruned(f"Training failed: {exc}") from exc
        train_time = time.time() - t0
        trial.set_user_attr("train_time", train_time)

        # 5. Evaluate on validation set
        skip = getattr(model, "max_lag", 0)
        val_y_true, val_y_pred = [], []
        for vds in val_sets:
            yp = model.predict(vds.u, vds.y, mode=predict_mode)
            yt = np.asarray(vds.y).ravel()
            yp_r = np.asarray(yp).ravel()
            n = min(len(yt), len(yp_r))
            s = min(skip, n)
            val_y_true.append(yt[s:n])
            val_y_pred.append(yp_r[s:n])

        agg_metrics = compute_all(
            np.concatenate(val_y_true),
            np.concatenate(val_y_pred),
            skip=0,
        )
        for k, v in agg_metrics.items():
            trial.set_user_attr(k, v)

        if wlogger is not None:
            wlogger.log_metrics(
                {f"hp_search/val/{k}": v for k, v in agg_metrics.items()}
            )
            wlogger.finish()

        value = agg_metrics[metric]
        # Guard against NaN — Optuna minimises, NaN breaks samplers
        if np.isnan(value) or np.isinf(value):
            raise optuna.TrialPruned(f"Metric {metric} is NaN/Inf")
        return value

    return objective
