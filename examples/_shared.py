"""Shared helpers for the example training and evaluation scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import MODEL_CONFIGS
from src.data import Dataset
from src.validation import Metrics
from src.visualization import plot_predictions

MODEL_CLASS_NAMES: dict[str, str] = {
    "narx": "NARX",
    "arima": "ARIMA",
    "exponential_smoothing": "ExponentialSmoothing",
    "random_forest": "RandomForest",
    "neural_network": "NeuralNetwork",
    "gru": "GRU",
    "lstm": "LSTM",
    "tcn": "TCN",
    "mamba": "Mamba",
    "neural_ode": "NeuralODE",
    "neural_sde": "NeuralSDE",
    "neural_cde": "NeuralCDE",
    "linear_physics": "LinearPhysics",
    "stribeck_physics": "StribeckPhysics",
    "hybrid_linear_beam": "HybridLinearBeam",
    "hybrid_nonlinear_cam": "HybridNonlinearCam",
    "ude": "UDE",
    "vanilla_node_2d": "VanillaNODE2D",
    "structured_node": "StructuredNODE",
    "adaptive_node": "AdaptiveNODE",
    "vanilla_ncde_2d": "VanillaNCDE2D",
    "structured_ncde": "StructuredNCDE",
    "adaptive_ncde": "AdaptiveNCDE",
    "vanilla_nsde_2d": "VanillaNSDE2D",
    "structured_nsde": "StructuredNSDE",
    "adaptive_nsde": "AdaptiveNSDE",
}

DEFAULT_MODEL_NAMES: list[str] = [
    "narx",
    "arima",
    "exponential_smoothing",
    "random_forest",
    "neural_network",
    "gru",
    "lstm",
    "tcn",
    "neural_ode",
    "linear_physics",
    "stribeck_physics",
    "hybrid_linear_beam",
    "ude",
    "vanilla_node_2d",
    "structured_node",
    "adaptive_node",
]

PHYSICS_MODELS = {"linear_physics", "stribeck_physics"}
CLASS_NAME_TO_MODEL_KEY = {class_name: model_key for model_key, class_name in MODEL_CLASS_NAMES.items()}


@dataclass(frozen=True)
class SplitBundle:
    dataset: Dataset
    train: Dataset
    val: Dataset
    test: Dataset


@dataclass(frozen=True)
class ModelArtifacts:
    model: Any
    model_key: str
    dataset_name: str
    y_osa: np.ndarray
    y_fr: np.ndarray
    metrics_osa: dict[str, float]
    metrics_fr: dict[str, float]
    checkpoint_path: Path | None = None
    plot_path: Path | None = None


def resolve_model_key(model_or_key: Any) -> str:
    if isinstance(model_or_key, str):
        return model_or_key
    model_key = CLASS_NAME_TO_MODEL_KEY.get(type(model_or_key).__name__)
    if model_key is None:
        raise ValueError(f"Unsupported model class: {type(model_or_key).__name__}")
    return model_key


def resolve_model_class(model_key: str):
    import src.models as models_pkg

    try:
        class_name = MODEL_CLASS_NAMES[model_key]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_CLASS_NAMES))
        raise ValueError(f"Unknown model '{model_key}'. Available: {available}") from exc
    return getattr(models_pkg, class_name)


def load_train_val_test(dataset_name: str, *, train_ratio: float = 0.7, val_ratio: float = 0.15) -> SplitBundle:
    dataset = Dataset.from_bab_experiment(dataset_name)
    train_ds, val_ds, test_ds = dataset.train_val_test_split(train=train_ratio, val=val_ratio)
    return SplitBundle(dataset=dataset, train=train_ds, val=val_ds, test=test_ds)


def describe_splits(train_ds: Dataset, val_ds: Dataset | None = None, test_ds: Dataset | None = None) -> str:
    parts = [f"{len(train_ds)} train"]
    if val_ds is not None:
        parts.append(f"{len(val_ds)} val")
    if test_ds is not None:
        parts.append(f"{len(test_ds)} test")
    return " / ".join(parts)


def build_model(
    model_key: str,
    *,
    epochs: int | None = None,
    learning_rate: float | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
):
    config = MODEL_CONFIGS[model_key]()
    if epochs is not None:
        config.epochs = int(epochs)
    if learning_rate is not None and hasattr(config, "learning_rate"):
        config.learning_rate = float(learning_rate)
    if wandb_project:
        config.wandb_project = wandb_project
        config.wandb_run_name = wandb_run_name
    model_cls = resolve_model_class(model_key)
    return model_cls(config)


def _fit_target(model_key: str, ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    if model_key in PHYSICS_MODELS and ds.y_dot is not None:
        return ds.u, np.column_stack([ds.y, ds.y_dot]).astype(float)
    return ds.arrays


def fit_model(model, train_ds: Dataset, val_ds: Dataset | None = None):
    model_key = resolve_model_key(model)
    val_data = _fit_target(model_key, val_ds) if val_ds is not None else None
    model.fit(_fit_target(model_key, train_ds), val_data=val_data)
    return model


def _to_output_series(prediction: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction, dtype=float)
    if pred.ndim == 2:
        pred = pred[:, 0]
    return pred.reshape(-1)


def _physics_initial_segment(ds: Dataset) -> np.ndarray:
    if len(ds.y) == 0:
        raise ValueError("Dataset must contain at least one target sample.")
    dt = 1.0 / ds.sampling_rate if ds.sampling_rate > 0 else 1.0
    if ds.y_dot is not None and len(ds.y_dot) > 0:
        return np.asarray([ds.y[0], ds.y[0] + float(ds.y_dot[0]) * dt], dtype=float)
    if len(ds.y) == 1:
        return np.asarray([ds.y[0]], dtype=float)
    return np.asarray(ds.y[:2], dtype=float)


def predict_model(model, test_ds: Dataset, *, model_key: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    model_key = resolve_model_key(model_key or model)
    y_osa = _to_output_series(model.predict(test_ds.u, test_ds.y, mode="OSA"))
    if model_key in PHYSICS_MODELS:
        y_fr = _to_output_series(model.predict_free_run(test_ds.u, _physics_initial_segment(test_ds)))
    else:
        y_fr = _to_output_series(model.predict(test_ds.u, test_ds.y, mode="FR"))
    return y_osa, y_fr


def evaluate_model(model, test_ds: Dataset, *, model_key: str | None = None) -> ModelArtifacts:
    model_key = resolve_model_key(model_key or model)
    y_osa, y_fr = predict_model(model, test_ds, model_key=model_key)
    return ModelArtifacts(
        model=model,
        model_key=model_key,
        dataset_name=test_ds.name,
        y_osa=y_osa,
        y_fr=y_fr,
        metrics_osa=Metrics.compute_all(test_ds.y, y_osa),
        metrics_fr=Metrics.compute_all(test_ds.y, y_fr),
    )


def train_and_evaluate(
    model_key: str,
    splits: SplitBundle,
    *,
    epochs: int | None = None,
    learning_rate: float | None = None,
    wandb_project: str | None = None,
    checkpoint_dir: str | Path | None = None,
    plot_dir: str | Path | None = None,
    plot_prefix: str | None = None,
) -> ModelArtifacts:
    model = build_model(
        model_key,
        epochs=epochs,
        learning_rate=learning_rate,
        wandb_project=wandb_project,
        wandb_run_name=f"{model_key}_{splits.dataset.name}" if wandb_project else None,
    )
    fit_model(model, splits.train, splits.val)
    artifacts = evaluate_model(model, splits.test, model_key=model_key)
    checkpoint_path = save_checkpoint(model, model_key, splits.dataset.name, checkpoint_dir) if checkpoint_dir else None
    plot_path = (
        save_prediction_plot(
            splits.test,
            {
                f"{model_key} OSA": artifacts.y_osa,
                f"{model_key} FR": artifacts.y_fr,
            },
            title=f"{model_key} on {splits.dataset.name}",
            plot_dir=plot_dir,
            filename=f"{plot_prefix or model_key}_{splits.dataset.name}.png",
        )
        if plot_dir
        else None
    )
    return ModelArtifacts(
        model=artifacts.model,
        model_key=artifacts.model_key,
        dataset_name=splits.dataset.name,
        y_osa=artifacts.y_osa,
        y_fr=artifacts.y_fr,
        metrics_osa=artifacts.metrics_osa,
        metrics_fr=artifacts.metrics_fr,
        checkpoint_path=checkpoint_path,
        plot_path=plot_path,
    )


def save_checkpoint(model, model_key: str, dataset_name: str, out_dir: str | Path) -> Path:
    out_path = Path(out_dir) / f"{model_key}_{dataset_name}.pt"
    model.save(out_path)
    return out_path


def save_prediction_plot(
    test_ds: Dataset,
    predictions: dict[str, np.ndarray],
    *,
    title: str,
    plot_dir: str | Path,
    filename: str,
) -> Path:
    plot_path = Path(plot_dir) / filename
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_predictions(
        test_ds.t,
        test_ds.y,
        predictions,
        title=title,
        save_path=str(plot_path),
    )
    return plot_path


def print_metric_summary(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    Metrics.summary(y_true, y_pred, name=name)
