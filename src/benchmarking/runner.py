"""Benchmark runner for model-to-model comparison on shared datasets."""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ..data import Dataset
from ..models import (
    NARX,
    NeuralODE,
    NeuralSDE,
    NeuralCDE,
    RandomForest,
    LinearPhysics,
    StribeckPhysics,
)
from ..models.blackbox_ode import VanillaNODE2D, StructuredNODE, AdaptiveNODE
from ..models.gru import GRU
from ..models.lstm import LSTM
from ..models.neural_network import NeuralNetwork
from ..models.tcn import TCN
from ..models.ude import UDE
from ..models.mamba import Mamba
from ..validation.metrics import Metrics


@dataclass(frozen=True)
class BenchmarkCase:
    """Single model benchmark configuration."""

    key: str
    name: str
    factory: Callable[[float], object]
    modes: tuple[str, ...] = ("OSA", "FR")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Dataset and split settings used for all benchmark cases."""

    datasets: tuple[str, ...] = ("multisine_05",)
    preprocess: bool = True
    resample_factor: int = 50
    train_ratio: float = 0.8
    end_ref_tolerance: float = 1e-8
    output_json: str = "benchmark_results.json"


def _base_case_factories() -> dict[str, BenchmarkCase]:
    """Canonical benchmark cases with balanced runtime and model diversity."""
    return {
        "narx": BenchmarkCase(
            key="narx",
            name="NARX",
            factory=lambda dt: NARX(nu=10, ny=10, poly_order=2, selection_criteria=10),
        ),
        "random_forest": BenchmarkCase(
            key="random_forest",
            name="RandomForest",
            factory=lambda dt: RandomForest(
                nu=10,
                ny=10,
                n_estimators=100,
                max_depth=12,
                random_state=42,
            ),
        ),
        "neural_ode": BenchmarkCase(
            key="neural_ode",
            name="NeuralODE",
            factory=lambda dt: NeuralODE(
                state_dim=1,
                input_dim=1,
                hidden_layers=[64, 64],
                dt=dt,
                solver="rk4",
                learning_rate=1e-3,
                epochs=1000,
                sequence_length=50,
                activation="selu",
                training_mode="full",
            ),
        ),
        "neural_sde": BenchmarkCase(
            key="neural_sde",
            name="NeuralSDE",
            factory=lambda dt: NeuralSDE(
                state_dim=1,
                input_dim=1,
                hidden_layers=[64, 64],
                diffusion_hidden_layers=[64, 64],
                dt=dt,
                solver="euler",
                learning_rate=1e-3,
                epochs=1000,
                sequence_length=50,
            ),
        ),
        "neural_cde": BenchmarkCase(
            key="neural_cde",
            name="NeuralCDE",
            factory=lambda dt: NeuralCDE(
                hidden_dim=32,
                input_dim=2,
                hidden_layers=[64, 64],
                interpolation="cubic",
                solver="rk4",
                learning_rate=1e-3,
                epochs=1000,
                sequence_length=50,
            ),
        ),
        "linear_physics": BenchmarkCase(
            key="linear_physics",
            name="LinearPhysics",
            factory=lambda dt: LinearPhysics(
                dt=dt,
                solver="rk4",
                learning_rate=1e-3,
                epochs=1000,
            ),
        ),
        "stribeck_physics": BenchmarkCase(
            key="stribeck_physics",
            name="StribeckPhysics",
            factory=lambda dt: StribeckPhysics(
                dt=dt,
                solver="rk4",
                learning_rate=1e-3,
                epochs=1000,
            ),
        ),
        "lstm": BenchmarkCase(
            key="lstm",
            name="LSTM",
            factory=lambda dt: LSTM(
                nu=10,
                ny=10,
                hidden_size=64,
                num_layers=2,
                dropout=0.1,
                learning_rate=1e-3,
                epochs=200,
                batch_size=32,
            ),
        ),
        "tcn": BenchmarkCase(
            key="tcn",
            name="TCN",
            factory=lambda dt: TCN(
                nu=10,
                ny=10,
                num_channels=[64, 64, 64, 64],
                kernel_size=3,
                dropout=0.1,
                learning_rate=1e-3,
                epochs=200,
                batch_size=32,
            ),
        ),
        "ude": BenchmarkCase(
            key="ude",
            name="UDE",
            factory=lambda dt: UDE(
                sampling_time=dt,
                tau=1.0,
                hidden_layers=[64, 64],
                learning_rate=1e-3,
                epochs=1000,
                sequence_length=50,
                integration_substeps=5,
                training_mode="full",
            ),
        ),
        "mamba": BenchmarkCase(
            key="mamba",
            name="Mamba",
            factory=lambda dt: Mamba(
                nu=10,
                ny=10,
                d_model=64,
                d_state=16,
                d_conv=4,
                n_layers=2,
                expand_factor=2,
                dropout=0.1,
                learning_rate=1e-3,
                epochs=200,
                batch_size=32,
            ),
        ),
        "gru": BenchmarkCase(
            key="gru",
            name="GRU",
            factory=lambda dt: GRU(
                nu=10,
                ny=10,
                hidden_size=64,
                num_layers=2,
                dropout=0.1,
                learning_rate=1e-3,
                epochs=200,
                batch_size=32,
            ),
        ),
        "neural_network": BenchmarkCase(
            key="neural_network",
            name="Neural Network",
            factory=lambda dt: NeuralNetwork(
                nu=10,
                ny=10,
                hidden_layers=[80, 80, 80],
                activation="selu",
                learning_rate=1e-3,
                epochs=200,
                batch_size=32,
            ),
        ),
        "vanilla_node_2d": BenchmarkCase(
            key="vanilla_node_2d",
            name="VanillaNODE2D",
            factory=lambda dt: VanillaNODE2D(
                hidden_dim=128,
                dt=dt,
                solver="rk4",
                learning_rate=0.01,
                epochs=5000,
                k_steps=20,
                batch_size=128,
            ),
        ),
        "structured_node": BenchmarkCase(
            key="structured_node",
            name="StructuredNODE",
            factory=lambda dt: StructuredNODE(
                hidden_dim=128,
                dt=dt,
                solver="rk4",
                learning_rate=0.01,
                epochs=5000,
                k_steps=20,
                batch_size=128,
            ),
        ),
        "adaptive_node": BenchmarkCase(
            key="adaptive_node",
            name="AdaptiveNODE",
            factory=lambda dt: AdaptiveNODE(
                hidden_dim=128,
                dt=dt,
                solver="rk4",
                learning_rate=0.005,
                epochs=5000,
                k_steps=20,
                batch_size=128,
            ),
        ),
    }


def build_benchmark_cases(selected: Iterable[str] | None = None) -> list[BenchmarkCase]:
    """Resolve benchmark cases by key."""
    cases = _base_case_factories()
    if selected is None:
        return list(cases.values())

    resolved: list[BenchmarkCase] = []
    unknown = []
    for key in selected:
        norm = key.strip().lower()
        if norm not in cases:
            unknown.append(norm)
            continue
        resolved.append(cases[norm])
    if unknown:
        available = ", ".join(sorted(cases))
        names = ", ".join(sorted(set(unknown)))
        raise ValueError(f"Unknown benchmark case(s): {names}. Available: {available}")
    return resolved


class BenchmarkRunner:
    """Runs full benchmark and emits structured results."""

    def __init__(self, cases: list[BenchmarkCase], config: BenchmarkConfig):
        if not cases:
            raise ValueError("At least one benchmark case is required")
        self.cases = cases
        self.config = config

    @staticmethod
    def _fit_model(model, u, y, wandb_run=None):
        """Call model.fit with optional W&B args when supported."""
        fit_sig = inspect.signature(model.fit)
        kwargs = {}
        if "verbose" in fit_sig.parameters:
            kwargs["verbose"] = True
        if wandb_run is not None and "wandb_run" in fit_sig.parameters:
            kwargs["wandb_run"] = wandb_run
            if "wandb_log_every" in fit_sig.parameters:
                kwargs["wandb_log_every"] = 1
        model.fit(u, y, **kwargs)

    @staticmethod
    def _evaluate_mode(model, mode: str, test_ds: Dataset) -> tuple[dict[str, float], int]:
        mode = mode.upper()
        if mode not in {"OSA", "FR"}:
            raise ValueError(f"Unsupported mode: {mode}")

        if mode == "OSA":
            y_pred = model.predict(test_ds.u, test_ds.y, mode="OSA")
        else:
            y_init = test_ds.y[: model.max_lag]
            y_pred = model.predict(test_ds.u, y_init, mode="FR")

        y_true = test_ds.y[model.max_lag :]
        metrics = Metrics.compute_all(y_true, y_pred)
        return metrics, len(y_pred)

    def run(self, wandb_run=None) -> list[dict[str, float | int | str]]:
        """Execute all benchmark cases on all datasets."""
        rows: list[dict[str, float | int | str]] = []

        for dataset_name in self.config.datasets:
            ds = Dataset.from_bab_experiment(
                dataset_name,
                preprocess=self.config.preprocess,
                resample_factor=self.config.resample_factor,
                end_ref_tolerance=self.config.end_ref_tolerance,
            )
            train_ds, test_ds = ds.split(self.config.train_ratio)
            dt = 1.0 / ds.sampling_rate

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "dataset/samples": len(ds),
                        "dataset/train_samples": len(train_ds),
                        "dataset/test_samples": len(test_ds),
                        "dataset/sampling_rate_hz": ds.sampling_rate,
                    }
                )

            for case_idx, case in enumerate(self.cases, 1):
                print(
                    f"\n{'='*60}\n"
                    f"[{case_idx}/{len(self.cases)}] Training {case.name} ..."
                    f"\n{'='*60}"
                )
                model = case.factory(dt)
                t0 = time.perf_counter()
                try:
                    self._fit_model(model, train_ds.u, train_ds.y, wandb_run=wandb_run)
                    fit_seconds = time.perf_counter() - t0
                    print(f"[{case.name}] Training done in {fit_seconds:.1f}s")
                except Exception as exc:
                    fit_seconds = time.perf_counter() - t0
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "model": case.name,
                            "mode": "ERROR",
                            "fit_seconds": fit_seconds,
                            "n_predictions": 0,
                            "MSE": np.nan,
                            "RMSE": np.nan,
                            "MAE": np.nan,
                            "R2": np.nan,
                            "NRMSE": np.nan,
                            "FIT%": np.nan,
                            "error": str(exc),
                        }
                    )
                    print(f"[benchmark] Skipping {case.name}: {exc}")
                    continue

                for mode in case.modes:
                    try:
                        metrics, n_pred = self._evaluate_mode(model, mode, test_ds)
                        row = {
                            "dataset": dataset_name,
                            "model": case.name,
                            "mode": mode.upper(),
                            "fit_seconds": fit_seconds,
                            "n_predictions": n_pred,
                            **metrics,
                        }
                        rows.append(row)

                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "benchmark/fit_seconds": fit_seconds,
                                    "benchmark/n_predictions": n_pred,
                                    **{
                                        f"benchmark/{mode.lower()}/{k.lower()}": v
                                        for k, v in metrics.items()
                                    },
                                }
                            )
                    except Exception as exc:
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "model": case.name,
                                "mode": f"{mode.upper()}_ERROR",
                                "fit_seconds": fit_seconds,
                                "n_predictions": 0,
                                "MSE": np.nan,
                                "RMSE": np.nan,
                                "MAE": np.nan,
                                "R2": np.nan,
                                "NRMSE": np.nan,
                                "FIT%": np.nan,
                                "error": str(exc),
                            }
                        )
                        print(f"[benchmark] {case.name} {mode.upper()} failed: {exc}")

        if wandb_run is not None and rows:
            try:
                import wandb

                columns = sorted({key for row in rows for key in row.keys()})
                table = wandb.Table(columns=columns)
                for row in rows:
                    table.add_data(*[row.get(c) for c in columns])
                wandb_run.log({"benchmark/results_table": table})
            except Exception:
                pass

        return rows

    def run_and_save(self, wandb_run=None) -> list[dict[str, float | int | str]]:
        """Run benchmark and persist result rows as JSON."""
        rows = self.run(wandb_run=wandb_run)
        output_path = Path(self.config.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "cases": [case.key for case in self.cases],
            "results": rows,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return rows


def summarize_results(
    rows: list[dict[str, float | int | str]],
    mode: str = "FR",
    metric: str = "FIT%",
) -> list[dict[str, float | int | str]]:
    """Return leaderboard rows sorted by metric for one prediction mode."""
    filtered = []
    for row in rows:
        if str(row.get("mode", "")).upper() != mode.upper():
            continue
        value = row.get(metric, np.nan)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            filtered.append(row)
    return sorted(filtered, key=lambda row: float(row.get(metric, 0.0)), reverse=True)
