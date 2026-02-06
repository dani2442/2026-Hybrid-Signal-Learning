#!/usr/bin/env python
"""Demo for bab_datasets loader + two hybrid models."""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src import Dataset, HybridLinearBeam, HybridNonlinearCam, Metrics


def main():
    dataset = Dataset.from_bab_experiment(
        "multisine_05",
        preprocess=True,
        resample_factor=50,
    )
    dt = 1.0 / dataset.sampling_rate

    wandb_run = None
    wandb_project = os.getenv("WANDB_PROJECT")
    if wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=wandb_project,
                config={
                    "dataset": dataset.name,
                    "samples": len(dataset),
                    "sampling_time": dt,
                },
            )
        except Exception as exc:
            print(f"W&B disabled: {exc}")

    linear = HybridLinearBeam(sampling_time=dt, tau=1.0, estimate_delta=True)
    linear.fit(dataset.u, dataset.y, wandb_run=wandb_run)

    nonlinear = HybridNonlinearCam(
        sampling_time=dt,
        R=0.02,
        r=0.005,
        e=0.002,
        L=0.12,
        I=5e-4,
        J=1e-4,
        k=25.0,
        delta=0.01,
        k_t=0.08,
        k_b=0.08,
        R_M=2.0,
        L_M=0.01,
        trainable_params=("J", "k", "delta", "k_t"),
        epochs=200,
    )
    nonlinear.fit(dataset.u, dataset.y, wandb_run=wandb_run)

    y0 = dataset.y[:2]
    y_hat_linear = linear.predict(dataset.u, y0, mode="FR")
    y_hat_nonlinear = nonlinear.predict(dataset.u, y0, mode="FR")

    print("Linear params:", linear.parameters())
    print("Nonlinear params:", nonlinear.parameters())

    Metrics.summary(dataset.y[2:], y_hat_linear, name="HybridLinearBeam FR")
    Metrics.summary(dataset.y[2:], y_hat_nonlinear, name="HybridNonlinearCam FR")

    if wandb_run is not None:
        wandb_run.log(
            {
                "eval/linear_fit_percent": Metrics.fit_percent(dataset.y[2:], y_hat_linear),
                "eval/nonlinear_fit_percent": Metrics.fit_percent(dataset.y[2:], y_hat_nonlinear),
                "eval/linear_r2": Metrics.r2(dataset.y[2:], y_hat_linear),
                "eval/nonlinear_r2": Metrics.r2(dataset.y[2:], y_hat_nonlinear),
            }
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
