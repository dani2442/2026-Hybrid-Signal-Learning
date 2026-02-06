#!/usr/bin/env python
"""Demo for bab_datasets loader + two hybrid models."""

from src import Dataset, HybridLinearBeam, HybridNonlinearCam, Metrics


def main():
    dataset = Dataset.from_bab_experiment(
        "multisine_05",
        preprocess=True,
        resample_factor=50,
    )
    dt = 1.0 / dataset.sampling_rate

    linear = HybridLinearBeam(sampling_time=dt, tau=1.0, estimate_delta=True)
    linear.fit(dataset.u, dataset.y)

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
    nonlinear.fit(dataset.u, dataset.y)

    y0 = dataset.y[:2]
    y_hat_linear = linear.predict(dataset.u, y0, mode="FR")
    y_hat_nonlinear = nonlinear.predict(dataset.u, y0, mode="FR")

    print("Linear params:", linear.parameters())
    print("Nonlinear params:", nonlinear.parameters())

    Metrics.summary(dataset.y[2:], y_hat_linear, name="HybridLinearBeam FR")
    Metrics.summary(dataset.y[2:], y_hat_nonlinear, name="HybridNonlinearCam FR")


if __name__ == "__main__":
    main()
