"""Linear physics-guided hybrid model for beam dynamics using torchsde."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .base import BaseModel
from .torchsde_utils import (
    ControlledPathMixin,
    inverse_softplus,
    optimize_with_adam,
    simulate_controlled_sde,
)


class _LinearBeamSDEFunc(ControlledPathMixin):
    """SDE drift for beam dynamics with zero diffusion."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        sampling_time: float,
        tau: float,
        estimate_delta: bool,
        init_params: Dict[str, float],
        device,
    ):
        import torch
        import torch.nn as nn

        self.tau = float(tau)
        self.estimate_delta = bool(estimate_delta)
        self._eps = 1e-8
        self._device = device

        self.raw_J = nn.Parameter(
            torch.tensor(
                inverse_softplus(init_params["J"]),
                dtype=torch.float64,
                device=device,
            )
        )
        self.raw_R = nn.Parameter(
            torch.tensor(
                inverse_softplus(init_params["R"]),
                dtype=torch.float64,
                device=device,
            )
        )
        self.raw_K = nn.Parameter(
            torch.tensor(
                inverse_softplus(init_params["K"]),
                dtype=torch.float64,
                device=device,
            )
        )

        if self.estimate_delta:
            self.delta = nn.Parameter(
                torch.tensor(float(init_params["delta"]), dtype=torch.float64, device=device)
            )
        else:
            self.delta = torch.tensor(0.0, dtype=torch.float64, device=device)

        self._init_control_path(
            dt=sampling_time,
            input_dim=1,
            device=device,
            dtype=torch.float64,
        )

    def parameters(self):
        params = [self.raw_J, self.raw_R, self.raw_K]
        if hasattr(self.delta, "requires_grad") and self.delta.requires_grad:
            params.append(self.delta)
        return params

    def train(self):
        return self

    def eval(self):
        return self

    def _decoded_params(self):
        import torch.nn.functional as F

        J = F.softplus(self.raw_J) + self._eps
        R = F.softplus(self.raw_R) + self._eps
        K = F.softplus(self.raw_K) + self._eps
        delta = self.delta if self.estimate_delta else self.delta.detach()
        return J, R, K, delta

    def f(self, t, y):
        import torch

        J, R, K, delta = self._decoded_params()
        theta = y[:, 0]
        omega = y[:, 1]
        voltage = self._u_at(t, y.shape[0])[:, 0]

        acc = (self.tau * voltage - R * omega - K * (theta + delta)) / J
        return torch.stack([omega, acc], dim=-1)

    def g(self, t, y):
        import torch

        return torch.zeros_like(y)

    def decoded_parameter_dict(self) -> Dict[str, float]:
        J, R, K, delta = self._decoded_params()
        return {
            "J": float(J.detach().cpu().item()),
            "R": float(R.detach().cpu().item()),
            "K": float(K.detach().cpu().item()),
            "delta": float(delta.detach().cpu().item()),
        }


class HybridLinearBeam(BaseModel):
    """
    Hybrid beam model:
        J*theta_ddot + R*theta_dot + K*(theta + delta) = tau*V

    Uses torchsde with zero diffusion and nn.Parameter-based optimization.
    """

    def __init__(
        self,
        sampling_time: float,
        tau: float = 1.0,
        estimate_delta: bool = True,
        ridge: float = 1e-8,
        learning_rate: float = 1e-2,
        lr: float | None = None,
        epochs: int = 600,
        integration_substeps: int = 1,
    ):
        super().__init__(nu=1, ny=2)
        if sampling_time <= 0:
            raise ValueError("sampling_time must be positive")
        if lr is not None:
            learning_rate = float(lr)
        self.sampling_time = float(sampling_time)
        self.tau = float(tau)
        self.estimate_delta = bool(estimate_delta)
        self.ridge = float(ridge)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.integration_substeps = int(integration_substeps)

        self.a1_: float = 0.0
        self.a0_: float = 0.0
        self.b0_: float = 0.0
        self.bias_: float = 0.0

        self.J_: float = 0.0
        self.R_: float = 0.0
        self.K_: float = 0.0
        self.delta_: float = 0.0

        self.training_loss_: list[float] = []
        self.sde_func_ = None
        self._device = None

    def _initial_guess(self, u: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        dt = self.sampling_time
        y_dot = np.gradient(y, dt)
        y_ddot = np.gradient(y_dot, dt)

        if self.estimate_delta:
            phi = np.column_stack([-y_dot, -y, u, np.ones_like(y)])
        else:
            phi = np.column_stack([-y_dot, -y, u])

        reg = phi.T @ phi + self.ridge * np.eye(phi.shape[1])
        theta = np.linalg.solve(reg, phi.T @ y_ddot)

        self.a1_ = float(theta[0])
        self.a0_ = float(theta[1])
        self.b0_ = float(theta[2])
        self.bias_ = float(theta[3]) if self.estimate_delta else 0.0

        b0_safe = self.b0_
        if np.isclose(b0_safe, 0.0):
            b0_safe = 1e-6

        J0 = max(self.tau / b0_safe, 1e-6)
        R0 = max(self.a1_ * J0, 1e-6)
        K0 = max(self.a0_ * J0, 1e-6)
        if self.estimate_delta and not np.isclose(self.a0_, 0.0):
            delta0 = float(-self.bias_ / self.a0_)
        else:
            delta0 = 0.0
        return {"J": float(J0), "R": float(R0), "K": float(K0), "delta": float(delta0)}

    def _simulate_theta_torch(self, u_t, theta0, omega0):
        import torch

        x0 = torch.stack([theta0, omega0])
        int_dt = (
            self.sampling_time / self.integration_substeps
            if self.integration_substeps > 1
            else None
        )
        state_path = simulate_controlled_sde(
            sde_func=self.sde_func_,
            u_path=u_t,
            x0=x0,
            dt=self.sampling_time,
            method="euler",
            integration_dt=int_dt,
        )
        return state_path[:, 0]

    def fit(
        self,
        u: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
        wandb_run=None,
        wandb_log_every: int = 1,
    ) -> "HybridLinearBeam":
        """Fit physical parameters by gradient descent through torchsde simulation."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install torch")

        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        if len(u) != len(y):
            raise ValueError("u and y must have same length")
        if len(y) < 5:
            raise ValueError("Need at least 5 samples to estimate derivatives")

        init_params = self._initial_guess(u=u, y=y)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sde_func_ = _LinearBeamSDEFunc(
            sampling_time=self.sampling_time,
            tau=self.tau,
            estimate_delta=self.estimate_delta,
            init_params=init_params,
            device=self._device,
        )

        u_t = torch.tensor(u, dtype=torch.float64, device=self._device).reshape(-1, 1)
        y_t = torch.tensor(y, dtype=torch.float64, device=self._device)
        theta0 = y_t[0]
        omega0 = (y_t[1] - y_t[0]) / self.sampling_time

        def _loss_fn():
            theta_hat = self._simulate_theta_torch(u_t=u_t, theta0=theta0, omega0=omega0)
            return torch.mean((theta_hat[self.max_lag :] - y_t[self.max_lag :]) ** 2)

        def _log_epoch(epoch: int, loss_value: float, grad_norm: float):
            if wandb_run is None or wandb_log_every <= 0 or epoch % wandb_log_every:
                return
            payload = {
                "train/epoch": epoch,
                "train/loss": loss_value,
                "train/grad_norm": grad_norm,
            }
            payload.update(
                {
                    f"params/{name}": value
                    for name, value in self.sde_func_.decoded_parameter_dict().items()
                }
            )
            wandb_run.log(payload, step=epoch)

        self.training_loss_ = optimize_with_adam(
            parameters=self.sde_func_.parameters(),
            loss_fn=_loss_fn,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            on_epoch_end=_log_epoch,
            verbose=verbose,
            progress_desc="Training HybridLinearBeam",
        )

        decoded = self.sde_func_.decoded_parameter_dict()
        self.J_ = decoded["J"]
        self.R_ = decoded["R"]
        self.K_ = decoded["K"]
        self.delta_ = decoded["delta"]

        self._is_fitted = True
        return self

    def _integrate_one_step(self, theta: float, omega: float, voltage: float) -> float:
        import torch

        u_step = torch.tensor(
            [[voltage], [voltage]], dtype=torch.float64, device=self._device
        )
        theta0 = torch.tensor(theta, dtype=torch.float64, device=self._device)
        omega0 = torch.tensor(omega, dtype=torch.float64, device=self._device)

        theta_path = self._simulate_theta_torch(u_t=u_step, theta0=theta0, omega0=omega0)
        return float(theta_path[-1].detach().cpu().item())

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction using measured theta(t)."""
        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        if len(u) != len(y):
            raise ValueError("u and y must have same length")
        if len(y) < self.max_lag + 1:
            return np.array([])

        dt = self.sampling_time
        y_hat = np.zeros_like(y, dtype=float)
        y_hat[:2] = y[:2]

        for k in range(1, len(y) - 1):
            theta_k = y[k]
            omega_k = (y[k] - y[k - 1]) / dt
            y_hat[k + 1] = self._integrate_one_step(theta=theta_k, omega=omega_k, voltage=u[k])

        return y_hat[self.max_lag :]

    def predict_free_run(self, u: np.ndarray, y_initial: np.ndarray) -> np.ndarray:
        """Free-run simulation with two initial output conditions."""
        import torch

        u = np.asarray(u, dtype=float).flatten()
        y_initial = np.asarray(y_initial, dtype=float).flatten()

        if len(y_initial) < self.max_lag:
            raise ValueError(f"Need at least {self.max_lag} initial outputs")
        if len(u) < self.max_lag:
            return np.array([])

        with torch.no_grad():
            u_t = torch.tensor(u, dtype=torch.float64, device=self._device).reshape(-1, 1)
            theta0 = torch.tensor(y_initial[0], dtype=torch.float64, device=self._device)
            omega0 = torch.tensor(
                (y_initial[1] - y_initial[0]) / self.sampling_time,
                dtype=torch.float64,
                device=self._device,
            )
            y_hat = self._simulate_theta_torch(u_t=u_t, theta0=theta0, omega0=omega0)
        return y_hat.detach().cpu().numpy()[self.max_lag :]

    def parameters(self) -> Dict[str, float]:
        """Return identified physical parameters."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        return {
            "J": self.J_,
            "R": self.R_,
            "K": self.K_,
            "delta": self.delta_,
            "tau": self.tau,
        }

    def __repr__(self) -> str:
        return (
            "HybridLinearBeam("
            f"dt={self.sampling_time}, tau={self.tau}, "
            f"estimate_delta={self.estimate_delta}, epochs={self.epochs})"
        )
