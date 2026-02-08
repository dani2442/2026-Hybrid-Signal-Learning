"""Universal Differential Equation (UDE) model.

Combines a known physics prior (linear beam dynamics) with a small neural
network that learns the unmodelled residual:

    dx/dt = f_physics(x, u) + f_nn(x, u)

The physics part uses the same second-order beam ODE as HybridLinearBeam:

    J θ̈ + R θ̇ + K (θ + δ) = τ V

but the neural correction f_nn captures any model mismatch (nonlinearities,
unmodelled friction, etc.) and is trained end-to-end through the ODE
integrator via back-propagation.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .base import BaseModel
from .torchsde_utils import (
    ControlledPathMixin,
    inverse_softplus,
    simulate_controlled_sde,
    train_sequence_batches,
)


class _UDEFunc(ControlledPathMixin):
    """SDE function for the UDE: physics drift + neural residual, zero diffusion."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        sampling_time: float,
        tau: float,
        init_params: Dict[str, float],
        hidden_layers: List[int],
        device,
        dtype,
    ):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        self._device = device
        self._dtype = dtype
        self.tau = float(tau)
        self._eps = 1e-8

        # ---------- Physical parameters (trainable, softplus-constrained) ----
        self.raw_J = nn.Parameter(
            torch.tensor(
                inverse_softplus(init_params["J"]),
                dtype=dtype, device=device,
            )
        )
        self.raw_R = nn.Parameter(
            torch.tensor(
                inverse_softplus(init_params["R"]),
                dtype=dtype, device=device,
            )
        )
        self.raw_K = nn.Parameter(
            torch.tensor(
                inverse_softplus(init_params["K"]),
                dtype=dtype, device=device,
            )
        )
        self.delta = nn.Parameter(
            torch.tensor(float(init_params["delta"]), dtype=dtype, device=device)
        )

        # ---------- Neural residual  f_nn([theta, omega, V]) → [d_theta, d_omega]
        input_size = 3  # theta, omega, voltage
        state_dim = 2   # corrections to [d_theta, d_omega]
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, state_dim))
        self.residual_net = nn.Sequential(*layers).to(dtype).to(device)

        # Small initial scale so physics dominates at the start
        for m in self.residual_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self._init_control_path(
            dt=sampling_time,
            input_dim=1,
            device=device,
            dtype=dtype,
        )

    # ----- parameter helpers -------------------------------------------------
    def _decoded_physics(self):
        import torch.nn.functional as F

        J = F.softplus(self.raw_J) + self._eps
        R = F.softplus(self.raw_R) + self._eps
        K = F.softplus(self.raw_K) + self._eps
        return J, R, K, self.delta

    def parameters(self):
        return [self.raw_J, self.raw_R, self.raw_K, self.delta] + list(
            self.residual_net.parameters()
        )

    def to(self, device):
        self.residual_net = self.residual_net.to(device)
        self._u_path = self._u_path.to(device)
        return self

    def train(self):
        self.residual_net.train()

    def eval(self):
        self.residual_net.eval()

    # ----- SDE interface -----------------------------------------------------
    def f(self, t, y):
        """Drift = physics + neural residual."""
        import torch

        J, R, K, delta = self._decoded_physics()
        theta = y[:, 0]
        omega = y[:, 1]
        voltage = self._u_at(t, y.shape[0])[:, 0]

        # Physics: [d_theta = omega,  d_omega = (tau*V - R*omega - K*(theta+delta))/J]
        acc_phys = (self.tau * voltage - R * omega - K * (theta + delta)) / J
        f_phys = torch.stack([omega, acc_phys], dim=-1)

        # Neural residual
        nn_input = torch.stack([theta, omega, voltage], dim=-1)
        f_nn = self.residual_net(nn_input)

        return f_phys + f_nn

    def g(self, t, y):
        import torch

        return torch.zeros_like(y)

    def decoded_parameter_dict(self) -> Dict[str, float]:
        J, R, K, delta = self._decoded_physics()
        return {
            "J": float(J.detach().cpu().item()),
            "R": float(R.detach().cpu().item()),
            "K": float(K.detach().cpu().item()),
            "delta": float(delta.detach().cpu().item()),
        }


class UDE(BaseModel):
    """
    Universal Differential Equation for system identification.

    Combines a known second-order physics prior with a neural network
    residual, both trained jointly through an ODE integrator.

    Dynamics (state = [θ, ω]):
        dθ/dt = ω
        dω/dt = (τV − Rω − K(θ+δ)) / J   +  f_nn(θ, ω, V)

    Args:
        sampling_time: dt of the discretised data (seconds).
        tau: Motor torque constant (fixed, not trained).
        hidden_layers: Widths of the residual MLP hidden layers.
        learning_rate: Adam learning rate.
        epochs: Training epochs (each sweeps ≤100 random subsequences).
        sequence_length: Subsequence length for BPTT.
        ridge: Tikhonov regularisation for the initial LS guess.
    """

    def __init__(
        self,
        sampling_time: float = 0.05,
        tau: float = 1.0,
        hidden_layers: List[int] | None = None,
        learning_rate: float = 1e-3,
        epochs: int = 200,
        sequence_length: int = 20,
        ridge: float = 1e-8,
    ):
        super().__init__(nu=1, ny=2)
        if sampling_time <= 0:
            raise ValueError("sampling_time must be positive")
        self.sampling_time = float(sampling_time)
        self.tau = float(tau)
        self.hidden_layers = hidden_layers or [64, 64]
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.sequence_length = int(sequence_length)
        self.ridge = float(ridge)

        self.sde_func_: _UDEFunc | None = None
        self._device = None
        self._dtype = None
        self.training_loss_: list[float] = []

    # ----- physics-based initial guess (same as HybridLinearBeam) -----------
    def _initial_guess(self, u: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        dt = self.sampling_time
        y_dot = np.gradient(y, dt)
        y_ddot = np.gradient(y_dot, dt)

        phi = np.column_stack([-y_dot, -y, u, np.ones_like(y)])
        reg = phi.T @ phi + self.ridge * np.eye(phi.shape[1])
        theta = np.linalg.solve(reg, phi.T @ y_ddot)

        a1, a0, b0, bias = float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3])
        b0_safe = b0 if not np.isclose(b0, 0.0) else 1e-6
        J0 = max(self.tau / b0_safe, 1e-6)
        R0 = max(a1 * J0, 1e-6)
        K0 = max(a0 * J0, 1e-6)
        delta0 = float(-bias / a0) if not np.isclose(a0, 0.0) else 0.0
        return {"J": J0, "R": R0, "K": K0, "delta": delta0}

    # ----- simulation helper ------------------------------------------------
    def _simulate_trajectory(self, u_path, x0):
        return simulate_controlled_sde(
            sde_func=self.sde_func_,
            u_path=u_path,
            x0=x0,
            dt=self.sampling_time,
            method="euler",
        )

    # ----- fit ---------------------------------------------------------------
    def fit(
        self,
        u: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
        wandb_run=None,
        wandb_log_every: int = 1,
    ) -> "UDE":
        """Train physics + neural parameters through the ODE integrator."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install torch")

        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        if len(u) != len(y):
            raise ValueError("u and y must have same length")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float64

        init_params = self._initial_guess(u, y)

        self.sde_func_ = _UDEFunc(
            sampling_time=self.sampling_time,
            tau=self.tau,
            init_params=init_params,
            hidden_layers=self.hidden_layers,
            device=self._device,
            dtype=self._dtype,
        )

        # For train_sequence_batches the state is 2-D [theta, omega].
        # We approximate omega from finite differences.
        dt = self.sampling_time
        omega = np.gradient(y, dt)
        y_state = np.column_stack([y, omega])  # (N, 2)
        u_2d = u.reshape(-1, 1)

        self.training_loss_ = train_sequence_batches(
            sde_func=self.sde_func_,
            simulate_fn=lambda u_seq, x0: self._simulate_trajectory(u_seq, x0),
            u=u_2d,
            y=y_state,
            input_dim=1,
            state_dim=2,
            sequence_length=self.sequence_length,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            device=self._device,
            dtype=self._dtype,
            verbose=verbose,
            progress_desc="Training UDE",
            wandb_run=wandb_run,
            wandb_log_every=wandb_log_every,
        )

        self._is_fitted = True
        return self

    # ----- predict ----------------------------------------------------------
    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction: use measured θ(k), ω(k) → θ(k+1)."""
        import torch

        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        dt = self.sampling_time
        omega = np.gradient(y, dt)

        predictions = []
        self.sde_func_.eval()
        with torch.no_grad():
            for t in range(len(y) - 1):
                x_t = torch.tensor(
                    [y[t], omega[t]], dtype=self._dtype, device=self._device,
                )
                u_t = torch.tensor(
                    [[u[t]], [u[t]]], dtype=self._dtype, device=self._device,
                )
                x_next = self._simulate_trajectory(u_t, x_t)
                predictions.append(float(x_next[-1, 0].cpu().item()))

        return np.asarray(predictions)

    def predict_free_run(
        self,
        u: np.ndarray,
        y_initial: np.ndarray,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Free-run simulation from initial conditions."""
        import torch

        u = np.asarray(u, dtype=float).flatten()
        y_init = np.asarray(y_initial, dtype=float).flatten()
        dt = self.sampling_time

        theta0 = y_init[0]
        omega0 = (y_init[1] - y_init[0]) / dt if len(y_init) >= 2 else 0.0

        self.sde_func_.eval()
        with torch.no_grad():
            u_t = torch.tensor(
                u.reshape(-1, 1), dtype=self._dtype, device=self._device,
            )
            x0 = torch.tensor(
                [theta0, omega0], dtype=self._dtype, device=self._device,
            )
            pred = self._simulate_trajectory(u_t, x0)

        y_hat = pred[:, 0].cpu().numpy()
        return y_hat[self.max_lag :]

    def parameters(self) -> Dict[str, float]:
        """Return identified physical parameters."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        return self.sde_func_.decoded_parameter_dict()

    def __repr__(self) -> str:
        return (
            f"UDE(dt={self.sampling_time}, tau={self.tau}, "
            f"hidden_layers={self.hidden_layers}, epochs={self.epochs})"
        )
