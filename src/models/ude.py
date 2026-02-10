"""Universal Differential Equation (UDE) model.

Combines a known physics prior (linear beam dynamics) with a small neural
residual acting only on acceleration:

    dθ/dt = ω
    dω/dt = (τV - Rω - K(θ + δ))/J + r_nn(ω)

The physics part uses the same second-order beam ODE as HybridLinearBeam:

    J θ̈ + R θ̇ + K (θ + δ) = τ V

and the neural correction r_nn captures unmodelled effects (e.g. nonlinear
friction) while keeping kinematics explicit in dθ/dt = ω.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .base import BaseModel
from .torchsde_utils import (
    ControlledPathMixin,
    inverse_softplus,
    train_sequence_batches,
)


class _UDEFunc(ControlledPathMixin):
    """SDE function for the UDE: physics drift + residual on dω/dt, zero diffusion."""

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

        # ---------- Neural residual r_nn(omega) → correction to d_omega
        input_size = 1  # omega only
        residual_dim = 1
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, residual_dim))
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
        """Drift with explicit kinematics and neural residual only in acceleration."""
        import torch

        J, R, K, delta = self._decoded_physics()
        theta = y[:, 0]
        omega = y[:, 1]
        voltage = self._u_at(t, y.shape[0])[:, 0]

        # Physics: [d_theta = omega, d_omega = (tau*V - R*omega - K*(theta+delta))/J]
        acc_phys = (self.tau * voltage - R * omega - K * (theta + delta)) / J
        # Neural residual correction to acceleration only: r_nn(omega)
        acc_res = self.residual_net(omega.unsqueeze(-1)).squeeze(-1)
        return torch.stack([omega, acc_phys + acc_res], dim=-1)

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

    Combines a known second-order physics prior with a neural residual on
    acceleration, both trained jointly through an ODE integrator.

    Dynamics (state = [θ, ω]):
        dθ/dt = ω
        dω/dt = (τV − Rω − K(θ+δ)) / J   +  f_nn(ω)

    Args:
        sampling_time: dt of the discretised data (seconds).
        tau: Motor torque constant (fixed, not trained).
        hidden_layers: Widths of the residual MLP hidden layers.
        learning_rate: Adam learning rate.
        epochs: Training epochs (each sweeps ≤100 random subsequences).
        sequence_length: Subsequence length for BPTT.
        ridge: Tikhonov regularisation for the initial LS guess.
        integration_substeps: RK4 sub-steps per sample interval.
        training_mode: "full" (single dataset) or "subsequence" (single/multi).
        omega_loss_weight: Relative weight for velocity error in training loss.
    """

    def __init__(
        self,
        sampling_time: float = 0.05,
        tau: float = 1.0,
        hidden_layers: List[int] | None = None,
        learning_rate: float = 1e-3,
        lr: float | None = None,
        epochs: int = 200,
        sequence_length: int = 20,
        sequences_per_epoch: int = 24,
        patience: int | None = 200,
        min_delta: float = 0.0,
        ridge: float = 1e-8,
        integration_substeps: int = 1,
        training_mode: str = "full",
        omega_loss_weight: float = 0.2,
    ):
        super().__init__(nu=1, ny=2)
        if sampling_time <= 0:
            raise ValueError("sampling_time must be positive")
        self.sampling_time = float(sampling_time)
        self.tau = float(tau)
        self.hidden_layers = hidden_layers or [64, 64]
        if lr is not None:
            learning_rate = float(lr)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.sequence_length = int(sequence_length)
        self.sequences_per_epoch = int(sequences_per_epoch)
        self.patience = None if patience is None else int(patience)
        self.min_delta = float(min_delta)
        self.ridge = float(ridge)
        self.integration_substeps = int(integration_substeps)
        if self.integration_substeps <= 0:
            raise ValueError("integration_substeps must be >= 1")
        if training_mode not in {"full", "subsequence"}:
            raise ValueError("training_mode must be 'full' or 'subsequence'")
        self.training_mode = training_mode
        self.omega_loss_weight = float(omega_loss_weight)
        if self.omega_loss_weight < 0:
            raise ValueError("omega_loss_weight must be non-negative")

        self.sde_func_: _UDEFunc | None = None
        self._device = None
        self._dtype = None
        self.training_loss_: list[float] = []

    @staticmethod
    def _tensor_to_numpy_safe(x) -> np.ndarray:
        x_cpu = x.detach().cpu()
        try:
            return x_cpu.numpy()
        except RuntimeError:
            return np.asarray(x_cpu.tolist())

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
        """Integrate the UDE forward using torchdiffeq (RK4).

        Much faster than the Python-level Euler loop because torchdiffeq
        builds the computation graph in C++ and RK4 is stable at dt ≈ 0.05
        without manual sub-stepping.
        """
        import torch
        from torchdiffeq import odeint

        self.sde_func_.set_control(u_path)
        N = u_path.shape[0]
        ts = (
            torch.arange(N, device=self._device, dtype=self._dtype)
            * self.sampling_time
        )
        x0_batch = x0.reshape(1, -1)  # (1, state_dim)
        step_size = self.sampling_time / float(self.integration_substeps)
        path = odeint(
            self.sde_func_.f,
            x0_batch,
            ts,
            method="rk4",
            options={"step_size": step_size},
        )
        return path[:, 0, :]  # (N, state_dim)

    # ----- full-trajectory training -----------------------------------------
    def _train_full_trajectory(
        self, u, y, verbose, wandb_run=None, wandb_log_every=1,
    ) -> list:
        """Train by integrating the full trajectory; weighted loss on θ and ω."""
        import torch
        import torch.optim as optim
        from tqdm.auto import tqdm

        dt = self.sampling_time
        u_t = torch.tensor(u.reshape(-1, 1), dtype=self._dtype, device=self._device)
        y_t = torch.tensor(y, dtype=self._dtype, device=self._device)  # (N,)
        omega_t = torch.tensor(np.gradient(y, dt), dtype=self._dtype, device=self._device)

        theta0 = y_t[0]
        omega0 = (y_t[1] - y_t[0]) / dt
        x0 = torch.tensor([theta0, omega0], dtype=self._dtype, device=self._device)

        params = list(self.sde_func_.parameters())
        optimizer = optim.Adam(params, lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=200, min_lr=1e-6,
        )
        loss_history: list[float] = []
        best_loss = float("inf")
        best_state: dict | None = None

        epoch_iter = range(self.epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training UDE (full)", unit="epoch")

        for epoch in epoch_iter:
            self.sde_func_.train()
            optimizer.zero_grad()

            pred = self._simulate_trajectory(u_t, x0)  # (N, 2)
            loss_theta = torch.mean((pred[:, 0] - y_t) ** 2)
            loss_omega = torch.mean((pred[:, 1] - omega_t) ** 2)
            loss = loss_theta + (self.omega_loss_weight * loss_omega)

            if torch.isnan(loss) or torch.isinf(loss):
                loss_history.append(float(loss.detach().cpu().item()))
                if verbose and hasattr(epoch_iter, "set_postfix"):
                    epoch_iter.set_postfix(loss=float("nan"))
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            lv = float(loss.detach().cpu().item())
            loss_history.append(lv)
            scheduler.step(lv)

            if lv < best_loss:
                best_loss = lv
                best_state = {id(p): p.data.clone() for p in params}

            if verbose and hasattr(epoch_iter, "set_postfix"):
                lr_now = optimizer.param_groups[0]["lr"]
                epoch_iter.set_postfix(loss=lv, lr=f"{lr_now:.1e}")
            if wandb_run and (epoch + 1) % wandb_log_every == 0:
                wandb_run.log(
                    {
                        "train/loss": lv,
                        "train/loss_theta": float(loss_theta.detach().cpu().item()),
                        "train/loss_omega": float(loss_omega.detach().cpu().item()),
                        "train/epoch": epoch + 1,
                    }
                )

        # Restore best weights
        if best_state is not None:
            for p in params:
                p.data.copy_(best_state[id(p)])

        return loss_history

    # ----- fit ---------------------------------------------------------------
    def fit(
        self,
        u: np.ndarray | Sequence[np.ndarray],
        y: np.ndarray | Sequence[np.ndarray],
        verbose: bool = True,
        wandb_run=None,
        wandb_log_every: int = 1,
    ) -> "UDE":
        """Train physics + neural parameters through the ODE integrator."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install torch")

        is_multi = isinstance(u, Sequence) and not isinstance(u, np.ndarray)
        if is_multi != (isinstance(y, Sequence) and not isinstance(y, np.ndarray)):
            raise ValueError("u and y must both be arrays or both be dataset lists")

        if is_multi:
            if len(u) != len(y):
                raise ValueError("u and y dataset lists must have the same length")
            u_data = [np.asarray(u_ds, dtype=float).flatten() for u_ds in u]
            y_data = [np.asarray(y_ds, dtype=float).flatten() for y_ds in y]
            for u_ds, y_ds in zip(u_data, y_data):
                if len(u_ds) != len(y_ds):
                    raise ValueError("Each dataset pair must have u and y with equal length")
            if self.training_mode == "full":
                raise ValueError(
                    "training_mode='full' does not support multi-dataset input. "
                    "Use training_mode='subsequence' for random-batch multi-dataset training."
                )
            init_u = u_data[0]
            init_y = y_data[0]
        else:
            u_data = np.asarray(u, dtype=float).flatten()
            y_data = np.asarray(y, dtype=float).flatten()
            if len(u_data) != len(y_data):
                raise ValueError("u and y must have same length")
            init_u = u_data
            init_y = y_data

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float32

        init_params = self._initial_guess(init_u, init_y)

        self.sde_func_ = _UDEFunc(
            sampling_time=self.sampling_time,
            tau=self.tau,
            init_params=init_params,
            hidden_layers=self.hidden_layers,
            device=self._device,
            dtype=self._dtype,
        )

        if self.training_mode == "full":
            self.training_loss_ = self._train_full_trajectory(
                u_data, y_data, verbose, wandb_run, wandb_log_every,
            )
        else:
            # Subsequence batching (original strategy)
            dt = self.sampling_time
            if is_multi:
                y_state = []
                u_2d = []
                for u_ds, y_ds in zip(u_data, y_data):
                    omega_ds = np.gradient(y_ds, dt)
                    y_state.append(np.column_stack([y_ds, omega_ds]))
                    u_2d.append(u_ds.reshape(-1, 1))
            else:
                omega = np.gradient(y_data, dt)
                y_state = np.column_stack([y_data, omega])  # (N, 2)
                u_2d = u_data.reshape(-1, 1)

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
                sequences_per_epoch=self.sequences_per_epoch,
                early_stopping_patience=self.patience,
                early_stopping_min_delta=self.min_delta,
                state_loss_weights=[1.0, self.omega_loss_weight],
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

        y_hat = self._tensor_to_numpy_safe(pred[:, 0])
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
