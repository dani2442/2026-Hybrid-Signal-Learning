"""Neural ODE model solved with torchsde and zero diffusion."""

from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseModel
from .torchsde_utils import (
    ControlledPathMixin,
    simulate_controlled_sde,
    train_sequence_batches,
)


class _ControlledNeuralODEFunc(ControlledPathMixin):
    """SDE function wrapper for drift-only dynamics with piecewise-constant input."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        hidden_layers: List[int],
        dt: float,
    ):
        import torch
        import torch.nn as nn

        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)

        input_size = self.state_dim + self.input_dim
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, self.state_dim))
        self.drift_net = nn.Sequential(*layers)

        for module in self.drift_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self._init_control_path(
            dt=dt,
            input_dim=self.input_dim,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def to(self, device):
        self.drift_net = self.drift_net.to(device)
        self._u_path = self._u_path.to(device)
        return self

    def parameters(self):
        return self.drift_net.parameters()

    def train(self):
        self.drift_net.train()

    def eval(self):
        self.drift_net.eval()

    def f(self, t, y):
        import torch

        u_t = self._u_at(t, y.shape[0])
        xu = torch.cat([y, u_t], dim=-1)
        return self.drift_net(xu)

    def g(self, t, y):
        import torch

        return torch.zeros_like(y)


class NeuralODE(BaseModel):
    """
    Neural ODE for continuous-time system identification solved with torchsde.

    Dynamics:
        dx = f_theta(x, u) dt
    """

    _SOLVER_MAP = {
        "euler": "euler",
        "rk4": "euler",
        "dopri5": "euler",
    }

    def __init__(
        self,
        state_dim: int = 1,
        input_dim: int = 1,
        hidden_layers: List[int] = [64, 64],
        solver: str = "rk4",
        dt: float = 0.05,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        sequence_length: int = 50,
    ):
        super().__init__(nu=input_dim, ny=state_dim)
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.hidden_layers = list(hidden_layers)
        self.solver = solver
        self.dt = float(dt)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.sequence_length = int(sequence_length)

        self.sde_func_ = None
        self._device = None
        self._dtype = None
        self.training_loss_: list[float] = []

        if self.solver not in self._SOLVER_MAP:
            supported = ", ".join(sorted(self._SOLVER_MAP.keys()))
            raise ValueError(f"Unknown solver: {self.solver}. Supported: {supported}")

    def _simulate_trajectory(self, u_path, x0):
        return simulate_controlled_sde(
            sde_func=self.sde_func_,
            u_path=u_path,
            x0=x0,
            dt=self.dt,
            method=self._SOLVER_MAP[self.solver],
        )

    def _integrate_one_step(self, x_t, u_t):
        import torch

        u_path = torch.cat([u_t, u_t], dim=0)
        x_path = self._simulate_trajectory(u_path=u_path, x0=x_t)
        return x_path[-1]

    def fit(
        self,
        u: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
        wandb_run=None,
        wandb_log_every: int = 1,
    ) -> "NeuralODE":
        """Train model parameters by backpropagating through torchsde integration."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install torch")

        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y = np.asarray(y, dtype=float).reshape(-1, self.state_dim)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float32
        self.sde_func_ = _ControlledNeuralODEFunc(
            state_dim=self.state_dim,
            input_dim=self.input_dim,
            hidden_layers=self.hidden_layers,
            dt=self.dt,
        ).to(self._device)

        self.training_loss_ = train_sequence_batches(
            sde_func=self.sde_func_,
            simulate_fn=lambda u_seq, x0: self._simulate_trajectory(u_seq, x0),
            u=u,
            y=y,
            input_dim=self.input_dim,
            state_dim=self.state_dim,
            sequence_length=self.sequence_length,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            device=self._device,
            dtype=self._dtype,
            verbose=verbose,
            progress_desc="Training NeuralODE",
            wandb_run=wandb_run,
            wandb_log_every=wandb_log_every,
        )

        self._is_fitted = True
        return self

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction using measured states."""
        import torch

        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y = np.asarray(y, dtype=float).reshape(-1, self.state_dim)

        predictions = []
        self.sde_func_.eval()
        with torch.no_grad():
            for t in range(len(y) - 1):
                x_t = torch.tensor(y[t], dtype=self._dtype, device=self._device)
                u_t = torch.tensor(u[t : t + 1], dtype=self._dtype, device=self._device)
                x_next = self._integrate_one_step(x_t=x_t, u_t=u_t)
                predictions.append(x_next.cpu().numpy())

        return np.asarray(predictions).reshape(-1)

    def predict_free_run(
        self,
        u: np.ndarray,
        y_initial: np.ndarray,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Free-run simulation from initial condition."""
        del show_progress  # retained for compatibility
        import torch

        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y_init = np.asarray(y_initial, dtype=float).reshape(-1, self.state_dim)

        self.sde_func_.eval()
        with torch.no_grad():
            u_t = torch.tensor(u, dtype=self._dtype, device=self._device)
            x0 = torch.tensor(y_init[0], dtype=self._dtype, device=self._device)
            pred = self._simulate_trajectory(u_path=u_t, x0=x0)
        return pred.cpu().numpy().reshape(-1)

    def __repr__(self) -> str:
        return (
            f"NeuralODE(state_dim={self.state_dim}, input_dim={self.input_dim}, "
            f"hidden_layers={self.hidden_layers}, solver='{self.solver}')"
        )
