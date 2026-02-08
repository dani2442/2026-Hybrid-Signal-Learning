"""Neural SDE model with learned drift and diffusion."""

from __future__ import annotations

from itertools import chain
from typing import List

import numpy as np

from .base import BaseModel
from .torchsde_utils import (
    ControlledPathMixin,
    simulate_controlled_sde,
    train_sequence_batches,
)


class _ControlledNeuralSDEFunc(ControlledPathMixin):
    """SDE function with piecewise-constant control input."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        hidden_layers: List[int],
        diffusion_hidden_layers: List[int],
        dt: float,
    ):
        import torch
        import torch.nn as nn

        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)

        self.drift_net = self._build_network(
            input_size=self.state_dim + self.input_dim,
            output_size=self.state_dim,
            hidden_layers=hidden_layers,
            final_bias=0.0,
        )
        self.diffusion_net = self._build_network(
            input_size=self.state_dim + self.input_dim,
            output_size=self.state_dim,
            hidden_layers=diffusion_hidden_layers,
            final_bias=-3.0,
        )
        self._init_control_path(
            dt=dt,
            input_dim=self.input_dim,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    @staticmethod
    def _build_network(
        input_size: int,
        output_size: int,
        hidden_layers: List[int],
        final_bias: float,
    ):
        import torch.nn as nn

        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        net = nn.Sequential(*layers)

        linear_layers = [m for m in net.modules() if isinstance(m, nn.Linear)]
        for idx, module in enumerate(linear_layers):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                if idx == len(linear_layers) - 1:
                    nn.init.constant_(module.bias, final_bias)
                else:
                    nn.init.zeros_(module.bias)
        return net

    def to(self, device):
        self.drift_net = self.drift_net.to(device)
        self.diffusion_net = self.diffusion_net.to(device)
        self._u_path = self._u_path.to(device)
        return self

    def parameters(self):
        return chain(self.drift_net.parameters(), self.diffusion_net.parameters())

    def train(self):
        self.drift_net.train()
        self.diffusion_net.train()

    def eval(self):
        self.drift_net.eval()
        self.diffusion_net.eval()

    def f(self, t, y):
        import torch

        u_t = self._u_at(t, y.shape[0])
        xu = torch.cat([y, u_t], dim=-1)
        return self.drift_net(xu)

    def g(self, t, y):
        import torch
        import torch.nn.functional as F

        u_t = self._u_at(t, y.shape[0])
        xu = torch.cat([y, u_t], dim=-1)
        return F.softplus(self.diffusion_net(xu)) + 1e-6


class NeuralSDE(BaseModel):
    """
    Neural SDE for continuous-time system identification.

    Dynamics:
        dx = f_theta(x, u) dt + g_phi(x, u) dW_t
    where both drift f_theta and diffusion g_phi are learned.
    """

    _SUPPORTED_SOLVERS = {"euler", "milstein", "srk"}

    def __init__(
        self,
        state_dim: int = 1,
        input_dim: int = 1,
        hidden_layers: List[int] = [64, 64],
        diffusion_hidden_layers: List[int] = [64, 64],
        solver: str = "euler",
        dt: float = 0.05,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        sequence_length: int = 50,
    ):
        super().__init__(nu=input_dim, ny=state_dim)
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.hidden_layers = list(hidden_layers)
        self.diffusion_hidden_layers = list(diffusion_hidden_layers)
        self.solver = solver
        self.dt = float(dt)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.sequence_length = int(sequence_length)

        if self.solver not in self._SUPPORTED_SOLVERS:
            supported = ", ".join(sorted(self._SUPPORTED_SOLVERS))
            raise ValueError(f"Unknown solver: {self.solver}. Supported: {supported}")

        self.sde_func_ = None
        self._device = None
        self._dtype = None
        self.training_loss_: list[float] = []

    def _simulate_trajectory(self, u_path, x0, deterministic=False):
        if deterministic:
            # Use drift-only (ODE) simulation for deterministic eval
            from .torchsde_utils import _euler_ode_integrate
            self.sde_func_.set_control(u_path)
            return _euler_ode_integrate(
                self.sde_func_, u_path, x0, self.dt, self.dt,
            )
        return simulate_controlled_sde(
            sde_func=self.sde_func_,
            u_path=u_path,
            x0=x0,
            dt=self.dt,
            method=self.solver,
        )

    def _integrate_one_step(self, x_t, u_t, deterministic=False):
        import torch

        u_path = torch.cat([u_t, u_t], dim=0)
        x_path = self._simulate_trajectory(
            u_path=u_path, x0=x_t, deterministic=deterministic,
        )
        return x_path[-1]

    def fit(
        self,
        u: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
        wandb_run=None,
        wandb_log_every: int = 1,
    ) -> "NeuralSDE":
        """Train drift and diffusion networks by backprop through torchsde."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install torch")

        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y = np.asarray(y, dtype=float).reshape(-1, self.state_dim)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float32
        self.sde_func_ = _ControlledNeuralSDEFunc(
            state_dim=self.state_dim,
            input_dim=self.input_dim,
            hidden_layers=self.hidden_layers,
            diffusion_hidden_layers=self.diffusion_hidden_layers,
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
            progress_desc="Training NeuralSDE",
            wandb_run=wandb_run,
            wandb_log_every=wandb_log_every,
        )

        self._is_fitted = True
        return self

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction using measured state at each step.

        Uses deterministic (drift-only) integration so predictions are
        reproducible and not degraded by accumulated stochastic noise.
        """
        import torch

        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y = np.asarray(y, dtype=float).reshape(-1, self.state_dim)

        predictions = []
        self.sde_func_.eval()
        with torch.no_grad():
            for t in range(len(y) - 1):
                x_t = torch.tensor(y[t], dtype=self._dtype, device=self._device)
                u_t = torch.tensor(u[t : t + 1], dtype=self._dtype, device=self._device)
                x_next = self._integrate_one_step(
                    x_t=x_t, u_t=u_t, deterministic=True,
                )
                predictions.append(x_next.cpu().numpy())

        return np.asarray(predictions).reshape(-1)

    def predict_free_run(
        self,
        u: np.ndarray,
        y_initial: np.ndarray,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Free-run simulation from initial condition.

        Uses deterministic (drift-only) integration so predictions are
        reproducible and not degraded by accumulated stochastic noise.
        """
        del show_progress  # retained for API compatibility
        import torch

        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y_init = np.asarray(y_initial, dtype=float).reshape(-1, self.state_dim)

        self.sde_func_.eval()
        with torch.no_grad():
            u_t = torch.tensor(u, dtype=self._dtype, device=self._device)
            x0 = torch.tensor(y_init[0], dtype=self._dtype, device=self._device)
            pred = self._simulate_trajectory(
                u_path=u_t, x0=x0, deterministic=True,
            )
        return pred.cpu().numpy().reshape(-1)

    def __repr__(self) -> str:
        return (
            f"NeuralSDE(state_dim={self.state_dim}, input_dim={self.input_dim}, "
            f"hidden_layers={self.hidden_layers}, "
            f"diffusion_hidden_layers={self.diffusion_hidden_layers}, "
            f"solver='{self.solver}')"
        )
