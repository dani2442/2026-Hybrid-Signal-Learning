"""Neural SDE model with learned drift and diffusion."""

from __future__ import annotations

from itertools import chain
from typing import List

import numpy as np
from tqdm.auto import tqdm

from .base import BaseModel


class _ControlledNeuralSDEFunc:
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

        self.dt = float(dt)
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
        self._u_path = torch.zeros(2, self.input_dim)

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

    def set_control(self, u_path):
        self._u_path = u_path

    def _u_at(self, t, batch_size):
        import torch

        idx = torch.clamp((t / self.dt).long(), min=0, max=self._u_path.shape[0] - 1)
        u_t = self._u_path[idx]
        if u_t.ndim == 1:
            u_t = u_t.unsqueeze(0)
        return u_t.expand(batch_size, -1)

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

        if self.solver not in self._SUPPORTED_SOLVERS:
            supported = ", ".join(sorted(self._SUPPORTED_SOLVERS))
            raise ValueError(f"Unknown solver: {self.solver}. Supported: {supported}")

        self.sde_func_ = None
        self._device = None
        self._dtype = None

    def _simulate_trajectory(self, u_path, x0):
        try:
            import torch
            import torchsde
        except ImportError:
            raise ImportError("torchsde required. Install with: pip install torchsde")

        self.sde_func_.set_control(u_path)
        ts = torch.arange(
            u_path.shape[0], dtype=u_path.dtype, device=u_path.device
        ) * self.dt
        x0_batch = x0.reshape(1, self.state_dim)
        x_path = torchsde.sdeint(
            self.sde_func_,
            x0_batch,
            ts,
            method=self.solver,
            dt=self.dt,
        )
        return x_path[:, 0, :]

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
        sequence_length: int = 20,
    ) -> "NeuralSDE":
        """Train drift and diffusion networks by backprop through torchsde."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install torch")

        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y = np.asarray(y, dtype=float).reshape(-1, self.state_dim)

        n_samples = len(y)
        n_sequences = n_samples - sequence_length
        if n_sequences <= 0:
            raise ValueError("Not enough data for given sequence length")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float32
        self.sde_func_ = _ControlledNeuralSDEFunc(
            state_dim=self.state_dim,
            input_dim=self.input_dim,
            hidden_layers=self.hidden_layers,
            diffusion_hidden_layers=self.diffusion_hidden_layers,
            dt=self.dt,
        ).to(self._device)

        optimizer = optim.Adam(self.sde_func_.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        epoch_iter = range(self.epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training NeuralSDE", unit="epoch")

        for _ in epoch_iter:
            self.sde_func_.train()
            total_loss = 0.0
            indices = np.random.permutation(n_sequences)[: min(100, n_sequences)]

            for idx in indices:
                y_seq = torch.tensor(
                    y[idx : idx + sequence_length], dtype=self._dtype, device=self._device
                )
                u_seq = torch.tensor(
                    u[idx : idx + sequence_length], dtype=self._dtype, device=self._device
                )

                optimizer.zero_grad()
                pred_seq = self._simulate_trajectory(u_path=u_seq, x0=y_seq[0])
                loss = criterion(pred_seq, y_seq)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu().item())

            if verbose and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=total_loss / len(indices))

        self._is_fitted = True
        return self

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction using measured state at each step."""
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
        del show_progress  # retained for API compatibility
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
            f"NeuralSDE(state_dim={self.state_dim}, input_dim={self.input_dim}, "
            f"hidden_layers={self.hidden_layers}, "
            f"diffusion_hidden_layers={self.diffusion_hidden_layers}, "
            f"solver='{self.solver}')"
        )
