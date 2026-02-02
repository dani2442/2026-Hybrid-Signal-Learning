"""Neural ODE model for continuous-time system identification."""

from typing import List, Optional, Callable
import numpy as np
from tqdm.auto import tqdm

from .base import BaseModel


class NeuralODE(BaseModel):
    """
    Neural ODE for continuous-time dynamical system identification.
    
    Models the system dynamics as: dx/dt = f_θ(x, u)
    where f_θ is a neural network.
    
    Args:
        state_dim: Dimension of the state (default 1 for scalar output)
        input_dim: Dimension of the input (default 1)
        hidden_layers: List of hidden layer sizes for the dynamics network
        solver: ODE solver ('euler', 'rk4', 'dopri5')
        dt: Time step for integration
        learning_rate: Learning rate for training
        epochs: Number of training epochs
    """

    def __init__(
        self,
        state_dim: int = 1,
        input_dim: int = 1,
        hidden_layers: List[int] = [64, 64],
        solver: str = "rk4",
        dt: float = 0.05,
        learning_rate: float = 1e-3,
        epochs: int = 100,
    ):
        super().__init__(nu=input_dim, ny=state_dim)
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.solver = solver
        self.dt = dt
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.dynamics_net_ = None
        self._device = None

    def _build_dynamics_net(self):
        """Build neural network for dynamics f(x, u)."""
        import torch
        import torch.nn as nn

        input_size = self.state_dim + self.input_dim
        layers = []
        prev_size = input_size

        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, self.state_dim))

        model = nn.Sequential(*layers)
        
        # Initialize weights
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        return model

    def _integrate_step(self, x: "torch.Tensor", u: "torch.Tensor") -> "torch.Tensor":
        """Single integration step using selected solver."""
        import torch

        if self.solver == "euler":
            # Euler method: x_{n+1} = x_n + dt * f(x_n, u_n)
            xu = torch.cat([x, u], dim=-1)
            dx = self.dynamics_net_(xu)
            return x + self.dt * dx

        elif self.solver == "rk4":
            # Runge-Kutta 4th order
            def f(x_):
                xu = torch.cat([x_, u], dim=-1)
                return self.dynamics_net_(xu)

            k1 = f(x)
            k2 = f(x + 0.5 * self.dt * k1)
            k3 = f(x + 0.5 * self.dt * k2)
            k4 = f(x + self.dt * k3)
            return x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        else:
            raise ValueError(f"Unknown solver: {self.solver}")

    def fit(
        self,
        u: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
        sequence_length: int = 20,
    ) -> "NeuralODE":
        """
        Train the Neural ODE model.
        
        Args:
            u: Input signal
            y: Output signal (treated as state observations)
            verbose: Show training progress
            sequence_length: Length of sequences for training
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install torch")

        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y = np.asarray(y, dtype=float).reshape(-1, self.state_dim)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dynamics_net_ = self._build_dynamics_net().to(self._device)

        # Create training sequences
        n_samples = len(y)
        n_sequences = n_samples - sequence_length

        if n_sequences <= 0:
            raise ValueError("Not enough data for given sequence length")

        optimizer = optim.Adam(self.dynamics_net_.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        epoch_iter = range(self.epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training NeuralODE", unit="epoch")

        for epoch in epoch_iter:
            self.dynamics_net_.train()
            total_loss = 0.0

            # Random sequence sampling
            indices = np.random.permutation(n_sequences)[:min(100, n_sequences)]

            for idx in indices:
                # Get sequence
                y_seq = torch.tensor(
                    y[idx : idx + sequence_length], dtype=torch.float32
                ).to(self._device)
                u_seq = torch.tensor(
                    u[idx : idx + sequence_length], dtype=torch.float32
                ).to(self._device)

                optimizer.zero_grad()

                # Forward integration
                x = y_seq[0:1]  # Initial state
                predictions = [x]

                for t in range(sequence_length - 1):
                    u_t = u_seq[t : t + 1]
                    x = self._integrate_step(x, u_t)
                    predictions.append(x)

                pred = torch.cat(predictions, dim=0)
                loss = criterion(pred, y_seq)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if verbose and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=total_loss / len(indices))

        self._is_fitted = True
        return self

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        One-step-ahead prediction.
        
        Uses actual previous state for each prediction.
        """
        import torch

        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y = np.asarray(y, dtype=float).reshape(-1, self.state_dim)

        n_samples = len(y)
        predictions = []

        self.dynamics_net_.eval()
        with torch.no_grad():
            for t in range(n_samples - 1):
                x_t = torch.tensor(y[t:t+1], dtype=torch.float32).to(self._device)
                u_t = torch.tensor(u[t:t+1], dtype=torch.float32).to(self._device)
                x_next = self._integrate_step(x_t, u_t)
                predictions.append(x_next.cpu().numpy())

        return np.concatenate(predictions, axis=0).flatten()

    def predict_free_run(
        self,
        u: np.ndarray,
        y_initial: np.ndarray,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Free-run simulation using only initial condition.
        
        Integrates the ODE forward using predicted states.
        """
        import torch

        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y_init = np.asarray(y_initial, dtype=float)

        if y_init.ndim == 1:
            y_init = y_init.reshape(-1, self.state_dim)

        n_steps = len(u) - 1
        
        self.dynamics_net_.eval()
        x = torch.tensor(y_init[0:1], dtype=torch.float32).to(self._device)
        predictions = [x.cpu().numpy()]

        sim_range = range(n_steps)
        if show_progress:
            sim_range = tqdm(sim_range, desc="NeuralODE simulation", unit="step")

        with torch.no_grad():
            for t in sim_range:
                u_t = torch.tensor(u[t:t+1], dtype=torch.float32).to(self._device)
                x = self._integrate_step(x, u_t)
                predictions.append(x.cpu().numpy())

        return np.concatenate(predictions, axis=0).flatten()

    def __repr__(self) -> str:
        return (
            f"NeuralODE(state_dim={self.state_dim}, input_dim={self.input_dim}, "
            f"hidden_layers={self.hidden_layers}, solver='{self.solver}')"
        )
