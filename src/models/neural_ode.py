"""Neural ODE model solved with torchdiffeq.

Uses proper ODE solvers (RK4, dopri5) instead of SDE integration
with zero diffusion.  Supports both subsequence-batched training
and full-trajectory training.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .base import BaseModel
from .torchsde_utils import train_sequence_batches


class _ODEFunc:
    """ODE dynamics f(t, x, u) with linear interpolation of inputs."""

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        hidden_layers: List[int],
        activation: str = "selu",
    ):
        import torch.nn as nn

        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)

        act_cls = {"selu": nn.SELU, "tanh": nn.Tanh, "relu": nn.ReLU}.get(
            activation.lower(), nn.SELU
        )

        input_size = self.state_dim + self.input_dim
        layers: list[nn.Module] = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(act_cls())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, self.state_dim))
        self.net = nn.Sequential(*layers)

        # Weight initialisation matching the activation
        nonlin_name = "selu" if activation.lower() == "selu" else "linear"
        for idx, module in enumerate(self.net.modules()):
            if isinstance(module, nn.Linear):
                if module == list(self.net.modules())[-1]:  # output layer
                    nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity=nonlin_name)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Control signal storage (set before odeint call)
        self._u_series = None  # (T, input_dim)
        self._t_series = None  # (T,)

    def to(self, device):
        self.net = self.net.to(device)
        return self

    def parameters(self):
        return self.net.parameters()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def set_control(self, u_path, t_path=None):
        """Store control signal and time grid for interpolation."""
        import torch

        if u_path.ndim == 1:
            u_path = u_path.reshape(-1, 1)
        self._u_series = u_path
        if t_path is not None:
            self._t_series = t_path
        else:
            self._t_series = torch.arange(
                u_path.shape[0], dtype=u_path.dtype, device=u_path.device
            ).float()

    def _u_at(self, t):
        """Linearly interpolate control signal at time t."""
        import torch

        ts = self._t_series
        us = self._u_series
        t_clamped = torch.clamp(t, ts[0], ts[-1])
        k = torch.searchsorted(ts, t_clamped, right=True)
        k_upper = torch.clamp(k, 1, len(ts) - 1)
        k_lower = k_upper - 1
        t_low, t_high = ts[k_lower], ts[k_upper]
        u_low, u_high = us[k_lower], us[k_upper]
        denom = t_high - t_low
        if denom.abs() < 1e-9:
            return u_low
        alpha = (t_clamped - t_low) / denom
        return u_low + alpha * (u_high - u_low)

    def __call__(self, t, y):
        """ODE right-hand side: dy/dt = f(t, y, u(t))."""
        import torch

        u_t = self._u_at(t)
        if u_t.ndim == 1:
            u_t = u_t.unsqueeze(0).expand(y.shape[0], -1)
        xu = torch.cat([y, u_t], dim=-1)
        return self.net(xu)


class NeuralODE(BaseModel):
    """
    Neural ODE for continuous-time system identification using torchdiffeq.

    Dynamics:
        dx/dt = f_theta(x, u(t))

    Uses proper ODE solvers (RK4, dopri5, euler) via torchdiffeq.
    Supports both subsequence-batched training and full-trajectory training.
    """

    _VALID_SOLVERS = {"euler", "rk4", "dopri5"}

    def __init__(
        self,
        state_dim: int = 1,
        input_dim: int = 1,
        hidden_layers: List[int] = [64, 64],
        solver: str = "rk4",
        dt: float = 0.05,
        learning_rate: float = 1e-3,
        lr: float | None = None,
        epochs: int = 100,
        sequence_length: int = 50,
        sequences_per_epoch: int = 24,
        activation: str = "selu",
        training_mode: str = "subsequence",
    ):
        super().__init__(nu=input_dim, ny=state_dim)
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.hidden_layers = list(hidden_layers)
        self.solver = solver
        self.dt = float(dt)
        if lr is not None:
            learning_rate = float(lr)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.sequence_length = int(sequence_length)
        self.sequences_per_epoch = int(sequences_per_epoch)
        self.activation = activation
        self.training_mode = training_mode  # "subsequence" or "full"

        self.ode_func_ = None
        self._device = None
        self._dtype = None
        self.training_loss_: list[float] = []

        if self.solver not in self._VALID_SOLVERS:
            supported = ", ".join(sorted(self._VALID_SOLVERS))
            raise ValueError(f"Unknown solver: {self.solver}. Supported: {supported}")

    def _simulate_trajectory(self, u_path, x0, t_path=None):
        """Integrate ODE from x0 given the control path u_path."""
        from torchdiffeq import odeint
        import torch

        n_steps = u_path.shape[0]
        # Build the integration time grid FIRST
        if t_path is not None:
            ts = t_path
        else:
            ts = torch.arange(n_steps, dtype=u_path.dtype, device=u_path.device) * self.dt

        # Control signal MUST use the same time grid as the ODE solver
        self.ode_func_.set_control(u_path, ts)

        x0_batch = x0 if x0.ndim == 2 else x0.reshape(1, -1)

        odeint_kwargs = {"method": self.solver}
        if self.solver in ("euler", "rk4"):
            odeint_kwargs["options"] = {"step_size": self.dt}
        else:
            odeint_kwargs["rtol"] = 1e-3
            odeint_kwargs["atol"] = 1e-3

        # odeint returns (T, batch, state_dim)
        pred = odeint(self.ode_func_, x0_batch, ts, **odeint_kwargs)
        return pred[:, 0, :]  # (T, state_dim)

    def _integrate_one_step(self, x_t, u_t):
        """Single OSA step."""
        import torch

        u_path = torch.cat([u_t, u_t], dim=0)
        x_path = self._simulate_trajectory(u_path=u_path, x0=x_t)
        return x_path[-1]

    def _train_full_trajectory(
        self, u, y, verbose, wandb_run=None, wandb_log_every=1,
    ) -> list:
        """Train by integrating the full trajectory each epoch."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from tqdm.auto import tqdm

        y_t = torch.tensor(y, dtype=self._dtype, device=self._device).reshape(-1, self.state_dim)
        u_t = torch.tensor(u, dtype=self._dtype, device=self._device).reshape(-1, self.input_dim)
        x0 = y_t[0:1]
        t_grid = torch.arange(len(y_t), dtype=self._dtype, device=self._device) * self.dt

        params = list(self.ode_func_.parameters())
        optimizer = optim.Adam(params, lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=200, min_lr=1e-6,
        )
        criterion = nn.MSELoss()
        loss_history = []
        best_loss = float("inf")
        best_state = None

        epoch_iter = range(self.epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training NeuralODE (full)", unit="epoch")

        for epoch in epoch_iter:
            self.ode_func_.train()
            optimizer.zero_grad()

            pred = self._simulate_trajectory(u_t, x0, t_grid)
            loss = criterion(pred, y_t)

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

            # Keep best model weights
            if lv < best_loss:
                best_loss = lv
                best_state = {k: v.clone() for k, v in self.ode_func_.net.state_dict().items()}

            if verbose and hasattr(epoch_iter, "set_postfix"):
                lr_now = optimizer.param_groups[0]["lr"]
                epoch_iter.set_postfix(loss=lv, lr=f"{lr_now:.1e}")
            if wandb_run and (epoch + 1) % wandb_log_every == 0:
                wandb_run.log({"train/loss": lv, "train/epoch": epoch + 1})

        # Restore best weights
        if best_state is not None:
            self.ode_func_.net.load_state_dict(best_state)

        return loss_history

    def fit(
        self,
        u: np.ndarray | Sequence[np.ndarray],
        y: np.ndarray | Sequence[np.ndarray],
        verbose: bool = True,
        wandb_run=None,
        wandb_log_every: int = 1,
    ) -> "NeuralODE":
        """Train model parameters by backpropagating through torchdiffeq."""
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
            u_data = [
                np.asarray(u_ds, dtype=float).reshape(-1, self.input_dim) for u_ds in u
            ]
            y_data = [
                np.asarray(y_ds, dtype=float).reshape(-1, self.state_dim) for y_ds in y
            ]
            for u_ds, y_ds in zip(u_data, y_data):
                if len(u_ds) != len(y_ds):
                    raise ValueError("Each dataset pair must have u and y with equal length")
        else:
            u_data = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
            y_data = np.asarray(y, dtype=float).reshape(-1, self.state_dim)
            if len(u_data) != len(y_data):
                raise ValueError("u and y must have the same length")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float32
        self.ode_func_ = _ODEFunc(
            state_dim=self.state_dim,
            input_dim=self.input_dim,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
        ).to(self._device)

        if self.training_mode == "full":
            if is_multi:
                raise ValueError(
                    "training_mode='full' does not support multi-dataset input. "
                    "Use training_mode='subsequence' for random-batch multi-dataset training."
                )
            self.training_loss_ = self._train_full_trajectory(
                u_data, y_data, verbose, wandb_run, wandb_log_every,
            )
        else:
            # Subsequence batching (original strategy)
            self.training_loss_ = train_sequence_batches(
                sde_func=self.ode_func_,
                simulate_fn=lambda u_seq, x0: self._simulate_trajectory(u_seq, x0),
                u=u_data,
                y=y_data,
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
                sequences_per_epoch=self.sequences_per_epoch,
            )

        self._is_fitted = True
        return self

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction using measured states."""
        import torch

        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y = np.asarray(y, dtype=float).reshape(-1, self.state_dim)

        predictions = []
        self.ode_func_.eval()
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
        del show_progress
        import torch

        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y_init = np.asarray(y_initial, dtype=float).reshape(-1, self.state_dim)

        self.ode_func_.eval()
        with torch.no_grad():
            u_t = torch.tensor(u, dtype=self._dtype, device=self._device)
            x0 = torch.tensor(y_init[0], dtype=self._dtype, device=self._device)
            pred = self._simulate_trajectory(u_path=u_t, x0=x0)
        return pred.cpu().numpy().reshape(-1)

    def __repr__(self) -> str:
        return (
            f"NeuralODE(state_dim={self.state_dim}, input_dim={self.input_dim}, "
            f"hidden_layers={self.hidden_layers}, solver='{self.solver}', "
            f"activation='{self.activation}', training_mode='{self.training_mode}')"
        )
