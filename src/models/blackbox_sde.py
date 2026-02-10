"""Black-box Neural SDE models with 2-D state [theta, omega].

This module mirrors the NODE family in ``blackbox_ode.py`` but adds a
learned diffusion term and trains with ``torchsde``.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseModel
from .blackbox_ode import _interp_u


def _build_vanilla_nsde(hidden_dim: int = 128):
    """Vanilla NSDE: drift NN -> [dtheta, domega], learned diagonal diffusion."""
    import torch
    import torch.nn as nn

    class _VanillaNSDEFunc(nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def __init__(self):
            super().__init__()
            self.drift_net = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.SELU(),
                nn.AlphaDropout(0.05),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SELU(),
                nn.AlphaDropout(0.05),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SELU(),
                nn.Linear(hidden_dim // 2, 2),
            )
            self.diff_net = nn.Sequential(
                nn.Linear(3, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 2),
            )
            self.u_series = None
            self.t_series = None
            self.batch_start_times = None

        def f(self, t, x):
            u_t = _interp_u(self, t, x)
            return self.drift_net(torch.cat([x, u_t], dim=1))

        def g(self, t, x):
            import torch.nn.functional as F

            u_t = _interp_u(self, t, x)
            raw = self.diff_net(torch.cat([x, u_t], dim=1))
            return F.softplus(raw) + 1e-6

    return _VanillaNSDEFunc()


def _build_structured_nsde(hidden_dim: int = 128):
    """Structured NSDE: dtheta=omega hardcoded, NN learns domega + diffusion."""
    import torch.nn as nn

    class _StructuredNSDEFunc(nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def __init__(self):
            super().__init__()
            self.acc_net = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.SELU(),
                nn.AlphaDropout(0.05),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SELU(),
                nn.AlphaDropout(0.05),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.diff_net = nn.Sequential(
                nn.Linear(3, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.u_series = None
            self.t_series = None
            self.batch_start_times = None

        def f(self, t, x):
            import torch

            u_t = _interp_u(self, t, x)
            theta, omega = x[:, 0:1], x[:, 1:2]
            acc = self.acc_net(torch.cat([theta, omega, u_t], dim=1))
            return torch.cat([omega, acc], dim=1)

        def g(self, t, x):
            import torch
            import torch.nn.functional as F

            u_t = _interp_u(self, t, x)
            theta, omega = x[:, 0:1], x[:, 1:2]
            sigma_acc = F.softplus(self.diff_net(torch.cat([theta, omega, u_t], dim=1))) + 1e-6
            sigma_theta = torch.zeros_like(sigma_acc)
            return torch.cat([sigma_theta, sigma_acc], dim=1)

    return _StructuredNSDEFunc()


def _build_adaptive_nsde(hidden_dim: int = 128):
    """Adaptive NSDE: structured drift with residual + learned diffusion."""
    import torch
    import torch.nn as nn

    class _AdaptiveNSDEFunc(nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def __init__(self):
            super().__init__()
            self.base_net = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.SELU(),
                nn.AlphaDropout(0.05),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SELU(),
                nn.AlphaDropout(0.05),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.residual_net = nn.Sequential(
                nn.Linear(3, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.diff_net = nn.Sequential(
                nn.Linear(3, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
            with torch.no_grad():
                self.residual_net[-1].weight.mul_(0.01)
                self.residual_net[-1].bias.zero_()

            self.u_series = None
            self.t_series = None
            self.batch_start_times = None

        def f(self, t, x):
            import torch

            u_t = _interp_u(self, t, x)
            theta, omega = x[:, 0:1], x[:, 1:2]
            inp = torch.cat([theta, omega, u_t], dim=1)
            acc = self.base_net(inp) + self.residual_net(inp)
            return torch.cat([omega, acc], dim=1)

        def g(self, t, x):
            import torch
            import torch.nn.functional as F

            u_t = _interp_u(self, t, x)
            theta, omega = x[:, 0:1], x[:, 1:2]
            sigma_acc = F.softplus(self.diff_net(torch.cat([theta, omega, u_t], dim=1))) + 1e-6
            sigma_theta = torch.zeros_like(sigma_acc)
            return torch.cat([sigma_theta, sigma_acc], dim=1)

    return _AdaptiveNSDEFunc()


def _odeint_call(func, x0, t_eval, solver: str, dt: float):
    from torchdiffeq import odeint

    kwargs = {"method": solver}
    if solver in ("euler", "rk4"):
        kwargs["options"] = {"step_size": dt}
    else:
        kwargs["rtol"] = 1e-3
        kwargs["atol"] = 1e-3
    return odeint(func, x0, t_eval, **kwargs)


def _sdeint_call(func, x0, t_eval, method: str, dt: float):
    import torchsde

    return torchsde.sdeint(func, x0, t_eval, method=method, dt=dt)


class _BlackboxSDE2D(BaseModel):
    """Shared wrapper for 2-D black-box Neural SDE models."""

    _VALID_SOLVERS = {"euler", "milstein", "srk"}
    _VALID_ODE_SOLVERS = {"euler", "rk4", "dopri5"}

    def __init__(
        self,
        sde_factory,
        hidden_dim: int = 128,
        dt: float = 0.05,
        solver: str = "euler",
        ode_solver: str = "rk4",
        learning_rate: float = 1e-2,
        epochs: int = 5000,
        k_steps: int = 20,
        batch_size: int = 128,
        training_mode: str = "shooting",
    ):
        super().__init__(nu=1, ny=2)
        self.sde_factory = sde_factory
        self.hidden_dim = int(hidden_dim)
        self.dt = float(dt)
        self.solver = solver
        self.ode_solver = ode_solver
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.k_steps = int(k_steps)
        self.batch_size = int(batch_size)
        self.training_mode = training_mode

        self.sde_func_ = None
        self._device = None
        self._dtype = None
        self.training_loss_: list[float] = []

        if self.solver not in self._VALID_SOLVERS:
            raise ValueError(f"Unknown SDE solver: {self.solver}. Use {sorted(self._VALID_SOLVERS)}")
        if self.ode_solver not in self._VALID_ODE_SOLVERS:
            raise ValueError(f"Unknown ODE solver: {self.ode_solver}. Use {sorted(self._VALID_ODE_SOLVERS)}")

    @staticmethod
    def _compute_velocity(y: np.ndarray, dt: float) -> np.ndarray:
        return np.gradient(y, dt)

    def _simulate(
        self,
        u_t,
        x0,
        t_grid,
        batch_start_times=None,
        deterministic: bool = False,
    ):
        self.sde_func_.u_series = u_t
        self.sde_func_.t_series = t_grid
        self.sde_func_.batch_start_times = batch_start_times

        x0_batch = x0 if x0.ndim == 2 else x0.unsqueeze(0)
        if deterministic:
            return _odeint_call(lambda t, y: self.sde_func_.f(t, y), x0_batch, t_grid, self.ode_solver, self.dt)
        return _sdeint_call(self.sde_func_, x0_batch, t_grid, self.solver, self.dt)

    def _train_shooting(self, u, y_sim, verbose):
        import torch
        import torch.optim as optim
        from tqdm.auto import tqdm

        u_t = torch.tensor(u.reshape(-1, 1), dtype=self._dtype, device=self._device)
        y_t = torch.tensor(y_sim, dtype=self._dtype, device=self._device)
        t_full = torch.arange(len(y_t), dtype=self._dtype, device=self._device) * self.dt

        K = min(self.k_steps, len(y_t) - 1)
        B = self.batch_size
        t_eval = torch.arange(K, dtype=self._dtype, device=self._device) * self.dt

        params = list(self.sde_func_.parameters())
        optimizer = optim.Adam(params, lr=self.learning_rate)

        history: list[float] = []
        epoch_iter = range(self.epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc=f"Training {self.__class__.__name__}", unit="epoch")

        for _ in epoch_iter:
            self.sde_func_.train()
            optimizer.zero_grad()

            max_start = len(y_t) - K
            start_idx_np = np.random.randint(0, max(1, max_start), size=B)
            start_idx = torch.as_tensor(start_idx_np, dtype=torch.long, device=self._device)
            x0 = y_t[start_idx]
            batch_start_times = t_full[start_idx].reshape(-1, 1)

            pred = self._simulate(
                u_t=u_t,
                x0=x0,
                t_grid=t_eval,
                batch_start_times=batch_start_times,
                deterministic=False,
            )  # (K, B, 2)
            targets = torch.stack([y_t[int(i) : int(i) + K] for i in start_idx.tolist()], dim=1)
            loss = torch.mean((pred - targets) ** 2)

            if torch.isnan(loss) or torch.isinf(loss):
                history.append(float("nan"))
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()
            lv = float(loss.detach().cpu().item())
            history.append(lv)

            if verbose and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=f"{lv:.6f}")

        return history

    def fit(self, u: np.ndarray, y: np.ndarray, verbose: bool = True) -> "_BlackboxSDE2D":
        import torch

        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float)

        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            y_pos = y.flatten()
            y_vel = self._compute_velocity(y_pos, self.dt)
            y_sim = np.column_stack([y_pos, y_vel])
        else:
            y_sim = y

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float32
        self.sde_func_ = self.sde_factory(self.hidden_dim).to(self._device)
        self.training_loss_ = self._train_shooting(u, y_sim, verbose)
        self._is_fitted = True
        return self

    def predict_osa(self, u: np.ndarray, y: np.ndarray, deterministic: bool = True) -> np.ndarray:
        import torch

        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float)
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            y_pos = y.flatten()
            y_vel = self._compute_velocity(y_pos, self.dt)
            y_sim = np.column_stack([y_pos, y_vel])
        else:
            y_sim = y

        preds: List[float] = []
        self.sde_func_.eval()
        with torch.no_grad():
            for k in range(len(y_sim) - 1):
                x0 = torch.tensor(y_sim[k : k + 1], dtype=self._dtype, device=self._device)
                u_seg = torch.tensor([[u[k]], [u[min(k + 1, len(u) - 1)]]], dtype=self._dtype, device=self._device)
                t_seg = torch.tensor([k * self.dt, (k + 1) * self.dt], dtype=self._dtype, device=self._device)
                out = self._simulate(u_seg, x0, t_seg, deterministic=deterministic)
                preds.append(float(out[-1, 0, 0].detach().cpu().item()))
        return np.asarray(preds)

    def predict_free_run(
        self,
        u: np.ndarray,
        y_initial: np.ndarray,
        return_2d: bool = False,
        deterministic: bool = True,
    ) -> np.ndarray:
        import torch

        u = np.asarray(u, dtype=float).flatten()
        y_init = np.asarray(y_initial, dtype=float)
        if y_init.ndim == 1 or (y_init.ndim == 2 and y_init.shape[1] == 1):
            y_pos = y_init.flatten()
            y_vel = self._compute_velocity(y_pos, self.dt)
            x0_np = np.array([[y_pos[0], y_vel[0]]])
        else:
            x0_np = y_init[0:1]

        self.sde_func_.eval()
        with torch.no_grad():
            u_t = torch.tensor(u.reshape(-1, 1), dtype=self._dtype, device=self._device)
            x0 = torch.tensor(x0_np, dtype=self._dtype, device=self._device)
            t_grid = torch.arange(len(u), dtype=self._dtype, device=self._device) * self.dt
            pred = self._simulate(u_t, x0, t_grid, deterministic=deterministic)

        out = pred[:, 0, :].detach().cpu().numpy()
        if return_2d:
            return out
        return out[:, 0]

    def simulate_full_2d(self, u: np.ndarray, y_sim_init: np.ndarray, deterministic: bool = True):
        return self.predict_free_run(u, y_sim_init, return_2d=True, deterministic=deterministic)


class VanillaNSDE2D(_BlackboxSDE2D):
    """Vanilla 2-D Neural SDE."""

    def __init__(
        self,
        hidden_dim: int = 128,
        dt: float = 0.05,
        solver: str = "euler",
        ode_solver: str = "rk4",
        learning_rate: float = 1e-2,
        epochs: int = 5000,
        k_steps: int = 20,
        batch_size: int = 128,
        training_mode: str = "shooting",
    ):
        super().__init__(
            sde_factory=_build_vanilla_nsde,
            hidden_dim=hidden_dim,
            dt=dt,
            solver=solver,
            ode_solver=ode_solver,
            learning_rate=learning_rate,
            epochs=epochs,
            k_steps=k_steps,
            batch_size=batch_size,
            training_mode=training_mode,
        )


class StructuredNSDE(_BlackboxSDE2D):
    """Structured 2-D Neural SDE with hardcoded dtheta=omega."""

    def __init__(
        self,
        hidden_dim: int = 128,
        dt: float = 0.05,
        solver: str = "euler",
        ode_solver: str = "rk4",
        learning_rate: float = 1e-2,
        epochs: int = 5000,
        k_steps: int = 20,
        batch_size: int = 128,
        training_mode: str = "shooting",
    ):
        super().__init__(
            sde_factory=_build_structured_nsde,
            hidden_dim=hidden_dim,
            dt=dt,
            solver=solver,
            ode_solver=ode_solver,
            learning_rate=learning_rate,
            epochs=epochs,
            k_steps=k_steps,
            batch_size=batch_size,
            training_mode=training_mode,
        )


class AdaptiveNSDE(_BlackboxSDE2D):
    """Adaptive 2-D Neural SDE with residual acceleration correction."""

    def __init__(
        self,
        hidden_dim: int = 128,
        dt: float = 0.05,
        solver: str = "euler",
        ode_solver: str = "rk4",
        learning_rate: float = 5e-3,
        epochs: int = 5000,
        k_steps: int = 20,
        batch_size: int = 128,
        training_mode: str = "shooting",
    ):
        super().__init__(
            sde_factory=_build_adaptive_nsde,
            hidden_dim=hidden_dim,
            dt=dt,
            solver=solver,
            ode_solver=ode_solver,
            learning_rate=learning_rate,
            epochs=epochs,
            k_steps=k_steps,
            batch_size=batch_size,
            training_mode=training_mode,
        )
