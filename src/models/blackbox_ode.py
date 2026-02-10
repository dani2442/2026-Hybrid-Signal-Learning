"""Black-box Neural ODE models with 2-D state [θ, θ̇].

Three architectures sharing the same training / prediction wrapper:

VanillaNODE2D
    Fully black-box: NN learns both dθ/dt and dθ̇/dt simultaneously.

StructuredNODE
    Kinematic constraint: dθ/dt = θ̇ is hardcoded.
    NN only learns the acceleration dθ̇/dt = f_NN(θ, θ̇, u).

AdaptiveNODE
    Like StructuredNODE but with an additional near-zero-initialized
    residual NN that can correct the base acceleration.

All use SELU / AlphaDropout, multiple-shooting training (random
K-step windows, batch of initial conditions), and torchdiffeq RK4.
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel


# ═══════════════════════════════════════════════════════════════════════
#  u-interpolation mixin  (avoids copy-paste across ODE funcs)
# ═══════════════════════════════════════════════════════════════════════

def _interp_u(model, t, x):
    """Linear-interpolate u at time *t*, supporting batch_start_times."""
    import torch

    if model.batch_start_times is not None:
        t_abs = model.batch_start_times + t
    else:
        t_abs = t * torch.ones_like(x[:, 0:1])

    k_idx = torch.searchsorted(model.t_series, t_abs.reshape(-1), right=True)
    k_idx = torch.clamp(k_idx, 1, len(model.t_series) - 1)
    t1 = model.t_series[k_idx - 1].unsqueeze(1)
    t2 = model.t_series[k_idx].unsqueeze(1)
    u1, u2 = model.u_series[k_idx - 1], model.u_series[k_idx]
    denom = (t2 - t1).clone()
    denom[denom < 1e-6] = 1.0
    alpha = (t_abs - t1) / denom
    return u1 + alpha * (u2 - u1)


# ═══════════════════════════════════════════════════════════════════════
#  ODE right-hand sides  (nn.Module for torchdiffeq)
# ═══════════════════════════════════════════════════════════════════════

def _build_vanilla(hidden_dim: int = 128):
    """Vanilla NODE: NN → [dθ, dω].  No kinematic prior."""
    import torch
    import torch.nn as nn

    class _VanillaFunc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
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
            self.u_series = None
            self.t_series = None
            self.batch_start_times = None

        def forward(self, t, x):
            u_t = _interp_u(self, t, x)
            return self.net(torch.cat([x, u_t], dim=1))

    return _VanillaFunc()


def _build_structured(hidden_dim: int = 128):
    """Structured NODE: dθ/dt = θ̇ (hardcoded), NN → dθ̇/dt."""
    import torch
    import torch.nn as nn

    class _StructuredFunc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
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
            self.u_series = None
            self.t_series = None
            self.batch_start_times = None

        def forward(self, t, x):
            u_t = _interp_u(self, t, x)
            th, thd = x[:, 0:1], x[:, 1:2]
            thdd = self.net(torch.cat([th, thd, u_t], dim=1))
            return torch.cat([thd, thdd], dim=1)

    return _StructuredFunc()


def _build_adaptive(hidden_dim: int = 128):
    """Adaptive NODE: structured base + near-zero residual correction."""
    import torch
    import torch.nn as nn

    class _AdaptiveFunc(nn.Module):
        def __init__(self):
            super().__init__()
            self.dynamics_net = nn.Sequential(
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
            self.adaptive_residual = nn.Sequential(
                nn.Linear(3, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
            # Initialise residual near zero
            with torch.no_grad():
                self.adaptive_residual[-1].weight.mul_(0.01)
                self.adaptive_residual[-1].bias.zero_()

            self.u_series = None
            self.t_series = None
            self.batch_start_times = None

        def forward(self, t, x):
            u_t = _interp_u(self, t, x)
            th, thd = x[:, 0:1], x[:, 1:2]
            inp = torch.cat([th, thd, u_t], dim=1)
            thdd = self.dynamics_net(inp) + self.adaptive_residual(inp)
            return torch.cat([thd, thdd], dim=1)

    return _AdaptiveFunc()


# ═══════════════════════════════════════════════════════════════════════
#  Shared training / prediction wrapper for 2-D state ODE models
# ═══════════════════════════════════════════════════════════════════════

class _BlackboxODE2D(BaseModel):
    """Shared wrapper for black-box 2-D state ODE models.

    State x = [θ, θ̇].
    ``fit()`` accepts 1-D y (position only, velocity is estimated
    via central differences) **or** 2-D y_sim = [θ, θ̇].
    Training uses multiple-shooting: a batch of B random windows of
    K steps each, sampled uniformly across the signal.
    Loss covers both position and velocity.
    """

    _VALID_SOLVERS = {"euler", "rk4", "dopri5"}

    def __init__(
        self,
        ode_factory,
        hidden_dim: int = 128,
        dt: float = 0.05,
        solver: str = "rk4",
        learning_rate: float = 1e-2,
        epochs: int = 5000,
        k_steps: int = 20,
        batch_size: int = 128,
        training_mode: str = "shooting",
    ):
        super().__init__(nu=1, ny=2)
        self.ode_factory = ode_factory
        self.hidden_dim = int(hidden_dim)
        self.dt = float(dt)
        self.solver = solver
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.k_steps = int(k_steps)
        self.batch_size = int(batch_size)
        self.training_mode = training_mode

        self.ode_func_ = None
        self._device = None
        self._dtype = None
        self.training_loss_: list[float] = []

        if self.solver not in self._VALID_SOLVERS:
            raise ValueError(
                f"Unknown solver: {solver}. Use: {sorted(self._VALID_SOLVERS)}"
            )

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_velocity(y: np.ndarray, dt: float) -> np.ndarray:
        """Central-difference velocity estimate."""
        return np.gradient(y, dt)

    def _simulate(self, u_t, x0, t_grid):
        """Integrate ODE from x0 along t_grid."""
        from torchdiffeq import odeint

        self.ode_func_.t_series = t_grid
        self.ode_func_.u_series = u_t
        self.ode_func_.batch_start_times = None

        x0_batch = x0 if x0.ndim == 2 else x0.unsqueeze(0)
        kw = {"method": self.solver}
        if self.solver in ("euler", "rk4"):
            kw["options"] = {"step_size": self.dt}
        else:
            kw["rtol"] = 1e-3
            kw["atol"] = 1e-3

        return odeint(self.ode_func_, x0_batch, t_grid, **kw)  # (T, B, 2)

    def _simulate_batch(self, u_t, x0_batch, t_grid, batch_start_times):
        """Integrate ODE for a batch of initial conditions (shooting)."""
        from torchdiffeq import odeint

        self.ode_func_.t_series = t_grid
        self.ode_func_.u_series = u_t
        self.ode_func_.batch_start_times = batch_start_times

        kw = {"method": self.solver}
        if self.solver in ("euler", "rk4"):
            kw["options"] = {"step_size": self.dt}
        else:
            kw["rtol"] = 1e-3
            kw["atol"] = 1e-3

        return odeint(self.ode_func_, x0_batch, t_grid, **kw)  # (K, B, 2)

    # ── multiple-shooting training ────────────────────────────────────

    def _train_shooting(self, u, y_sim, verbose, wandb_run=None, wandb_log_every=1):
        """Multiple-shooting training on random K-step windows."""
        import torch
        import torch.optim as optim
        from tqdm.auto import tqdm

        u_t = torch.tensor(u.reshape(-1, 1), dtype=self._dtype, device=self._device)
        y_t = torch.tensor(y_sim, dtype=self._dtype, device=self._device)  # (N, 2)
        t_full = torch.arange(len(y_t), dtype=self._dtype, device=self._device) * self.dt

        K = min(self.k_steps, len(y_t) - 1)
        B = self.batch_size
        dt_local = self.dt
        t_eval = torch.arange(K, dtype=self._dtype, device=self._device) * dt_local

        params = list(self.ode_func_.parameters())
        optimizer = optim.Adam(params, lr=self.learning_rate)

        loss_history: list[float] = []
        epoch_iter = range(self.epochs)
        if verbose:
            epoch_iter = tqdm(
                epoch_iter,
                desc=f"Training {self.__class__.__name__}",
                unit="epoch",
            )

        for epoch in epoch_iter:
            self.ode_func_.train()
            optimizer.zero_grad()

            # Random batch of starting indices
            max_start = len(y_t) - K
            start_idx = np.random.randint(0, max(1, max_start), size=B)
            x0 = y_t[start_idx]  # (B, 2)
            batch_start_times = t_full[start_idx].reshape(-1, 1)  # (B, 1)

            pred = self._simulate_batch(u_t, x0, t_eval, batch_start_times)
            # pred: (K, B, 2)

            # Targets: (K, B, 2)
            targets = torch.stack(
                [y_t[i : i + K] for i in start_idx], dim=1,
            )
            loss = torch.mean((pred - targets) ** 2)

            if torch.isnan(loss) or torch.isinf(loss):
                loss_history.append(float("nan"))
                if verbose and hasattr(epoch_iter, "set_postfix"):
                    epoch_iter.set_postfix(loss="nan")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()

            lv = loss.item()
            loss_history.append(lv)

            if verbose and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=f"{lv:.6f}")
            if wandb_run and (epoch + 1) % wandb_log_every == 0:
                wandb_run.log({"train/loss": lv, "train/epoch": epoch + 1})

        return loss_history

    # ── multi-dataset training ────────────────────────────────────────

    def fit_multi(
        self,
        datasets: list[dict],
        verbose: bool = True,
        wandb_run=None,
        wandb_log_every: int = 1,
    ) -> "_BlackboxODE2D":
        """Train on multiple datasets simultaneously.

        Parameters
        ----------
        datasets : list of dict
            Each dict has keys ``"t"``, ``"u"``, ``"y"`` as torch tensors
            already on the correct device, and ``"name"`` (str).
        """
        import torch
        import torch.optim as optim
        from tqdm.auto import tqdm

        # Auto-initialise if not already done (e.g. called without fit())
        if self._device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._dtype = torch.float32
        if self.ode_func_ is None:
            self.ode_func_ = self.ode_factory(self.hidden_dim).to(self._device)

        K = self.k_steps
        B = self.batch_size

        params = list(self.ode_func_.parameters())
        optimizer = optim.Adam(params, lr=self.learning_rate)

        loss_history: list[float] = []
        epoch_iter = range(self.epochs)
        if verbose:
            epoch_iter = tqdm(
                epoch_iter,
                desc=f"Training {self.__class__.__name__}",
                unit="epoch",
            )

        for epoch in epoch_iter:
            self.ode_func_.train()
            optimizer.zero_grad()

            # Pick a random dataset
            ds = datasets[np.random.randint(len(datasets))]
            t_ds, u_ds, y_ds = ds["t"], ds["u"], ds["y"]
            dt_local = (t_ds[1] - t_ds[0]).item()
            t_eval = torch.arange(K, dtype=self._dtype, device=self._device) * dt_local

            self.ode_func_.u_series = u_ds
            self.ode_func_.t_series = t_ds

            max_start = len(t_ds) - K
            start_idx = np.random.randint(0, max(1, max_start), size=B)
            x0 = y_ds[start_idx]
            self.ode_func_.batch_start_times = t_ds[start_idx].reshape(-1, 1)

            pred = odeint_call(self.ode_func_, x0, t_eval, self.solver, self.dt)
            targets = torch.stack([y_ds[i : i + K] for i in start_idx], dim=1)
            loss = torch.mean((pred - targets) ** 2)

            if torch.isnan(loss) or torch.isinf(loss):
                loss_history.append(float("nan"))
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()

            lv = loss.item()
            loss_history.append(lv)
            if verbose and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=f"{lv:.6f}", ds=ds.get("name", ""))

        self.training_loss_ = loss_history
        self._is_fitted = True
        return self

    # ── fit / predict ─────────────────────────────────────────────────

    def fit(
        self,
        u: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
        wandb_run=None,
        wandb_log_every: int = 1,
    ) -> "_BlackboxODE2D":
        import torch

        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float)

        # Accept 1D (position only) or 2D (position + velocity)
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            y_pos = y.flatten()
            y_vel = self._compute_velocity(y_pos, self.dt)
            y_sim = np.column_stack([y_pos, y_vel])
        else:
            y_sim = y  # already 2D

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float32
        self.ode_func_ = self.ode_factory(self.hidden_dim).to(self._device)

        self.training_loss_ = self._train_shooting(
            u, y_sim, verbose, wandb_run, wandb_log_every,
        )
        self._is_fitted = True
        return self

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead (returns position only for compatibility)."""
        import torch

        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float)
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            y_pos = y.flatten()
            y_vel = self._compute_velocity(y_pos, self.dt)
            y_sim = np.column_stack([y_pos, y_vel])
        else:
            y_sim = y

        predictions = []
        self.ode_func_.eval()
        with torch.no_grad():
            for k in range(len(y_sim) - 1):
                x0 = torch.tensor(
                    y_sim[k : k + 1], dtype=self._dtype, device=self._device,
                )
                u_seg = torch.tensor(
                    [[u[k]], [u[min(k + 1, len(u) - 1)]]],
                    dtype=self._dtype, device=self._device,
                )
                t_seg = torch.tensor(
                    [k * self.dt, (k + 1) * self.dt],
                    dtype=self._dtype, device=self._device,
                )
                pred = self._simulate(u_seg, x0, t_seg)  # (2, 1, 2)
                predictions.append(pred[-1, 0, 0].cpu().item())

        return np.asarray(predictions)

    def predict_free_run(
        self,
        u: np.ndarray,
        y_initial: np.ndarray,
        return_2d: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Free-run simulation.  By default returns position only."""
        import torch

        u = np.asarray(u, dtype=float).flatten()
        y_init = np.asarray(y_initial, dtype=float)
        if y_init.ndim == 1 or (y_init.ndim == 2 and y_init.shape[1] == 1):
            y_pos = y_init.flatten()
            y_vel = self._compute_velocity(y_pos, self.dt)
            x0_np = np.array([[y_pos[0], y_vel[0]]])
        else:
            x0_np = y_init[0:1]

        self.ode_func_.eval()
        with torch.no_grad():
            u_t = torch.tensor(
                u.reshape(-1, 1), dtype=self._dtype, device=self._device,
            )
            x0 = torch.tensor(x0_np, dtype=self._dtype, device=self._device)
            t_grid = torch.arange(
                len(u), dtype=self._dtype, device=self._device,
            ) * self.dt
            pred = self._simulate(u_t, x0, t_grid)  # (T, 1, 2)

        result = pred[:, 0, :].cpu().numpy()  # (T, 2)
        if return_2d:
            return result
        return result[:, 0]  # position only

    def simulate_full_2d(
        self, u: np.ndarray, y_sim_init: np.ndarray,
    ) -> np.ndarray:
        """Full-trajectory simulation returning [θ, θ̇] (T×2)."""
        return self.predict_free_run(u, y_sim_init, return_2d=True)


# ═══════════════════════════════════════════════════════════════════════
#  Helper for fit_multi (torchdiffeq call)
# ═══════════════════════════════════════════════════════════════════════

def odeint_call(func, x0, t_eval, solver, dt):
    from torchdiffeq import odeint
    kw = {"method": solver}
    if solver in ("euler", "rk4"):
        kw["options"] = {"step_size": dt}
    else:
        kw["rtol"] = 1e-3
        kw["atol"] = 1e-3
    return odeint(func, x0, t_eval, **kw)


# ═══════════════════════════════════════════════════════════════════════
#  Public model classes
# ═══════════════════════════════════════════════════════════════════════

class VanillaNODE2D(_BlackboxODE2D):
    """Vanilla Neural ODE with 2-D state.

    Both derivatives are learned by the NN — no kinematic prior.
    Input: [θ, θ̇, u]  →  Output: [dθ/dt, dθ̇/dt]
    """
    def __init__(self, hidden_dim=128, dt=0.05, solver="rk4",
                 learning_rate=0.01, epochs=5000,
                 k_steps=20, batch_size=128, training_mode="shooting"):
        super().__init__(
            ode_factory=_build_vanilla, hidden_dim=hidden_dim, dt=dt,
            solver=solver, learning_rate=learning_rate, epochs=epochs,
            k_steps=k_steps, batch_size=batch_size, training_mode=training_mode,
        )

    def __repr__(self):
        return (f"VanillaNODE2D(hidden={self.hidden_dim}, K={self.k_steps}, "
                f"B={self.batch_size}, epochs={self.epochs})")


class StructuredNODE(_BlackboxODE2D):
    """Structured Neural ODE with kinematic constraint.

    dθ/dt = θ̇   (hardcoded)
    dθ̇/dt = f_NN(θ, θ̇, u)
    """
    def __init__(self, hidden_dim=128, dt=0.05, solver="rk4",
                 learning_rate=0.01, epochs=5000,
                 k_steps=20, batch_size=128, training_mode="shooting"):
        super().__init__(
            ode_factory=_build_structured, hidden_dim=hidden_dim, dt=dt,
            solver=solver, learning_rate=learning_rate, epochs=epochs,
            k_steps=k_steps, batch_size=batch_size, training_mode=training_mode,
        )

    def __repr__(self):
        return (f"StructuredNODE(hidden={self.hidden_dim}, K={self.k_steps}, "
                f"B={self.batch_size}, epochs={self.epochs})")


class AdaptiveNODE(_BlackboxODE2D):
    """Adaptive Neural ODE: structured base + near-zero residual.

    dθ/dt = θ̇
    dθ̇/dt = f_base(θ, θ̇, u) + f_residual(θ, θ̇, u)

    Residual path initialised near zero so early training is stable.
    """
    def __init__(self, hidden_dim=128, dt=0.05, solver="rk4",
                 learning_rate=0.005, epochs=5000,
                 k_steps=20, batch_size=128, training_mode="shooting"):
        super().__init__(
            ode_factory=_build_adaptive, hidden_dim=hidden_dim, dt=dt,
            solver=solver, learning_rate=learning_rate, epochs=epochs,
            k_steps=k_steps, batch_size=batch_size, training_mode=training_mode,
        )

    def __repr__(self):
        return (f"AdaptiveNODE(hidden={self.hidden_dim}, K={self.k_steps}, "
                f"B={self.batch_size}, epochs={self.epochs})")
