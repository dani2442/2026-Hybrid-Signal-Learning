"""Physics-based ODE models for system identification.

Two continuous-time models solved with torchdiffeq:

LinearPhysODE
    Second-order linear beam:
        J·θ̈ + R·θ̇ + K·(θ + δ) = τ·V

StribeckPhysODE
    Same linear beam + Stribeck friction:
        J·θ̈ + R·θ̇ + K·(θ + δ) + F_stribeck(θ̇) = τ·V
    where  F_stribeck = (Fc + (Fs−Fc)·exp(−(θ̇/vs)²))·sign(θ̇) + b·θ̇

Parameters are log-parameterised so they remain strictly positive
during optimisation.  Training integrates the full trajectory with
torchdiffeq RK4 and back-propagates through the solver.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import BaseModel


# ═══════════════════════════════════════════════════════════════════════
#  ODE right-hand sides  (nn.Module, works with torchdiffeq.odeint)
# ═══════════════════════════════════════════════════════════════════════

def _build_linear_ode():
    """Return a fresh LinearPhysODE nn.Module."""
    import torch
    import torch.nn as nn

    class LinearPhysODE(nn.Module):
        """J·θ̈ + R·θ̇ + K·(θ+δ) = τ·V"""

        def __init__(self):
            super().__init__()
            self.log_J = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
            self.log_R = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
            self.log_K = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))
            self.delta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.log_Tau = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))
            self.u_series = None
            self.t_series = None
            self.batch_start_times = None

        def get_params(self):
            J = torch.exp(self.log_J)
            R = torch.exp(self.log_R)
            K = torch.exp(self.log_K)
            Tau = torch.exp(self.log_Tau)
            return J, R, K, self.delta, Tau

        def forward(self, t, x):
            J, R, K, delta, Tau = self.get_params()
            if self.batch_start_times is not None:
                t_abs = self.batch_start_times + t
            else:
                t_abs = t * torch.ones_like(x[:, 0:1])

            k_idx = torch.searchsorted(self.t_series, t_abs.reshape(-1), right=True)
            k_idx = torch.clamp(k_idx, 1, len(self.t_series) - 1)
            t1 = self.t_series[k_idx - 1].unsqueeze(1)
            t2 = self.t_series[k_idx].unsqueeze(1)
            u1, u2 = self.u_series[k_idx - 1], self.u_series[k_idx]
            denom = t2 - t1
            denom = denom.clone()
            denom[denom < 1e-6] = 1.0
            alpha = (t_abs - t1) / denom
            u_t = u1 + alpha * (u2 - u1)

            th, thd = x[:, 0:1], x[:, 1:2]
            thdd = (Tau * u_t - R * thd - K * (th + delta)) / J
            return torch.cat([thd, thdd], dim=1)

    return LinearPhysODE()


def _build_stribeck_ode():
    """Return a fresh StribeckPhysODE nn.Module."""
    import torch
    import torch.nn as nn

    class StribeckPhysODE(nn.Module):
        """J·θ̈ + R·θ̇ + K·(θ+δ) + F_stribeck = τ·V"""

        def __init__(self):
            super().__init__()
            self.log_J = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
            self.log_R = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
            self.log_K = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))
            self.delta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.log_Tau = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))
            # Stribeck friction parameters
            self.log_Fc = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
            self.log_Fs = nn.Parameter(torch.tensor(np.log(0.2), dtype=torch.float32))
            self.log_vs = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
            self.log_b = nn.Parameter(torch.tensor(np.log(0.01), dtype=torch.float32))
            self.u_series = None
            self.t_series = None
            self.batch_start_times = None

        def get_params(self):
            J = torch.exp(self.log_J)
            R = torch.exp(self.log_R)
            K = torch.exp(self.log_K)
            Tau = torch.exp(self.log_Tau)
            Fc = torch.exp(self.log_Fc)
            Fs = torch.exp(self.log_Fs)
            vs = torch.exp(self.log_vs)
            b = torch.exp(self.log_b)
            return J, R, K, self.delta, Tau, Fc, Fs, vs, b

        def forward(self, t, x):
            J, R, K, delta, Tau, Fc, Fs, vs, b = self.get_params()
            if self.batch_start_times is not None:
                t_abs = self.batch_start_times + t
            else:
                t_abs = t * torch.ones_like(x[:, 0:1])

            k_idx = torch.searchsorted(self.t_series, t_abs.reshape(-1), right=True)
            k_idx = torch.clamp(k_idx, 1, len(self.t_series) - 1)
            t1 = self.t_series[k_idx - 1].unsqueeze(1)
            t2 = self.t_series[k_idx].unsqueeze(1)
            u1, u2 = self.u_series[k_idx - 1], self.u_series[k_idx]
            denom = t2 - t1
            denom = denom.clone()
            denom[denom < 1e-6] = 1.0
            alpha = (t_abs - t1) / denom
            u_t = u1 + alpha * (u2 - u1)

            th, thd = x[:, 0:1], x[:, 1:2]
            sgn = torch.tanh(thd / 1e-3)
            F_str = (Fc + (Fs - Fc) * torch.exp(-(thd / vs) ** 2)) * sgn + b * thd
            thdd = (Tau * u_t - R * thd - K * (th + delta) - F_str) / J
            return torch.cat([thd, thdd], dim=1)

    return StribeckPhysODE()


# ═══════════════════════════════════════════════════════════════════════
#  Shared training / prediction logic
# ═══════════════════════════════════════════════════════════════════════

class _PhysODEModel(BaseModel):
    """Shared wrapper for physics-based ODE models.

    State is 2-D  x = [θ, θ̇].  Only θ is observed so:
      - ω = θ̇  is estimated from the first two y samples at init
      - Training loss is MSE on θ only
      - Predictions return θ
    """

    _VALID_SOLVERS = {"euler", "rk4", "dopri5"}

    def __init__(
        self,
        ode_factory,
        dt: float = 0.05,
        solver: str = "rk4",
        learning_rate: float = 1e-3,
        epochs: int = 1000,
        sequence_length: int = 50,
        training_mode: str = "full",
    ):
        super().__init__(nu=1, ny=1)
        self.ode_factory = ode_factory
        self.dt = float(dt)
        self.solver = solver
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.sequence_length = int(sequence_length)
        self.training_mode = training_mode

        self.ode_func_ = None
        self._device = None
        self._dtype = None
        self.training_loss_: list[float] = []

        if self.solver not in self._VALID_SOLVERS:
            raise ValueError(
                f"Unknown solver: {solver}. Use: {sorted(self._VALID_SOLVERS)}"
            )

    # ── integration ───────────────────────────────────────────────────

    def _simulate(self, u_t, x0, t_grid=None):
        """Integrate ODE over time grid given control signal u_t."""
        from torchdiffeq import odeint
        import torch

        n = u_t.shape[0]
        if t_grid is None:
            t_grid = torch.arange(n, dtype=self._dtype, device=self._device) * self.dt

        # set control signal on the ODE func
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

        pred = odeint(self.ode_func_, x0_batch, t_grid, **kw)  # (T, B, 2)
        return pred[:, 0, :]  # (T, 2)

    # ── full-trajectory training ──────────────────────────────────────

    def _train_full(self, u, y, verbose, wandb_run=None, wandb_log_every=1):
        import torch
        import torch.optim as optim
        from tqdm.auto import tqdm

        u_t = torch.tensor(u.reshape(-1, 1), dtype=self._dtype, device=self._device)
        y_t = torch.tensor(y, dtype=self._dtype, device=self._device)
        t_grid = torch.arange(len(y_t), dtype=self._dtype, device=self._device) * self.dt

        # Initial state: θ₀ = y[0],  ω₀ ≈ (y[1]-y[0]) / dt
        theta0 = y_t[0]
        omega0 = (y_t[1] - y_t[0]) / self.dt if len(y_t) > 1 else torch.tensor(0.0)
        x0 = torch.tensor(
            [theta0.item(), omega0.item()], dtype=self._dtype, device=self._device,
        ).unsqueeze(0)

        params = list(self.ode_func_.parameters())
        optimizer = optim.Adam(params, lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=200, min_lr=1e-6,
        )

        loss_history: list[float] = []
        best_loss = float("inf")
        best_state: dict | None = None

        epoch_iter = range(self.epochs)
        if verbose:
            epoch_iter = tqdm(
                epoch_iter,
                desc=f"Training {self.__class__.__name__} (full)",
                unit="epoch",
            )

        for epoch in epoch_iter:
            self.ode_func_.train()
            optimizer.zero_grad()

            pred = self._simulate(u_t, x0, t_grid)  # (T, 2)
            loss = torch.mean((pred[:, 0] - y_t) ** 2)  # MSE on θ only

            if torch.isnan(loss) or torch.isinf(loss):
                loss_history.append(float("nan"))
                if verbose and hasattr(epoch_iter, "set_postfix"):
                    epoch_iter.set_postfix(loss="nan")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            lv = loss.item()
            loss_history.append(lv)
            scheduler.step(lv)

            if lv < best_loss:
                best_loss = lv
                best_state = {
                    k: v.clone() for k, v in self.ode_func_.state_dict().items()
                }

            if verbose and hasattr(epoch_iter, "set_postfix"):
                lr_now = optimizer.param_groups[0]["lr"]
                epoch_iter.set_postfix(loss=lv, lr=f"{lr_now:.1e}")
            if wandb_run and (epoch + 1) % wandb_log_every == 0:
                wandb_run.log({"train/loss": lv, "train/epoch": epoch + 1})

        if best_state is not None:
            self.ode_func_.load_state_dict(best_state)

        return loss_history

    # ── subsequence training ──────────────────────────────────────────

    def _train_subsequence(self, u, y, verbose, wandb_run=None, wandb_log_every=1):
        """Train on random subsequences from one or multiple datasets."""
        import torch
        import torch.optim as optim
        from tqdm.auto import tqdm

        seq_len = int(self.sequence_length)

        def _prepare_dataset(u_arr, y_arr):
            u_np = np.asarray(u_arr, dtype=float).flatten()
            y_np = np.asarray(y_arr, dtype=float).flatten()
            if len(u_np) != len(y_np):
                raise ValueError("Each dataset pair must have u and y with equal length")
            max_start = len(y_np) - seq_len - 1
            if max_start <= 0:
                return None
            u_t = torch.tensor(u_np.reshape(-1, 1), dtype=self._dtype, device=self._device)
            y_t = torch.tensor(y_np, dtype=self._dtype, device=self._device)
            t_full = torch.arange(len(y_t), dtype=self._dtype, device=self._device) * self.dt
            return (u_t, y_t, t_full, int(max_start))

        datasets = []
        is_multi = isinstance(u, Sequence) and not isinstance(u, np.ndarray)
        if is_multi:
            if not (isinstance(y, Sequence) and not isinstance(y, np.ndarray)):
                raise ValueError("When u is a sequence of datasets, y must also be a sequence")
            if len(u) != len(y):
                raise ValueError("u and y dataset lists must have the same length")
            for u_ds, y_ds in zip(u, y):
                prepared = _prepare_dataset(u_ds, y_ds)
                if prepared is not None:
                    datasets.append(prepared)
        else:
            prepared = _prepare_dataset(u, y)
            if prepared is not None:
                datasets.append(prepared)

        if not datasets:
            raise ValueError("Not enough data for given sequence length")

        ds_weights = np.asarray([max_start for _, _, _, max_start in datasets], dtype=float)
        ds_weights = ds_weights / np.sum(ds_weights)

        params = list(self.ode_func_.parameters())
        optimizer = optim.Adam(params, lr=self.learning_rate)

        loss_history: list[float] = []
        epoch_iter = range(self.epochs)
        if verbose:
            epoch_iter = tqdm(
                epoch_iter,
                desc=f"Training {self.__class__.__name__} (subseq)",
                unit="epoch",
            )

        for epoch in epoch_iter:
            self.ode_func_.train()
            optimizer.zero_grad()

            # Random dataset + random subsequence
            ds_idx = int(np.random.choice(len(datasets), p=ds_weights))
            u_t, y_t, t_full, max_start = datasets[ds_idx]
            start = int(np.random.randint(0, max(1, max_start)))
            end = start + seq_len
            u_seq = u_t[start:end]
            y_seq = y_t[start:end]
            t_seq = t_full[start:end]

            theta0 = y_seq[0]
            omega0 = (y_seq[1] - y_seq[0]) / self.dt if len(y_seq) > 1 else torch.tensor(0.0)
            x0 = torch.tensor(
                [theta0.item(), omega0.item()],
                dtype=self._dtype, device=self._device,
            ).unsqueeze(0)

            pred = self._simulate(u_seq, x0, t_seq)
            loss = torch.mean((pred[:, 0] - y_seq) ** 2)

            if torch.isnan(loss) or torch.isinf(loss):
                loss_history.append(float("nan"))
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            lv = loss.item()
            loss_history.append(lv)
            if verbose and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=lv)

        return loss_history

    # ── fit / predict ─────────────────────────────────────────────────

    def fit(
        self,
        u: np.ndarray | Sequence[np.ndarray],
        y: np.ndarray | Sequence[np.ndarray],
        verbose: bool = True,
        wandb_run=None,
        wandb_log_every: int = 1,
    ) -> "_PhysODEModel":
        import torch

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
        else:
            u_data = np.asarray(u, dtype=float).flatten()
            y_data = np.asarray(y, dtype=float).flatten()
            if len(u_data) != len(y_data):
                raise ValueError("u and y must have same length")

        if is_multi and self.training_mode == "full":
            raise ValueError(
                "training_mode='full' does not support multi-dataset input. "
                "Use training_mode='subsequence' for random-batch multi-dataset training."
            )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float32
        self.ode_func_ = self.ode_factory().to(self._device)

        if self.training_mode == "full":
            self.training_loss_ = self._train_full(
                u_data, y_data, verbose, wandb_run, wandb_log_every,
            )
        else:
            self.training_loss_ = self._train_subsequence(
                u_data, y_data, verbose, wandb_run, wandb_log_every,
            )

        self._is_fitted = True
        return self

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead: integrate one dt from measured state at each step."""
        import torch

        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()

        predictions = []
        self.ode_func_.eval()
        with torch.no_grad():
            for k in range(len(y) - 1):
                theta_k = y[k]
                omega_k = (y[k] - y[k - 1]) / self.dt if k > 0 else (y[1] - y[0]) / self.dt
                x0 = torch.tensor(
                    [[theta_k, omega_k]], dtype=self._dtype, device=self._device,
                )
                u_seg = torch.tensor(
                    [[u[k]], [u[min(k + 1, len(u) - 1)]]],
                    dtype=self._dtype, device=self._device,
                )
                pred = self._simulate(u_seg, x0)  # (2, 2)
                predictions.append(pred[-1, 0].cpu().item())

        return np.asarray(predictions)

    def predict_free_run(
        self, u: np.ndarray, y_initial: np.ndarray, **kwargs,
    ) -> np.ndarray:
        """Free-run simulation from initial condition."""
        import torch

        u = np.asarray(u, dtype=float).flatten()
        y_init = np.asarray(y_initial, dtype=float).flatten()

        theta0 = y_init[0]
        omega0 = (y_init[1] - y_init[0]) / self.dt if len(y_init) > 1 else 0.0

        self.ode_func_.eval()
        with torch.no_grad():
            u_t = torch.tensor(
                u.reshape(-1, 1), dtype=self._dtype, device=self._device,
            )
            x0 = torch.tensor(
                [[theta0, omega0]], dtype=self._dtype, device=self._device,
            )
            pred = self._simulate(u_t, x0)  # (T, 2)

        return pred[:, 0].cpu().numpy()  # return θ only


# ═══════════════════════════════════════════════════════════════════════
#  Public model classes
# ═══════════════════════════════════════════════════════════════════════

class LinearPhysics(_PhysODEModel):
    """Linear 2nd-order beam model.

    Physics:  J·θ̈ + R·θ̇ + K·(θ + δ) = τ·V

    5 learnable parameters: J, R, K, δ, τ  (log-parameterised for J,R,K,τ).
    """

    def __init__(
        self,
        dt: float = 0.05,
        solver: str = "rk4",
        learning_rate: float = 1e-3,
        lr: float | None = None,
        epochs: int = 1000,
        sequence_length: int = 50,
        training_mode: str = "full",
    ):
        if lr is not None:
            learning_rate = float(lr)

        super().__init__(
            ode_factory=_build_linear_ode,
            dt=dt,
            solver=solver,
            learning_rate=learning_rate,
            epochs=epochs,
            sequence_length=sequence_length,
            training_mode=training_mode,
        )

    def __repr__(self):
        return (
            f"LinearPhysics(solver='{self.solver}', epochs={self.epochs}, "
            f"lr={self.learning_rate}, training_mode='{self.training_mode}')"
        )


class StribeckPhysics(_PhysODEModel):
    """Linear beam + Stribeck friction model.

    Physics:
        J·θ̈ + R·θ̇ + K·(θ + δ) + F_stribeck(θ̇) = τ·V
        F_stribeck = (Fc + (Fs−Fc)·exp(−(θ̇/vs)²))·sign(θ̇) + b·θ̇

    9 learnable parameters: J, R, K, δ, τ, Fc, Fs, vs, b.
    """

    def __init__(
        self,
        dt: float = 0.05,
        solver: str = "rk4",
        learning_rate: float = 1e-3,
        lr: float | None = None,
        epochs: int = 1000,
        sequence_length: int = 50,
        training_mode: str = "full",
    ):
        if lr is not None:
            learning_rate = float(lr)

        super().__init__(
            ode_factory=_build_stribeck_ode,
            dt=dt,
            solver=solver,
            learning_rate=learning_rate,
            epochs=epochs,
            sequence_length=sequence_length,
            training_mode=training_mode,
        )

    def __repr__(self):
        return (
            f"StribeckPhysics(solver='{self.solver}', epochs={self.epochs}, "
            f"lr={self.learning_rate}, training_mode='{self.training_mode}')"
        )
