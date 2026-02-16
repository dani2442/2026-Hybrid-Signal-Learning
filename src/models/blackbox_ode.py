"""Black-box Neural ODE / SDE models with 2-D state [θ, θ̇].

Nine architectures sharing one training / prediction base:

ODE (zero diffusion)            SDE (learned diffusion)
─────────────────               ────────────────────────
VanillaNODE2D                   VanillaNSDE2D
StructuredNODE                  StructuredNSDE
AdaptiveNODE                    AdaptiveNSDE

CDE-inspired variants are registered by ``blackbox_cde.py``.

All use SELU / AlphaDropout, multiple-shooting training, torchsde Euler.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..config import BlackboxODE2DConfig, BlackboxSDE2DConfig
from .base import BaseModel, SHOOTING_GRAD_CLIP
from .torchsde_utils import interp_u


# ═══════════════════════════════════════════════════════════════════════
#  Shared network builders
# ═══════════════════════════════════════════════════════════════════════

def _selu_block(in_dim: int, hidden_dim: int, out_dim: int):
    """4-layer SELU + AlphaDropout network used across all blackbox dynamics."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.SELU(), nn.AlphaDropout(0.05),
        nn.Linear(hidden_dim, hidden_dim), nn.SELU(), nn.AlphaDropout(0.05),
        nn.Linear(hidden_dim, hidden_dim // 2), nn.SELU(),
        nn.Linear(hidden_dim // 2, out_dim),
    )


def _tanh_block(in_dim: int, hidden_dim: int, out_dim: int):
    """Small Tanh network for diffusion / residual branches."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim // 2), nn.Tanh(),
        nn.Linear(hidden_dim // 2, out_dim),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Dynamics right-hand sides  (nn.Module for torchsde f/g interface)
# ═══════════════════════════════════════════════════════════════════════

# -- ODE variants (zero diffusion) ------------------------------------

def _build_vanilla(hidden_dim: int = 128):
    """Fully black-box: NN → [dθ, dθ̇]."""
    import torch, torch.nn as nn

    class _Func(nn.Module):
        noise_type = "diagonal"; sde_type = "ito"
        def __init__(self):
            super().__init__()
            self.net = _selu_block(3, hidden_dim, 2)
            self.u_series = self.t_series = self.batch_start_times = None
        def f(self, t, x):
            u_t = interp_u(self, t, x)
            return self.net(torch.cat([x, u_t], dim=1))
        def g(self, t, x):
            return torch.zeros_like(x)

    return _Func()


def _build_structured(hidden_dim: int = 128):
    """Kinematic constraint dθ/dt = θ̇, NN → dθ̇/dt."""
    import torch, torch.nn as nn

    class _Func(nn.Module):
        noise_type = "diagonal"; sde_type = "ito"
        def __init__(self):
            super().__init__()
            self.net = _selu_block(3, hidden_dim, 1)
            self.u_series = self.t_series = self.batch_start_times = None
        def f(self, t, x):
            u_t = interp_u(self, t, x)
            thd = x[:, 1:2]
            thdd = self.net(torch.cat([x, u_t], dim=1))
            return torch.cat([thd, thdd], dim=1)
        def g(self, t, x):
            return torch.zeros_like(x)

    return _Func()


def _build_adaptive(hidden_dim: int = 128):
    """Structured base + near-zero residual correction."""
    import torch, torch.nn as nn

    class _Func(nn.Module):
        noise_type = "diagonal"; sde_type = "ito"
        def __init__(self):
            super().__init__()
            self.dynamics_net = _selu_block(3, hidden_dim, 1)
            self.adaptive_residual = _tanh_block(3, hidden_dim, 1)
            with torch.no_grad():
                self.adaptive_residual[-1].weight.mul_(0.01)
                self.adaptive_residual[-1].bias.zero_()
            self.u_series = self.t_series = self.batch_start_times = None
        def f(self, t, x):
            u_t = interp_u(self, t, x)
            thd = x[:, 1:2]
            inp = torch.cat([x, u_t], dim=1)
            thdd = self.dynamics_net(inp) + self.adaptive_residual(inp)
            return torch.cat([thd, thdd], dim=1)
        def g(self, t, x):
            return torch.zeros_like(x)

    return _Func()


# -- SDE variants (learned diffusion) ---------------------------------

def _build_vanilla_nsde(hidden_dim: int = 128):
    """Fully black-box drift + learned diagonal diffusion."""
    import torch, torch.nn as nn

    class _Func(nn.Module):
        noise_type = "diagonal"; sde_type = "ito"
        def __init__(self):
            super().__init__()
            self.drift_net = _selu_block(3, hidden_dim, 2)
            self.diff_net = _tanh_block(3, hidden_dim, 2)
            self.u_series = self.t_series = self.batch_start_times = None
        def f(self, t, x):
            u_t = interp_u(self, t, x)
            return self.drift_net(torch.cat([x, u_t], dim=1))
        def g(self, t, x):
            import torch.nn.functional as F
            u_t = interp_u(self, t, x)
            return F.softplus(self.diff_net(torch.cat([x, u_t], dim=1))) + 1e-6

    return _Func()


def _build_structured_nsde(hidden_dim: int = 128):
    """Structured drift (dθ=ω) + diffusion on acceleration only."""
    import torch, torch.nn as nn

    class _Func(nn.Module):
        noise_type = "diagonal"; sde_type = "ito"
        def __init__(self):
            super().__init__()
            self.acc_net = _selu_block(3, hidden_dim, 1)
            self.diff_net = _tanh_block(3, hidden_dim, 1)
            self.u_series = self.t_series = self.batch_start_times = None
        def f(self, t, x):
            u_t = interp_u(self, t, x)
            theta, omega = x[:, 0:1], x[:, 1:2]
            acc = self.acc_net(torch.cat([theta, omega, u_t], dim=1))
            return torch.cat([omega, acc], dim=1)
        def g(self, t, x):
            import torch.nn.functional as F
            u_t = interp_u(self, t, x)
            theta, omega = x[:, 0:1], x[:, 1:2]
            sigma = F.softplus(self.diff_net(torch.cat([theta, omega, u_t], dim=1))) + 1e-6
            return torch.cat([torch.zeros_like(sigma), sigma], dim=1)

    return _Func()


def _build_adaptive_nsde(hidden_dim: int = 128):
    """Adaptive drift (structured + residual) + learned diffusion."""
    import torch, torch.nn as nn

    class _Func(nn.Module):
        noise_type = "diagonal"; sde_type = "ito"
        def __init__(self):
            super().__init__()
            self.base_net = _selu_block(3, hidden_dim, 1)
            self.residual_net = _tanh_block(3, hidden_dim, 1)
            self.diff_net = _tanh_block(3, hidden_dim, 1)
            with torch.no_grad():
                self.residual_net[-1].weight.mul_(0.01)
                self.residual_net[-1].bias.zero_()
            self.u_series = self.t_series = self.batch_start_times = None
        def f(self, t, x):
            u_t = interp_u(self, t, x)
            theta, omega = x[:, 0:1], x[:, 1:2]
            inp = torch.cat([theta, omega, u_t], dim=1)
            return torch.cat([omega, self.base_net(inp) + self.residual_net(inp)], dim=1)
        def g(self, t, x):
            import torch.nn.functional as F
            u_t = interp_u(self, t, x)
            theta, omega = x[:, 0:1], x[:, 1:2]
            sigma = F.softplus(self.diff_net(torch.cat([theta, omega, u_t], dim=1))) + 1e-6
            return torch.cat([torch.zeros_like(sigma), sigma], dim=1)

    return _Func()


# Factory name → builder function
_FACTORIES: Dict[str, Any] = {
    "vanilla": _build_vanilla,
    "structured": _build_structured,
    "adaptive": _build_adaptive,
    "vanilla_nsde": _build_vanilla_nsde,
    "structured_nsde": _build_structured_nsde,
    "adaptive_nsde": _build_adaptive_nsde,
}


# ═══════════════════════════════════════════════════════════════════════
#  Shared wrapper
# ═══════════════════════════════════════════════════════════════════════

class _BlackboxODE2D(BaseModel):
    """Shared base for all black-box 2-D state models (ODE, SDE, CDE).

    Sub-classes only need to set ``_factory_name`` and, if they use a
    different config dataclass, override ``_make_default_config``.
    """

    _factory_name: str = "vanilla"

    def __init__(self, config=None, **kwargs):
        if config is None:
            config = self._make_default_config(**kwargs)
        super().__init__(config)
        self.func_ = None
        self._device = None
        self._dtype = None

    @staticmethod
    def _make_default_config(**kwargs):
        return BlackboxODE2DConfig(**kwargs)

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_velocity(y, dt):
        return np.gradient(y, dt)

    def _simulate(self, u_t, x0, t_grid, batch_start_times=None):
        """Integrate dynamics using torchsde (SDE or zero-noise ODE)."""
        import torchsde
        self.func_.u_series = u_t
        self.func_.t_series = t_grid
        self.func_.batch_start_times = batch_start_times
        x0_batch = x0 if x0.ndim == 2 else x0.unsqueeze(0)
        return torchsde.sdeint(
            self.func_, x0_batch, t_grid,
            method="euler", dt=self.config.dt,
        )

    def _simulate_deterministic(self, u_t, x0, t_grid):
        """Euler-integrate drift f() only — deterministic, no BM overhead."""
        import torch
        self.func_.u_series = u_t
        self.func_.t_series = t_grid
        self.func_.batch_start_times = None
        x0_batch = x0 if x0.ndim == 2 else x0.unsqueeze(0)
        dt = self.config.dt
        x = x0_batch.clone()
        trajectory = [x]
        for i in range(1, len(t_grid)):
            dx = self.func_.f(t_grid[i - 1], x)
            x = x + dx * dt
            trajectory.append(x)
        return torch.stack(trajectory, dim=0)

    # ── training ──────────────────────────────────────────────────────

    def _fit(self, u, y, *, val_data=None, logger=None):
        import torch
        c = self.config
        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float)

        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            y_pos = y.flatten()
            y_sim = np.column_stack([y_pos, self._compute_velocity(y_pos, c.dt)])
        else:
            y_sim = y

        self._device = self._resolve_torch_device()
        self._dtype = torch.float32

        factory = _FACTORIES[self._factory_name]
        self.func_ = factory(c.hidden_dim).to(self._device)
        self.training_loss_ = self._train_shooting(u, y_sim, logger)

    def _train_shooting(self, u, y_sim, logger):
        import torch, torch.optim as optim
        from tqdm.auto import tqdm
        c = self.config

        u_t = torch.tensor(u.reshape(-1, 1), dtype=self._dtype, device=self._device)
        y_t = torch.tensor(y_sim, dtype=self._dtype, device=self._device)
        t_full = torch.arange(len(y_t), dtype=self._dtype, device=self._device) * c.dt

        K = min(c.k_steps, len(y_t) - 1)
        B = c.batch_size
        t_eval = torch.arange(K, dtype=self._dtype, device=self._device) * c.dt

        params = list(self.func_.parameters())
        optimizer = optim.Adam(params, lr=c.learning_rate)
        history = []

        it = range(c.epochs)
        if c.verbose:
            it = tqdm(it, desc=f"Training {self.__class__.__name__}", unit="epoch")

        for epoch in it:
            self.func_.train()
            optimizer.zero_grad()

            max_start = len(y_t) - K
            start_idx = np.random.randint(0, max(1, max_start), size=B)
            x0 = y_t[start_idx]
            batch_start_times = t_full[start_idx].reshape(-1, 1)

            pred = self._simulate(u_t, x0, t_eval, batch_start_times)
            targets = torch.stack([y_t[i:i + K] for i in start_idx], dim=1)
            loss = torch.mean((pred - targets) ** 2)

            if torch.isnan(loss) or torch.isinf(loss):
                history.append(float("nan"))
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, SHOOTING_GRAD_CLIP)
            optimizer.step()

            loss_val = loss.item()
            history.append(loss_val)

            if c.verbose and hasattr(it, "set_postfix"):
                it.set_postfix(loss=f"{loss_val:.6f}")
            if logger and c.wandb_log_every > 0 and (epoch + 1) % c.wandb_log_every == 0:
                logger.log_metrics(
                    {"train/loss": loss_val, "train/epoch": epoch + 1},
                    step=epoch + 1,
                )

        return history

    # ── predict ───────────────────────────────────────────────────────

    def predict_osa(self, u, y):
        import torch
        c = self.config
        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float)
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            y_pos = y.flatten()
            y_sim = np.column_stack([y_pos, self._compute_velocity(y_pos, c.dt)])
        else:
            y_sim = y

        preds = []
        self.func_.eval()
        with torch.no_grad():
            for k in range(len(y_sim) - 1):
                x0 = torch.tensor(y_sim[k:k+1], dtype=self._dtype, device=self._device)
                u_seg = torch.tensor(
                    [[u[k]], [u[min(k + 1, len(u) - 1)]]],
                    dtype=self._dtype, device=self._device,
                )
                t_seg = torch.tensor(
                    [k * c.dt, (k + 1) * c.dt],
                    dtype=self._dtype, device=self._device,
                )
                pred = self._simulate_deterministic(u_seg, x0, t_seg)
                preds.append(pred[-1, 0, 0].cpu().item())
        return np.asarray(preds)

    def predict_free_run(self, u, y_initial):
        import torch
        c = self.config
        u = np.asarray(u, dtype=float).flatten()
        y_init = np.asarray(y_initial, dtype=float)
        if y_init.ndim == 1 or (y_init.ndim == 2 and y_init.shape[1] == 1):
            y_pos = y_init.flatten()
            x0_np = np.array([[y_pos[0], self._compute_velocity(y_pos, c.dt)[0]]])
        else:
            x0_np = y_init[0:1]

        self.func_.eval()
        with torch.no_grad():
            u_t = torch.tensor(u.reshape(-1, 1), dtype=self._dtype, device=self._device)
            x0 = torch.tensor(x0_np, dtype=self._dtype, device=self._device)
            t_grid = torch.arange(len(u), dtype=self._dtype, device=self._device) * c.dt
            pred = self._simulate_deterministic(u_t, x0, t_grid)
        return pred[:, 0, 0].cpu().numpy()

    # ── save / load ───────────────────────────────────────────────────

    def _collect_state(self) -> Dict[str, Any]:
        if self.func_ is None:
            return {}
        return {"func": self.func_.state_dict()}

    def _restore_state(self, state):
        # Accept legacy keys from older saved models
        sd = state.get("func") or state.get("ode_func") or state.get("sde_func")
        self.func_.load_state_dict(sd)

    def _build_for_load(self):
        c = self.config
        self._device = self._resolve_torch_device("cpu")
        self._dtype = torch.float32
        factory = _FACTORIES[self._factory_name]
        self.func_ = factory(c.hidden_dim).to(self._device)

    def __repr__(self):
        c = self.config
        return (f"{type(self).__name__}(hidden={c.hidden_dim}, "
                f"K={c.k_steps}, epochs={c.epochs})")


# ═══════════════════════════════════════════════════════════════════════
#  Public ODE model classes
# ═══════════════════════════════════════════════════════════════════════

class VanillaNODE2D(_BlackboxODE2D):
    """Vanilla Neural ODE with 2-D state. No kinematic prior."""
    _factory_name = "vanilla"


class StructuredNODE(_BlackboxODE2D):
    """Structured NODE: dθ/dt=θ̇ hardcoded, NN → dθ̇/dt."""
    _factory_name = "structured"


class AdaptiveNODE(_BlackboxODE2D):
    """Adaptive NODE: structured base + near-zero residual correction."""
    _factory_name = "adaptive"


# ═══════════════════════════════════════════════════════════════════════
#  Public SDE model classes
# ═══════════════════════════════════════════════════════════════════════

class VanillaNSDE2D(_BlackboxODE2D):
    """Vanilla 2-D Neural SDE."""
    _factory_name = "vanilla_nsde"

    @staticmethod
    def _make_default_config(**kwargs):
        return BlackboxSDE2DConfig(**kwargs)


class StructuredNSDE(_BlackboxODE2D):
    """Structured 2-D NSDE: dθ=ω hardcoded, NN → dω + diffusion."""
    _factory_name = "structured_nsde"

    @staticmethod
    def _make_default_config(**kwargs):
        return BlackboxSDE2DConfig(**kwargs)


class AdaptiveNSDE(_BlackboxODE2D):
    """Adaptive 2-D NSDE: structured + residual + learned diffusion."""
    _factory_name = "adaptive_nsde"

    @staticmethod
    def _make_default_config(**kwargs):
        return BlackboxSDE2DConfig(**kwargs)
