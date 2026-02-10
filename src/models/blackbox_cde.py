"""CDE-inspired black-box models with 2-D state [theta, omega].

These models follow the same interface as the 2-D NODE family in
``blackbox_ode.py`` but augment the dynamics with the control derivative
du/dt, which is the core signal used by controlled differential equations.

Architectures
-------------
VanillaNCDE2D
    Fully black-box: NN learns both derivatives [dtheta, domega]
    from [theta, omega, u, du_dt].

StructuredNCDE
    Kinematic constraint: dtheta = omega is hardcoded.
    NN only learns domega from [theta, omega, u, du_dt].

AdaptiveNCDE
    Structured base plus a near-zero residual correction on domega.
"""

from __future__ import annotations

from .blackbox_ode import _BlackboxODE2D


def _interp_u_and_du(model, t, x):
    """Linear interpolation of u(t) and estimate of du/dt at time *t*."""
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
    u_t = u1 + alpha * (u2 - u1)
    du_t = (u2 - u1) / denom
    return u_t, du_t


def _build_vanilla_ncde(hidden_dim: int = 128):
    """Vanilla NCDE-inspired model: NN -> [dtheta, domega]."""
    import torch
    import torch.nn as nn

    class _VanillaNCDEFunc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, hidden_dim),
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
            u_t, du_t = _interp_u_and_du(self, t, x)
            return self.net(torch.cat([x, u_t, du_t], dim=1))

    return _VanillaNCDEFunc()


def _build_structured_ncde(hidden_dim: int = 128):
    """Structured NCDE-inspired model: dtheta = omega, NN -> domega."""
    import torch
    import torch.nn as nn

    class _StructuredNCDEFunc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, hidden_dim),
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
            u_t, du_t = _interp_u_and_du(self, t, x)
            theta, omega = x[:, 0:1], x[:, 1:2]
            acc = self.net(torch.cat([theta, omega, u_t, du_t], dim=1))
            return torch.cat([omega, acc], dim=1)

    return _StructuredNCDEFunc()


def _build_adaptive_ncde(hidden_dim: int = 128):
    """Adaptive NCDE-inspired: structured base + near-zero residual."""
    import torch
    import torch.nn as nn

    class _AdaptiveNCDEFunc(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_net = nn.Sequential(
                nn.Linear(4, hidden_dim),
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
                nn.Linear(4, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
            with torch.no_grad():
                self.residual_net[-1].weight.mul_(0.01)
                self.residual_net[-1].bias.zero_()

            self.u_series = None
            self.t_series = None
            self.batch_start_times = None

        def forward(self, t, x):
            u_t, du_t = _interp_u_and_du(self, t, x)
            theta, omega = x[:, 0:1], x[:, 1:2]
            inp = torch.cat([theta, omega, u_t, du_t], dim=1)
            acc = self.base_net(inp) + self.residual_net(inp)
            return torch.cat([omega, acc], dim=1)

    return _AdaptiveNCDEFunc()


class VanillaNCDE2D(_BlackboxODE2D):
    """Vanilla 2-D NCDE-inspired model."""

    def __init__(
        self,
        hidden_dim: int = 128,
        dt: float = 0.05,
        solver: str = "rk4",
        learning_rate: float = 1e-2,
        epochs: int = 5000,
        k_steps: int = 20,
        batch_size: int = 128,
        training_mode: str = "shooting",
    ):
        super().__init__(
            ode_factory=_build_vanilla_ncde,
            hidden_dim=hidden_dim,
            dt=dt,
            solver=solver,
            learning_rate=learning_rate,
            epochs=epochs,
            k_steps=k_steps,
            batch_size=batch_size,
            training_mode=training_mode,
        )

    def __repr__(self):
        return (
            f"VanillaNCDE2D(hidden={self.hidden_dim}, K={self.k_steps}, "
            f"B={self.batch_size}, epochs={self.epochs})"
        )


class StructuredNCDE(_BlackboxODE2D):
    """Structured 2-D NCDE-inspired model with hardcoded kinematics."""

    def __init__(
        self,
        hidden_dim: int = 128,
        dt: float = 0.05,
        solver: str = "rk4",
        learning_rate: float = 1e-2,
        epochs: int = 5000,
        k_steps: int = 20,
        batch_size: int = 128,
        training_mode: str = "shooting",
    ):
        super().__init__(
            ode_factory=_build_structured_ncde,
            hidden_dim=hidden_dim,
            dt=dt,
            solver=solver,
            learning_rate=learning_rate,
            epochs=epochs,
            k_steps=k_steps,
            batch_size=batch_size,
            training_mode=training_mode,
        )

    def __repr__(self):
        return (
            f"StructuredNCDE(hidden={self.hidden_dim}, K={self.k_steps}, "
            f"B={self.batch_size}, epochs={self.epochs})"
        )


class AdaptiveNCDE(_BlackboxODE2D):
    """Adaptive 2-D NCDE-inspired model with residual correction."""

    def __init__(
        self,
        hidden_dim: int = 128,
        dt: float = 0.05,
        solver: str = "rk4",
        learning_rate: float = 5e-3,
        epochs: int = 5000,
        k_steps: int = 20,
        batch_size: int = 128,
        training_mode: str = "shooting",
    ):
        super().__init__(
            ode_factory=_build_adaptive_ncde,
            hidden_dim=hidden_dim,
            dt=dt,
            solver=solver,
            learning_rate=learning_rate,
            epochs=epochs,
            k_steps=k_steps,
            batch_size=batch_size,
            training_mode=training_mode,
        )

    def __repr__(self):
        return (
            f"AdaptiveNCDE(hidden={self.hidden_dim}, K={self.k_steps}, "
            f"B={self.batch_size}, epochs={self.epochs})"
        )
