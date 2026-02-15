"""CDE-inspired black-box models with 2-D state [θ, θ̇].

Augments dynamics with du/dt (control derivative).

VanillaNCDE2D  - NN → [dθ, dω] from [θ, ω, u, du/dt]
StructuredNCDE - dθ=ω hardcoded, NN → dω
AdaptiveNCDE   - structured + near-zero residual
"""

from __future__ import annotations

from .blackbox_ode import _BlackboxODE2D, _FACTORIES, _selu_block, _tanh_block
from .torchsde_utils import interp_u


def _build_vanilla_ncde(hidden_dim: int = 128):
    """Fully black-box CDE: NN → [dθ, dω] from [θ, ω, u, du/dt]."""
    import torch, torch.nn as nn

    class _Func(nn.Module):
        noise_type = "diagonal"; sde_type = "ito"
        def __init__(self):
            super().__init__()
            self.net = _selu_block(4, hidden_dim, 2)
            self.u_series = self.t_series = self.batch_start_times = None
        def f(self, t, x):
            u_t, du_t = interp_u(self, t, x, return_du=True)
            return self.net(torch.cat([x, u_t, du_t], dim=1))
        def g(self, t, x):
            return torch.zeros_like(x)

    return _Func()


def _build_structured_ncde(hidden_dim: int = 128):
    """Structured CDE: dθ=ω hardcoded, NN → dω from [θ,ω,u,du/dt]."""
    import torch, torch.nn as nn

    class _Func(nn.Module):
        noise_type = "diagonal"; sde_type = "ito"
        def __init__(self):
            super().__init__()
            self.net = _selu_block(4, hidden_dim, 1)
            self.u_series = self.t_series = self.batch_start_times = None
        def f(self, t, x):
            u_t, du_t = interp_u(self, t, x, return_du=True)
            theta, omega = x[:, 0:1], x[:, 1:2]
            acc = self.net(torch.cat([theta, omega, u_t, du_t], dim=1))
            return torch.cat([omega, acc], dim=1)
        def g(self, t, x):
            return torch.zeros_like(x)

    return _Func()


def _build_adaptive_ncde(hidden_dim: int = 128):
    """Adaptive CDE: structured + near-zero residual from [θ,ω,u,du/dt]."""
    import torch, torch.nn as nn

    class _Func(nn.Module):
        noise_type = "diagonal"; sde_type = "ito"
        def __init__(self):
            super().__init__()
            self.base_net = _selu_block(4, hidden_dim, 1)
            self.residual_net = _tanh_block(4, hidden_dim, 1)
            with torch.no_grad():
                self.residual_net[-1].weight.mul_(0.01)
                self.residual_net[-1].bias.zero_()
            self.u_series = self.t_series = self.batch_start_times = None
        def f(self, t, x):
            u_t, du_t = interp_u(self, t, x, return_du=True)
            theta, omega = x[:, 0:1], x[:, 1:2]
            inp = torch.cat([theta, omega, u_t, du_t], dim=1)
            return torch.cat([omega, self.base_net(inp) + self.residual_net(inp)], dim=1)
        def g(self, t, x):
            return torch.zeros_like(x)

    return _Func()


# Register CDE factories alongside ODE/SDE ones
_FACTORIES["vanilla_ncde"] = _build_vanilla_ncde
_FACTORIES["structured_ncde"] = _build_structured_ncde
_FACTORIES["adaptive_ncde"] = _build_adaptive_ncde


class VanillaNCDE2D(_BlackboxODE2D):
    """Vanilla 2-D NCDE-inspired model."""
    _factory_name = "vanilla_ncde"

    @staticmethod
    def _make_default_config():
        from ..config import BlackboxCDE2DConfig
        return BlackboxCDE2DConfig()


class StructuredNCDE(_BlackboxODE2D):
    """Structured 2-D NCDE: dθ=ω hardcoded, NN → dω from [θ,ω,u,du/dt]."""
    _factory_name = "structured_ncde"

    @staticmethod
    def _make_default_config():
        from ..config import BlackboxCDE2DConfig
        return BlackboxCDE2DConfig()


class AdaptiveNCDE(_BlackboxODE2D):
    """Adaptive 2-D NCDE: structured + near-zero residual."""
    _factory_name = "adaptive_ncde"

    @staticmethod
    def _make_default_config():
        from ..config import BlackboxCDE2DConfig
        return BlackboxCDE2DConfig()
