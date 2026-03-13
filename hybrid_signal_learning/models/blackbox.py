"""Black-box Neural ODE models."""

from __future__ import annotations

import torch
from torch import nn

from .base import InterpNeuralODEBase, NN_VARIANTS, NnVariantConfig, _build_selu_mlp


class BlackBoxODE(InterpNeuralODEBase):
    def __init__(self, variant_name: str = "base") -> None:
        super().__init__()
        if variant_name not in NN_VARIANTS:
            raise ValueError(f"Unknown variant '{variant_name}'")
        self.variant_name = variant_name
        self.net = _build_selu_mlp(input_dim=3, output_dim=2, variant=NN_VARIANTS[variant_name])

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        xb, squeeze = self._as_batch(x)
        u_t = self._interp_u(t, xb)
        dx = self.net(torch.cat([xb, u_t], dim=1))
        return dx.squeeze(0) if squeeze else dx


class StructuredBlackBoxODE(InterpNeuralODEBase):
    """Kinematic constraint: dx0 = x1 (hardcoded), dx1 = NN(theta, theta_dot, u).

    Same capacity as BlackBoxODE but the kinematic prior reduces the NN to
    predicting only the acceleration, which generally improves sample efficiency.
    """

    def __init__(self, variant_name: str = "base") -> None:
        super().__init__()
        if variant_name not in NN_VARIANTS:
            raise ValueError(f"Unknown variant '{variant_name}'")
        self.variant_name = variant_name
        self.net = _build_selu_mlp(input_dim=3, output_dim=1, variant=NN_VARIANTS[variant_name])

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        xb, squeeze = self._as_batch(x)
        u_t = self._interp_u(t, xb)

        th = xb[:, 0:1]
        thd = xb[:, 1:2]
        thdd = self.net(torch.cat([th, thd, u_t], dim=1))

        dx = torch.cat([thd, thdd], dim=1)
        return dx.squeeze(0) if squeeze else dx


class AdaptiveBlackBoxODE(InterpNeuralODEBase):
    """Kinematic constraint with base dynamics NN + near-zero residual correction NN.

    Both paths receive [theta, theta_dot, u].  The residual network's final
    layer is initialised near zero so it doesn't destabilise early training.
    """

    def __init__(self, variant_name: str = "base") -> None:
        super().__init__()
        if variant_name not in NN_VARIANTS:
            raise ValueError(f"Unknown variant '{variant_name}'")
        self.variant_name = variant_name

        cfg = NN_VARIANTS[variant_name]
        self.dynamics_net = _build_selu_mlp(input_dim=3, output_dim=1, variant=cfg)

        # Smaller residual network (half hidden dim, depth 1, tanh activation)
        res_hidden = max(16, cfg.hidden_dim // 2)
        self.adaptive_residual = nn.Sequential(
            nn.Linear(3, res_hidden),
            nn.Tanh(),
            nn.Linear(res_hidden, 1),
        )
        # Initialise residual path near zero
        with torch.no_grad():
            self.adaptive_residual[-1].weight.mul_(0.01)
            self.adaptive_residual[-1].bias.zero_()

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        xb, squeeze = self._as_batch(x)
        u_t = self._interp_u(t, xb)

        th = xb[:, 0:1]
        thd = xb[:, 1:2]
        nn_input = torch.cat([th, thd, u_t], dim=1)

        thdd = self.dynamics_net(nn_input) + self.adaptive_residual(nn_input)

        dx = torch.cat([thd, thdd], dim=1)
        return dx.squeeze(0) if squeeze else dx
