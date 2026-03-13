"""Universal Differential Equation: dx/dt = A·x + B·u + NN(x, u).

Adapted from models/hybrid/ude.py. Combines a learnable linear state-space
model with a neural-network correction term.  The kinematic constraint
dx0 = x1 is enforced, so the linear + NN part only predicts acceleration.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import InterpNeuralODEBase, NN_VARIANTS, NnVariantConfig, _build_selu_mlp


class UDEODE(InterpNeuralODEBase):
    """UDE for 2-D state: [theta, theta_dot].

    Dynamics::

        thd  = theta_dot                          (kinematic constraint)
        thdd = A @ x + B * u + NN([theta, theta_dot, u])

    where A ∈ ℝ^{1×2} and B ∈ ℝ^{1×1} are learnable linear matrices
    and NN is a SELU MLP residual.
    """

    def __init__(self, variant_name: str = "base") -> None:
        super().__init__()
        if variant_name not in NN_VARIANTS:
            raise ValueError(f"Unknown variant '{variant_name}'")
        self.variant_name = variant_name

        # Learnable linear dynamics (maps state to acceleration contribution)
        self.A = nn.Parameter(torch.randn(1, 2) * 0.1)
        self.B = nn.Parameter(torch.randn(1, 1) * 0.1)

        # Neural residual
        self.net = _build_selu_mlp(input_dim=3, output_dim=1, variant=NN_VARIANTS[variant_name])

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        xb, squeeze = self._as_batch(x)
        u_t = self._interp_u(t, xb)

        th = xb[:, 0:1]
        thd = xb[:, 1:2]

        # Linear contribution to acceleration
        linear_acc = (xb @ self.A.T) + (u_t @ self.B.T)  # [B, 1]

        # Neural residual contribution to acceleration
        nn_acc = self.net(torch.cat([th, thd, u_t], dim=1))  # [B, 1]

        thdd = linear_acc + nn_acc
        dx = torch.cat([thd, thdd], dim=1)
        return dx.squeeze(0) if squeeze else dx
