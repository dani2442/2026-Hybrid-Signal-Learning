"""Hybrid models combining physics priors and neural networks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .base import InterpNeuralODEBase, NN_VARIANTS, _build_selu_mlp


class HybridJointODE(InterpNeuralODEBase):
    def __init__(self, variant_name: str = "base") -> None:
        super().__init__()
        if variant_name not in NN_VARIANTS:
            raise ValueError(f"Unknown variant '{variant_name}'")
        self.variant_name = variant_name

        self.log_J = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
        self.log_R = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
        self.log_K = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.log_Tau = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))

        self.net = _build_selu_mlp(input_dim=3, output_dim=1, variant=NN_VARIANTS[variant_name])

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        xb, squeeze = self._as_batch(x)

        J = torch.exp(self.log_J)
        R = torch.exp(self.log_R)
        K = torch.exp(self.log_K)
        Tau = torch.exp(self.log_Tau)

        u_t = self._interp_u(t, xb)
        th = xb[:, 0:1]
        thd = xb[:, 1:2]

        thdd_phys = (Tau * u_t - R * thd - K * (th + self.delta)) / J
        thdd_res = self.net(torch.cat([th, thd, u_t], dim=1))
        thdd = thdd_phys + thdd_res
        dx = torch.cat([thd, thdd], dim=1)

        return dx.squeeze(0) if squeeze else dx


class HybridJointStribeckODE(InterpNeuralODEBase):
    """
    thdd = stribeck_physics(theta, theta_dot, u) + NN residual
    where:
      J*thdd + R*thd + K*(th+delta) + F_str(thd) = Tau*u
    """

    def __init__(self, variant_name: str = "base") -> None:
        super().__init__()
        if variant_name not in NN_VARIANTS:
            raise ValueError(f"Unknown variant '{variant_name}'")
        self.variant_name = variant_name

        self.log_J = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
        self.log_R = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
        self.log_K = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.log_Tau = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))

        self.log_Fc = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
        self.log_Fs = nn.Parameter(torch.tensor(np.log(0.2), dtype=torch.float32))
        self.log_vs = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
        self.log_b = nn.Parameter(torch.tensor(np.log(0.01), dtype=torch.float32))

        self.net = _build_selu_mlp(input_dim=3, output_dim=1, variant=NN_VARIANTS[variant_name])

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        xb, squeeze = self._as_batch(x)

        J = torch.exp(self.log_J)
        R = torch.exp(self.log_R)
        K = torch.exp(self.log_K)
        Tau = torch.exp(self.log_Tau)
        Fc = torch.exp(self.log_Fc)
        Fs = torch.exp(self.log_Fs)
        vs = torch.exp(self.log_vs)
        b = torch.exp(self.log_b)

        u_t = self._interp_u(t, xb)
        th = xb[:, 0:1]
        thd = xb[:, 1:2]

        sgn = torch.tanh(thd / 1e-3)
        f_str = (Fc + (Fs - Fc) * torch.exp(-((thd / vs) ** 2))) * sgn + b * thd

        thdd_phys = (Tau * u_t - R * thd - K * (th + self.delta) - f_str) / J
        thdd_res = self.net(torch.cat([th, thd, u_t], dim=1))
        thdd = thdd_phys + thdd_res
        dx = torch.cat([thd, thdd], dim=1)

        return dx.squeeze(0) if squeeze else dx


class HybridFrozenPhysODE(InterpNeuralODEBase):
    def __init__(self, frozen_phys_params: dict[str, float], variant_name: str = "base") -> None:
        super().__init__()
        if variant_name not in NN_VARIANTS:
            raise ValueError(f"Unknown variant '{variant_name}'")
        self.variant_name = variant_name

        for key in ("J", "R", "K", "delta", "Tau"):
            if key not in frozen_phys_params:
                raise ValueError(f"Missing frozen parameter '{key}'")

        self.register_buffer("J0", torch.tensor(float(frozen_phys_params["J"]), dtype=torch.float32))
        self.register_buffer("R0", torch.tensor(float(frozen_phys_params["R"]), dtype=torch.float32))
        self.register_buffer("K0", torch.tensor(float(frozen_phys_params["K"]), dtype=torch.float32))
        self.register_buffer("delta0", torch.tensor(float(frozen_phys_params["delta"]), dtype=torch.float32))
        self.register_buffer("Tau0", torch.tensor(float(frozen_phys_params["Tau"]), dtype=torch.float32))

        self.net = _build_selu_mlp(input_dim=3, output_dim=1, variant=NN_VARIANTS[variant_name])

    def frozen_phys_params(self) -> dict[str, float]:
        return {
            "J": float(self.J0.detach().cpu()),
            "R": float(self.R0.detach().cpu()),
            "K": float(self.K0.detach().cpu()),
            "delta": float(self.delta0.detach().cpu()),
            "Tau": float(self.Tau0.detach().cpu()),
        }

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        xb, squeeze = self._as_batch(x)
        u_t = self._interp_u(t, xb)

        th = xb[:, 0:1]
        thd = xb[:, 1:2]

        thdd_phys = (self.Tau0 * u_t - self.R0 * thd - self.K0 * (th + self.delta0)) / self.J0
        thdd_res = self.net(torch.cat([th, thd, u_t], dim=1))
        thdd = thdd_phys + thdd_res

        dx = torch.cat([thd, thdd], dim=1)
        return dx.squeeze(0) if squeeze else dx


class HybridFrozenStribeckPhysODE(InterpNeuralODEBase):
    """thdd = frozen stribeck physics(theta, theta_dot, u) + NN residual."""

    def __init__(self, frozen_phys_params: dict[str, float], variant_name: str = "base") -> None:
        super().__init__()
        if variant_name not in NN_VARIANTS:
            raise ValueError(f"Unknown variant '{variant_name}'")
        self.variant_name = variant_name

        for key in ("J", "R", "K", "delta", "Tau", "Fc", "Fs", "vs", "b"):
            if key not in frozen_phys_params:
                raise ValueError(f"Missing frozen parameter '{key}'")

        self.register_buffer("J0", torch.tensor(float(frozen_phys_params["J"]), dtype=torch.float32))
        self.register_buffer("R0", torch.tensor(float(frozen_phys_params["R"]), dtype=torch.float32))
        self.register_buffer("K0", torch.tensor(float(frozen_phys_params["K"]), dtype=torch.float32))
        self.register_buffer("delta0", torch.tensor(float(frozen_phys_params["delta"]), dtype=torch.float32))
        self.register_buffer("Tau0", torch.tensor(float(frozen_phys_params["Tau"]), dtype=torch.float32))
        self.register_buffer("Fc0", torch.tensor(float(frozen_phys_params["Fc"]), dtype=torch.float32))
        self.register_buffer("Fs0", torch.tensor(float(frozen_phys_params["Fs"]), dtype=torch.float32))
        self.register_buffer("vs0", torch.tensor(float(frozen_phys_params["vs"]), dtype=torch.float32))
        self.register_buffer("b0", torch.tensor(float(frozen_phys_params["b"]), dtype=torch.float32))

        self.net = _build_selu_mlp(input_dim=3, output_dim=1, variant=NN_VARIANTS[variant_name])

    def frozen_phys_params(self) -> dict[str, float]:
        return {
            "J": float(self.J0.detach().cpu()),
            "R": float(self.R0.detach().cpu()),
            "K": float(self.K0.detach().cpu()),
            "delta": float(self.delta0.detach().cpu()),
            "Tau": float(self.Tau0.detach().cpu()),
            "Fc": float(self.Fc0.detach().cpu()),
            "Fs": float(self.Fs0.detach().cpu()),
            "vs": float(self.vs0.detach().cpu()),
            "b": float(self.b0.detach().cpu()),
        }

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        xb, squeeze = self._as_batch(x)
        u_t = self._interp_u(t, xb)

        th = xb[:, 0:1]
        thd = xb[:, 1:2]

        sgn = torch.tanh(thd / 1e-3)
        f_str = (self.Fc0 + (self.Fs0 - self.Fc0) * torch.exp(-((thd / self.vs0) ** 2))) * sgn + self.b0 * thd
        thdd_phys = (self.Tau0 * u_t - self.R0 * thd - self.K0 * (th + self.delta0) - f_str) / self.J0
        thdd_res = self.net(torch.cat([th, thd, u_t], dim=1))
        thdd = thdd_phys + thdd_res

        dx = torch.cat([thd, thdd], dim=1)
        return dx.squeeze(0) if squeeze else dx
