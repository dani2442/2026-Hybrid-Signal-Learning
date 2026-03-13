"""Linear and Stribeck physics ODE models."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .base import InterpNeuralODEBase


class LinearPhysODE(InterpNeuralODEBase):
    """J*thdd + R*thd + K*(th + delta) = Tau*u"""

    def __init__(self) -> None:
        super().__init__()
        self.log_J = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
        self.log_R = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
        self.log_K = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.log_Tau = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))

    def get_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        J = torch.exp(self.log_J)
        R = torch.exp(self.log_R)
        K = torch.exp(self.log_K)
        Tau = torch.exp(self.log_Tau)
        return J, R, K, self.delta, Tau

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        xb, squeeze = self._as_batch(x)
        J, R, K, delta, Tau = self.get_params()
        u_t = self._interp_u(t, xb)

        th = xb[:, 0:1]
        thd = xb[:, 1:2]
        thdd = (Tau * u_t - R * thd - K * (th + delta)) / J

        dx = torch.cat([thd, thdd], dim=1)
        return dx.squeeze(0) if squeeze else dx


class StribeckPhysODE(InterpNeuralODEBase):
    """J*thdd + R*thd + K*(th + delta) + F_str = Tau*u"""

    def __init__(self) -> None:
        super().__init__()
        self.log_J = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
        self.log_R = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
        self.log_K = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.log_Tau = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))

        self.log_Fc = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
        self.log_Fs = nn.Parameter(torch.tensor(np.log(0.2), dtype=torch.float32))
        self.log_vs = nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))
        self.log_b = nn.Parameter(torch.tensor(np.log(0.01), dtype=torch.float32))

    def get_params(self) -> tuple[torch.Tensor, ...]:
        J = torch.exp(self.log_J)
        R = torch.exp(self.log_R)
        K = torch.exp(self.log_K)
        Tau = torch.exp(self.log_Tau)
        Fc = torch.exp(self.log_Fc)
        Fs = torch.exp(self.log_Fs)
        vs = torch.exp(self.log_vs)
        b = torch.exp(self.log_b)
        return J, R, K, self.delta, Tau, Fc, Fs, vs, b

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        xb, squeeze = self._as_batch(x)
        J, R, K, delta, Tau, Fc, Fs, vs, b = self.get_params()
        u_t = self._interp_u(t, xb)

        th = xb[:, 0:1]
        thd = xb[:, 1:2]

        sgn = torch.tanh(thd / 1e-3)
        f_str = (Fc + (Fs - Fc) * torch.exp(-((thd / vs) ** 2))) * sgn + b * thd
        thdd = (Tau * u_t - R * thd - K * (th + delta) - f_str) / J

        dx = torch.cat([thd, thdd], dim=1)
        return dx.squeeze(0) if squeeze else dx


def extract_linear_params(model: LinearPhysODE) -> dict[str, float]:
    J, R, K, delta, Tau = model.get_params()
    return {
        "J": float(J.detach().cpu()),
        "R": float(R.detach().cpu()),
        "K": float(K.detach().cpu()),
        "delta": float(delta.detach().cpu()),
        "Tau": float(Tau.detach().cpu()),
    }


def extract_stribeck_params(model: StribeckPhysODE) -> dict[str, float]:
    J, R, K, delta, Tau, Fc, Fs, vs, b = model.get_params()
    return {
        "J": float(J.detach().cpu()),
        "R": float(R.detach().cpu()),
        "K": float(K.detach().cpu()),
        "delta": float(delta.detach().cpu()),
        "Tau": float(Tau.detach().cpu()),
        "Fc": float(Fc.detach().cpu()),
        "Fs": float(Fs.detach().cpu()),
        "vs": float(vs.detach().cpu()),
        "b": float(b.detach().cpu()),
    }
