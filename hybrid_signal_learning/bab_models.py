from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


MODEL_KEYS = [
    "linear",
    "stribeck",
    "blackbox",
    "structured_blackbox",
    "adaptive_blackbox",
    "ct_esn",
    "hybrid_joint",
    "hybrid_joint_stribeck",
    "hybrid_frozen",
    "hybrid_frozen_stribeck",
]


@dataclass(frozen=True, slots=True)
class NnVariantConfig:
    hidden_dim: int
    depth: int
    dropout: float = 0.05


NN_VARIANTS: dict[str, NnVariantConfig] = {
    "compact": NnVariantConfig(hidden_dim=64, depth=2, dropout=0.05),
    "base": NnVariantConfig(hidden_dim=128, depth=3, dropout=0.05),
    "wide": NnVariantConfig(hidden_dim=256, depth=3, dropout=0.05),
    "deep": NnVariantConfig(hidden_dim=128, depth=5, dropout=0.05),
}


def uses_nn_variant(model_key: str) -> bool:
    return model_key in {
        "blackbox",
        "structured_blackbox",
        "adaptive_blackbox",
        "hybrid_joint",
        "hybrid_joint_stribeck",
        "hybrid_frozen",
        "hybrid_frozen_stribeck",
    }


def iter_model_specs(model_keys: list[str], nn_variants: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for mk in model_keys:
        if mk not in MODEL_KEYS:
            raise ValueError(f"Unsupported model key '{mk}'. Supported: {MODEL_KEYS}")
        if uses_nn_variant(mk):
            for vv in nn_variants:
                if vv not in NN_VARIANTS:
                    raise ValueError(f"Unsupported NN variant '{vv}'. Supported: {sorted(NN_VARIANTS)}")
                specs.append((mk, vv))
        else:
            specs.append((mk, "physics"))
    return specs


class InterpNeuralODEBase(nn.Module):
    """torchdiffeq-compatible base with piecewise-linear interpolation of u(t)."""

    def __init__(self) -> None:
        super().__init__()
        self.u_series: torch.Tensor | None = None
        self.t_series: torch.Tensor | None = None
        self.batch_start_times: torch.Tensor | None = None

    def set_series(self, t_series: torch.Tensor, u_series: torch.Tensor) -> None:
        if t_series.ndim != 1:
            raise ValueError("t_series must be 1D")
        if u_series.ndim != 2 or u_series.shape[1] != 1:
            raise ValueError("u_series must have shape (N,1)")
        if t_series.shape[0] != u_series.shape[0]:
            raise ValueError("t_series and u_series length mismatch")
        self.t_series = t_series
        self.u_series = u_series

    def set_batch_start_times(self, batch_start_times: torch.Tensor | None) -> None:
        self.batch_start_times = batch_start_times

    def _as_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.ndim == 2:
            return x, False
        if x.ndim == 1:
            return x.unsqueeze(0), True
        raise ValueError(f"Expected x with ndim 1 or 2, got {x.ndim}")

    def _interp_u(self, t: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
        if self.t_series is None or self.u_series is None:
            raise RuntimeError("Call set_series(t_series, u_series) before integration")

        if self.batch_start_times is not None:
            t_abs = self.batch_start_times + t
        else:
            t_abs = t * torch.ones_like(x_batch[:, 0:1])

        k_idx = torch.searchsorted(self.t_series, t_abs.reshape(-1), right=True)
        k_idx = torch.clamp(k_idx, 1, len(self.t_series) - 1)

        t1 = self.t_series[k_idx - 1].unsqueeze(1)
        t2 = self.t_series[k_idx].unsqueeze(1)
        u1 = self.u_series[k_idx - 1]
        u2 = self.u_series[k_idx]

        denom = t2 - t1
        denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
        alpha = (t_abs - t1) / denom
        return u1 + alpha * (u2 - u1)


def _build_selu_mlp(input_dim: int, output_dim: int, variant: NnVariantConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim

    for _ in range(variant.depth):
        layers.append(nn.Linear(in_dim, variant.hidden_dim))
        layers.append(nn.SELU())
        if variant.dropout > 0.0:
            layers.append(nn.AlphaDropout(variant.dropout))
        in_dim = variant.hidden_dim

    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


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


class ContinuousTimeESN(InterpNeuralODEBase):
    """Continuous-time Echo State Network with kinematic constraint.

    The ODE state is augmented: z = [theta, theta_dot, r_0, ..., r_{D-1}]
    where r is the reservoir hidden state of dimension ``reservoir_dim``.

    Because the integration state is larger than obs_dim=2, callers must:
      * Use :meth:`prepare_x0` to build a proper initial condition.
      * Slice predictions to ``[..., :obs_dim]`` after integration (the
        standard ``train_model`` / ``simulate_full_rollout`` already do this
        via ``obs_dim``).

    Unlike the MLP-based models this class does **not** use :data:`NN_VARIANTS`;
    pass ``reservoir_dim``, ``spectral_radius``, ``input_scale``, and
    ``leak_rate`` directly.
    """

    # Sentinel so the rest of the codebase can detect augmented-state models.
    augmented_state: bool = True

    def __init__(
        self,
        reservoir_dim: int = 200,
        spectral_radius: float = 0.9,
        input_scale: float = 0.5,
        leak_rate: float = 1.0,
        state_dim: int = 2,
        input_dim: int = 1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim

        # Learnable leak rate (clamped in forward to [0.1, 10])
        self.leak_rate = nn.Parameter(torch.tensor(float(leak_rate)))

        # Fixed sparse reservoir matrix
        W = torch.randn(reservoir_dim, reservoir_dim) * 0.1
        mask = torch.rand_like(W) < 0.8
        W[mask] = 0.0
        eigvals = torch.linalg.eigvals(W).abs()
        if eigvals.max() > 0:
            W = W * (spectral_radius / eigvals.max())
        self.register_buffer("W_res", W)

        # Learnable input weights
        self.W_in = nn.Parameter(
            torch.randn(reservoir_dim, state_dim + input_dim) * input_scale
        )

        # Readout: reservoir state + skip connection → acceleration
        self.W_out = nn.Linear(reservoir_dim + state_dim + input_dim, 1, bias=True)

    # ------------------------------------------------------------------
    # Helpers for augmented state
    # ------------------------------------------------------------------
    def init_reservoir(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        dev = device or self.W_res.device
        return torch.zeros(batch_size, self.reservoir_dim, device=dev)

    def prepare_x0(self, y0: torch.Tensor) -> torch.Tensor:
        """Concatenate physical initial state with zero reservoir state.

        Parameters
        ----------
        y0 : Tensor of shape ``(B, state_dim)`` or ``(state_dim,)``
        """
        if y0.ndim == 1:
            y0 = y0.unsqueeze(0)
        r0 = self.init_reservoir(y0.shape[0], device=y0.device)
        return torch.cat([y0[:, : self.state_dim], r0], dim=1)

    # ------------------------------------------------------------------
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        zb, squeeze = self._as_batch(z)

        x = zb[:, : self.state_dim]
        r = zb[:, self.state_dim :]

        u_t = self._interp_u(t, x)

        xu = torch.cat([x, u_t], dim=1)
        lr = torch.clamp(self.leak_rate, 0.1, 10.0)
        dr = lr * (-r + torch.tanh(r @ self.W_res.T + xu @ self.W_in.T))

        readout_input = torch.cat([r, x, u_t], dim=1)
        acceleration = self.W_out(readout_input)

        # Kinematic constraint: dx0 = x1
        velocity = x[:, 1:2]
        dx = torch.cat([velocity, acceleration], dim=1)

        dz = torch.cat([dx, dr], dim=1)
        return dz.squeeze(0) if squeeze else dz


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


def build_model(
    model_key: str,
    *,
    nn_variant: str = "base",
    frozen_phys_params: dict[str, float] | None = None,
    esn_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    if model_key == "linear":
        return LinearPhysODE()
    if model_key == "stribeck":
        return StribeckPhysODE()
    if model_key == "blackbox":
        return BlackBoxODE(variant_name=nn_variant)
    if model_key == "structured_blackbox":
        return StructuredBlackBoxODE(variant_name=nn_variant)
    if model_key == "adaptive_blackbox":
        return AdaptiveBlackBoxODE(variant_name=nn_variant)
    if model_key == "ct_esn":
        kw = esn_kwargs or {}
        return ContinuousTimeESN(**kw)
    if model_key == "hybrid_joint":
        return HybridJointODE(variant_name=nn_variant)
    if model_key == "hybrid_joint_stribeck":
        return HybridJointStribeckODE(variant_name=nn_variant)
    if model_key == "hybrid_frozen":
        if frozen_phys_params is None:
            raise ValueError("frozen_phys_params required for hybrid_frozen")
        return HybridFrozenPhysODE(frozen_phys_params=frozen_phys_params, variant_name=nn_variant)
    if model_key == "hybrid_frozen_stribeck":
        if frozen_phys_params is None:
            raise ValueError("frozen_phys_params required for hybrid_frozen_stribeck")
        return HybridFrozenStribeckPhysODE(frozen_phys_params=frozen_phys_params, variant_name=nn_variant)
    raise ValueError(f"Unsupported model key '{model_key}'")


def model_label(model_key: str, nn_variant: str) -> str:
    if uses_nn_variant(model_key):
        return f"{model_key}__{nn_variant}"
    return model_key


def save_model_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    model_key: str,
    nn_variant: str,
    run_idx: int,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_key": model_key,
        "nn_variant": nn_variant,
        "run_idx": int(run_idx),
        "seed": int(seed),
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }

    if isinstance(model, (HybridFrozenPhysODE, HybridFrozenStribeckPhysODE)):
        payload["frozen_phys_params"] = model.frozen_phys_params()

    if isinstance(model, ContinuousTimeESN):
        payload["esn_kwargs"] = {
            "reservoir_dim": model.reservoir_dim,
            "state_dim": model.state_dim,
            "input_dim": model.input_dim,
        }

    torch.save(payload, out)


def load_model_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    model_key = ckpt["model_key"]
    nn_variant = ckpt.get("nn_variant", "base")

    frozen_phys_params = ckpt.get("frozen_phys_params")
    esn_kwargs = ckpt.get("esn_kwargs")
    model = build_model(
        model_key,
        nn_variant=nn_variant,
        frozen_phys_params=frozen_phys_params,
        esn_kwargs=esn_kwargs,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    meta = {
        "model_key": model_key,
        "nn_variant": nn_variant,
        "run_idx": int(ckpt.get("run_idx", -1)),
        "seed": int(ckpt.get("seed", -1)),
        "extra": ckpt.get("extra", {}),
    }
    if frozen_phys_params is not None:
        meta["frozen_phys_params"] = frozen_phys_params
    if esn_kwargs is not None:
        meta["esn_kwargs"] = esn_kwargs
    return model, meta


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
