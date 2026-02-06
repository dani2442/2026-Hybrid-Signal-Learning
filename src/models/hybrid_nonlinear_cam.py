"""Nonlinear cam-bar-motor hybrid model solved with torchsde."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .base import BaseModel


class _NonlinearCamSDEFunc:
    """SDE drift for nonlinear cam-bar-motor model with zero diffusion."""

    noise_type = "diagonal"
    sde_type = "ito"
    _POSITIVE_PARAMS = {"R", "r", "e", "L", "I", "J", "k", "k_t", "R_M", "L_M"}

    def __init__(
        self,
        sampling_time: float,
        params: Dict[str, float],
        trainable_params: Iterable[str],
        device,
    ):
        import torch
        import torch.nn as nn

        self.dt = float(sampling_time)
        self._eps = 1e-8
        self._device = device
        self._trainable = set(trainable_params)

        self._raw_params: Dict[str, nn.Parameter] = {}
        self._const_params: Dict[str, torch.Tensor] = {}
        for name, value in params.items():
            if name in self._trainable:
                init_raw = (
                    self._inv_softplus(value)
                    if name in self._POSITIVE_PARAMS
                    else float(value)
                )
                self._raw_params[name] = nn.Parameter(
                    torch.tensor(init_raw, dtype=torch.float64, device=device)
                )
            else:
                self._const_params[name] = torch.tensor(
                    float(value), dtype=torch.float64, device=device
                )

        self._u_path = torch.zeros(2, 1, dtype=torch.float64, device=device)

    @staticmethod
    def _inv_softplus(x: float) -> float:
        x = max(float(x), 1e-8)
        if x > 30.0:
            return x
        return float(np.log(np.expm1(x)))

    def parameters(self):
        return list(self._raw_params.values())

    def train(self):
        return self

    def eval(self):
        return self

    def set_control(self, u_path):
        if u_path.ndim == 1:
            u_path = u_path.reshape(-1, 1)
        self._u_path = u_path

    def _decode_param(self, name: str):
        import torch.nn.functional as F

        if name in self._raw_params:
            raw = self._raw_params[name]
            if name in self._POSITIVE_PARAMS:
                return F.softplus(raw) + self._eps
            return raw
        return self._const_params[name]

    def _decoded_params(self):
        all_names = set(self._raw_params.keys()) | set(self._const_params.keys())
        return {name: self._decode_param(name) for name in all_names}

    def _u_at(self, t, batch_size):
        import torch

        idx = torch.clamp((t / self.dt).long(), min=0, max=self._u_path.shape[0] - 1)
        u_t = self._u_path[idx]
        if u_t.ndim == 1:
            u_t = u_t.unsqueeze(0)
        return u_t.expand(batch_size, -1)

    @staticmethod
    def _safe_div(num, den, eps: float = 1e-8):
        import torch

        den_safe = torch.where(torch.abs(den) < eps, torch.full_like(den, eps), den)
        return num / den_safe

    @staticmethod
    def _geometry_terms(theta, p):
        import torch

        Rp = p["R"] + p["r"]
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)

        inside = Rp**2 - (p["e"] ** 2) * (sin_t**2)
        S = torch.sqrt(torch.clamp(inside, min=1e-10))
        sin_phi = torch.clamp((p["e"] * sin_t) / Rp, -1.0 + 1e-8, 1.0 - 1e-8)
        cos_phi = torch.sqrt(torch.clamp(1.0 - sin_phi**2, min=1e-8))

        y_geom = S - p["e"] * cos_t - (Rp - p["e"])
        A = -(p["e"] ** 2) * sin_t * cos_t / S + p["e"] * sin_t
        B = (
            p["e"] * cos_t
            - (p["e"] ** 2) * (cos_t**2 - sin_t**2) / S
            - (p["e"] ** 4) * (sin_t**2) * (cos_t**2) / (S**3)
        )
        return y_geom, cos_phi, A, B

    def _theta_ddot(self, theta, omega, current, p):
        y_geom, cos_phi, A, B = self._geometry_terms(theta, p)
        inertial_coeff = (4.0 * p["I"]) / (p["L"] ** 2 * cos_phi)
        spring_coeff = (2.0 * p["k"]) / (p["L"] * cos_phi)
        m_eff = p["J"] + inertial_coeff * A
        rhs = (
            p["k_t"] * current
            - spring_coeff * (y_geom + p["delta"])
            - inertial_coeff * B * (omega**2)
        )
        return self._safe_div(rhs, m_eff)

    def f(self, t, y):
        import torch

        p = self._decoded_params()
        theta = y[:, 0]
        omega = y[:, 1]
        current = y[:, 2]
        u_t = self._u_at(t, y.shape[0])[:, 0]

        acc = self._theta_ddot(theta, omega, current, p)
        i_dot = (
            -(p["R_M"] / p["L_M"]) * current
            + (p["k_b"] / p["L_M"]) * omega
            + u_t / p["L_M"]
        )

        return torch.stack([omega, acc, i_dot], dim=-1)

    def g(self, t, y):
        import torch

        return torch.zeros_like(y)

    def decoded_parameter_dict(self) -> Dict[str, float]:
        params = self._decoded_params()
        return {name: float(value.detach().cpu().item()) for name, value in params.items()}


class HybridNonlinearCam(BaseModel):
    """
    Hybrid model for nonlinear cam-bar-motor dynamics.

    Uses torchsde for integration with zero diffusion and nn.Parameter trainables.
    """

    _POSITIVE_PARAMS = {"R", "r", "e", "L", "I", "J", "k", "k_t", "R_M", "L_M"}

    def __init__(
        self,
        sampling_time: float,
        R: float,
        r: float,
        e: float,
        L: float,
        I: float,
        J: float,
        k: float,
        delta: float,
        k_t: float,
        k_b: float,
        R_M: float,
        L_M: float,
        trainable_params: Iterable[str] = ("J", "k", "delta", "k_t", "k_b"),
        learning_rate: float = 2e-2,
        epochs: int = 600,
    ):
        super().__init__(nu=1, ny=2)
        if sampling_time <= 0:
            raise ValueError("sampling_time must be positive")

        self.sampling_time = float(sampling_time)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.trainable_params = tuple(trainable_params)

        self.params_: Dict[str, float] = {
            "R": float(R),
            "r": float(r),
            "e": float(e),
            "L": float(L),
            "I": float(I),
            "J": float(J),
            "k": float(k),
            "delta": float(delta),
            "k_t": float(k_t),
            "k_b": float(k_b),
            "R_M": float(R_M),
            "L_M": float(L_M),
        }
        unknown = set(self.trainable_params) - set(self.params_.keys())
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown trainable params: {names}")

        self.initial_current_: float = 0.0
        self.training_loss_: list[float] = []
        self.sde_func_ = None
        self._device = None

    def _simulate_state_torch(self, u_t, theta0, omega0, current0):
        try:
            import torch
            import torchsde
        except ImportError:
            raise ImportError("torchsde required. Install with: pip install torchsde")

        self.sde_func_.set_control(u_t)
        ts = torch.arange(
            u_t.shape[0], dtype=torch.float64, device=self._device
        ) * self.sampling_time
        x0 = torch.stack([theta0, omega0, current0]).reshape(1, 3)
        path = torchsde.sdeint(
            self.sde_func_,
            x0,
            ts,
            method="euler",
            dt=self.sampling_time,
        )
        return path[:, 0, :]

    def fit(self, u: np.ndarray, y: np.ndarray) -> "HybridNonlinearCam":
        """Fit selected parameters by minimizing free-run error with torchsde."""
        try:
            import torch
            import torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install torch")

        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        if len(u) != len(y):
            raise ValueError("u and y must have same length")
        if len(y) < self.max_lag + 2:
            raise ValueError("Need at least 4 samples")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sde_func_ = _NonlinearCamSDEFunc(
            sampling_time=self.sampling_time,
            params=self.params_,
            trainable_params=self.trainable_params,
            device=self._device,
        )

        u_t = torch.tensor(u, dtype=torch.float64, device=self._device).reshape(-1, 1)
        y_t = torch.tensor(y, dtype=torch.float64, device=self._device)
        theta0 = y_t[0]
        omega0 = (y_t[1] - y_t[0]) / self.sampling_time
        current0 = torch.tensor(
            self.initial_current_, dtype=torch.float64, device=self._device
        )

        optim_vars = self.sde_func_.parameters()
        self.training_loss_ = []
        if optim_vars:
            optimizer = optim.Adam(optim_vars, lr=self.learning_rate)
            for _ in range(self.epochs):
                optimizer.zero_grad()
                state_path = self._simulate_state_torch(
                    u_t=u_t, theta0=theta0, omega0=omega0, current0=current0
                )
                theta_hat = state_path[:, 0]
                loss = torch.mean((theta_hat[self.max_lag :] - y_t[self.max_lag :]) ** 2)
                loss.backward()
                optimizer.step()
                self.training_loss_.append(float(loss.detach().cpu().item()))

        self.params_.update(self.sde_func_.decoded_parameter_dict())
        self._is_fitted = True
        return self

    def _integrate_one_step(self, theta: float, omega: float, current: float, voltage: float):
        import torch

        u_step = torch.tensor(
            [[voltage], [voltage]], dtype=torch.float64, device=self._device
        )
        theta0 = torch.tensor(theta, dtype=torch.float64, device=self._device)
        omega0 = torch.tensor(omega, dtype=torch.float64, device=self._device)
        current0 = torch.tensor(current, dtype=torch.float64, device=self._device)
        state_path = self._simulate_state_torch(
            u_t=u_step, theta0=theta0, omega0=omega0, current0=current0
        )
        next_state = state_path[-1]
        return (
            float(next_state[0].detach().cpu().item()),
            float(next_state[2].detach().cpu().item()),
        )

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction using measured theta and internal current state."""
        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        if len(u) != len(y):
            raise ValueError("u and y must have same length")
        if len(y) < self.max_lag + 1:
            return np.array([])

        dt = self.sampling_time
        y_hat = np.zeros_like(y, dtype=float)
        y_hat[:2] = y[:2]

        current = self.initial_current_
        for k in range(1, len(y) - 1):
            theta_k = y[k]
            omega_k = (y[k] - y[k - 1]) / dt
            theta_next, current = self._integrate_one_step(
                theta=theta_k, omega=omega_k, current=current, voltage=u[k]
            )
            y_hat[k + 1] = theta_next

        return y_hat[self.max_lag :]

    def predict_free_run(self, u: np.ndarray, y_initial: np.ndarray) -> np.ndarray:
        """Free-run simulation using two initial angle samples."""
        import torch

        u = np.asarray(u, dtype=float).flatten()
        y_initial = np.asarray(y_initial, dtype=float).flatten()

        if len(y_initial) < self.max_lag:
            raise ValueError(f"Need at least {self.max_lag} initial outputs")
        if len(u) < self.max_lag:
            return np.array([])

        with torch.no_grad():
            u_t = torch.tensor(u, dtype=torch.float64, device=self._device).reshape(-1, 1)
            theta0 = torch.tensor(y_initial[0], dtype=torch.float64, device=self._device)
            omega0 = torch.tensor(
                (y_initial[1] - y_initial[0]) / self.sampling_time,
                dtype=torch.float64,
                device=self._device,
            )
            current0 = torch.tensor(
                self.initial_current_, dtype=torch.float64, device=self._device
            )
            state_path = self._simulate_state_torch(
                u_t=u_t, theta0=theta0, omega0=omega0, current0=current0
            )
            y_hat = state_path[:, 0]

        return y_hat.detach().cpu().numpy()[self.max_lag :]

    def parameters(self) -> Dict[str, float]:
        """Return current physical parameter values."""
        return dict(self.params_)

    def __repr__(self) -> str:
        trainable = ",".join(self.trainable_params)
        return (
            "HybridNonlinearCam("
            f"dt={self.sampling_time}, trainable=[{trainable}], epochs={self.epochs})"
        )
