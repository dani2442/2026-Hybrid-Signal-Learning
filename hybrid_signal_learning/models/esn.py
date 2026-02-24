"""Continuous-time Echo State Network."""

from __future__ import annotations

import torch
from torch import nn

from .base import InterpNeuralODEBase


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
