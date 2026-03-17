"""Vision models: encoder, decoders, and the end-to-end wrapper.

Encoder A  — ``EncoderThetaNet``: ResNet-50 backbone → θ (position only).
Encoder B  — ``PoseResNet50``:    ResNet-50 + deconv head → heatmaps → keypoints → θ.
Decoder    — ``DecoderKeypointsMLP``: [θ, θ̇] state → keypoint coordinates.
Decoder    — ``DecoderFrameDeconv``: [θ, θ̇] state → RGB image (transpose conv).
Composite  — ``EncOdeDecModel``: encoder + finite-difference init + ODE dynamics + optional decoder.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..models.base import resolve_device


# ═══════════════════════════════════════════════════════════════════════
#  Encoder A: image → position regression (θ)
# ═══════════════════════════════════════════════════════════════════════

class EncoderThetaNet(nn.Module):
    """ResNet-50 backbone with a small MLP head predicting ``θ`` (position).

    Velocity ``θ̇`` is **not** predicted directly; it is obtained via
    finite differences of two consecutive encoder outputs in the
    composite ``EncOdeDecModel``.

    Parameters
    ----------
    pretrained:
        Use ImageNet-pretrained backbone weights.
    state_dim:
        Output dimensionality (default 1 for ``θ`` only).
    """

    def __init__(self, pretrained: bool = True, state_dim: int = 1) -> None:
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        # Remove the final FC layer; keep everything up to avgpool.
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 2048, 1, 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, state_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, 3, H, W)`` images to ``(B, state_dim)`` position vectors."""
        feat = self.features(x)
        return self.head(feat)


# ═══════════════════════════════════════════════════════════════════════
#  Encoder B: image → heatmaps → keypoints (PoseResNet50)
# ═══════════════════════════════════════════════════════════════════════

class PoseResNet50(nn.Module):
    """ResNet-50 backbone + deconv head producing keypoint heatmaps.

    Architecture mirrors DLC-style pose estimation: the backbone produces
    feature maps, three transposed-conv layers upsample them, and a final
    ``1×1`` conv produces *num_keypoints* heatmaps.

    Parameters
    ----------
    num_keypoints:
        Number of output heatmaps (default 2: beam_left, beam_right).
    pretrained:
        Use ImageNet-pretrained backbone weights.
    """

    def __init__(self, num_keypoints: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        # Keep layers up to (but not including) avgpool.
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # → (B, 2048, H/32, W/32)

        # Deconv head: 2048→256→256→256
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv2d(256, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return heatmaps ``(B, K, H', W')`` from images ``(B, 3, H, W)``."""
        feat = self.backbone(x)
        up = self.deconv(feat)
        return self.heatmap_head(up)


# ═══════════════════════════════════════════════════════════════════════
#  Decoder: state → keypoint coordinates
# ═══════════════════════════════════════════════════════════════════════

class DecoderKeypointsMLP(nn.Module):
    """Small MLP mapping state ``[θ, θ̇]`` to keypoint coordinates.

    Parameters
    ----------
    state_dim:
        Input dimensionality (default 2).
    output_dim:
        Number of scalars to predict (default 4: ``[x_l, y_l, x_r, y_r]``).
    hidden_dim:
        Hidden layer width.
    """

    def __init__(
        self,
        state_dim: int = 2,
        output_dim: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, state_dim)`` → ``(B, output_dim)``."""
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════
#  Decoder: state → RGB image
# ═══════════════════════════════════════════════════════════════════════

class DecoderFrameDeconv(nn.Module):
    """Transpose-conv decoder mapping state ``[θ, θ̇]`` to RGB frames.

    The decoder expands the state to a low-resolution latent map and upsamples
    it by four ``×2`` transpose-convolution stages (overall ``×16``).
    Therefore ``frame_height`` and ``frame_width`` must be divisible by 16.
    """

    def __init__(
        self,
        *,
        state_dim: int = 2,
        frame_height: int = 96,
        frame_width: int = 96,
        out_channels: int = 3,
        base_channels: int = 256,
    ) -> None:
        super().__init__()
        if frame_height % 16 != 0 or frame_width % 16 != 0:
            raise ValueError(
                "frame_height and frame_width must be divisible by 16 "
                f"(got {frame_height}x{frame_width})."
            )
        if base_channels < 32:
            raise ValueError("base_channels must be >= 32.")

        self.frame_height = int(frame_height)
        self.frame_width = int(frame_width)
        self.base_h = self.frame_height // 16
        self.base_w = self.frame_width // 16
        self.base_channels = int(base_channels)

        latent_dim = self.base_channels * self.base_h * self.base_w
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
            nn.ReLU(inplace=True),
        )

        c0 = self.base_channels
        c1 = c0 // 2
        c2 = c1 // 2
        c3 = c2 // 2
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c0, c1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c3, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, state_dim)`` → ``(B, 3, H, W)`` in ``[0, 1]``."""
        z = self.fc(x)
        z = z.reshape(x.shape[0], self.base_channels, self.base_h, self.base_w)
        return self.deconv(z)


# ═══════════════════════════════════════════════════════════════════════
#  End-to-end wrapper: Encoder → ODE → Decoder
# ═══════════════════════════════════════════════════════════════════════

class EncOdeDecModel(nn.Module):
    """Composite model: image pair → finite-difference init → ODE rollout → observation.

    This module composes an encoder, an ODE dynamics model (from the main
    library), and an optional decoder.  The initial state ``[θ, θ̇]`` is
    built from two consecutive frames with finite differences and then
    integrated by the ODE dynamics before optional decoding.

    Parameters
    ----------
    encoder:
        Maps an image tensor ``(B, 3, H, W)`` to position ``(B, 1)``.
    ode_func:
        A torchsde-compatible dynamics module (must expose ``f``, ``g``,
        and ``u_series`` / ``t_series`` / ``batch_start_times`` attributes).
    decoder:
        Optional module mapping state vectors to observation predictions.
    dt:
        Integration time step (also used for the finite-difference
        velocity computation).
    """

    def __init__(
        self,
        encoder: nn.Module,
        ode_func: nn.Module,
        decoder: Optional[nn.Module] = None,
        dt: float = 0.05,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.ode_func = ode_func
        self.decoder = decoder
        self.dt = dt

    # ── forward helpers ───────────────────────────────────────────────

    def encode(self, y0: torch.Tensor) -> torch.Tensor:
        """Encode a single image to position ``(B, 1)``."""
        return self.encoder(y0)

    def encode_initial_state(
        self, y0: torch.Tensor, y_prev: torch.Tensor
    ) -> torch.Tensor:
        """Encode two consecutive frames and compute initial ``[θ, θ̇]``.

        Parameters
        ----------
        y0:
            Current frame ``(B, 3, H, W)``.
        y_prev:
            Previous frame ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Initial state ``(B, 2)`` as ``[θ, θ̇]`` where
            ``θ̇ = (θ₀ − θ₋₁) / dt``.
        """
        theta_0 = self.encoder(y0)
        theta_prev = self.encoder(y_prev)
        theta_dot_0 = (theta_0 - theta_prev) / self.dt
        return torch.cat([theta_0, theta_dot_0], dim=-1)

    def rollout(
        self,
        x0: torch.Tensor,
        u_seq: torch.Tensor,
        k_steps: int,
        *,
        control_series: Optional[torch.Tensor] = None,
        start_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Integrate ODE dynamics from ``x0`` for ``k_steps``.

        Parameters
        ----------
        x0:
            Initial state ``(B, 2)``.
        u_seq:
            Control inputs ``(B, K, 1)`` or ``(K, 1)`` (broadcast).
        k_steps:
            Number of integration steps.
        control_series:
            Optional full control trajectory ``(N, 1)`` for batched subsequence
            rollout from a shared absolute timeline.
        start_idx:
            Optional batch start indices into ``control_series``.

        Returns
        -------
        torch.Tensor
            State trajectory ``(K, B, 2)``.
        """
        device = x0.device
        t_eval = torch.arange(k_steps, dtype=x0.dtype, device=device) * self.dt

        if control_series is not None and start_idx is not None:
            control_series = control_series.to(device)
            if control_series.ndim == 1:
                control_series = control_series.reshape(-1, 1)
            start_idx = start_idx.to(device=device, dtype=x0.dtype).reshape(-1, 1)
            if start_idx.shape[0] != x0.shape[0]:
                raise ValueError(
                    f"Batch mismatch: x0 has B={x0.shape[0]}, start_idx has B={start_idx.shape[0]}"
                )
            self.ode_func.u_series = control_series
            self.ode_func.t_series = (
                torch.arange(
                    control_series.shape[0],
                    dtype=x0.dtype,
                    device=device,
                )
                * self.dt
            )
            self.ode_func.batch_start_times = start_idx * self.dt

            x = x0
            trajectory = [x]
            for step in range(1, k_steps):
                dx = self.ode_func.f(t_eval[step - 1], x)
                x = x + dx * self.dt
                trajectory.append(x)
            self.ode_func.batch_start_times = None
            return torch.stack(trajectory, dim=0)

        # ``u_seq`` can be shared ``(K, 1)`` or per-sample ``(B, K, 1)``.
        # For per-sample controls we integrate each sample separately so every
        # batch element sees its own control path.
        if u_seq.ndim == 2:
            self.ode_func.u_series = u_seq.to(device)
            self.ode_func.t_series = t_eval
            self.ode_func.batch_start_times = None

            x = x0
            trajectory = [x]
            for step in range(1, k_steps):
                dx = self.ode_func.f(t_eval[step - 1], x)
                x = x + dx * self.dt
                trajectory.append(x)
            return torch.stack(trajectory, dim=0)

        if u_seq.ndim != 3:
            raise ValueError(f"u_seq must have shape (K, 1) or (B, K, 1); got {tuple(u_seq.shape)}")

        if u_seq.shape[0] != x0.shape[0]:
            raise ValueError(
                f"Batch mismatch: x0 has B={x0.shape[0]}, u_seq has B={u_seq.shape[0]}"
            )

        batch_traj = []
        for b in range(x0.shape[0]):
            self.ode_func.u_series = u_seq[b].to(device)
            self.ode_func.t_series = t_eval
            self.ode_func.batch_start_times = None

            x = x0[b : b + 1]
            trajectory_b = [x]
            for step in range(1, k_steps):
                dx = self.ode_func.f(t_eval[step - 1], x)
                x = x + dx * self.dt
                trajectory_b.append(x)
            batch_traj.append(torch.stack(trajectory_b, dim=0))  # (K, 1, 2)

        return torch.cat(batch_traj, dim=1)  # (K, B, 2)

    def decode(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Decode state sequence to observations.

        Parameters
        ----------
        x_seq:
            ``(K, B, 2)`` state trajectory.

        Returns
        -------
        torch.Tensor
            ``(K, B, out_dim)`` for vector decoders, or
            ``(K, B, C, H, W)`` for image decoders.
        """
        if self.decoder is None:
            raise RuntimeError("No decoder attached to this model.")
        K, B, D = x_seq.shape
        flat = x_seq.reshape(K * B, D)
        y_hat = self.decoder(flat)
        if y_hat.ndim == 2:
            return y_hat.reshape(K, B, -1)
        if y_hat.ndim == 4:
            c, h, w = y_hat.shape[1], y_hat.shape[2], y_hat.shape[3]
            return y_hat.reshape(K, B, c, h, w)
        raise RuntimeError(
            "Decoder output must be rank-2 (vector) or rank-4 (image), "
            f"got shape {tuple(y_hat.shape)}."
        )

    def forward(
        self,
        y0: torch.Tensor,
        u_seq: torch.Tensor,
        k_steps: int,
        *,
        y_prev: Optional[torch.Tensor] = None,
        control_series: Optional[torch.Tensor] = None,
        start_idx: Optional[torch.Tensor] = None,
        return_states: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Full forward: encode pair → finite-diff init → rollout → decode.

        Parameters
        ----------
        y0:
            Current frame ``(B, 3, H, W)``.
        u_seq:
            Control inputs ``(B, K, 1)`` or ``(K, 1)``.
        k_steps:
            Number of ODE integration steps.
        y_prev:
            Previous frame ``(B, 3, H, W)`` for finite-difference initial velocity.
            If *None*, falls back to using ``y0`` for both.
        control_series:
            Optional full control trajectory used with ``start_idx`` for
            batched subsequence rollout.
        start_idx:
            Optional batch start indices into ``control_series``.

        Returns a dict with keys:
          - ``"x0_hat"``: initial state ``(B, 2)`` as ``[θ, θ̇]``.
          - ``"x_seq_hat"``: predicted state trajectory ``(K, B, 2)``.
          - ``"y_seq_hat"``: decoded observations ``(K, B, out_dim)`` (if decoder exists).
        """
        if y_prev is None:
            y_prev = y0
        x0_hat = self.encode_initial_state(y0, y_prev)
        x_seq_hat = self.rollout(
            x0_hat,
            u_seq,
            k_steps,
            control_series=control_series,
            start_idx=start_idx,
        )
        out: Dict[str, torch.Tensor] = {
            "x0_hat": x0_hat,
            "x_seq_hat": x_seq_hat,
        }
        if self.decoder is not None:
            out["y_seq_hat"] = self.decode(x_seq_hat)
        return out
