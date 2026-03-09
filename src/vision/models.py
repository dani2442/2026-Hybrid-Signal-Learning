"""Vision models: encoder, decoder, pose estimator, and end-to-end wrapper.

Encoder A  — ``EncoderThetaNet``: ResNet-50 backbone → 2-D state regression.
Encoder B  — ``PoseResNet50``:    ResNet-50 + deconv head → heatmaps → keypoints → θ.
Decoder    — ``DecoderKeypointsMLP``: state → keypoint coordinates.
Composite  — ``EncOdeDecModel``:  encoder + ODE dynamics + optional decoder.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..models.base import resolve_device


# ═══════════════════════════════════════════════════════════════════════
#  Encoder A: image → state regression (θ, θ̇)
# ═══════════════════════════════════════════════════════════════════════

class EncoderThetaNet(nn.Module):
    """ResNet-50 backbone with a small MLP head predicting ``[θ, θ̇]``.

    Parameters
    ----------
    pretrained:
        Use ImageNet-pretrained backbone weights.
    state_dim:
        Output dimensionality (default 2 for ``[θ, θ̇]``).
    """

    def __init__(self, pretrained: bool = True, state_dim: int = 2) -> None:
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        # Remove the final FC layer; keep everything up to avgpool.
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 2048, 1, 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, state_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, 3, H, W)`` images to ``(B, state_dim)`` state vectors."""
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
#  End-to-end wrapper: Encoder → ODE → Decoder
# ═══════════════════════════════════════════════════════════════════════

class EncOdeDecModel(nn.Module):
    """Composite model: image → state → ODE rollout → (optional) observation.

    This module composes an encoder, an ODE dynamics model (from the main
    library), and an optional decoder.  It is designed so that gradients
    flow end-to-end through ``encoder → odeint → decoder``.

    Parameters
    ----------
    encoder:
        Maps an image tensor to an initial state vector.
    ode_func:
        A torchsde-compatible dynamics module (must expose ``f``, ``g``,
        and ``u_series`` / ``t_series`` / ``batch_start_times`` attributes).
    decoder:
        Optional module mapping state vectors to observation predictions.
    dt:
        Integration time step.
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
        """Encode initial image(s) to state ``(B, 2)``."""
        return self.encoder(y0)

    def rollout(
        self,
        x0: torch.Tensor,
        u_seq: torch.Tensor,
        k_steps: int,
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

        Returns
        -------
        torch.Tensor
            State trajectory ``(K, B, 2)``.
        """
        device = x0.device
        t_eval = torch.arange(k_steps, dtype=x0.dtype, device=device) * self.dt

        # Prepare shared u for the dynamics module
        if u_seq.ndim == 3:
            # Use first batch element (all batches share same u in typical setup)
            u_flat = u_seq[0]
        else:
            u_flat = u_seq

        self.ode_func.u_series = u_flat.to(device)
        self.ode_func.t_series = t_eval
        self.ode_func.batch_start_times = None

        # Euler integration (differentiable, no BM overhead)
        x = x0
        trajectory = [x]
        for step in range(1, k_steps):
            dx = self.ode_func.f(t_eval[step - 1], x)
            x = x + dx * self.dt
            trajectory.append(x)
        return torch.stack(trajectory, dim=0)  # (K, B, 2)

    def decode(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Decode state sequence to observations.

        Parameters
        ----------
        x_seq:
            ``(K, B, 2)`` state trajectory.

        Returns
        -------
        torch.Tensor
            ``(K, B, out_dim)`` observation predictions.
        """
        if self.decoder is None:
            raise RuntimeError("No decoder attached to this model.")
        K, B, D = x_seq.shape
        flat = x_seq.reshape(K * B, D)
        y_hat = self.decoder(flat)
        return y_hat.reshape(K, B, -1)

    def forward(
        self,
        y0: torch.Tensor,
        u_seq: torch.Tensor,
        k_steps: int,
        *,
        return_states: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Full forward: encode → rollout → (optional) decode.

        Returns a dict with keys:
          - ``"x0_hat"``: encoded initial state ``(B, 2)``.
          - ``"x_seq_hat"``: predicted state trajectory ``(K, B, 2)``.
          - ``"y_seq_hat"``: decoded observations ``(K, B, out_dim)`` (if decoder exists).
        """
        x0_hat = self.encode(y0)
        x_seq_hat = self.rollout(x0_hat, u_seq, k_steps)
        out: Dict[str, torch.Tensor] = {
            "x0_hat": x0_hat,
            "x_seq_hat": x_seq_hat,
        }
        if self.decoder is not None:
            out["y_seq_hat"] = self.decode(x_seq_hat)
        return out
