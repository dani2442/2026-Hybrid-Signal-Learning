"""Shared constants for model training."""

DEFAULT_GRAD_CLIP: float = 1.0
"""Default gradient clipping norm for most training loops."""

SHOOTING_GRAD_CLIP: float = 10.0
"""Gradient clipping norm for multiple-shooting training."""

NORM_EPS: float = 1e-8
"""Epsilon guard for z-score normalisation denominators."""
