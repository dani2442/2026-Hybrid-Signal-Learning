"""Model definitions for hybrid signal learning.

This sub-package provides all model classes and helpers.  The public API
is fully backwards-compatible with the previous single-file ``models.py``.
"""

# ── base building blocks ──────────────────────────────────────────────
from .base import (
    InterpNeuralODEBase,
    NN_VARIANTS,
    NnVariantConfig,
    _build_selu_mlp,
    uses_nn_variant,
)

# ── existing ODE models ──────────────────────────────────────────────
from .physics import (
    LinearPhysODE,
    StribeckPhysODE,
    extract_linear_params,
    extract_stribeck_params,
)
from .blackbox import (
    AdaptiveBlackBoxODE,
    BlackBoxODE,
    StructuredBlackBoxODE,
)
from .esn import ContinuousTimeESN
from .hybrid import (
    HybridFrozenPhysODE,
    HybridFrozenStribeckPhysODE,
    HybridJointODE,
    HybridJointStribeckODE,
)

# ── new ODE model ────────────────────────────────────────────────────
from .ude import UDEODE

# ── new differential equation models (optional deps) ─────────────────
from .neural_sde import BlackBoxSDE

# ── new discrete models ──────────────────────────────────────────────
from .sequence import (
    GRUSeqModel,
    LSTMSeqModel,
    MambaSeqModel,
    SequenceModelBase,
    TCNSeqModel,
)
from .feedforward import FeedForwardNN

# ── factory / checkpoint / iteration ─────────────────────────────────
from .factory import (
    MODEL_KEYS,
    build_model,
    iter_model_specs,
    load_model_checkpoint,
    model_label,
    save_model_checkpoint,
    write_json,
)

__all__ = [
    # base
    "InterpNeuralODEBase",
    "NnVariantConfig",
    "NN_VARIANTS",
    "_build_selu_mlp",
    "uses_nn_variant",
    # physics
    "LinearPhysODE",
    "StribeckPhysODE",
    "extract_linear_params",
    "extract_stribeck_params",
    # blackbox ODE
    "BlackBoxODE",
    "StructuredBlackBoxODE",
    "AdaptiveBlackBoxODE",
    # ESN
    "ContinuousTimeESN",
    # hybrid
    "HybridJointODE",
    "HybridJointStribeckODE",
    "HybridFrozenPhysODE",
    "HybridFrozenStribeckPhysODE",
    # UDE (new)
    "UDEODE",
    # SDE (new)
    "BlackBoxSDE",
    # sequence (new)
    "SequenceModelBase",
    "GRUSeqModel",
    "LSTMSeqModel",
    "TCNSeqModel",
    "MambaSeqModel",
    # feedforward (new)
    "FeedForwardNN",
    # factory / helpers
    "MODEL_KEYS",
    "build_model",
    "iter_model_specs",
    "model_label",
    "save_model_checkpoint",
    "load_model_checkpoint",
    "write_json",
]
