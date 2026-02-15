"""Configuration dataclasses for all models.

Each model has a typed config that controls its hyperparameters.
Configs support serialisation to/from dicts for checkpointing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BaseConfig:
    """Shared fields for every model."""

    nu: int = 1
    ny: int = 1
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    verbose: bool = True
    device: str = "auto"          # "auto", "cpu", "cuda", "cuda:0", …
    seed: Optional[int] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_log_every: int = 1
    early_stopping_patience: Optional[int] = None

    # ── serialisation helpers ─────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert config to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BaseConfig":
        """Create config from a dict, ignoring unknown keys."""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ─────────────────────────────────────────────────────────────────────
# Classical / statistical
# ─────────────────────────────────────────────────────────────────────

@dataclass
class NARXConfig(BaseConfig):
    nu: int = 1
    ny: int = 1
    poly_order: int = 2
    selection_criteria: float = 0.01


@dataclass
class ARIMAConfig(BaseConfig):
    order: tuple = (1, 0, 1)
    nu: int = 0


@dataclass
class ExponentialSmoothingConfig(BaseConfig):
    trend: Optional[str] = "add"
    seasonal: Optional[str] = None
    seasonal_periods: Optional[int] = None
    nu: int = 0


# ─────────────────────────────────────────────────────────────────────
# Discrete ML — feed-forward
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RandomForestConfig(BaseConfig):
    nu: int = 5
    ny: int = 5
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42


@dataclass
class NeuralNetworkConfig(BaseConfig):
    nu: int = 5
    ny: int = 5
    hidden_layers: List[int] = field(default_factory=lambda: [80, 80, 80])
    activation: str = "selu"
    normalize: bool = False


# ─────────────────────────────────────────────────────────────────────
# Discrete ML — sequence
# ─────────────────────────────────────────────────────────────────────

@dataclass
class GRUConfig(BaseConfig):
    nu: int = 10
    ny: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class LSTMConfig(BaseConfig):
    nu: int = 10
    ny: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class TCNConfig(BaseConfig):
    nu: int = 10
    ny: int = 10
    num_channels: List[int] = field(default_factory=lambda: [64, 64, 64, 64])
    kernel_size: int = 3
    dropout: float = 0.1


@dataclass
class MambaConfig(BaseConfig):
    nu: int = 10
    ny: int = 10
    d_model: int = 64
    d_state: int = 16
    d_conv: int = 4
    n_layers: int = 2
    expand_factor: int = 2
    dropout: float = 0.1


# ─────────────────────────────────────────────────────────────────────
# Continuous-time
# ─────────────────────────────────────────────────────────────────────

@dataclass
class NeuralODEConfig(BaseConfig):
    state_dim: int = 1
    input_dim: int = 1
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64])
    solver: str = "euler"
    dt: float = 0.05
    sequence_length: int = 50
    sequences_per_epoch: int = 24
    activation: str = "selu"
    training_mode: str = "subsequence"


@dataclass
class NeuralSDEConfig(BaseConfig):
    state_dim: int = 1
    input_dim: int = 1
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64])
    diffusion_hidden_layers: List[int] = field(default_factory=lambda: [64, 64])
    solver: str = "euler"
    dt: float = 0.05
    sequence_length: int = 50
    sequences_per_epoch: int = 24


@dataclass
class NeuralCDEConfig(BaseConfig):
    hidden_dim: int = 32
    input_dim: int = 2
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64])
    interpolation: str = "cubic"
    solver: str = "rk4"
    rtol: float = 1e-4
    atol: float = 1e-5
    sequence_length: int = 50
    sequences_per_epoch: int = 24


# ─────────────────────────────────────────────────────────────────────
# Physics / hybrid
# ─────────────────────────────────────────────────────────────────────

@dataclass
class LinearPhysicsConfig(BaseConfig):
    dt: float = 0.05
    solver: str = "euler"
    epochs: int = 1000
    sequence_length: int = 50
    training_mode: str = "full"


@dataclass
class StribeckPhysicsConfig(BaseConfig):
    dt: float = 0.05
    solver: str = "euler"
    epochs: int = 1000
    sequence_length: int = 50
    training_mode: str = "full"


@dataclass
class HybridLinearBeamConfig(BaseConfig):
    dt: float = 0.05
    tau: float = 1.0
    estimate_delta: bool = True
    ridge: float = 1e-8
    epochs: int = 600
    learning_rate: float = 1e-2
    integration_substeps: int = 1


@dataclass
class HybridNonlinearCamConfig(BaseConfig):
    dt: float = 0.05
    epochs: int = 600
    learning_rate: float = 2e-2
    integration_substeps: int = 20
    # Physical parameters
    R: float = 0.015
    r: float = 0.005
    e: float = 0.005
    L: float = 0.2
    I: float = 0.01
    J: float = 0.001
    k: float = 100.0
    delta: float = 0.0
    k_t: float = 0.1
    k_b: float = 0.1
    R_M: float = 1.0
    L_M: float = 0.01
    trainable_params: List[str] = field(
        default_factory=lambda: ["J", "k", "delta", "k_t", "k_b"]
    )
    initial_current: float = 0.0


@dataclass
class UDEConfig(BaseConfig):
    dt: float = 0.05
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64])
    solver: str = "euler"
    epochs: int = 1000
    sequence_length: int = 50
    training_mode: str = "full"
    activation: str = "selu"


# ─────────────────────────────────────────────────────────────────────
# Black-box 2-D variants
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BlackboxODE2DConfig(BaseConfig):
    state_dim: int = 2
    input_dim: int = 1
    hidden_dim: int = 128
    solver: str = "euler"
    dt: float = 0.05
    k_steps: int = 20
    epochs: int = 5000
    batch_size: int = 128


@dataclass
class BlackboxSDE2DConfig(BaseConfig):
    state_dim: int = 2
    input_dim: int = 1
    hidden_dim: int = 128
    diffusion_hidden_dim: int = 64
    solver: str = "euler"
    dt: float = 0.05
    k_steps: int = 20
    epochs: int = 5000
    batch_size: int = 128


@dataclass
class BlackboxCDE2DConfig(BaseConfig):
    state_dim: int = 2
    input_dim: int = 1
    hidden_dim: int = 128
    solver: str = "euler"
    dt: float = 0.05
    k_steps: int = 20
    epochs: int = 5000
    batch_size: int = 128


# ─────────────────────────────────────────────────────────────────────
# Registry: model name → config class
# ─────────────────────────────────────────────────────────────────────

MODEL_CONFIGS: Dict[str, type] = {
    "narx": NARXConfig,
    "arima": ARIMAConfig,
    "exponential_smoothing": ExponentialSmoothingConfig,
    "random_forest": RandomForestConfig,
    "neural_network": NeuralNetworkConfig,
    "gru": GRUConfig,
    "lstm": LSTMConfig,
    "tcn": TCNConfig,
    "mamba": MambaConfig,
    "neural_ode": NeuralODEConfig,
    "neural_sde": NeuralSDEConfig,
    "neural_cde": NeuralCDEConfig,
    "linear_physics": LinearPhysicsConfig,
    "stribeck_physics": StribeckPhysicsConfig,
    "hybrid_linear_beam": HybridLinearBeamConfig,
    "hybrid_nonlinear_cam": HybridNonlinearCamConfig,
    "ude": UDEConfig,
    "vanilla_node_2d": BlackboxODE2DConfig,
    "structured_node": BlackboxODE2DConfig,
    "adaptive_node": BlackboxODE2DConfig,
    "vanilla_ncde_2d": BlackboxCDE2DConfig,
    "structured_ncde": BlackboxCDE2DConfig,
    "adaptive_ncde": BlackboxCDE2DConfig,
    "vanilla_nsde_2d": BlackboxSDE2DConfig,
    "structured_nsde": BlackboxSDE2DConfig,
    "adaptive_nsde": BlackboxSDE2DConfig,
}
