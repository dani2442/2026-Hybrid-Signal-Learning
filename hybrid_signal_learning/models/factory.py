"""Model factory, checkpoint save/load, iteration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .base import NN_VARIANTS, NnVariantConfig, uses_nn_variant
from .blackbox import AdaptiveBlackBoxODE, BlackBoxODE, StructuredBlackBoxODE
from .esn import ContinuousTimeESN
from .feedforward import FeedForwardNN
from .hybrid import (
    HybridFrozenPhysODE,
    HybridFrozenStribeckPhysODE,
    HybridJointODE,
    HybridJointStribeckODE,
)
from .physics import LinearPhysODE, StribeckPhysODE
from .sequence import GRUSeqModel, LSTMSeqModel, MambaSeqModel, TCNSeqModel
from .ude import UDEODE

# ── Optional models (imported lazily to avoid hard dependency) ────────
_BlackBoxSDE = None


def _get_sde_cls():
    global _BlackBoxSDE
    if _BlackBoxSDE is None:
        from .neural_sde import BlackBoxSDE

        _BlackBoxSDE = BlackBoxSDE
    return _BlackBoxSDE


# ─────────────────────────────────────────────────────────────────────
# Model keys
# ─────────────────────────────────────────────────────────────────────

MODEL_KEYS = [
    # ── physics (no NN variant) ──
    "linear",
    "stribeck",
    # ── ODE black-box ──
    "blackbox",
    "structured_blackbox",
    "adaptive_blackbox",
    # ── ODE reservoir ──
    "ct_esn",
    # ── ODE hybrid ──
    "hybrid_joint",
    "hybrid_joint_stribeck",
    "hybrid_frozen",
    "hybrid_frozen_stribeck",
    # ── ODE universal differential equation (NEW) ──
    "ude",
    # ── stochastic differential equations (NEW) ──
    "neural_sde",
    # ── discrete sequence models (NEW) ──
    "gru",
    "lstm",
    "tcn",
    "mamba",
    # ── discrete feedforward (NEW) ──
    "feedforward_nn",
]


# ─────────────────────────────────────────────────────────────────────
# Iteration helpers
# ─────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────


def build_model(
    model_key: str,
    *,
    nn_variant: str = "base",
    frozen_phys_params: dict[str, float] | None = None,
    esn_kwargs: dict[str, Any] | None = None,
    feedforward_lag: int = 10,
) -> nn.Module:
    """Construct an (untrained) model by key.

    Parameters
    ----------
    model_key : str
        One of :data:`MODEL_KEYS`.
    nn_variant : str
        NN size variant (``"compact"`` / ``"base"`` / ``"wide"`` / ``"deep"``).
    frozen_phys_params : dict | None
        Required for ``hybrid_frozen*`` models.
    esn_kwargs : dict | None
        Extra kwargs forwarded to :class:`ContinuousTimeESN`.
    feedforward_lag : int
        Lag order for ``feedforward_nn``.
    """
    # ── physics (no NN) ──
    if model_key == "linear":
        return LinearPhysODE()
    if model_key == "stribeck":
        return StribeckPhysODE()

    # ── ODE black-box ──
    if model_key == "blackbox":
        return BlackBoxODE(variant_name=nn_variant)
    if model_key == "structured_blackbox":
        return StructuredBlackBoxODE(variant_name=nn_variant)
    if model_key == "adaptive_blackbox":
        return AdaptiveBlackBoxODE(variant_name=nn_variant)

    # ── ODE reservoir ──
    if model_key == "ct_esn":
        kw = esn_kwargs or {}
        return ContinuousTimeESN(**kw)

    # ── ODE hybrid ──
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

    # ── UDE (NEW) ──
    if model_key == "ude":
        return UDEODE(variant_name=nn_variant)

    # ── Neural SDE (NEW, optional dep) ──
    if model_key == "neural_sde":
        return _get_sde_cls()(variant_name=nn_variant)

    # ── Discrete sequence models (NEW) ──
    if model_key == "gru":
        return GRUSeqModel(variant_name=nn_variant)
    if model_key == "lstm":
        return LSTMSeqModel(variant_name=nn_variant)
    if model_key == "tcn":
        return TCNSeqModel(variant_name=nn_variant)
    if model_key == "mamba":
        return MambaSeqModel(variant_name=nn_variant)

    # ── Feedforward NN (NEW) ──
    if model_key == "feedforward_nn":
        return FeedForwardNN(variant_name=nn_variant, lag=feedforward_lag)

    raise ValueError(f"Unsupported model key '{model_key}'")


# ─────────────────────────────────────────────────────────────────────
# Label helper
# ─────────────────────────────────────────────────────────────────────


def model_label(model_key: str, nn_variant: str) -> str:
    if uses_nn_variant(model_key):
        return f"{model_key}__{nn_variant}"
    return model_key


# ─────────────────────────────────────────────────────────────────────
# Checkpoint save / load
# ─────────────────────────────────────────────────────────────────────


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

    if isinstance(model, FeedForwardNN):
        payload["feedforward_lag"] = model.lag

    torch.save(payload, out)


def load_model_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    model_key = ckpt["model_key"]
    nn_variant = ckpt.get("nn_variant", "base")

    frozen_phys_params = ckpt.get("frozen_phys_params")
    esn_kwargs = ckpt.get("esn_kwargs")
    feedforward_lag = ckpt.get("feedforward_lag", 10)

    model = build_model(
        model_key,
        nn_variant=nn_variant,
        frozen_phys_params=frozen_phys_params,
        esn_kwargs=esn_kwargs,
        feedforward_lag=feedforward_lag,
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


# ─────────────────────────────────────────────────────────────────────
# JSON helper (kept for backwards compat)
# ─────────────────────────────────────────────────────────────────────


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
