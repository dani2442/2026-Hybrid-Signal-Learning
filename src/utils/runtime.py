"""Runtime utilities for device resolution and reproducibility."""

import random

import numpy as np
import torch


def resolve_device(device: str = "auto") -> str:
    """Resolve runtime device from an explicit value or auto-detection."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def seed_all(seed: int) -> None:
    """Seed all supported random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
