import os
import torch
import numpy as np

DEFAULT_PROXY_URL = "http://proxy.nhr.fau.de:80"
def ensure_proxy_env(proxy_url: str = DEFAULT_PROXY_URL) -> None:
    """Configure proxy environment variables when they are not set."""
    os.environ.setdefault("HTTP_PROXY", proxy_url)
    os.environ.setdefault("HTTPS_PROXY", proxy_url)


def resolve_device(device: str = "auto") -> str:
    """Resolve runtime device from an explicit value or auto-detection."""
    if device != "auto":
        return device

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"

    print("Using CPU")
    return "cpu"


def seed_all(seed: int) -> None:
    """Seed supported random number generators."""
    torch.manual_seed(seed)
    np.random.seed(seed)