"""Dataset loading from files and remote sources."""

from __future__ import annotations

import os
from typing import Dict, Optional
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np

from .preprocessing import (
    downsample_optional,
    estimate_dt_and_fs,
    estimate_y_dot,
    find_end_before_ref_zero,
    find_trigger_start,
    shift_time_to_zero,
    slice_optional,
)


# ─────────────────────────────────────────────────────────────────────
# BAB dataset registry
# ─────────────────────────────────────────────────────────────────────

BAB_DATASET_REGISTRY: Dict[str, Dict[str, str]] = {
    "rampa_positiva": {
        "filename": "01_rampa_positiva.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/01_rampa_positiva.mat",
    },
    "rampa_negativa": {
        "filename": "02_rampa_negativa.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/02_rampa_negativa.mat",
    },
    "random_steps_01": {
        "filename": "03_random_steps_01.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/03_random_steps_01.mat",
    },
    "random_steps_02": {
        "filename": "03_random_steps_02.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/03_random_steps_02.mat",
    },
    "random_steps_03": {
        "filename": "03_random_steps_03.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/03_random_steps_03.mat",
    },
    "random_steps_04": {
        "filename": "03_random_steps_04.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/03_random_steps_04.mat",
    },
    "swept_sine": {
        "filename": "04_swept_sine.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/04_swept_sine.mat",
    },
    "multisine_05": {
        "filename": "05_multisine_01.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/05_multisine_01.mat",
    },
    "multisine_06": {
        "filename": "06_multisine_02.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/06_multisine_02.mat",
    },
}

BAB_ALIASES: Dict[str, str] = {
    "01_rampa_positiva": "rampa_positiva",
    "02_rampa_negativa": "rampa_negativa",
    "03_random_steps_01": "random_steps_01",
    "03_random_steps_02": "random_steps_02",
    "03_random_steps_03": "random_steps_03",
    "03_random_steps_04": "random_steps_04",
    "04_swept_sine": "swept_sine",
    "05_multisine_01": "multisine_05",
    "06_multisine_02": "multisine_06",
}


def _resolve_bab_name(name: str) -> str:
    """Resolve BAB aliases to canonical keys."""
    if name in BAB_DATASET_REGISTRY:
        return name
    return BAB_ALIASES.get(name, name)


def list_bab_experiments() -> list[str]:
    """Return sorted canonical BAB experiment keys."""
    return sorted(BAB_DATASET_REGISTRY.keys())


def _build_dataset(
    t: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    name: str,
    y_ref: Optional[np.ndarray] = None,
    y_filt: Optional[np.ndarray] = None,
    trigger: Optional[np.ndarray] = None,
    y_dot_method: str = "savgol",
    savgol_window: int = 51,
    savgol_poly: int = 3,
    fallback_fs: float = 1.0,
):
    """Create a Dataset with consistent dt/fs/y_dot handling."""
    from .dataset import Dataset

    dt, fs = estimate_dt_and_fs(t, default_fs=fallback_fs)
    y_dot = estimate_y_dot(
        y,
        dt if dt > 0 else 1.0,
        method=y_dot_method,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
    )
    return Dataset(
        t=t, u=u, y=y,
        y_ref=y_ref, y_dot=y_dot, y_filt=y_filt,
        trigger=trigger, name=name, sampling_rate=fs,
    )


def from_mat(
    filepath: str,
    time_key: str = "time",
    u_key: str = "u",
    y_key: str = "y",
    y_ref_key: str = "yref",
    trigger_key: str = "trigger",
    y_filt_key: str = "yf",
):
    """Load dataset from a ``.mat`` file."""
    try:
        import scipy.io
    except ImportError as exc:
        raise ImportError("scipy required: pip install scipy") from exc

    data = scipy.io.loadmat(filepath)
    t = data[time_key].ravel()
    u = data[u_key].ravel()
    y = data[y_key].ravel()

    y_ref = None
    if y_ref_key in data and data[y_ref_key].size > 0:
        y_ref = data[y_ref_key].ravel()
    elif "ref" in data and data["ref"].size > 0:
        y_ref = data["ref"].ravel()

    trigger = data[trigger_key].ravel() if trigger_key in data and data[trigger_key].size > 0 else None
    y_filt = data[y_filt_key].ravel() if y_filt_key in data and data[y_filt_key].size > 0 else None

    return _build_dataset(
        t=t, u=u, y=y,
        name=os.path.basename(filepath),
        y_ref=y_ref, y_filt=y_filt, trigger=trigger,
    )


def from_url(url: str, save_path: Optional[str] = None, **kwargs):
    """Download and load dataset from URL."""
    filename = save_path or os.path.basename(url)
    if not os.path.exists(filename):
        urlretrieve(url, filename)
    return from_mat(filename, **kwargs)


def from_bab_experiment(
    name: str,
    preprocess: bool = True,
    end_idx: Optional[int] = None,
    resample_factor: int = 50,
    end_ref_tolerance: float = 1e-8,
    y_dot_method: str = "savgol",
    savgol_window: int = 51,
    savgol_poly: int = 3,
    data_dir: Optional[str] = None,
):
    """Load a BAB experiment by key or alias."""
    try:
        import scipy.io
    except ImportError as exc:
        raise ImportError("scipy required: pip install scipy") from exc

    resolved = _resolve_bab_name(name)
    if resolved not in BAB_DATASET_REGISTRY:
        available = ", ".join(list_bab_experiments())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    entry = BAB_DATASET_REGISTRY[resolved]
    local_data_dir = data_dir or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data")
    )
    os.makedirs(local_data_dir, exist_ok=True)

    filepath = os.path.join(local_data_dir, entry["filename"])
    if not os.path.exists(filepath):
        try:
            urlretrieve(entry["url"], filepath)
        except URLError as exc:
            raise RuntimeError(
                f"Failed to download {entry['filename']} from {entry['url']}"
            ) from exc

    data = scipy.io.loadmat(filepath)
    t = np.asarray(data["time"]).ravel()
    u = np.asarray(data["u"]).ravel()
    y = np.asarray(data["y"]).ravel()
    trigger = np.asarray(data["trigger"]).ravel() if "trigger" in data else None
    y_ref = np.asarray(data["ref"]).ravel() if "ref" in data else None
    y_filt = np.asarray(data["yf"]).ravel() if "yf" in data else None

    if not preprocess:
        return _build_dataset(
            t=t, u=u, y=y, y_ref=y_ref, y_filt=y_filt,
            trigger=trigger, name=resolved,
            y_dot_method=y_dot_method,
            savgol_window=savgol_window,
            savgol_poly=savgol_poly,
        )

    start_idx = find_trigger_start(trigger)
    processed_end = end_idx
    if processed_end is None:
        processed_end = find_end_before_ref_zero(y_ref, tolerance=end_ref_tolerance)
        if processed_end <= start_idx:
            processed_end = len(t)
    processed_end = min(int(processed_end), len(t))

    t = t[start_idx:processed_end]
    u = u[start_idx:processed_end]
    y = y[start_idx:processed_end]
    y_ref = slice_optional(y_ref, start_idx, processed_end)
    y_filt = slice_optional(y_filt, start_idx, processed_end)
    trigger = slice_optional(trigger, start_idx, processed_end)

    t = downsample_optional(t, resample_factor)
    u = downsample_optional(u, resample_factor)
    y = downsample_optional(y, resample_factor)
    y_ref = downsample_optional(y_ref, resample_factor)
    y_filt = downsample_optional(y_filt, resample_factor)
    trigger = downsample_optional(trigger, resample_factor)

    return _build_dataset(
        t=shift_time_to_zero(t),
        u=u, y=y, name=resolved,
        y_ref=y_ref, y_filt=y_filt, trigger=trigger,
        y_dot_method=y_dot_method,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
    )


def from_bab_experiments(names: list[str], **kwargs):
    """Load multiple BAB experiments into a :class:`DatasetCollection`.

    Parameters
    ----------
    names : list[str]
        BAB experiment keys or aliases.
    **kwargs
        Forwarded to :func:`from_bab_experiment`.
    """
    from .dataset import DatasetCollection

    if not names:
        raise ValueError("At least one dataset name is required.")
    return DatasetCollection([from_bab_experiment(name, **kwargs) for name in names])
