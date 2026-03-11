"""Unified asset registry and download helpers for BAB datasets.

Centralises all remote-asset URLs (sensors, videos, labels) and provides a
single ``ensure_asset`` helper that downloads files lazily into the correct
sub-directory of the project ``data/`` tree:

    data/
      sensors/   – .mat sensor recordings
      videos/    – video files (.MOV)
      labels/    – keypoint / theta / true-label CSVs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional
from urllib.error import URLError
from urllib.request import urlretrieve


# ─────────────────────────────────────────────────────────────────────
# Default root directories
# ─────────────────────────────────────────────────────────────────────

def _project_data_root() -> Path:
    """Return ``<project>/data`` using this file's location."""
    return Path(__file__).resolve().parents[2] / "data"


def sensors_dir(data_root: Optional[str] = None) -> Path:
    root = Path(data_root) if data_root else _project_data_root()
    return root / "sensors"


def videos_dir(data_root: Optional[str] = None) -> Path:
    root = Path(data_root) if data_root else _project_data_root()
    return root / "videos"


def labels_dir(data_root: Optional[str] = None) -> Path:
    root = Path(data_root) if data_root else _project_data_root()
    return root / "labels"


# ─────────────────────────────────────────────────────────────────────
# Sensor (.mat) registry
# ─────────────────────────────────────────────────────────────────────

SENSOR_REGISTRY: Dict[str, Dict[str, str]] = {
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

SENSOR_ALIASES: Dict[str, str] = {
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


# ─────────────────────────────────────────────────────────────────────
# Video registry
# ─────────────────────────────────────────────────────────────────────

VIDEO_REGISTRY: Dict[str, Dict[str, str]] = {
    "swept_sine": {
        "filename": "swept_sine.MOV",
        "url": "https://osf.io/download/6988598dcc6c050425c72a29/",
    },
    "multisine_05": {
        "filename": "multisine_01.MOV",
        "url": "https://osf.io/download/6wuxp/",
    },
    "multisine_06": {
        "filename": "multisine_02.MOV",
        "url": "https://osf.io/download/698859903a9f3f11e6c72c5b/",
    },
    "rampa_positiva": {
        "filename": "rampa_positiva.MOV",
        "url": "https://osf.io/download/6988598ecc6c050425c72a2b/",
    },
    "rampa_negativa": {
        "filename": "rampa_negativa.MOV",
        "url": "https://osf.io/download/698859902f1925341e85bb72/",
    },
    "random_steps_01": {
        "filename": "random_steps_W.MOV",
        "url": "https://osf.io/download/6988598efa846778d990453a/",
    },
    "random_steps_02": {
        "filename": "random_steps_X.MOV",
        "url": "https://osf.io/download/698859902f1925341e85bb74/",
    },
    "random_steps_03": {
        "filename": "random_steps_Y.MOV",
        "url": "https://osf.io/download/wujtg/",
    },
    "random_steps_04": {
        "filename": "random_steps_Z.MOV",
        "url": "https://osf.io/download/69885990fa846778d990453c/",
    },
}

# Dataset key → video base name (without extension).
DATASET_VIDEO_MAP: Dict[str, str] = {
    "multisine_05": "multisine_01",
    "multisine_06": "multisine_02",
    "random_steps_01": "random_steps_W",
    "random_steps_02": "random_steps_X",
    "random_steps_03": "random_steps_Y",
    "random_steps_04": "random_steps_Z",
}

# Manually annotated LED-on frame indices (trigger sync).
VIDEO_LED_FRAME_MAP: Dict[str, int] = {
    "swept_sine": 291,
    "rampa_positiva": 415,
    "rampa_negativa": 266,
    "random_steps_W": 273,
    "random_steps_X": 375,
    "random_steps_Y": 238,
    "random_steps_Z": 288,
    "multisine_01": 279,
    "multisine_02": 260,
}


# ─────────────────────────────────────────────────────────────────────
# Label CSV registry
# ─────────────────────────────────────────────────────────────────────

KEYPOINT_LABEL_REGISTRY: Dict[str, Dict[str, str]] = {}

THETA_LABEL_REGISTRY: Dict[str, Dict[str, str]] = {}

TRUE_LABEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "swept_sine": {
        "filename": "swept_sine_true_labels.csv",
    },
    "multisine_05": {
        "filename": "multisine_05_true_labels.csv",
    },
    "multisine_06": {
        "filename": "multisine_06_true_labels.csv",
    },
    "rampa_positiva": {
        "filename": "rampa_positiva_true_labels.csv",
    },
    "rampa_negativa": {
        "filename": "rampa_negativa_true_labels.csv",
    },
    "random_steps_01": {
        "filename": "random_steps_01_true_labels.csv",
    },
    "random_steps_02": {
        "filename": "random_steps_02_true_labels.csv",
    },
    "random_steps_03": {
        "filename": "random_steps_03_true_labels.csv",
    },
    "random_steps_04": {
        "filename": "random_steps_04_true_labels.csv",
    },
}


# ─────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────

def _download(url: str, dest: Path, desc: str = "") -> None:
    """Download *url* to *dest* with optional tqdm progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = desc or dest.name

    try:
        from tqdm.auto import tqdm  # type: ignore

        pbar = None

        def _hook(b: int, bs: int, tot: int) -> None:
            nonlocal pbar
            if pbar is None:
                pbar = tqdm(total=tot, unit="B", unit_scale=True,
                            desc=f"Downloading {label}")
            pbar.update(max(0, b * bs - pbar.n))

        urlretrieve(url, str(dest), reporthook=_hook)
        if pbar is not None:
            pbar.close()
    except ImportError:
        print(f"Downloading {label} …")
        urlretrieve(url, str(dest))
        print("Done.")


# ─────────────────────────────────────────────────────────────────────
# Public ensure_* helpers
# ─────────────────────────────────────────────────────────────────────

def resolve_sensor_name(name: str) -> str:
    """Resolve sensor dataset aliases to canonical keys."""
    if name in SENSOR_REGISTRY:
        return name
    return SENSOR_ALIASES.get(name, name)


def resolve_video_name(
    dataset_name: str,
    override_map: Optional[Dict[str, str]] = None,
) -> str:
    """Map a sensor dataset key to the corresponding video base name."""
    if override_map and dataset_name in override_map:
        return override_map[dataset_name]
    if dataset_name in DATASET_VIDEO_MAP:
        return DATASET_VIDEO_MAP[dataset_name]
    return dataset_name


def resolve_led_frame(video_name: str, override: Optional[int] = None) -> int:
    """Resolve LED-on frame index for a video."""
    if override is not None:
        return int(override)
    return int(VIDEO_LED_FRAME_MAP.get(video_name, 0))


def ensure_sensor(dataset_name: str, *, data_root: Optional[str] = None) -> Path:
    """Return path to the sensor .mat file, downloading if absent."""
    resolved = resolve_sensor_name(dataset_name)
    if resolved not in SENSOR_REGISTRY:
        available = ", ".join(sorted(SENSOR_REGISTRY.keys()))
        raise ValueError(f"Unknown sensor dataset '{dataset_name}'. Available: {available}")
    entry = SENSOR_REGISTRY[resolved]
    dest = sensors_dir(data_root) / entry["filename"]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            _download(entry["url"], dest, desc=entry["filename"])
        except (URLError, OSError) as exc:
            raise RuntimeError(
                f"Failed to download {entry['filename']} from {entry['url']}"
            ) from exc
    return dest


def ensure_video(
    dataset_name: str,
    *,
    video_path: Optional[str] = None,
    data_root: Optional[str] = None,
) -> Path:
    """Return path to the video file, downloading from OSF if absent."""
    if video_path:
        p = Path(video_path)
        if not p.exists():
            raise FileNotFoundError(f"Video not found: {p}")
        return p

    if dataset_name not in VIDEO_REGISTRY:
        raise ValueError(
            f"No auto-download entry for dataset '{dataset_name}'. "
            "Pass video_path explicitly."
        )
    entry = VIDEO_REGISTRY[dataset_name]
    dest = videos_dir(data_root) / entry["filename"]
    if not dest.exists():
        print(f"Video not in {dest.parent} – downloading from OSF …")
        _download(entry["url"], dest, desc=entry["filename"])
    return dest


def ensure_label(
    registry: Dict[str, Dict[str, str]],
    dataset_name: str,
    *,
    data_root: Optional[str] = None,
) -> Optional[Path]:
    """Return path to a label CSV, downloading if absent. None if unavailable."""
    if dataset_name not in registry:
        return None
    entry = registry[dataset_name]
    dest = labels_dir(data_root) / entry["filename"]
    if not dest.exists():
        url = entry.get("url")
        if not url:
            return None
        print(f"Downloading {entry['filename']} …")
        _download(url, dest)
    return dest


def ensure_keypoint_labels(dataset_name: str, **kw) -> Optional[Path]:
    return ensure_label(KEYPOINT_LABEL_REGISTRY, dataset_name, **kw)


def ensure_theta_labels(dataset_name: str, **kw) -> Optional[Path]:
    return ensure_label(THETA_LABEL_REGISTRY, dataset_name, **kw)


def ensure_true_labels(dataset_name: str, **kw) -> Optional[Path]:
    return ensure_label(TRUE_LABEL_REGISTRY, dataset_name, **kw)


def resolve_video_path(
    dataset_name: str,
    *,
    video_dir: Optional[str] = None,
    video_path: Optional[str] = None,
    video_map: Optional[Dict[str, str]] = None,
    data_root: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve final video path and canonical video name.

    Tries, in order:
    1. Explicit *video_path*.
    2. Look in *video_dir* for ``<video_name>.*``.
    3. Auto-download via the registry and store in ``data/videos/``.
    """
    video_name = resolve_video_name(dataset_name, override_map=video_map)

    if video_path is not None:
        return str(video_path), video_name

    if video_dir is not None:
        vd = Path(video_dir)
        for ext in (".mp4", ".avi", ".mkv", ".mov", ".MOV"):
            candidate = vd / f"{video_name}{ext}"
            if candidate.exists():
                return str(candidate), video_name
        raise FileNotFoundError(
            f"No video found for '{video_name}' in {video_dir} "
            "(tried .mp4, .avi, .mkv, .mov, .MOV)"
        )

    # Auto-download
    path = ensure_video(dataset_name, data_root=data_root)
    return str(path), video_name
