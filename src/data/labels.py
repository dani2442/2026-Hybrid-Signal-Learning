"""Label CSV parsing utilities for BAB keypoint and theta data.

Supports three CSV formats:
- **Simple**: header row with ``beam_left_x, beam_left_y, …``
- **DLC multi-header**: 3-row DLC header (scorer / bodyparts / coords)
- **DLC CollectedData**: ``labeled-data, video_name, imgXXXX.png, …``
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict

import numpy as np


KEYPOINT_COLUMN_LABELS = ("beam_left_x", "beam_left_y", "beam_right_x", "beam_right_y")
SENSOR_STATE_LABELS = ("theta_deg", "theta_dot_deg_s")


def _float_or(value: str | None, default: float = np.nan) -> float:
    try:
        return float(value if value not in (None, "") else default)
    except ValueError:
        return default


def _index_of(values: list[str], name: str, *, default: int = -1) -> int:
    return values.index(name) if name in values else default


# ─────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────

def keypoints_to_theta(kp_left: np.ndarray, kp_right: np.ndarray) -> np.ndarray:
    """Compute beam angle θ (degrees) from left/right keypoint coordinates.

    θ = arctan2(y_right − y_left, x_right − x_left)
    """
    kp_left = np.atleast_2d(kp_left)
    kp_right = np.atleast_2d(kp_right)
    dx = kp_right[:, 0] - kp_left[:, 0]
    dy = kp_right[:, 1] - kp_left[:, 1]
    return np.degrees(np.arctan2(dy, dx)).squeeze()


def interpolate_missing_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Fill NaN gaps in keypoint coordinate trajectories via linear interpolation."""
    arr = np.asarray(keypoints, dtype=float).copy()
    if arr.ndim != 2:
        raise ValueError("keypoints must be a 2D array.")
    n = arr.shape[0]
    x = np.arange(n)
    for k in range(arr.shape[1]):
        y = arr[:, k]
        valid = np.isfinite(y)
        if not np.any(valid):
            continue
        arr[:, k] = np.interp(x, x[valid], y[valid])
    return arr


# ─────────────────────────────────────────────────────────────────────
# CSV parsers
# ─────────────────────────────────────────────────────────────────────

def parse_keypoint_labels_csv(
    csv_path: str, fps: float = 30.0
) -> Dict[str, np.ndarray]:
    """Parse keypoint labels from simple, DLC, or CollectedData CSV formats.

    Returns a dict with keys:
      ``frame``, ``t_s``, ``keypoints`` (N,4), ``theta_deg``, and
      optionally ``confidence_min``.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Keypoint labels CSV not found: {csv_path}")

    with path.open(newline="") as f:
        all_rows = [row for row in csv.reader(f) if any(c.strip() for c in row)]
    if not all_rows:
        raise ValueError(f"Empty labels CSV: {csv_path}")

    first = [c.strip().lower() for c in all_rows[0]]

    # Simple header (single row with column names)
    simple_header = any(
        k in first
        for k in ("beam_left_x", "beam_right_x", "theta_deg", "t_s")
    )
    if simple_header:
        return _parse_simple_keypoint_csv(csv_path, fps=fps)

    # Detect DLC CollectedData format: data rows start with "labeled-data"
    if len(all_rows) > 3:
        sample_row = [c.strip().lower() for c in all_rows[3]]
        if sample_row and sample_row[0] == "labeled-data":
            return _parse_dlc_collected_data_csv(all_rows, fps=fps)

    # Standard DLC multi-header
    return _parse_dlc_multiheader_csv(all_rows, fps=fps)


def _parse_simple_keypoint_csv(
    csv_path: str, fps: float
) -> Dict[str, np.ndarray]:
    rows: list[dict] = []
    with Path(csv_path).open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
            )

    if not rows:
        raise ValueError(f"No rows found in labels CSV: {csv_path}")

    n = len(rows)
    frame = np.arange(n, dtype=int)
    if "frame" in rows[0]:
        frame = np.asarray(
            [int(_float_or(r.get("frame"), i)) for i, r in enumerate(rows)],
            dtype=int,
        )

    t_s = frame.astype(float) / float(fps)
    if "t_s" in rows[0]:
        t_s = np.asarray(
            [_float_or(r.get("t_s"), i / float(fps)) for i, r in enumerate(rows)],
            dtype=float,
        )

    def _col(name: str) -> np.ndarray:
        return np.asarray([_float_or(r.get(name)) for r in rows], dtype=float)

    xl = _col("beam_left_x")
    yl = _col("beam_left_y")
    xr = _col("beam_right_x")
    yr = _col("beam_right_y")
    keypoints = np.column_stack([xl, yl, xr, yr])

    conf_min = None
    if "confidence_min" in rows[0]:
        conf_min = _col("confidence_min")
    elif "beam_left_likelihood" in rows[0] and "beam_right_likelihood" in rows[0]:
        conf_min = np.minimum(
            _col("beam_left_likelihood"), _col("beam_right_likelihood")
        )

    if "theta_deg" in rows[0]:
        theta = _col("theta_deg")
    else:
        theta = keypoints_to_theta(keypoints[:, :2], keypoints[:, 2:4]).astype(float)

    theta_dot = None
    if "theta_dot_deg_s" in rows[0]:
        theta_dot = _col("theta_dot_deg_s")

    out: Dict[str, np.ndarray] = {
        "frame": frame,
        "t_s": t_s,
        "keypoints": keypoints,
        "theta_deg": theta,
    }
    if theta_dot is not None:
        out["theta_dot_deg_s"] = theta_dot
    if conf_min is not None:
        out["confidence_min"] = conf_min
    return out


def _parse_dlc_multiheader_csv(
    rows: list[list[str]], fps: float
) -> Dict[str, np.ndarray]:
    if len(rows) < 4:
        raise ValueError("DLC-style CSV must contain 3 header rows plus data rows.")

    bodyparts = [(c or "").strip().lower() for c in rows[1]]
    coords = [(c or "").strip().lower() for c in rows[2]]
    col_names = [f"{bp}_{co}" if bp and co else "" for bp, co in zip(bodyparts, coords)]

    def _find(name: str) -> int:
        idx = _index_of(col_names, name)
        if idx < 0:
            raise ValueError(
                f"Column '{name}' not found in DLC labels CSV."
            )
        return idx

    idx_xl = _find("beam_left_x")
    idx_yl = _find("beam_left_y")
    idx_xr = _find("beam_right_x")
    idx_yr = _find("beam_right_y")
    idx_ll = _index_of(col_names, "beam_left_likelihood")
    idx_rl = _index_of(col_names, "beam_right_likelihood")

    data_rows = rows[3:]
    n = len(data_rows)
    frame = np.arange(n, dtype=int)
    t_s = frame.astype(float) / float(fps)

    def _safe_float(row: list[str], idx: int) -> float:
        return np.nan if idx < 0 or idx >= len(row) else _float_or(row[idx])

    xl = np.asarray([_safe_float(r, idx_xl) for r in data_rows], dtype=float)
    yl = np.asarray([_safe_float(r, idx_yl) for r in data_rows], dtype=float)
    xr = np.asarray([_safe_float(r, idx_xr) for r in data_rows], dtype=float)
    yr = np.asarray([_safe_float(r, idx_yr) for r in data_rows], dtype=float)
    keypoints = np.column_stack([xl, yl, xr, yr])
    theta = keypoints_to_theta(keypoints[:, :2], keypoints[:, 2:4]).astype(float)

    out: Dict[str, np.ndarray] = {
        "frame": frame,
        "t_s": t_s,
        "keypoints": keypoints,
        "theta_deg": theta,
    }
    if idx_ll >= 0 and idx_rl >= 0:
        ll = np.asarray([_safe_float(r, idx_ll) for r in data_rows], dtype=float)
        rl = np.asarray([_safe_float(r, idx_rl) for r in data_rows], dtype=float)
        out["confidence_min"] = np.minimum(ll, rl)
    return out


def _parse_dlc_collected_data_csv(
    rows: list[list[str]], fps: float
) -> Dict[str, np.ndarray]:
    """Parse DLC CollectedData format (hand-annotated sparse labels).

    Format::

        scorer,        ,           , Dani_F,      Dani_F,      Dani_F,      Dani_F
        bodyparts,     ,           , beam_left,   beam_left,   beam_right,  beam_right
        coords,        ,           , x,           y,           x,           y
        labeled-data,  video_name, imgXXXX.png,  xl,          yl,          xr,          yr
    """
    if len(rows) < 4:
        raise ValueError(
            "DLC CollectedData CSV must have 3 header rows plus data."
        )

    # Determine column offset: find first real bodypart in row 1
    bodyparts = [(c or "").strip().lower() for c in rows[1]]
    data_col_start = 0
    for i, bp in enumerate(bodyparts):
        if bp in ("beam_left", "beam_right"):
            data_col_start = i
            break

    # Build col_names from the bodypart + coord headers
    coords_row = [(c or "").strip().lower() for c in rows[2]]
    col_map: dict[str, int] = {}
    for i in range(data_col_start, min(len(bodyparts), len(coords_row))):
        bp = bodyparts[i]
        co = coords_row[i]
        if bp and co:
            col_map[f"{bp}_{co}"] = i

    for needed in ("beam_left_x", "beam_left_y", "beam_right_x", "beam_right_y"):
        if needed not in col_map:
            raise ValueError(
                f"Column '{needed}' not found in CollectedData CSV."
            )

    idx_xl = col_map["beam_left_x"]
    idx_yl = col_map["beam_left_y"]
    idx_xr = col_map["beam_right_x"]
    idx_yr = col_map["beam_right_y"]

    data_rows = rows[3:]
    frames_list: list[int] = []
    xl_list: list[float] = []
    yl_list: list[float] = []
    xr_list: list[float] = []
    yr_list: list[float] = []

    frame_re = re.compile(r"img(\d+)\.png", re.IGNORECASE)

    for row in data_rows:
        match = next(
            (m for cell in row[:data_col_start] if (m := frame_re.search(cell.strip()))),
            None,
        )
        if match is None:
            continue

        def _sf(idx: int) -> float:
            return np.nan if idx >= len(row) else _float_or(row[idx])

        frames_list.append(int(match.group(1)))
        xl_list.append(_sf(idx_xl))
        yl_list.append(_sf(idx_yl))
        xr_list.append(_sf(idx_xr))
        yr_list.append(_sf(idx_yr))

    if not frames_list:
        raise ValueError("No valid data rows found in CollectedData CSV.")

    frame = np.asarray(frames_list, dtype=int)
    t_s = frame.astype(float) / float(fps)
    keypoints = np.column_stack([xl_list, yl_list, xr_list, yr_list]).astype(float)
    theta = keypoints_to_theta(keypoints[:, :2], keypoints[:, 2:4]).astype(float)

    return {
        "frame": frame,
        "t_s": t_s,
        "keypoints": keypoints,
        "theta_deg": theta,
    }


def load_theta_csv(
    theta_csv_path: str, fps: float = 30.0
) -> Dict[str, np.ndarray]:
    """Load a theta CSV with ``t_s``/``theta_deg`` and optional ``theta_dot_deg_s``."""
    path = Path(theta_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Theta CSV not found: {theta_csv_path}")

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
            for r in reader
        ]
    if not rows:
        raise ValueError(f"Empty theta CSV: {theta_csv_path}")

    t_vals: list[float] = []
    th_vals: list[float] = []
    theta_dot_vals: list[float] | None = [] if "theta_dot_deg_s" in rows[0] else None
    for i, r in enumerate(rows):
        t_vals.append(_float_or(r.get("t_s"), i / float(fps)))
        th_vals.append(_float_or(r.get("theta_deg")))
        if theta_dot_vals is not None:
            theta_dot_vals.append(_float_or(r.get("theta_dot_deg_s")))

    out = {
        "t_s": np.asarray(t_vals, dtype=float),
        "theta_deg": np.asarray(th_vals, dtype=float),
    }
    if theta_dot_vals is not None:
        out["theta_dot_deg_s"] = np.asarray(theta_dot_vals, dtype=float)
    return out
