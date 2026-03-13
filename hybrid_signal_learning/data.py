from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

import bab_datasets as nod


@dataclass(slots=True)
class ExperimentData:
    name: str
    u: np.ndarray
    y: np.ndarray
    y_ref: np.ndarray
    y_dot: np.ndarray
    y_sim: np.ndarray
    t: np.ndarray
    Ts: float
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(slots=True)
class TrainRolloutData:
    u: np.ndarray
    y_sim: np.ndarray
    t: np.ndarray
    Ts: float
    segments: list[tuple[str, int, int]]


def is_core_dataset(name: str) -> bool:
    return ("multisine" in name) or ("random_steps" in name)


def list_datasets(include: Iterable[str] | None = None) -> list[str]:
    available = nod.list_experiments()
    if include is None:
        return available

    include_set = set(include)
    filtered = [name for name in available if name in include_set]
    missing = include_set.difference(filtered)
    if missing:
        raise ValueError(f"Requested datasets not available: {sorted(missing)}")
    return filtered


def load_protocol2_datasets(
    *,
    y_dot_method: str = "central",
    resample_factor: int = 50,
    zoom_last_n: int = 200,
    include: Iterable[str] | None = None,
) -> dict[str, ExperimentData]:
    """
    Protocol 2 only:
    - core datasets (multisine/random_steps): temporal 50/50 train/test split
    - non-core datasets: test-only
    """

    dataset_names = list_datasets(include=include)
    data_map: dict[str, ExperimentData] = {}

    for name in dataset_names:
        loaded = nod.load_experiment(
            name,
            preprocess=True,
            plot=False,
            end_idx=None,
            resample_factor=resample_factor,
            zoom_last_n=zoom_last_n,
            y_dot_method=y_dot_method,
        )
        u, y, y_ref, y_dot = loaded
        Ts = float(loaded.sampling_time)

        n = len(u)
        split_i = n // 2

        if is_core_dataset(name):
            train_idx = np.arange(0, split_i, dtype=int)
            test_idx = np.arange(split_i, n, dtype=int)
        else:
            train_idx = np.array([], dtype=int)
            test_idx = np.arange(0, n, dtype=int)

        y_sim = np.column_stack([y, y_dot])
        t = np.arange(n, dtype=float) * Ts

        data_map[name] = ExperimentData(
            name=name,
            u=np.asarray(u, dtype=float),
            y=np.asarray(y, dtype=float),
            y_ref=np.asarray(y_ref, dtype=float),
            y_dot=np.asarray(y_dot, dtype=float),
            y_sim=np.asarray(y_sim, dtype=float),
            t=np.asarray(t, dtype=float),
            Ts=Ts,
            train_idx=train_idx,
            test_idx=test_idx,
        )

    return data_map


def build_train_rollout_data(data_map: dict[str, ExperimentData]) -> TrainRolloutData:
    """
    Concatenate only train partitions of protocol-2 datasets, preserving boundaries in `segments`.
    """

    if not data_map:
        raise ValueError("Empty dataset map")

    train_u_parts: list[np.ndarray] = []
    train_y_parts: list[np.ndarray] = []
    segments: list[tuple[str, int, int]] = []
    ts_values: list[float] = []

    cursor = 0
    for name, ds in data_map.items():
        if ds.train_idx.size == 0:
            continue

        u_part = ds.u[ds.train_idx]
        y_part = ds.y_sim[ds.train_idx]

        if u_part.size == 0:
            continue

        train_u_parts.append(u_part)
        train_y_parts.append(y_part)
        seg_len = int(len(u_part))
        segments.append((name, cursor, cursor + seg_len))
        cursor += seg_len
        ts_values.append(ds.Ts)

    if not train_u_parts:
        raise RuntimeError("No train samples found. Protocol-2 split produced empty training set.")

    Ts = ts_values[0]
    if not np.allclose(np.asarray(ts_values), Ts, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"Inconsistent sampling times in training partitions: {ts_values}")

    u = np.concatenate(train_u_parts, axis=0)
    y_sim = np.concatenate(train_y_parts, axis=0)
    t = np.arange(len(u), dtype=float) * Ts

    return TrainRolloutData(
        u=np.asarray(u, dtype=float),
        y_sim=np.asarray(y_sim, dtype=float),
        t=np.asarray(t, dtype=float),
        Ts=float(Ts),
        segments=segments,
    )


def compute_valid_start_indices(
    segments: list[tuple[str, int, int]],
    k_steps: int,
) -> np.ndarray:
    valid: list[int] = []
    for _, s0, s1 in segments:
        seg_len = s1 - s0
        if seg_len <= k_steps:
            continue
        valid.extend(range(s0, s1 - k_steps))

    starts = np.asarray(valid, dtype=int)
    if starts.size == 0:
        raise RuntimeError(
            f"No valid k-step starts. Check segments ({len(segments)}) and k_steps={k_steps}."
        )
    return starts


def split_label(ds: ExperimentData, idx: np.ndarray) -> str:
    if idx.size == 0:
        return "none"
    if idx.size == len(ds.t):
        return "all"
    return f"{idx[0]}:{idx[-1]} ({idx.size} samples)"


def summarize_protocol2(data_map: dict[str, ExperimentData]) -> str:
    lines = ["Protocol-2 split summary:"]
    for name, ds in data_map.items():
        lines.append(
            f"- {name}: train={split_label(ds, ds.train_idx)}, test={split_label(ds, ds.test_idx)}"
        )
    return "\n".join(lines)
