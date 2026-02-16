"""Dataset container for system identification data.

This module provides a thin data container.  All loading logic lives in
:mod:`src.data.loaders` and preprocessing in :mod:`src.data.preprocessing`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Dataset:
    """Immutable container for a single experiment's time-series data.

    Parameters
    ----------
    t : np.ndarray
        Time vector.
    u : np.ndarray
        Input (control) signal.
    y : np.ndarray
        Output (response) signal.
    y_ref : np.ndarray | None
        Reference output if available.
    y_dot : np.ndarray | None
        Estimated derivative of *y*.
    y_filt : np.ndarray | None
        Filtered output.
    trigger : np.ndarray | None
        Trigger channel.
    name : str
        Human-readable identifier.
    sampling_rate : float
        Sampling frequency [Hz].
    """

    t: np.ndarray
    u: np.ndarray
    y: np.ndarray
    y_ref: Optional[np.ndarray] = None
    y_dot: Optional[np.ndarray] = None
    y_filt: Optional[np.ndarray] = None
    trigger: Optional[np.ndarray] = None
    name: str = "unnamed"
    sampling_rate: float = 1.0

    # ── derived properties ────────────────────────────────────────────

    @property
    def n_samples(self) -> int:
        return len(self.t)

    @property
    def dt(self) -> float:
        if len(self.t) < 2:
            return 1.0 / self.sampling_rate
        return float(np.median(np.diff(self.t)))

    @property
    def arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(t, u, y)`` tuple."""
        return self.t, self.u, self.y

    # ── splitting ─────────────────────────────────────────────────────

    def split(
        self, train_ratio: float = 0.7
    ) -> Tuple["Dataset", "Dataset"]:
        """Split into train / test."""
        n = self.n_samples
        idx = int(n * train_ratio)
        return self._slice(0, idx), self._slice(idx, n)

    def train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple["Dataset", "Dataset", "Dataset"]:
        """Split into train / val / test."""
        n = self.n_samples
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            self._slice(0, n_train),
            self._slice(n_train, n_train + n_val),
            self._slice(n_train + n_val, n),
        )

    # ── helpers ───────────────────────────────────────────────────────

    def _slice(self, start: int, end: int) -> "Dataset":
        """Return a sub-dataset for index range ``[start, end)``."""
        def _s(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            return arr[start:end] if arr is not None else None

        return Dataset(
            t=self.t[start:end],
            u=self.u[start:end],
            y=self.y[start:end],
            y_ref=_s(self.y_ref),
            y_dot=_s(self.y_dot),
            y_filt=_s(self.y_filt),
            trigger=_s(self.trigger),
            name=self.name,
            sampling_rate=self.sampling_rate,
        )

    def create_time_vector(self, n: Optional[int] = None) -> np.ndarray:
        """Generate a uniform time vector starting at zero."""
        n = n or self.n_samples
        return np.linspace(0, (n - 1) * self.dt, n)

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name!r}, n={self.n_samples}, "
            f"dt={self.dt:.4f}, fs={self.sampling_rate:.1f})"
        )


# ─────────────────────────────────────────────────────────────────────
# Multi-dataset wrapper
# ─────────────────────────────────────────────────────────────────────

class DatasetCollection:
    """Convenience wrapper around a list of :class:`Dataset` objects.

    Supports iteration, indexing, and bulk operations.

    Parameters
    ----------
    datasets : list[Dataset]
        Individual datasets.
    """

    def __init__(self, datasets: List[Dataset]) -> None:
        if not datasets:
            raise ValueError("DatasetCollection requires at least one dataset.")
        self._datasets = list(datasets)

    # ── class methods ─────────────────────────────────────────────────

    @classmethod
    def from_bab_experiments(cls, names: List[str], **kwargs) -> "DatasetCollection":
        """Load multiple BAB experiments into a collection.

        Parameters
        ----------
        names : list[str]
            BAB experiment keys or aliases.
        **kwargs
            Forwarded to :func:`loaders.from_bab_experiment`.
        """
        from .loaders import from_bab_experiment
        return cls([from_bab_experiment(n, **kwargs) for n in names])

    # ── list-like API ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._datasets)

    def __getitem__(self, index: int) -> Dataset:
        return self._datasets[index]

    def __iter__(self):
        return iter(self._datasets)

    @property
    def names(self) -> List[str]:
        return [d.name for d in self._datasets]

    # ── bulk helpers ──────────────────────────────────────────────────

    def train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[List["Dataset"], List["Dataset"], List["Dataset"]]:
        """Split every dataset and return three lists."""
        trains, vals, tests = [], [], []
        for ds in self._datasets:
            tr, va, te = ds.train_val_test_split(train_ratio, val_ratio)
            trains.append(tr)
            vals.append(va)
            tests.append(te)
        return trains, vals, tests

    def as_train_tuples(
        self,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return ``[(u, y), ...]`` for all datasets — suitable for
        ``BaseModel.fit(train_data=...)``."""
        return [(d.u, d.y) for d in self._datasets]

    def concatenated(self) -> Tuple[np.ndarray, np.ndarray]:
        """Concatenate all datasets into single ``(u, y)`` arrays."""
        u = np.concatenate([d.u for d in self._datasets])
        y = np.concatenate([d.y for d in self._datasets])
        return u, y

    def summary(self) -> Dict[str, int]:
        """Return ``{name: n_samples}`` mapping."""
        return {d.name: d.n_samples for d in self._datasets}

    def __repr__(self) -> str:
        items = ", ".join(f"{d.name}(n={d.n_samples})" for d in self._datasets)
        return f"DatasetCollection([{items}])"
