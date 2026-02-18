"""PyTorch Dataset wrappers for windowed training and full-sequence
evaluation.

* :class:`WindowedTrainDataset` — draws random sub-windows of a fixed
  length from the full signal.  A new random offset is sampled on every
  call to ``__getitem__``, giving the model a different view each epoch.
* :class:`FullSequenceDataset` — returns the entire signal in one item.
  Used for validation and test evaluation where we want deterministic,
  full-horizon predictions.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class WindowedTrainDataset(TorchDataset):
    """Yields random sub-windows of a fixed length.

    Parameters
    ----------
    u : np.ndarray
        Input signal  (1-D or 2-D with shape ``[T, ...]``).
    y : np.ndarray
        Output signal (same leading dimension as *u*).
    window_size : int
        Number of samples per window.
    samples_per_epoch : int | None
        Virtual dataset length.  Defaults to ``T // window_size``
        (non-overlapping coverage count).
    """

    def __init__(
        self,
        u: np.ndarray,
        y: np.ndarray,
        window_size: int,
        samples_per_epoch: int | None = None,
    ) -> None:
        super().__init__()
        self._u = np.asarray(u, dtype=np.float32)
        self._y = np.asarray(y, dtype=np.float32)
        if self._u.shape[0] != self._y.shape[0]:
            raise ValueError(
                f"u and y must have same length, got {self._u.shape[0]} "
                f"and {self._y.shape[0]}"
            )
        self._T = self._u.shape[0]
        self._window = min(window_size, self._T)
        self._len = samples_per_epoch or max(1, self._T // self._window)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, _index: int):
        max_start = self._T - self._window
        start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        end = start + self._window
        return (
            torch.from_numpy(self._u[start:end]),
            torch.from_numpy(self._y[start:end]),
        )


class FullSequenceDataset(TorchDataset):
    """Returns the full signal as a single sample.

    Parameters
    ----------
    u : np.ndarray
        Input signal.
    y : np.ndarray
        Output signal.
    """

    def __init__(self, u: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        self._u = torch.from_numpy(np.asarray(u, dtype=np.float32))
        self._y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        if self._u.shape[0] != self._y.shape[0]:
            raise ValueError(
                f"u and y must have same length, got {self._u.shape[0]} "
                f"and {self._y.shape[0]}"
            )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int):
        return self._u, self._y
