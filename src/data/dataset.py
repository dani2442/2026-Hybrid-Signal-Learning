"""Dataset loading and preprocessing utilities."""

import os
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.request import urlretrieve

import numpy as np


@dataclass
class Dataset:
    """
    Container for system identification data.
    
    Attributes:
        t: Time vector
        u: Input signal
        y: Output signal
        y_ref: Reference signal (optional)
        name: Dataset name
        sampling_rate: Sampling frequency in Hz
    """
    t: np.ndarray
    u: np.ndarray
    y: np.ndarray
    y_ref: Optional[np.ndarray] = None
    name: str = ""
    sampling_rate: float = 1.0

    def __post_init__(self):
        """Validate data shapes."""
        self.t = np.asarray(self.t).flatten()
        self.u = np.asarray(self.u).flatten()
        self.y = np.asarray(self.y).flatten()
        
        if self.y_ref is not None:
            self.y_ref = np.asarray(self.y_ref).flatten()

        assert len(self.t) == len(self.u) == len(self.y), \
            "t, u, y must have the same length"

    def __len__(self) -> int:
        return len(self.t)

    @classmethod
    def from_mat(
        cls,
        filepath: str,
        time_key: str = "time",
        u_key: str = "u",
        y_key: str = "y",
        y_ref_key: str = "yref",
    ) -> "Dataset":
        """
        Load dataset from a .mat file.
        
        Args:
            filepath: Path to .mat file
            time_key: Key for time vector
            u_key: Key for input signal
            y_key: Key for output signal
            y_ref_key: Key for reference signal
        """
        try:
            import scipy.io
        except ImportError:
            raise ImportError("scipy required. Install with: pip install scipy")

        data = scipy.io.loadmat(filepath)
        
        t = data[time_key].flatten()
        u = data[u_key].flatten()
        y = data[y_key].flatten()
        
        y_ref = None
        if y_ref_key in data and data[y_ref_key].size > 0:
            y_ref = data[y_ref_key].flatten()

        # Estimate sampling rate
        dt = np.median(np.diff(t))
        fs = 1.0 / dt if dt > 0 else 1.0

        return cls(
            t=t, u=u, y=y, y_ref=y_ref,
            name=os.path.basename(filepath),
            sampling_rate=fs
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        save_path: Optional[str] = None,
        **kwargs,
    ) -> "Dataset":
        """
        Download and load dataset from URL.
        
        Args:
            url: URL to .mat file
            save_path: Local path to save file (optional)
            **kwargs: Additional arguments for from_mat()
        """
        filename = save_path or os.path.basename(url)
        
        if not os.path.exists(filename):
            print(f"Downloading {url}...")
            urlretrieve(url, filename)

        return cls.from_mat(filename, **kwargs)

    @classmethod
    def from_helon_github(cls, filename: str) -> "Dataset":
        """
        Load dataset from Helon's sysid GitHub repository.
        
        Args:
            filename: Name of the .mat file (e.g., '05_multisine_01.mat')
        """
        base_url = "https://raw.githubusercontent.com/helonayala/sysid/main/data"
        url = f"{base_url}/{filename}"
        return cls.from_url(url)

    def preprocess(
        self,
        trigger_key: Optional[str] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        resample_factor: int = 1,
        detrend: bool = False,
        normalize: bool = False,
    ) -> "Dataset":
        """
        Preprocess the dataset.
        
        Args:
            trigger_key: If provided, starts from first non-zero trigger
            start_idx: Start index for slicing
            end_idx: End index for slicing
            resample_factor: Downsample by this factor
            detrend: Remove mean from signals
            normalize: Normalize to zero mean, unit variance
            
        Returns:
            New preprocessed Dataset
        """
        # Determine start/end indices
        s_idx = start_idx or 0
        e_idx = end_idx or len(self.t)

        # Slice
        t = self.t[s_idx:e_idx]
        u = self.u[s_idx:e_idx]
        y = self.y[s_idx:e_idx]
        y_ref = self.y_ref[s_idx:e_idx] if self.y_ref is not None else None

        # Resample
        if resample_factor > 1:
            t = t[::resample_factor]
            u = u[::resample_factor]
            y = y[::resample_factor]
            if y_ref is not None:
                y_ref = y_ref[::resample_factor]

        # Detrend
        if detrend:
            u = u - np.mean(u)
            y = y - np.mean(y)
            if y_ref is not None:
                y_ref = y_ref - np.mean(y_ref)

        # Normalize
        if normalize:
            u_std = np.std(u) or 1.0
            y_std = np.std(y) or 1.0
            u = (u - np.mean(u)) / u_std
            y = (y - np.mean(y)) / y_std
            if y_ref is not None:
                y_ref = (y_ref - np.mean(y_ref)) / y_std

        # Adjust time to start at 0
        t = t - t[0]

        new_fs = self.sampling_rate / resample_factor

        return Dataset(
            t=t, u=u, y=y, y_ref=y_ref,
            name=self.name, sampling_rate=new_fs
        )

    def split(
        self, ratio: float = 0.8
    ) -> Tuple["Dataset", "Dataset"]:
        """
        Split dataset into train and test sets.
        
        Args:
            ratio: Fraction for training set
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        n = len(self.t)
        split_idx = int(n * ratio)

        train = Dataset(
            t=self.t[:split_idx],
            u=self.u[:split_idx],
            y=self.y[:split_idx],
            y_ref=self.y_ref[:split_idx] if self.y_ref is not None else None,
            name=f"{self.name}_train",
            sampling_rate=self.sampling_rate,
        )

        test = Dataset(
            t=self.t[split_idx:] - self.t[split_idx],
            u=self.u[split_idx:],
            y=self.y[split_idx:],
            y_ref=self.y_ref[split_idx:] if self.y_ref is not None else None,
            name=f"{self.name}_test",
            sampling_rate=self.sampling_rate,
        )

        return train, test

    def __repr__(self) -> str:
        return (
            f"Dataset(name='{self.name}', samples={len(self)}, "
            f"fs={self.sampling_rate:.1f}Hz)"
        )
