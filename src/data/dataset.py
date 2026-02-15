"""Dataset loading and preprocessing utilities."""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np


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

BAB_ALIASES = {
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
    """Resolve bab_datasets aliases to canonical keys."""
    if name in BAB_DATASET_REGISTRY:
        return name
    return BAB_ALIASES.get(name, name)


def _find_trigger_start(trigger: Optional[np.ndarray]) -> int:
    """Find first active trigger index."""
    if trigger is None:
        return 0
    idx = np.where(np.asarray(trigger).flatten() != 0)[0]
    if idx.size == 0:
        return 0
    return int(idx[0])


def _find_end_before_ref_zero(
    y_ref: Optional[np.ndarray], tolerance: float = 1e-8
) -> int:
    """Find last index before reference goes to zero."""
    if y_ref is None:
        return -1
    y_ref = np.asarray(y_ref).flatten()
    for i in range(len(y_ref) - 1, -1, -1):
        if np.abs(y_ref[i]) > tolerance:
            return int(i + 1)
    return -1


def _estimate_y_dot(
    y: np.ndarray,
    dt: float,
    method: str = "savgol",
    savgol_window: int = 51,
    savgol_poly: int = 3,
) -> np.ndarray:
    """Estimate output derivative from sampled output."""
    y = np.asarray(y, dtype=float).flatten()
    if len(y) < 3 or dt <= 0:
        return np.zeros_like(y)

    if method == "central":
        return np.gradient(y, dt)

    if method == "savgol":
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            return np.gradient(y, dt)

        w = max(5, int(savgol_window))
        if w % 2 == 0:
            w += 1
        if w >= len(y):
            w = len(y) - 1 if len(y) % 2 == 0 else len(y)
        if w < 5:
            return np.gradient(y, dt)
        poly = min(int(savgol_poly), w - 1)
        return savgol_filter(
            y,
            window_length=w,
            polyorder=poly,
            deriv=1,
            delta=dt,
            mode="interp",
        )

    raise ValueError(f"Unknown y_dot estimation method: {method}")


def _slice_optional(arr: Optional[np.ndarray], start: int, end: int) -> Optional[np.ndarray]:
    """Slice optional array safely."""
    if arr is None:
        return None
    return arr[start:end]


def _downsample_optional(arr: Optional[np.ndarray], factor: int) -> Optional[np.ndarray]:
    """Downsample optional array when factor > 1."""
    if arr is None or factor <= 1:
        return arr
    return arr[::factor]


def _shift_time_to_zero(t: np.ndarray) -> np.ndarray:
    """Shift time vector so it starts at zero."""
    if len(t) == 0:
        return t
    return t - t[0]


def _estimate_dt_and_fs(t: np.ndarray, default_fs: float = 1.0) -> tuple[float, float]:
    """Estimate sampling time and rate from time vector."""
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
    fs = 1.0 / dt if dt > 0 else float(default_fs)
    return float(dt), float(fs)


@dataclass
class Dataset:
    """
    Container for system identification data.

    Attributes:
        t: Time vector
        u: Input signal
        y: Output signal
        y_ref: Reference signal (optional)
        y_dot: Output derivative signal (optional)
        y_filt: Optional filtered output from source dataset
        trigger: Optional trigger signal
        name: Dataset name
        sampling_rate: Sampling frequency in Hz
    """

    t: np.ndarray
    u: np.ndarray
    y: np.ndarray
    y_ref: Optional[np.ndarray] = None
    y_dot: Optional[np.ndarray] = None
    y_filt: Optional[np.ndarray] = None
    trigger: Optional[np.ndarray] = None
    name: str = ""
    sampling_rate: float = 1.0

    def __post_init__(self):
        """Validate data shapes."""
        self.t = np.asarray(self.t).flatten()
        self.u = np.asarray(self.u).flatten()
        self.y = np.asarray(self.y).flatten()

        if self.y_ref is not None:
            self.y_ref = np.asarray(self.y_ref).flatten()
        if self.y_dot is not None:
            self.y_dot = np.asarray(self.y_dot).flatten()
        if self.y_filt is not None:
            self.y_filt = np.asarray(self.y_filt).flatten()
        if self.trigger is not None:
            self.trigger = np.asarray(self.trigger).flatten()

        assert len(self.t) == len(self.u) == len(self.y), "t, u, y must have the same length"
        if self.y_ref is not None:
            assert len(self.y_ref) == len(self.t), "y_ref must match t length"
        if self.y_dot is not None:
            assert len(self.y_dot) == len(self.t), "y_dot must match t length"
        if self.y_filt is not None:
            assert len(self.y_filt) == len(self.t), "y_filt must match t length"
        if self.trigger is not None:
            assert len(self.trigger) == len(self.t), "trigger must match t length"

    def __len__(self) -> int:
        return len(self.t)

    @classmethod
    def list_bab_experiments(cls) -> list[str]:
        """List available experiment keys from bab_datasets."""
        return sorted(BAB_DATASET_REGISTRY.keys())

    @classmethod
    def _build_dataset(
        cls,
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
    ) -> "Dataset":
        """Create a Dataset with consistent dt/fs/y_dot handling."""
        dt, fs = _estimate_dt_and_fs(t, default_fs=fallback_fs)
        y_dot = _estimate_y_dot(
            y,
            dt if dt > 0 else 1.0,
            method=y_dot_method,
            savgol_window=savgol_window,
            savgol_poly=savgol_poly,
        )
        return cls(
            t=t,
            u=u,
            y=y,
            y_ref=y_ref,
            y_dot=y_dot,
            y_filt=y_filt,
            trigger=trigger,
            name=name,
            sampling_rate=fs,
        )

    @classmethod
    def from_mat(
        cls,
        filepath: str,
        time_key: str = "time",
        u_key: str = "u",
        y_key: str = "y",
        y_ref_key: str = "yref",
        trigger_key: str = "trigger",
        y_filt_key: str = "yf",
    ) -> "Dataset":
        """
        Load dataset from a .mat file.

        Args:
            filepath: Path to .mat file
            time_key: Key for time vector
            u_key: Key for input signal
            y_key: Key for output signal
            y_ref_key: Key for reference signal
            trigger_key: Key for trigger signal
            y_filt_key: Key for filtered output
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
        elif "ref" in data and data["ref"].size > 0:
            y_ref = data["ref"].flatten()

        trigger = None
        if trigger_key in data and data[trigger_key].size > 0:
            trigger = data[trigger_key].flatten()

        y_filt = None
        if y_filt_key in data and data[y_filt_key].size > 0:
            y_filt = data[y_filt_key].flatten()

        return cls._build_dataset(
            t=t,
            u=u,
            y=y,
            name=os.path.basename(filepath),
            y_ref=y_ref,
            y_filt=y_filt,
            trigger=trigger,
            y_dot_method="savgol",
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
    def from_bab_experiment(
        cls,
        name: str,
        preprocess: bool = True,
        end_idx: Optional[int] = None,
        resample_factor: int = 50,
        end_ref_tolerance: float = 1e-8,
        y_dot_method: str = "savgol",
        savgol_window: int = 51,
        savgol_poly: int = 3,
        data_dir: Optional[str] = None,
    ) -> "Dataset":
        """
        Load a bab_datasets experiment by key or alias.

        Args:
            name: Experiment key (e.g., 'multisine_05') or alias (e.g., '05_multisine_01')
            preprocess: Apply trigger-based start/end and resampling
            end_idx: Optional manual end index before resampling
            resample_factor: Downsample factor when preprocess=True
            end_ref_tolerance: Threshold used to detect end based on reference signal
            y_dot_method: Derivative method ('savgol' or 'central')
            savgol_window: Savitzky-Golay window length
            savgol_poly: Savitzky-Golay polynomial order
            data_dir: Local directory containing dataset files
        """
        try:
            import scipy.io
        except ImportError:
            raise ImportError("scipy required. Install with: pip install scipy")

        resolved = _resolve_bab_name(name)
        if resolved not in BAB_DATASET_REGISTRY:
            available = ", ".join(cls.list_bab_experiments())
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
        t = np.asarray(data["time"]).flatten()
        u = np.asarray(data["u"]).flatten()
        y = np.asarray(data["y"]).flatten()
        trigger = np.asarray(data["trigger"]).flatten() if "trigger" in data else None
        y_ref = np.asarray(data["ref"]).flatten() if "ref" in data else None
        y_filt = np.asarray(data["yf"]).flatten() if "yf" in data else None

        if not preprocess:
            return cls._build_dataset(
                t=t,
                u=u,
                y=y,
                y_ref=y_ref,
                y_filt=y_filt,
                trigger=trigger,
                name=resolved,
                y_dot_method=y_dot_method,
                savgol_window=savgol_window,
                savgol_poly=savgol_poly,
            )

        start_idx = _find_trigger_start(trigger)
        processed_end = end_idx
        if processed_end is None:
            processed_end = _find_end_before_ref_zero(y_ref, tolerance=end_ref_tolerance)
            if processed_end <= start_idx:
                processed_end = len(t)
        processed_end = min(int(processed_end), len(t))

        t = t[start_idx:processed_end]
        u = u[start_idx:processed_end]
        y = y[start_idx:processed_end]
        y_ref = _slice_optional(y_ref, start_idx, processed_end)
        y_filt = _slice_optional(y_filt, start_idx, processed_end)
        trigger = _slice_optional(trigger, start_idx, processed_end)

        t = _downsample_optional(t, resample_factor)
        u = _downsample_optional(u, resample_factor)
        y = _downsample_optional(y, resample_factor)
        y_ref = _downsample_optional(y_ref, resample_factor)
        y_filt = _downsample_optional(y_filt, resample_factor)
        trigger = _downsample_optional(trigger, resample_factor)

        return cls._build_dataset(
            t=_shift_time_to_zero(t),
            u=u,
            y=y,
            name=resolved,
            y_ref=y_ref,
            y_filt=y_filt,
            trigger=trigger,
            y_dot_method=y_dot_method,
            savgol_window=savgol_window,
            savgol_poly=savgol_poly,
        )

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
            trigger_key: If "trigger", starts from first non-zero trigger
            start_idx: Start index for slicing
            end_idx: End index for slicing
            resample_factor: Downsample by this factor
            detrend: Remove mean from signals
            normalize: Normalize to zero mean, unit variance

        Returns:
            New preprocessed Dataset
        """
        s_idx = start_idx if start_idx is not None else 0
        if (
            start_idx is None
            and trigger_key == "trigger"
            and self.trigger is not None
        ):
            s_idx = _find_trigger_start(self.trigger)
        e_idx = end_idx if end_idx is not None else len(self.t)

        t = self.t[s_idx:e_idx]
        u = self.u[s_idx:e_idx]
        y = self.y[s_idx:e_idx]
        y_ref = _slice_optional(self.y_ref, s_idx, e_idx)
        y_filt = _slice_optional(self.y_filt, s_idx, e_idx)
        trigger = _slice_optional(self.trigger, s_idx, e_idx)

        t = _downsample_optional(t, resample_factor)
        u = _downsample_optional(u, resample_factor)
        y = _downsample_optional(y, resample_factor)
        y_ref = _downsample_optional(y_ref, resample_factor)
        y_filt = _downsample_optional(y_filt, resample_factor)
        trigger = _downsample_optional(trigger, resample_factor)

        if detrend:
            u = u - np.mean(u)
            y = y - np.mean(y)
            if y_ref is not None:
                y_ref = y_ref - np.mean(y_ref)
            if y_filt is not None:
                y_filt = y_filt - np.mean(y_filt)

        if normalize:
            u_std = np.std(u) or 1.0
            y_std = np.std(y) or 1.0
            u = (u - np.mean(u)) / u_std
            y_mean = np.mean(y)
            y = (y - y_mean) / y_std
            if y_ref is not None:
                y_ref = (y_ref - y_mean) / y_std
            if y_filt is not None:
                y_filt = (y_filt - y_mean) / y_std

        fallback_fs = self.sampling_rate / max(resample_factor, 1)
        return type(self)._build_dataset(
            t=_shift_time_to_zero(t),
            u=u,
            y=y,
            name=self.name,
            y_ref=y_ref,
            y_filt=y_filt,
            trigger=trigger,
            y_dot_method="central",
            fallback_fs=fallback_fs,
        )

    def _subset(self, start: int, end: int, name: str, reset_time: bool = False) -> "Dataset":
        """Return sliced dataset view with optional time reset."""
        t = self.t[start:end]
        if reset_time:
            t = _shift_time_to_zero(t)
        return Dataset(
            t=t,
            u=self.u[start:end],
            y=self.y[start:end],
            y_ref=_slice_optional(self.y_ref, start, end),
            y_dot=_slice_optional(self.y_dot, start, end),
            y_filt=_slice_optional(self.y_filt, start, end),
            trigger=_slice_optional(self.trigger, start, end),
            name=name,
            sampling_rate=self.sampling_rate,
        )

    @property
    def arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(u, y)`` tuple ready for :meth:`BaseModel.fit`."""
        return self.u, self.y

    def split(self, ratio: float = 0.8) -> Tuple["Dataset", "Dataset"]:
        """
        Split dataset into train and test sets.

        Args:
            ratio: Fraction for training set

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        n = len(self.t)
        split_idx = int(n * ratio)

        train = self._subset(0, split_idx, name=f"{self.name}_train")
        test = self._subset(split_idx, n, name=f"{self.name}_test", reset_time=True)

        return train, test

    def train_val_test_split(
        self,
        train: float = 0.7,
        val: float = 0.15,
    ) -> Tuple["Dataset", "Dataset", "Dataset"]:
        """Split into train / validation / test sets.

        Args:
            train: Fraction for training (default 0.70).
            val:   Fraction for validation (default 0.15).
                   Test gets ``1 - train - val``.

        Returns:
            ``(train_ds, val_ds, test_ds)``
        """
        if train + val >= 1.0:
            raise ValueError("train + val must be < 1.0")
        n = len(self.t)
        i1 = int(n * train)
        i2 = int(n * (train + val))

        train_ds = self._subset(0, i1, name=f"{self.name}_train")
        val_ds = self._subset(i1, i2, name=f"{self.name}_val", reset_time=True)
        test_ds = self._subset(i2, n, name=f"{self.name}_test", reset_time=True)
        return train_ds, val_ds, test_ds

    def __repr__(self) -> str:
        return (
            f"Dataset(name='{self.name}', samples={len(self)}, "
            f"fs={self.sampling_rate:.1f}Hz)"
        )
