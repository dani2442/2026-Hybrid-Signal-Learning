import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import scipy.io
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from urllib.error import URLError

DATASET_REGISTRY: Dict[str, Dict[str, str]] = {
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

ALIASES = {
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


@dataclass
class InputOutputData:
    name: str
    u: np.ndarray
    y: np.ndarray
    sampling_time: float
    y_ref: Optional[np.ndarray] = None
    y_raw: Optional[np.ndarray] = None
    y_filt: Optional[np.ndarray] = None
    y_dot: Optional[np.ndarray] = None
    state_initialization_window_length: Optional[int] = None

    def __iter__(self):
        yield self.u
        yield self.y
        yield self.y_ref
        yield self.y_dot

    def __getitem__(self, item):
        return InputOutputData(
            name=f"{self.name}[{item}]",
            u=self.u[item],
            y=self.y[item],
            sampling_time=self.sampling_time,
            y_ref=None if self.y_ref is None else self.y_ref[item],
            y_raw=None if self.y_raw is None else self.y_raw[item],
            y_filt=None if self.y_filt is None else self.y_filt[item],
            y_dot=None if self.y_dot is None else self.y_dot[item],
            state_initialization_window_length=self.state_initialization_window_length,
        )

    def __repr__(self) -> str:
        base = (
            f"InputOutputData \"{self.name}\" u.shape={self.u.shape} y.shape={self.y.shape}\n"
            f"sampling_time={self.sampling_time:.4f}"
        )
        if self.y_ref is not None:
            base += f" y_ref.shape={self.y_ref.shape}"
        if self.state_initialization_window_length is not None:
            base += f" state_initialization_window_length={self.state_initialization_window_length}"
        return base

    def time_vector(self) -> np.ndarray:
        return np.arange(len(self.y)) * float(self.sampling_time)


def list_experiments():
    return sorted(DATASET_REGISTRY.keys())


def _resolve_name(name: str) -> str:
    if name in DATASET_REGISTRY:
        return name
    if name in ALIASES:
        return ALIASES[name]
    return name


def _default_data_dir() -> str:
    pkg_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(pkg_dir, "..", "data"))


def _download_if_needed(url: str, filename: str, data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        try:
            urlretrieve(url, path)
        except URLError as exc:
            raise RuntimeError(
                f"Failed to download {filename}. Place the file in {data_dir} and retry."
            ) from exc
    return path


def _load_mat(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    data = scipy.io.loadmat(path)
    t = data["time"].flatten()
    u = data["u"].flatten()
    y = data["y"].flatten()
    trig = data["trigger"].flatten()
    y_filt = data["yf"].flatten() if "yf" in data else None
    y_ref = data["ref"].flatten() if "ref" in data else None
    return t, u, y, trig, y_filt, y_ref


def _find_trigger_start(trig: np.ndarray) -> int:
    idx = np.where(trig != 0)[0]
    if idx.size == 0:
        return 0
    return int(idx[0])


def _find_end_before_ref_zero(y_ref: np.ndarray, tolerance: float = 1e-8) -> int:
    if y_ref is None:
        return -1
    # Scan from the end until we find a nonzero reference (with tolerance)
    for i in range(len(y_ref) - 1, -1, -1):
        if np.abs(y_ref[i]) > tolerance:
            return int(i + 1)
    return -1


def _plot_zoom_windows(
    y: np.ndarray,
    trig_proc: Optional[np.ndarray],
    start_idx: int,
    end_idx: int,
    n_samples: int = 200,
):
    n_samples = int(max(50, n_samples))

    # Window around start (start in the middle)
    half = n_samples // 2
    s0 = max(0, start_idx - half)
    s1 = min(len(y), s0 + n_samples)

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(s0, s1), y[s0:s1], color="red", label="y")
    if trig_proc is not None:
        plt.plot(np.arange(s0, s1), trig_proc[s0:s1], color="black", alpha=0.6, label="trigger_proc")
    plt.title(f"Zoom near start (samples {s0}:{s1})")
    plt.xlabel("Sample")
    plt.ylabel("Signal")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # Window at the very end of the full signal
    e1 = max(0, len(y) - n_samples)
    e2 = len(y)
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(e1, e2), y[e1:e2], color="red", label="y")
    if trig_proc is not None:
        plt.plot(np.arange(e1, e2), trig_proc[e1:e2], color="black", alpha=0.6, label="trigger_proc")
    plt.title(f"Zoom near end (samples {e1}:{e2})")
    plt.xlabel("Sample")
    plt.ylabel("Signal")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def _plot_signals(t, u, y, y_ref, y_dot, trig_raw, trig_proc, title: str):
    plt.figure(figsize=(10, 8))

    plt.subplot(4, 1, 1)
    plt.plot(t, u, color="blue", label="Input (u)")
    plt.ylabel("Input")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.title(f"Data: {title}")

    plt.subplot(4, 1, 2)
    plt.plot(t, y, color="red", label="Output (y)")
    if y_ref is not None:
        plt.plot(t, y_ref, color="gray", alpha=0.7, label="Reference (y_ref)")
    plt.ylabel("Output")
    plt.xlabel("Time (s)")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.subplot(4, 1, 3)
    if trig_raw is None:
        trig_raw = np.zeros_like(t)
    plt.plot(t, trig_raw, color="black", label="Trigger (raw)")
    if trig_proc is None:
        trig_proc = np.zeros_like(t)
    plt.plot(t, trig_proc, color="purple", alpha=0.7, label="Trigger (processed)")
    plt.ylabel("Trigger")
    plt.xlabel("Time (s)")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(t, y_dot, color="tab:green", label="Velocity (y_dot)")
    plt.ylabel("Velocity")
    plt.xlabel("Time (s)")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def _estimate_y_dot(
    y: np.ndarray,
    ts: float,
    method: str = "savgol",
    savgol_window: int = 51,
    savgol_poly: int = 3,
    spline_s: float = 0.0,
    butter_cutoff_hz: float = 5.0,
    butter_order: int = 4,
):
    if method == "savgol":
        w = max(5, int(savgol_window))
        if w % 2 == 0:
            w += 1
        return savgol_filter(y, window_length=w, polyorder=savgol_poly, deriv=1, delta=ts, mode="interp")
    if method == "central":
        return np.gradient(y, ts)
    if method == "spline":
        t = np.arange(len(y)) * ts
        spl = UnivariateSpline(t, y, s=spline_s)
        return spl.derivative()(t)
    if method == "butter":
        nyq = 0.5 / ts
        cutoff = min(max(butter_cutoff_hz, 1e-6), nyq * 0.99)
        b, a = butter(butter_order, cutoff / nyq, btype="low")
        y_smooth = filtfilt(b, a, y)
        return np.gradient(y_smooth, ts)
    if method == "tvreg":
        try:
            from tvregdiff import TVRegDiff
        except Exception as exc:
            raise ValueError(
                "tvreg method requires 'tvregdiff' package. Install it or choose another method."
            ) from exc
        return TVRegDiff(y, 1.0 / ts, 1e-2, itern=200)
    raise ValueError(f"Unknown y_dot method '{method}'.")


def _round_ts(ts: float, decimals: int = 4) -> float:
    return float(np.round(ts, int(decimals)))


def load_experiment(
    name: str,
    preprocess: bool = True,
    plot: bool = False,
    end_idx: Optional[int] = None,
    resample_factor: int = 50,
    zoom_last_n: int = 200,
    end_ref_tolerance: float = 1e-8,
    y_dot_method: str = "savgol",
    savgol_window: int = 51,
    savgol_poly: int = 3,
    spline_s: float = 0.0,
    butter_cutoff_hz: float = 5.0,
    butter_order: int = 4,
    data_dir: Optional[str] = None,
) -> InputOutputData:
    """
    Load a dataset by key and return InputOutputData.

    Example:
        import bab_datasets as nod
        data = nod.load_experiment("multisine_05", preprocess=True, plot=True)
        u, y = data
    """
    name = _resolve_name(name)
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_experiments()}")

    entry = DATASET_REGISTRY[name]
    data_dir = data_dir or _default_data_dir()
    path = _download_if_needed(entry["url"], entry["filename"], data_dir)

    t, u, y, trig, y_filt, y_ref = _load_mat(path)
    t_full = t
    u_full = u
    y_full = y
    y_ref_full = y_ref
    trig_full = trig

    ts = _round_ts(float(np.average(np.diff(t))))
    y_dot_full = _estimate_y_dot(
        y_full,
        ts,
        method=y_dot_method,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
        spline_s=spline_s,
        butter_cutoff_hz=butter_cutoff_hz,
        butter_order=butter_order,
    )

    if not preprocess:
        y_dot = _estimate_y_dot(
            y,
            ts,
            method=y_dot_method,
            savgol_window=savgol_window,
            savgol_poly=savgol_poly,
            spline_s=spline_s,
            butter_cutoff_hz=butter_cutoff_hz,
            butter_order=butter_order,
        )
        y_out = y
        return InputOutputData(
            name=name,
            u=u,
            y=y_out,
            sampling_time=ts,
            y_ref=y_ref,
            y_raw=y,
            y_filt=y_filt,
            y_dot=y_dot,
        )

    start_idx = _find_trigger_start(trig)

    if end_idx is None:
        if y_ref is not None:
            end_idx = _find_end_before_ref_zero(y_ref, tolerance=end_ref_tolerance)
        if end_idx is None or end_idx < 0 or end_idx <= start_idx:
            end_idx = len(u)

    # build processed trigger for clarity
    trig_proc = np.zeros_like(trig)
    trig_proc[start_idx:end_idx] = 1.0

    if plot:
        _plot_signals(t_full, u_full, y_full, y_ref_full, y_dot_full, trig_full, trig_proc, entry["filename"])
        _plot_zoom_windows(y_full, trig_proc, start_idx, end_idx, n_samples=zoom_last_n)

    u = u[start_idx:end_idx]
    y = y[start_idx:end_idx]
    if y_filt is not None:
        y_filt = y_filt[start_idx:end_idx]
    if y_ref is not None:
        y_ref = y_ref[start_idx:end_idx]
    t = t[start_idx:end_idx] - t[start_idx]

    if resample_factor and resample_factor > 1:
        u = u[::resample_factor]
        y = y[::resample_factor]
        t = t[::resample_factor]
        if y_filt is not None:
            y_filt = y_filt[::resample_factor]
        if y_ref is not None:
            y_ref = y_ref[::resample_factor]
        trig_proc = trig_proc[start_idx:end_idx][::resample_factor]
        ts = _round_ts(float(np.average(np.diff(t))))

    y_dot = _estimate_y_dot(
        y,
        ts,
        method=y_dot_method,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
        spline_s=spline_s,
        butter_cutoff_hz=butter_cutoff_hz,
        butter_order=butter_order,
    )

    if plot:
        # Resampled plots (u, y+y_ref, velocity)
        plt.figure(figsize=(10, 6))
        plt.subplot(3, 1, 1)
        plt.plot(t, u, label="u (resampled)")
        plt.ylabel("Input")
        plt.legend(loc="upper right")
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(t, y, label="y (resampled)")
        if y_ref is not None:
            plt.plot(t, y_ref, label="y_ref (resampled)")
        plt.ylabel("Output")
        plt.legend(loc="upper right")
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(t, y_dot, label="y_dot (resampled)")
        plt.ylabel("Velocity")
        plt.xlabel("Time (s)")
        plt.legend(loc="upper right")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    y_out = y
    return InputOutputData(
        name=name,
        u=u,
        y=y_out,
        sampling_time=ts,
        y_ref=y_ref,
        y_raw=y,
        y_filt=y_filt,
        y_dot=y_dot,
    )
