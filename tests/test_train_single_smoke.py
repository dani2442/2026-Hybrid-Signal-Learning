from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from src.data import Dataset
from examples import train_single


def _make_synthetic_dataset(n_samples: int = 180) -> Dataset:
    t = np.linspace(0.0, 1.8, n_samples)
    u = np.sin(2.0 * np.pi * 1.3 * t)
    y = 0.8 * np.sin(2.0 * np.pi * 1.3 * t + 0.2)
    return Dataset(t=t, u=u, y=y, name="synthetic", sampling_rate=float(n_samples))


def test_train_single_smoke_with_synthetic_data(tmp_path, monkeypatch):
    synthetic = _make_synthetic_dataset()

    monkeypatch.setattr(
        train_single.Dataset,
        "from_bab_experiment",
        classmethod(lambda cls, *args, **kwargs: synthetic),
    )
    monkeypatch.setattr(train_single, "plot_predictions", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_single.py",
            "--model",
            "narx",
            "--dataset",
            "synthetic",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--out-dir",
            "checkpoints",
        ],
    )

    train_single.main()

    assert (tmp_path / "checkpoints" / "narx_synthetic.pt").exists()


def test_train_single_script_mode_help_runs_without_pythonpath():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "examples/train_single.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--model" in result.stdout
