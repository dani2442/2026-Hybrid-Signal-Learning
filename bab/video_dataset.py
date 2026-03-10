"""Video frame dataset for autoencoder training."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class BABVideoDataset(Dataset):
    """Loads video frames, resizes to target size, and normalises to [0, 1].

    Frames are cached in memory on first access for fast epoch iteration.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    size : tuple[int, int]
        Target (width, height) for resizing.  Default ``(256, 256)``.
    """

    def __init__(self, video_path: str, size: tuple[int, int] = (256, 256)):
        super().__init__()
        self.video_path = video_path
        self.size = size
        self.frames: list[np.ndarray] = []

        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Could not open: {video_path}"
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in tqdm(range(total), desc="Loading frames", leave=False):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, size)
            self.frames.append(frame.astype(np.float32) / 255.0)

        cap.release()
        print(f"BABVideoDataset: {len(self.frames)} frames @ {size[0]}×{size[1]}")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns (3, H, W) float32 tensor in [0, 1]."""
        return torch.from_numpy(self.frames[idx]).permute(2, 0, 1)
