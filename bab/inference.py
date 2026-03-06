"""Preprocessing, keypoint extraction, and video analysis for pose estimation."""

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .models import DLCResNet50, PoseResNet50

# DLC preprocessing: RGB float32 [0, 255] minus ImageNet mean (no std normalization)
DLC_MEAN_PIXEL = np.array([123.68, 116.779, 103.939], dtype=np.float32)

# Standard ImageNet preprocessing (for PoseResNet50 / Part E models)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

KEYPOINT_NAMES = ["beam_left", "beam_right"]

# DLC default stride and location-refinement scaling factor
DLC_STRIDE = 8.0
DLC_LOCREF_STDEV = 7.2801


def preprocess_frame_dlc(frame_bgr: np.ndarray) -> torch.Tensor:
    """
    DLC-compatible preprocessing: RGB [0,255] minus mean. No resize (native resolution).
    This matches what DLC TF uses internally with ResNet v1.
    """
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img - DLC_MEAN_PIXEL
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def preprocess_frame_imagenet(frame_bgr: np.ndarray, input_size=(256, 256)) -> torch.Tensor:
    """ImageNet preprocessing for PoseResNet50 (Part E models)."""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def heatmaps_to_keypoints(heatmaps: torch.Tensor, orig_h: int, orig_w: int,
                           locref: torch.Tensor = None) -> dict:
    """
    Extracts keypoint coordinates from heatmaps using argmax + location refinement.
    Matches DLC's ``argmax_pose_predict``:

        x = col * stride + 0.5 * stride + locref_x * locref_stdev
        y = row * stride + 0.5 * stride + locref_y * locref_stdev

    Args:
        heatmaps: (B, K, Hm, Wm) part prediction heatmaps
        locref: (B, 2K, Hm, Wm) raw location refinement offsets (optional).
                For keypoint k, channel 2k = x offset, channel 2k+1 = y offset.
    """
    B, K, Hm, Wm = heatmaps.shape
    results = {}

    for k in range(K):
        hmap = heatmaps[0, k]

        # Argmax to find peak location in heatmap space
        flat_idx = hmap.reshape(-1).argmax()
        row = (flat_idx // Wm).item()
        col = (flat_idx % Wm).item()

        # DLC coordinate formula: pos = argmax * stride + 0.5 * stride + offset
        if locref is not None:
            dx = locref[0, 2 * k,     row, col].item() * DLC_LOCREF_STDEV
            dy = locref[0, 2 * k + 1, row, col].item() * DLC_LOCREF_STDEV
        else:
            dx, dy = 0.0, 0.0

        x_orig = col * DLC_STRIDE + 0.5 * DLC_STRIDE + dx
        y_orig = row * DLC_STRIDE + 0.5 * DLC_STRIDE + dy

        confidence = torch.sigmoid(hmap.max()).item()

        name = KEYPOINT_NAMES[k] if k < len(KEYPOINT_NAMES) else f"kp_{k}"
        results[name] = (x_orig, y_orig, confidence)

    return results


def compute_theta(keypoints: dict) -> float:
    """theta = arctan2(yr - yl, xr - xl) in degrees."""
    xl, yl, _ = keypoints["beam_left"]
    xr, yr, _ = keypoints["beam_right"]
    return np.degrees(np.arctan2(yr - yl, xr - xl))


@torch.no_grad()
def predict_frame_dlc(model: DLCResNet50, frame_bgr: np.ndarray, device="cpu"):
    """Inference with the converted DLC model (native resolution, DLC preprocessing)."""
    orig_h, orig_w = frame_bgr.shape[:2]
    inp = preprocess_frame_dlc(frame_bgr).to(device)
    out = model(inp)
    locref = out.get("locref_pred", None)
    return heatmaps_to_keypoints(out["part_pred"], orig_h, orig_w, locref=locref)


@torch.no_grad()
def predict_frame(model: PoseResNet50, frame_bgr: np.ndarray,
                  input_size=(256, 256), device="cpu"):
    """Inference with the PoseResNet50 model (resized, ImageNet preprocessing)."""
    orig_h, orig_w = frame_bgr.shape[:2]
    inp = preprocess_frame_imagenet(frame_bgr, input_size).to(device)
    heatmaps = model(inp)
    return heatmaps_to_keypoints(heatmaps, orig_h, orig_w)


def analyze_video_pytorch(
    model,
    video_path: str,
    output_csv: str = "theta_pytorch.csv",
    fps: float = 30.0,
    device: str = "cpu",
    pcut: float = 0.1,
    model_type: str = "dlc",
    input_size: tuple = (256, 256),
) -> pd.DataFrame:
    """
    Equivalent to deeplabcut.analyze_videos() + theta extraction, in pure PyTorch.

    Args:
        model: DLCResNet50 or PoseResNet50 instance
        video_path: path to input video
        output_csv: path to save results CSV
        fps: video frame rate
        device: "cpu" or "cuda" or "mps"
        pcut: confidence threshold
        model_type: "dlc" for DLCResNet50, "vanilla" for PoseResNet50
        input_size: resize target for PoseResNet50 (ignored for DLC)

    Returns:
        DataFrame with columns: frame, t_s, beam_left_x/y, beam_right_x/y,
        confidence_min, theta_deg
    """
    model.eval()
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open: {video_path}"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Analyzing {total_frames} frames (model_type={model_type})...")

    records = []
    for frame_idx in tqdm(range(total_frames), desc="Inference"):
        ret, frame = cap.read()
        if not ret:
            break

        if model_type == "dlc":
            kps = predict_frame_dlc(model, frame, device)
        else:
            kps = predict_frame(model, frame, input_size, device)

        xl, yl, cl = kps["beam_left"]
        xr, yr, cr = kps["beam_right"]
        conf_min = min(cl, cr)

        records.append({
            "frame": frame_idx,
            "t_s": frame_idx / fps,
            "beam_left_x": xl, "beam_left_y": yl,
            "beam_right_x": xr, "beam_right_y": yr,
            "confidence_min": conf_min,
            "theta_deg": compute_theta(kps) if conf_min >= pcut else np.nan,
        })

    cap.release()

    df = pd.DataFrame(records)
    df["theta_deg"] = df["theta_deg"].interpolate(limit_direction="both")
    df.to_csv(output_csv, index=False)

    low = df["confidence_min"].lt(pcut).sum()
    print(f"Saved: {output_csv}  ({len(df)} frames, {low} with low confidence)")
    return df
