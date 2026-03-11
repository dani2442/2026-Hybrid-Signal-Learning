#!/usr/bin/env python
"""Automated beam-endpoint labeling for BAB videos.

Detects the beam (bright metallic bar) in selected video frames using
image-processing and writes a DLC-CollectedData-compatible CSV with
beam_left / beam_right keypoint coordinates.

Detection strategy (validated against swept_sine hand-labels, θ error < 0.4°):
  • Right endpoint: bright-pixel weighted centroid in a vertical strip at x ≈ 1822.
  • Left endpoint: dense brightness sampling across the beam mid-section
    (x = 200–1700, where no clamp/support clutter exists), iterative
    RANSAC line-fit, extrapolated to x ≈ 85.

Usage
-----
    # Label ~100 evenly-spaced frames from multisine_05:
    python examples/label_video.py --dataset multisine_05 --n-frames 100

    # Custom video + output:
    python examples/label_video.py --video-path data/videos/foo.MOV \\
        --dataset foo --n-frames 120 --out labels/foo_labels.csv

    # Validate against existing labels (swept_sine):
    python examples/label_video.py --dataset swept_sine --validate
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Beam detection
# ---------------------------------------------------------------------------

def detect_beam_endpoints(
    gray: np.ndarray,
    x_left: int = 85,
    x_right: int = 1822,
    y_lo: int = 180,
    y_hi: int = 520,
    threshold: int = 220,
) -> tuple[float, float, float, float]:
    """Detect beam-left and beam-right (x, y) in a grayscale 1080×1920 frame.

    Returns (xl, yl, xr, yr) in pixel coordinates, or NaNs on failure.
    """
    strip_hw = 10

    # ---- Right side: direct bright-pixel centroid (very accurate) ----
    r_strip = (
        gray[y_lo:y_hi, x_right - strip_hw : x_right + strip_hw + 1]
        .astype(float)
        .mean(axis=1)
    )
    r_bright = r_strip > threshold
    if not np.any(r_bright):
        thr = float(np.percentile(r_strip, 90))
        r_bright = r_strip > thr
    else:
        thr = float(threshold)

    ys_arr = np.arange(len(r_strip), dtype=float)
    weights = np.maximum(0, r_strip - thr) * r_bright
    if np.any(weights > 0):
        yr = float(np.average(ys_arr, weights=weights)) + y_lo
    else:
        return float(x_left), np.nan, float(x_right), np.nan

    # ---- Left side: mid-section line fit → extrapolate to x_left ----
    sample_xs = np.arange(200, 1700, 50)
    points: list[tuple[int, float]] = []

    for cx in sample_xs:
        col = (
            gray[y_lo:y_hi, cx - 8 : cx + 9].astype(float).mean(axis=1)
        )
        bright = col > threshold
        if not np.any(bright):
            continue
        ys_local = np.arange(len(col), dtype=float)
        w = np.maximum(0, col - threshold) * bright
        y_det = float(np.average(ys_local, weights=w)) + y_lo
        points.append((int(cx), y_det))

    if len(points) < 5:
        # Fallback: assume roughly horizontal beam
        return float(x_left), yr, float(x_right), yr

    pts = np.array(points)

    # Iterative RANSAC-style outlier removal
    for _ in range(3):
        coeffs = np.polyfit(pts[:, 0], pts[:, 1], 1)
        pred = np.polyval(coeffs, pts[:, 0])
        res = np.abs(pts[:, 1] - pred)
        thr_r = max(5.0, float(np.percentile(res, 75)))
        mask = res < thr_r
        if np.sum(mask) >= 5:
            pts = pts[mask]

    coeffs = np.polyfit(pts[:, 0], pts[:, 1], 1)
    yl = float(np.polyval(coeffs, x_left))

    return float(x_left), yl, float(x_right), yr


# ---------------------------------------------------------------------------
# Label a video
# ---------------------------------------------------------------------------

def label_video(
    video_path: str,
    n_frames: int = 100,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect beam endpoints in *n_frames* evenly-spaced frames.

    Returns
    -------
    frame_indices : (N,) int array
    keypoints : (N, 4) float array – [xl, yl, xr, yr] per frame
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None or end_frame > total:
        end_frame = total

    indices = np.linspace(start_frame, end_frame - 1, n_frames, dtype=int)
    indices = np.unique(indices)

    frame_indices = []
    keypoints = []

    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        xl, yl, xr, yr = detect_beam_endpoints(gray)
        if np.isnan(yl) or np.isnan(yr):
            continue
        frame_indices.append(int(fi))
        keypoints.append([xl, yl, xr, yr])

    cap.release()
    return np.array(frame_indices, dtype=int), np.array(keypoints, dtype=float)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_dlc_collected_csv(
    csv_path: str | Path,
    frame_indices: np.ndarray,
    keypoints: np.ndarray,
    *,
    scorer: str = "auto_detector",
    video_folder: str = "video",
) -> None:
    """Write labels in DLC CollectedData CSV format."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        # Header rows
        w.writerow(["scorer", "", "", scorer, scorer, scorer, scorer])
        w.writerow(["bodyparts", "", "", "beam_left", "beam_left", "beam_right", "beam_right"])
        w.writerow(["coords", "", "", "x", "y", "x", "y"])

        for i, fi in enumerate(frame_indices):
            img_name = f"img{int(fi):04d}.png"
            xl, yl, xr, yr = keypoints[i]
            w.writerow([
                "labeled-data", video_folder, img_name,
                f"{xl:.6f}", f"{yl:.6f}", f"{xr:.6f}", f"{yr:.6f}",
            ])

    print(f"Wrote {len(frame_indices)} labels to {path}")


def write_simple_csv(
    csv_path: str | Path,
    frame_indices: np.ndarray,
    keypoints: np.ndarray,
    fps: float = 30.0,
) -> None:
    """Write a simpler CSV with columns: frame, t_s, beam_left_x, beam_left_y,
    beam_right_x, beam_right_y, theta_deg."""
    from src.data.labels import keypoints_to_theta

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    thetas = keypoints_to_theta(keypoints[:, :2], keypoints[:, 2:4])

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame", "t_s",
            "beam_left_x", "beam_left_y",
            "beam_right_x", "beam_right_y",
            "theta_deg",
        ])
        for i, fi in enumerate(frame_indices):
            w.writerow([
                int(fi),
                f"{fi / fps:.6f}",
                f"{keypoints[i, 0]:.4f}",
                f"{keypoints[i, 1]:.4f}",
                f"{keypoints[i, 2]:.4f}",
                f"{keypoints[i, 3]:.4f}",
                f"{thetas[i]:.6f}",
            ])

    print(f"Wrote {len(frame_indices)} labels (simple CSV) to {path}")


# ---------------------------------------------------------------------------
# Validation against existing labels
# ---------------------------------------------------------------------------

def validate_against_labels(
    video_path: str,
    labels_csv: str,
    fps: float = 30.0,
) -> None:
    """Run detection on frames from an existing label CSV and report accuracy."""
    from src.data.labels import parse_keypoint_labels_csv

    labels = parse_keypoint_labels_csv(labels_csv, fps=fps)
    kp_true = labels["keypoints"]
    fi_true = labels["frame"]

    cap = cv2.VideoCapture(video_path)
    errors_l, errors_r = [], []

    for i, fi in enumerate(fi_true):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, yl, _, yr = detect_beam_endpoints(gray)
        errors_l.append(yl - kp_true[i, 1])
        errors_r.append(yr - kp_true[i, 3])

    cap.release()
    errors_l = np.array(errors_l)
    errors_r = np.array(errors_r)

    print(f"\nValidation on {len(errors_l)} labeled frames:")
    print(f"  Left y  — mean: {np.nanmean(errors_l):+.2f} px, "
          f"std: {np.nanstd(errors_l):.2f}, 90th ‰: {np.nanpercentile(np.abs(errors_l), 90):.2f}")
    print(f"  Right y — mean: {np.nanmean(errors_r):+.2f} px, "
          f"std: {np.nanstd(errors_r):.2f}, 90th ‰: {np.nanpercentile(np.abs(errors_r), 90):.2f}")

    # Theta error
    beam_span = 1822 - 85
    theta_true = np.degrees(np.arctan2(kp_true[:, 3] - kp_true[:, 1], kp_true[:, 2] - kp_true[:, 0]))
    theta_det = np.degrees(np.arctan2(
        (kp_true[:, 3] + errors_r) - (kp_true[:, 1] + errors_l),
        np.full(len(kp_true), float(beam_span)),
    ))
    te = theta_det - theta_true
    print(f"  θ error — mean: {np.nanmean(te):+.4f}°, "
          f"std: {np.nanstd(te):.4f}, max: {np.nanmax(np.abs(te)):.4f}, "
          f"90th ‰: {np.nanpercentile(np.abs(te), 90):.4f}°")


# ---------------------------------------------------------------------------
# Diagnostic montage
# ---------------------------------------------------------------------------

def save_diagnostic_montage(
    video_path: str,
    frame_indices: np.ndarray,
    keypoints: np.ndarray,
    out_path: str | Path,
    n_show: int = 12,
) -> None:
    """Save a montage of sample frames with detected keypoints overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = np.linspace(0, len(frame_indices) - 1, min(n_show, len(frame_indices)), dtype=int)

    cap = cv2.VideoCapture(video_path)
    ncols = min(4, len(sel))
    nrows = int(np.ceil(len(sel) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for j, si in enumerate(sel):
        fi = int(frame_indices[si])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        ax = axes[j // ncols, j % ncols]
        if ret:
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            xl, yl, xr, yr = keypoints[si]
            ax.scatter([xl, xr], [yl, yr], c=["lime", "red"], s=60, marker="x", zorder=5)
            ax.plot([xl, xr], [yl, yr], c="yellow", linewidth=1.0, alpha=0.7, zorder=4)
        ax.set_title(f"frame {fi}", fontsize=8)
        ax.axis("off")

    for j in range(len(sel), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle("Auto-detected beam endpoints", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic montage to {out_path}")

    cap.release()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatically label beam endpoints in BAB video frames."
    )
    parser.add_argument("--dataset", default="multisine_05",
                        help="Dataset key (used to resolve video path).")
    parser.add_argument("--video-path", default=None,
                        help="Explicit video path (overrides --dataset lookup).")
    parser.add_argument("--n-frames", type=int, default=100,
                        help="Number of evenly-spaced frames to label.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--out", default=None,
                        help="Output CSV path. Default: data/labels/<dataset>_true_labels.csv")
    parser.add_argument("--format", choices=["dlc", "simple", "both"], default="both",
                        help="Output format: DLC CollectedData, simple CSV, or both.")
    parser.add_argument("--validate", action="store_true",
                        help="Validate detection against existing labels instead of labeling.")
    parser.add_argument("--montage", action="store_true", default=True,
                        help="Save a diagnostic frame montage (default: True).")
    parser.add_argument("--no-montage", dest="montage", action="store_false")
    args = parser.parse_args()

    from src.data.registry import ensure_video, ensure_true_labels

    # Resolve video
    if args.video_path:
        video_path = args.video_path
    else:
        video_path = str(ensure_video(args.dataset))
    print(f"Video: {video_path}")

    # --validate mode
    if args.validate:
        labels_csv = ensure_true_labels(args.dataset)
        if labels_csv is None:
            print(f"No existing labels found for '{args.dataset}'.")
            sys.exit(1)
        from src.data.video import get_video_fps
        fps = get_video_fps(video_path)
        validate_against_labels(video_path, str(labels_csv), fps=fps)
        return

    # Label frames
    print(f"Labeling {args.n_frames} frames …")
    fi, kp = label_video(
        video_path,
        n_frames=args.n_frames,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    print(f"Successfully detected beam in {len(fi)}/{args.n_frames} frames.")

    # Write output
    out_base = args.out or f"data/labels/{args.dataset}_true_labels.csv"
    out_path = Path(out_base)

    from src.data.video import get_video_fps
    fps = get_video_fps(video_path)

    if args.format in ("dlc", "both"):
        write_dlc_collected_csv(out_path, fi, kp, video_folder=args.dataset)
    if args.format in ("simple", "both"):
        simple_path = out_path.with_stem(out_path.stem + "_simple") if args.format == "both" else out_path
        write_simple_csv(simple_path, fi, kp, fps=fps)

    # Diagnostic montage
    if args.montage:
        montage_path = out_path.with_suffix(".png")
        save_diagnostic_montage(video_path, fi, kp, montage_path)

    # Summary stats
    from src.data.labels import keypoints_to_theta
    thetas = keypoints_to_theta(kp[:, :2], kp[:, 2:4])
    print(f"\nSummary:")
    print(f"  Frames labeled: {len(fi)}")
    print(f"  Frame range: {fi[0]}–{fi[-1]}")
    print(f"  θ range: [{np.min(thetas):.2f}°, {np.max(thetas):.2f}°]")
    print(f"  beam_left_y range: [{kp[:,1].min():.1f}, {kp[:,1].max():.1f}]")
    print(f"  beam_right_y range: [{kp[:,3].min():.1f}, {kp[:,3].max():.1f}]")


if __name__ == "__main__":
    main()
