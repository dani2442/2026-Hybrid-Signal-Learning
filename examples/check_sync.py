#!/usr/bin/env python
"""Synchronization diagnostic for BAB video + sensor data.

Downloads the video and DLC label CSVs into data/ if not already present,
then produces two diagnostic figures:

  1. Sample video frames with DLC keypoint labels overlaid.
  2. Sensor theta vs. video-derived theta (time-alignment comparison).

Usage
-----
    # Basic (swept_sine with auto-download):
    python examples/check_sync.py

    # Explicit video path / different dataset:
    python examples/check_sync.py --dataset swept_sine \
        --video-path /path/to/swept_sine.MOV \
        --out-dir results/sync_check/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_frame_overlays(
    frames: np.ndarray,
    frame_timestamps: np.ndarray,
    keypoints_video: np.ndarray | None,
    sample_count: int,
    out_path: Path,
    *,
    labeled_frame_indices: np.ndarray | None = None,
) -> None:
    """Plot a grid of sample frames with keypoint labels overlaid.

    Parameters
    ----------
    labeled_frame_indices : array of int, optional
        Local frame indices that have actual (not interpolated) DLC
        labels.  When provided the plot preferentially samples from
        these frames so that the keypoint dots match the true beam
        position visible in the frame.
    """
    import matplotlib.pyplot as plt

    n_total = len(frames)
    if n_total == 0:
        print("No frames to plot.")
        return

    # Choose which frames to display.  When sparse labeled indices are
    # available, sample from those to guarantee keypoints match frames.
    if labeled_frame_indices is not None and len(labeled_frame_indices) >= sample_count:
        # Pick evenly-spaced entries from the sorted labeled indices
        lfi = np.sort(np.unique(labeled_frame_indices))
        sel = np.linspace(0, len(lfi) - 1, sample_count, dtype=int)
        idxs = lfi[sel]
    else:
        idxs = np.linspace(0, n_total - 1, sample_count, dtype=int)

    ncols = min(sample_count, 4)
    nrows = int(np.ceil(sample_count / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if sample_count == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for j, fi in enumerate(idxs):
        ax = axes[j // ncols, j % ncols]
        frame = frames[fi]
        if frame.ndim == 2:
            ax.imshow(frame, cmap="gray")
        else:
            ax.imshow(frame)

        t = float(frame_timestamps[fi]) if fi < len(frame_timestamps) else fi / 30.0
        is_labeled = (
            labeled_frame_indices is not None
            and int(fi) in set(labeled_frame_indices.tolist())
        )
        tag = " [labeled]" if is_labeled else " [interp]"
        ax.set_title(f"frame {fi}  t={t:.2f}s{tag}", fontsize=8)
        ax.axis("off")

        # Overlay keypoints — coordinates are in the frame's own pixel space
        if keypoints_video is not None and fi < len(keypoints_video):
            kp = keypoints_video[fi]
            if np.all(np.isfinite(kp)):
                xl, yl = kp[0], kp[1]
                xr, yr = kp[2], kp[3]
                ax.scatter([xl, xr], [yl, yr], c=["lime", "red"], s=60, marker="o",
                           label="beam L/R", zorder=5)
                ax.plot([xl, xr], [yl, yr], c="yellow", linewidth=1.5, zorder=4)

    # Hide unused axes
    for j in range(len(idxs), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lime",
                   markersize=8, label="beam_left"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markersize=8, label="beam_right"),
    ]
    if keypoints_video is not None:
        fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9)

    fig.suptitle("Sample frames with DLC keypoint labels", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_theta_comparison(
    sensor_t: np.ndarray,
    sensor_theta: np.ndarray,
    video_t: np.ndarray | None,
    video_theta_raw: np.ndarray | None,
    video_theta_aligned: np.ndarray | None,
    theta_sensor_from_video: np.ndarray | None,
    alignment_info: dict | None,
    out_path: Path,
) -> None:
    """Plot sensor theta vs video-derived theta for sync verification."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # --- Top: full timelines (video raw vs. sensor) ---
    ax0 = axes[0]
    ax0.plot(sensor_t, sensor_theta, linewidth=0.8, label="sensor θ (raw)", alpha=0.9)
    if video_t is not None and video_theta_raw is not None:
        valid = np.isfinite(video_theta_raw)
        ax0.plot(video_t[valid], video_theta_raw[valid], linewidth=0.8,
                 linestyle="--", label="video θ (DLC, pre-align)", alpha=0.85)
    ax0.set_xlabel("time [s]")
    ax0.set_ylabel("theta [deg]")
    ax0.set_title("Full time series: sensor vs. video theta (before alignment)")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    # --- Bottom: aligned overlay on common sensor time axis ---
    ax1 = axes[1]
    ax1.plot(sensor_t, sensor_theta, linewidth=1.0, label="sensor θ")
    if theta_sensor_from_video is not None and len(theta_sensor_from_video) == len(sensor_t):
        valid = np.isfinite(theta_sensor_from_video)
        ax1.plot(sensor_t[valid], theta_sensor_from_video[valid],
                 linewidth=1.0, linestyle="--", label="video θ (aligned → sensor time)")

    info_str = ""
    if alignment_info:
        off = alignment_info.get("offset_s", np.nan)
        sign = alignment_info.get("sign", 1)
        alpha = alignment_info.get("alpha", np.nan)
        beta = alignment_info.get("beta", np.nan)
        corr = alignment_info.get("corr", np.nan)
        rmse = alignment_info.get("rmse", np.nan)
        info_str = (
            f"offset={off:.3f}s  sign={sign:+d}  "
            f"alpha={alpha:.3f}  beta={beta:.3f}  "
            f"corr={corr:.3f}  RMSE={rmse:.3f}°"
        )
    ax1.set_xlabel("sensor time [s]")
    ax1.set_ylabel("theta [deg]")
    ax1.set_title(f"Aligned overlay on sensor timescale\n{info_str}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BAB video–sensor synchronization diagnostic."
    )
    parser.add_argument("--dataset", default="swept_sine",
                        help="Sensor dataset key (default: swept_sine).")
    parser.add_argument("--video-path", default=None,
                        help="Explicit path to video file (skips auto-download).")
    parser.add_argument("--video-fps", type=float, default=None,
                        help="Video FPS (default: auto-detect from file).")
    parser.add_argument("--resample-factor", type=int, default=33,
                        help="Sensor downsample factor (default: 33 ≈ 30 Hz).")
    parser.add_argument("--no-auto-match-fps", action="store_true",
                        help="Disable automatic resample-factor selection.")
    parser.add_argument("--frame-height", type=int, default=None,
                        help="Frame height for loading (default: original res).")
    parser.add_argument("--frame-width", type=int, default=None,
                        help="Frame width for loading (default: original res).")
    parser.add_argument("--frame-samples", type=int, default=8,
                        help="Number of sample frames to overlay keypoints on.")
    parser.add_argument("--out-dir", default="results/sync_check",
                        help="Output directory for diagnostic plots.")
    parser.add_argument("--no-led-sync", action="store_true",
                        help="Disable LED trigger crop of video.")
    parser.add_argument("--no-align", action="store_true",
                        help="Skip theta offset/sign/scale alignment.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== BAB Sync Check: dataset='{args.dataset}' ===")
    print(f"Output directory: {out_dir.resolve()}\n")

    # ------------------------------------------------------------------
    # 1. Ensure files are present (use unified src/data registry)
    # ------------------------------------------------------------------
    from src.data.registry import (
        ensure_keypoint_labels,
        ensure_theta_labels,
        ensure_true_labels,
        ensure_video,
    )

    video_path = ensure_video(args.dataset, video_path=args.video_path)
    kp_csv_path = ensure_keypoint_labels(args.dataset)
    theta_csv_path = ensure_theta_labels(args.dataset)
    true_csv_path = ensure_true_labels(args.dataset)

    # If no dedicated keypoint CSV exists, fall back to true-labels CSV
    # (DLC CollectedData format contains keypoint coordinates).
    if kp_csv_path is None and true_csv_path is not None:
        kp_csv_path = true_csv_path

    print(f"Video:          {video_path}")
    print(f"Keypoint CSV:   {kp_csv_path or '(none)'}")
    print(f"Theta CSV:      {theta_csv_path or '(none)'}")
    print(f"True labels:    {true_csv_path or '(none)'}")

    # ------------------------------------------------------------------
    # 2. Load sensor + video via the project's loader
    #    KEY FIX: pass frame_height=None / frame_width=None so frames
    #    are loaded at their ORIGINAL resolution, matching the DLC
    #    keypoint pixel coordinates.
    # ------------------------------------------------------------------
    from src.vision.datasets import load_bab_with_video

    print("\nLoading sensor data + video frames …")
    loaded = load_bab_with_video(
        args.dataset,
        video_path=str(video_path),
        resample_factor=args.resample_factor,
        video_fps=args.video_fps,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
        preprocess=True,
        use_led_sync=not args.no_led_sync,
        keypoint_labels_csv=str(kp_csv_path) if kp_csv_path else None,
        theta_labels_csv=str(theta_csv_path) if theta_csv_path else None,
        align_theta=not args.no_align,
        auto_match_video_fps=not args.no_auto_match_fps,
        return_aux=True,
    )
    data, frames, frame_idx_map, aux = loaded

    print(f"  Sensor samples : {len(data)}")
    print(f"  Video frames   : {len(frames)}")
    print(f"  Frame shape    : {frames.shape[1:] if frames.ndim >= 3 else '?'}")
    print(f"  Sensor fs      : {data.sampling_rate:.2f} Hz")

    actual_fps = aux.get("video_fps", 30.0)
    actual_factor = aux.get("sensor_resample_factor", args.resample_factor)
    is_sparse = aux.get("is_sparse_labels", False)
    print(f"  Video FPS      : {actual_fps:.3f}")
    print(f"  Resample factor: {actual_factor} (sensor → ~{1000/actual_factor:.1f} Hz)")
    if is_sparse:
        print(f"  Label type     : SPARSE (hand-labeled)")

    if "theta_alignment" in aux:
        al = aux["theta_alignment"]
        print(f"\nTheta alignment:")
        print(f"  offset  = {al['offset_s']:.3f} s")
        print(f"  sign    = {al['sign']:+d}")
        print(f"  alpha   = {al['alpha']:.4f}  beta = {al['beta']:.4f}")
        print(f"  corr    = {al['corr']:.4f}")
        print(f"  RMSE    = {al['rmse']:.3f} °")

    # ------------------------------------------------------------------
    # 3. Extract things for plotting
    # ------------------------------------------------------------------
    # Prefer the sparse (non-interpolated) keypoints for overlays so
    # that keypoint dots always match the actual beam position in each frame.
    keypoints_video: np.ndarray | None = aux.get("keypoints_video_sparse")
    if keypoints_video is None:
        keypoints_video = aux.get("keypoints_video")
    labeled_frame_indices: np.ndarray | None = aux.get("labeled_frame_indices")
    frame_timestamps = np.arange(len(frames), dtype=float) / actual_fps

    video_t_segment = aux.get("theta_video_t_segment")
    video_theta_raw = aux.get("theta_video_raw_segment")
    video_theta_aligned = aux.get("theta_video_aligned")
    theta_sensor_from_video = aux.get("theta_sensor_from_video")
    alignment_info = aux.get("theta_alignment")

    # ------------------------------------------------------------------
    # 4. Plot 1: sample frames with keypoint overlays
    # ------------------------------------------------------------------
    print("\nPlotting frame overlays …")
    _plot_frame_overlays(
        frames,
        frame_timestamps,
        keypoints_video,
        sample_count=args.frame_samples,
        out_path=out_dir / "frame_overlay.png",
        labeled_frame_indices=labeled_frame_indices,
    )

    # ------------------------------------------------------------------
    # 5. Plot 2: theta comparison
    # ------------------------------------------------------------------
    print("Plotting theta comparison …")
    _plot_theta_comparison(
        sensor_t=data.t,
        sensor_theta=data.y,
        video_t=video_t_segment,
        video_theta_raw=video_theta_raw,
        video_theta_aligned=video_theta_aligned,
        theta_sensor_from_video=theta_sensor_from_video,
        alignment_info=alignment_info,
        out_path=out_dir / "theta_comparison.png",
    )

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print(f"\nDone. Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
