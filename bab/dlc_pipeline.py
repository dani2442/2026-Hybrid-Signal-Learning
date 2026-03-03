"""DeepLabCut pipeline wrappers: training, analysis, and theta extraction."""

import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


def run_dlc_training(config_path: str, video_path: str,
                     maxiters: int = 500, displayiters: int = 1,
                     lr_init: float = 0.0005, batch_size: int = 8,
                     weight_decay: float = 0.05):
    """
    Run the full DLC training pipeline: extract frames, configure pose_cfg,
    train the network, and evaluate.

    Args:
        config_path: path to DLC config.yaml
        video_path: path to the input video
        maxiters: maximum training iterations
        displayiters: print loss every N iterations
        lr_init: initial learning rate
        batch_size: training batch size
        weight_decay: L2 regularization
    """
    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions

    # Configure bodyparts
    cfg = auxiliaryfunctions.read_config(config_path)
    cfg["bodyparts"] = ["beam_left", "beam_right"]
    cfg["numframes2pick"] = 50
    auxiliaryfunctions.write_config(config_path, cfg)
    print(f"bodyparts: {cfg['bodyparts']}, numframes2pick: {cfg['numframes2pick']}")

    # Extract frames
    deeplabcut.extract_frames(config_path, mode="automatic", algo="uniform",
                              userfeedback=False)

    # Update pose_cfg.yaml
    pose_cfg_path = str(
        Path(config_path).parent / "dlc-models" / "iteration-0"
        / "bab_bar_2pts_dlc3Feb24-trainset95shuffle1" / "train" / "pose_cfg.yaml"
    )
    with open(pose_cfg_path) as f:
        pose_cfg = yaml.safe_load(f)
    pose_cfg["multi_step"] = [[lr_init, 1030000]]
    pose_cfg["lr_init"] = lr_init
    pose_cfg["display_iters"] = displayiters
    pose_cfg["batch_size"] = batch_size
    pose_cfg["weight_decay"] = weight_decay
    with open(pose_cfg_path, "w") as f:
        yaml.dump(pose_cfg, f, default_flow_style=False)
    print(f"pose_cfg.yaml updated: lr={lr_init}, batch_size={batch_size}, "
          f"weight_decay={weight_decay}")

    # Train and evaluate
    deeplabcut.train_network(config_path, maxiters=maxiters, displayiters=displayiters)
    deeplabcut.evaluate_network(config_path)


def run_dlc_analysis(config_path: str, video_path: str):
    """
    Run DLC video analysis and create a labeled video.
    Ensures H5 and metadata pickle files exist.

    Args:
        config_path: path to DLC config.yaml
        video_path: path to the input video
    """
    import deeplabcut

    deeplabcut.analyze_videos(config_path, [video_path], save_as_csv=True)

    # Ensure H5 + _meta.pickle exist
    _base = sorted(
        [f for f in Path(video_path).parent.glob("swept_sine_readyDLC*.csv")
         if "_theta" not in f.name and "_meta" not in f.name],
        key=lambda p: p.stat().st_mtime,
    )
    if _base:
        _csv = str(_base[-1])
        _h5 = _csv.replace(".csv", ".h5")
        if not Path(_h5).exists():
            _df = pd.read_csv(_csv, header=[0, 1, 2], index_col=0)
            _df.to_hdf(_h5, key="df_with_missing", format="table", mode="w")
            print(f"Created H5 from CSV: {_h5}")

        _meta_path = _csv.replace(".csv", "_meta.pickle")
        if not Path(_meta_path).exists():
            _cap = cv2.VideoCapture(video_path)
            _nx = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            _ny = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _nframes = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            _fps = _cap.get(cv2.CAP_PROP_FPS)
            _cap.release()
            metadata = {"data": {
                "start": 0, "stop": 0, "nframes": _nframes, "fps": _fps,
                "frame_dimensions": (_ny, _nx),
                "cropping": False, "cropping_parameters": [0, _nx, 0, _ny],
            }}
            with open(_meta_path, "wb") as f:
                pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)
            print(f"Created metadata: {_meta_path}")

    deeplabcut.create_labeled_video(config_path, [video_path])


def extract_theta_from_dlc(video_path: str, working_dir: str,
                           fps: float = 30.0, pcutoff: float = 0.01) -> pd.DataFrame:
    """
    Read DLC analysis results and compute beam angle theta(t).

    Args:
        video_path: path to the input video (analysis files are in the same dir)
        working_dir: project working directory
        fps: video frame rate
        pcutoff: confidence threshold for filtering

    Returns:
        DataFrame with columns: t_s, theta_deg
    """
    video_dir = Path(video_path).parent

    # Try H5 first, fall back to CSV
    h5_files = sorted(
        [f for f in video_dir.glob("swept_sine_readyDLC*.h5")
         if "_theta" not in f.name and "_meta" not in f.name],
        key=lambda p: p.stat().st_mtime,
    )
    csv_files = sorted(
        [f for f in video_dir.glob("swept_sine_readyDLC*.csv")
         if "_theta" not in f.name and "_meta" not in f.name],
        key=lambda p: p.stat().st_mtime,
    )

    if h5_files:
        data_path = str(h5_files[-1])
        print("Using H5:", data_path)
        df = pd.read_hdf(data_path)
    elif csv_files:
        data_path = str(csv_files[-1])
        print("Using CSV:", data_path)
        df = pd.read_csv(data_path, header=[0, 1, 2], index_col=0)
    else:
        raise FileNotFoundError(f"No DLC analysis file found in {video_dir}")

    scorer = df.columns.get_level_values(0)[0]

    xl = df[(scorer, "beam_left", "x")]
    yl = df[(scorer, "beam_left", "y")]
    xr = df[(scorer, "beam_right", "x")]
    yr = df[(scorer, "beam_right", "y")]

    has_p = (
        (scorer, "beam_left", "likelihood") in df.columns
        and (scorer, "beam_right", "likelihood") in df.columns
    )
    if has_p:
        pl = df[(scorer, "beam_left", "likelihood")]
        pr = df[(scorer, "beam_right", "likelihood")]
        mask = (pl > pcutoff) & (pr > pcutoff)
    else:
        mask = pd.Series(True, index=df.index)

    theta = np.arctan2((yr - yl), (xr - xl))
    theta_deg = np.degrees(theta)
    theta_deg_f = theta_deg.copy()
    theta_deg_f[~mask] = np.nan
    theta_deg_f = theta_deg_f.interpolate(limit_direction="both")

    t = df.index.to_numpy() / fps

    results_df = pd.DataFrame({"t_s": t, "theta_deg": theta_deg_f})

    out_path = str(Path(data_path).with_suffix("")) + "_theta.csv"
    results_df.to_csv(out_path, index=False)
    print(f"pcutoff: {pcutoff}, frames with confidence > {pcutoff}: {mask.sum()}/{len(df)}")
    print(f"Saved: {out_path}")

    return results_df
