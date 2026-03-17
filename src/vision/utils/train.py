"""Mode runners for the BAB multimodal training CLI.

Each ``_run_*`` function owns one high-level pipeline mode so
``examples/run_pipeline.py`` can stay focused on CLI setup and data loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.visualization.pipeline_plots import (
    plot_sensor_to_video_coords,
    plot_sensor_to_video_overlay,
)

from ..datasets import FrameStateDataset, WindowedSequenceDataset
from ..eval import evaluate_decoder, evaluate_end2end
from ..models import DecoderKeypointsMLP, EncOdeDecModel
from ..pipeline_utils import (
    evaluate_and_plot_encoder,
    evaluate_and_plot_ode_free_run,
    evaluate_ode_test_rollout,
    load_ode_func,
    require_keypoints_sensor,
    resolve_ode_training_targets,
    save_json,
    save_metrics_csv,
    select_evenly_spaced,
    split_indices,
    states_from_sensor,
    train_image_decoder_from_sensor,
    train_ode_model_separate,
)
from ..train import train_encoder, train_end_to_end


Runner = Callable[[Any, Any, Any, np.ndarray, np.ndarray, dict[str, Any], Path], None]

_MODES_WITH_ENCODER = {
    "encoder_only",
    "enc_ode",
    "enc_ode_dec",
    "separate",
    "enc_dec",
}


def mode_requires_encoder(config) -> bool:
    """Return whether the selected mode needs an encoder instance."""
    if config.mode in _MODES_WITH_ENCODER:
        return True
    return config.mode == "ode_retrain" and bool(config.ode_use_encoder_labels)


def _sensor_dt(data) -> float:
    return 1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0


def _run_encoder(config, encoder, data, frames, frame_idx_map, aux, run_dir) -> None:
    """Train and evaluate only the image-to-sensor encoder."""
    assert encoder is not None

    frame_ds = FrameStateDataset(data, frames, frame_idx_map)
    train_ds, val_ds, test_ds = frame_ds.split()
    print(f"Splits: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    result = train_encoder(
        encoder,
        train_ds,
        val_ds,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        device=config.device,
        seed=config.seed,
        wandb_project=config.wandb_project,
        wandb_run_name=f"enc_{config.dataset}",
        checkpoint_dir=str(run_dir),
    )
    print(f"Best val loss: {result['best_val_loss']:.6f}")

    metrics, _ = evaluate_and_plot_encoder(
        encoder,
        test_ds,
        aux,
        run_dir,
        plot_dataset=frame_ds,
        device=config.device,
    )
    print("Video->sensor metrics:", metrics)


def _run_decoder(config, _encoder, data, frames, frame_idx_map, _aux, run_dir) -> None:
    """Train and evaluate only the sensor-to-image decoder."""
    decoder_run = train_image_decoder_from_sensor(
        config,
        data,
        frames,
        frame_idx_map,
        run_dir,
        run_name=f"dec_img_{config.dataset}",
    )
    print(f"Decoder best val loss: {decoder_run['result']['best_val_loss']:.6f}")
    print("Sensor->video metrics:", decoder_run["metrics"])


def _run_enc_ode(config, encoder, data, frames, frame_idx_map, aux, run_dir) -> None:
    """Train the encoder and ODE jointly, without a decoder head."""
    _run_enc_ode_common(
        config,
        encoder,
        data,
        frames,
        frame_idx_map,
        aux,
        run_dir,
        with_decoder=False,
    )


def _run_enc_ode_dec(config, encoder, data, frames, frame_idx_map, aux, run_dir) -> None:
    """Train the encoder, ODE, and keypoint decoder jointly."""
    _run_enc_ode_common(
        config,
        encoder,
        data,
        frames,
        frame_idx_map,
        aux,
        run_dir,
        with_decoder=True,
    )


def _run_enc_ode_common(
    config,
    encoder,
    data,
    frames,
    frame_idx_map,
    aux,
    run_dir,
    *,
    with_decoder: bool,
) -> None:
    """Shared implementation for the joint encoder/ODE modes."""
    assert encoder is not None

    keypoints_sensor = require_keypoints_sensor(aux) if with_decoder else None
    seq_ds = WindowedSequenceDataset(
        data,
        frames,
        frame_idx_map,
        k_steps=config.k_steps,
        keypoints_sensor=keypoints_sensor,
    )
    train_ds, val_ds, test_ds = seq_ds.split()
    print(f"Sequence splits: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
    print(
        "Joint training mode: "
        f"{config.joint_training_mode} "
        f"(steps_per_epoch={config.joint_steps_per_epoch})"
    )

    ode_func = load_ode_func(config)
    decoder = DecoderKeypointsMLP() if with_decoder else None
    composite = EncOdeDecModel(
        encoder=encoder,
        ode_func=ode_func,
        decoder=decoder,
        dt=_sensor_dt(data),
    )

    result = train_end_to_end(
        composite,
        train_ds,
        val_ds,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        device=config.device,
        seed=config.seed,
        freeze_ode_epochs=config.freeze_ode_epochs,
        training_mode=config.joint_training_mode,
        steps_per_epoch=config.joint_steps_per_epoch,
        wandb_project=config.wandb_project,
        wandb_run_name=f"{config.mode}_{config.dataset}",
        checkpoint_dir=str(run_dir),
    )
    print(f"Best val loss: {result['best_val_loss']:.6f}")

    # Reuse the per-frame dataset for encoder metrics and plots.
    frame_ds = FrameStateDataset(data, frames, frame_idx_map)
    _, _, frame_test_ds = frame_ds.split()
    enc_metrics, _ = evaluate_and_plot_encoder(
        encoder,
        frame_test_ds,
        aux,
        run_dir,
        plot_dataset=frame_ds,
        device=config.device,
    )

    ode_metrics = evaluate_ode_test_rollout(
        data,
        test_ds,
        composite,
        run_dir,
        device=config.device,
    )

    if decoder is not None and keypoints_sensor is not None:
        # Decoder metrics use the same flat train/val/test split as the image decoder.
        states = states_from_sensor(data)
        _, _, test_idx = split_indices(len(states))
        dec_metrics = evaluate_decoder(
            decoder,
            states[test_idx],
            keypoints_sensor[test_idx],
            device=config.device,
        )
        save_metrics_csv(dec_metrics, run_dir / "metrics_sensor_to_video.csv")
        plot_sensor_to_video_coords(
            run_dir / "plot_sensor_to_video_coords.png",
            decoder,
            states[test_idx],
            keypoints_sensor[test_idx],
            t=np.asarray(data.t, dtype=float)[test_idx],
            device=config.device,
        )
        overlay_sel = select_evenly_spaced(test_idx, n=config.plot_count)
        plot_sensor_to_video_overlay(
            run_dir / "plot_sensor_to_video_overlay.png",
            decoder,
            states,
            keypoints_sensor,
            frames,
            frame_idx_map,
            overlay_sel,
            device=config.device,
        )

    e2e_metrics = evaluate_end2end(composite, test_ds, device=config.device)
    save_metrics_csv(e2e_metrics, run_dir / "metrics_end2end.csv")
    print("Video->sensor metrics:", enc_metrics)
    print("Sensor->future sensor metrics:", ode_metrics)
    if decoder is not None and keypoints_sensor is not None:
        print("Sensor->video metrics saved to metrics_sensor_to_video.csv")


def _run_ode_dec(config, _encoder, data, frames, frame_idx_map, _aux, run_dir) -> None:
    """Train the ODE and image decoder separately, then report both."""
    decoder_run = train_image_decoder_from_sensor(
        config,
        data,
        frames,
        frame_idx_map,
        run_dir,
        run_name=f"dec_ode_{config.dataset}",
    )
    ode_model, tr, va, te = train_ode_model_separate(config, data, run_dir)
    ode_eval = evaluate_and_plot_ode_free_run(
        ode_model,
        data,
        te,
        run_dir,
        use_sensor_y_dot=config.ode_init_from_y_dot,
    )

    save_json(
        {
            "ode_train_points": int(len(tr)),
            "ode_val_points": int(len(va)),
            "ode_test_points": int(len(te)),
            "decoder_best_val_loss": float(decoder_run["result"]["best_val_loss"]),
        },
        run_dir / "ode_dec_summary.json",
    )
    print(f"Decoder best val loss: {decoder_run['result']['best_val_loss']:.6f}")
    print("Sensor->future sensor metrics:", ode_eval["metrics"])
    print("Sensor->video metrics:", decoder_run["metrics"])


def _run_separate(config, encoder, data, frames, frame_idx_map, aux, run_dir) -> None:
    """Train encoder, ODE, and image decoder independently, then evaluate all."""
    assert encoder is not None

    frame_ds = FrameStateDataset(data, frames, frame_idx_map)
    enc_train_ds, enc_val_ds, enc_test_ds = frame_ds.split()
    print(
        f"Encoder splits: {len(enc_train_ds)} train / "
        f"{len(enc_val_ds)} val / {len(enc_test_ds)} test"
    )

    enc_result = train_encoder(
        encoder,
        enc_train_ds,
        enc_val_ds,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        device=config.device,
        seed=config.seed,
        wandb_project=config.wandb_project,
        wandb_run_name=f"enc_sep_{config.dataset}",
        checkpoint_dir=str(run_dir),
    )
    print(f"Encoder best val loss: {enc_result['best_val_loss']:.6f}")

    enc_metrics, _ = evaluate_and_plot_encoder(
        encoder,
        enc_test_ds,
        aux,
        run_dir,
        plot_dataset=frame_ds,
        device=config.device,
    )

    # Optionally swap the ODE targets with encoder predictions before training.
    y_override = resolve_ode_training_targets(config, encoder, frame_ds, data)
    ode_model, tr, va, te = train_ode_model_separate(
        config,
        data,
        run_dir,
        y_override=y_override,
    )

    source = ("finite-difference theta", "sensor y_dot")[config.ode_init_from_y_dot]
    print(f"Using {source} for ODE rollout initial theta_dot.")
    ode_eval = evaluate_and_plot_ode_free_run(
        ode_model,
        data,
        te,
        run_dir,
        use_sensor_y_dot=config.ode_init_from_y_dot,
        encoder=encoder,
        frame_dataset=frame_ds,
        device=config.device,
    )
    ode_metrics = ode_eval["metrics"]

    decoder_run = train_image_decoder_from_sensor(
        config,
        data,
        frames,
        frame_idx_map,
        run_dir,
        run_name=f"dec_sep_{config.dataset}",
    )
    print(f"Decoder best val loss: {decoder_run['result']['best_val_loss']:.6f}")
    dec_metrics = decoder_run["metrics"]

    save_json(
        {
            "encoder_best_val_loss": float(enc_result["best_val_loss"]),
            "ode_train_points": int(len(tr)),
            "ode_val_points": int(len(va)),
            "ode_test_points": int(len(te)),
            "decoder_best_val_loss": float(decoder_run["result"]["best_val_loss"]),
        },
        run_dir / "separate_training_summary.json",
    )
    print("Video->sensor metrics:", enc_metrics)
    print("Sensor->future sensor (free-run) metrics:", ode_metrics)
    if "video_chain_metrics" in ode_eval:
        print("Video->sensor->future sensor metrics:", ode_eval["video_chain_metrics"])
    print("Sensor->video metrics:", dec_metrics)


def _run_ode(config, encoder, data, frames, frame_idx_map, _aux, run_dir) -> None:
    """Train only the ODE, either from ground-truth labels or an encoder checkpoint."""
    y_override = None
    if config.ode_use_encoder_labels:
        if encoder is None:
            raise ValueError("ode_retrain with ode_use_encoder_labels=True requires an encoder.")
        if not config.encoder_checkpoint:
            raise ValueError(
                "--encoder-checkpoint is required when ode_use_encoder_labels=True "
                "for --mode ode_retrain."
            )

        import torch

        print(f"Loading encoder checkpoint: {config.encoder_checkpoint}")
        encoder.load_state_dict(
            torch.load(config.encoder_checkpoint, map_location="cpu", weights_only=True)
        )
        encoder.eval()

        frame_ds = FrameStateDataset(data, frames, frame_idx_map)
        y_override = resolve_ode_training_targets(config, encoder, frame_ds, data)

    ode_model, _tr, _va, te = train_ode_model_separate(
        config,
        data,
        run_dir,
        y_override=y_override,
    )

    source = ("finite-difference theta", "sensor y_dot")[config.ode_init_from_y_dot]
    print(f"Using {source} for ODE rollout initial theta_dot.")
    ode_eval = evaluate_and_plot_ode_free_run(
        ode_model,
        data,
        te,
        run_dir,
        use_sensor_y_dot=config.ode_init_from_y_dot,
    )
    _print_ode_parameter_summary(ode_model)
    print("Sensor->future sensor (free-run) metrics:", ode_eval["metrics"])


def _run_enc_dec(config, encoder, data, frames, frame_idx_map, aux, run_dir) -> None:
    """Placeholder for the planned encoder+decoder mode without an ODE block."""
    del config, encoder, data, frames, frame_idx_map, aux, run_dir
    raise NotImplementedError("enc_dec mode is not implemented yet.")


def _print_ode_parameter_summary(ode_model) -> None:
    if not hasattr(ode_model, "ode_func_") or ode_model.ode_func_ is None:
        return

    print("\nLearned ODE parameters:")
    for name, param in ode_model.ode_func_.named_parameters():
        val = param.detach().cpu().numpy()
        suffix = f"  (exp = {np.exp(val):.6f})" if "log_" in name else ""
        print(f"  {name} = {val:.4f}{suffix}")


MODE_RUNNERS: dict[str, Runner] = {
    "encoder_only": _run_encoder,      # Train and evaluate only the encoder (video->sensor).
    "decoder_only": _run_decoder,      # Train and evaluate only the decoder (sensor->video).
    "enc_ode": _run_enc_ode,           # Train encoder + ODE together, no decoder.
    "enc_ode_dec": _run_enc_ode_dec,   # Train encoder + ODE + decoder together.
    "ode_dec": _run_ode_dec,           # Train ODE + decoder together.
    "separate": _run_separate,         # Train encoder, ODE, and decoder separately; then evaluate all.
    "ode_retrain": _run_ode,           # Train only the ODE, using labels or a trained encoder.
    "enc_dec": _run_enc_dec,           # Train encoder + decoder together, no ODE.
}


__all__ = ["MODE_RUNNERS", "mode_requires_encoder"]
