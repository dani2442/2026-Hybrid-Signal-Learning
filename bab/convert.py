"""TF checkpoint inspection and DLC TF -> PyTorch weight conversion."""

import os

import torch

from .models import DLCResNet50


def inspect_tf_checkpoint(ckpt_path: str):
    """Lists all variable names and shapes in a TensorFlow checkpoint."""
    import tensorflow as tf

    reader = tf.train.load_checkpoint(ckpt_path)
    shape_map = reader.get_variable_to_shape_map()
    for name in sorted(shape_map.keys()):
        print(f"  {name:70s} {shape_map[name]}")
    print(f"\nTotal: {len(shape_map)} variables")
    return reader, shape_map


def build_tf_to_pytorch_mapping(num_keypoints=2):
    """
    Build a dictionary mapping TF variable names -> (PyTorch key, transform_type).

    transform_type:
      'conv'   — permute HWIO -> OIHW
      'deconv' — permute HWOI -> IOHW
      'direct' — copy as-is
    """
    mapping = {}

    # Stem
    mapping["resnet_v1_50/conv1/weights"] = ("conv1.weight", "conv")
    for suffix, pt_suffix in [("gamma", "weight"), ("beta", "bias"),
                               ("moving_mean", "running_mean"),
                               ("moving_variance", "running_var")]:
        mapping[f"resnet_v1_50/conv1/BatchNorm/{suffix}"] = (f"bn1.{pt_suffix}", "direct")

    # Blocks 1-4
    block_units = {"block1": 3, "block2": 4, "block3": 6, "block4": 3}
    for bname, n_units in block_units.items():
        for ui in range(n_units):
            tf_prefix = f"resnet_v1_50/{bname}/unit_{ui+1}/bottleneck_v1"
            pt_prefix = f"{bname}.{ui}"

            for ci, cname in enumerate(["conv1", "conv2", "conv3"], start=1):
                mapping[f"{tf_prefix}/{cname}/weights"] = (
                    f"{pt_prefix}.{cname}.weight", "conv")
                for suffix, pt_suffix in [("gamma", "weight"), ("beta", "bias"),
                                           ("moving_mean", "running_mean"),
                                           ("moving_variance", "running_var")]:
                    mapping[f"{tf_prefix}/{cname}/BatchNorm/{suffix}"] = (
                        f"{pt_prefix}.bn{ci}.{pt_suffix}", "direct")

            # Shortcut (only first unit of each block has learned shortcut)
            if ui == 0:
                mapping[f"{tf_prefix}/shortcut/weights"] = (
                    f"{pt_prefix}.shortcut.0.weight", "conv")
                for suffix, pt_suffix in [("gamma", "weight"), ("beta", "bias"),
                                           ("moving_mean", "running_mean"),
                                           ("moving_variance", "running_var")]:
                    mapping[f"{tf_prefix}/shortcut/BatchNorm/{suffix}"] = (
                        f"{pt_prefix}.shortcut.1.{pt_suffix}", "direct")

    # Prediction heads
    mapping["pose/part_pred/block4/weights"] = ("part_pred.weight", "deconv")
    mapping["pose/part_pred/block4/biases"] = ("part_pred.bias", "direct")
    mapping["pose/locref_pred/block4/weights"] = ("locref_pred.weight", "deconv")
    mapping["pose/locref_pred/block4/biases"] = ("locref_pred.bias", "direct")

    return mapping


def convert_tf_weight(tf_array, mode):
    """Convert a single TF numpy array to PyTorch tensor format."""
    if mode == "conv":
        return torch.from_numpy(tf_array.transpose(3, 2, 0, 1).copy())
    elif mode == "deconv":
        return torch.from_numpy(tf_array.transpose(3, 2, 0, 1).copy())
    elif mode == "direct":
        return torch.from_numpy(tf_array.copy())
    else:
        raise ValueError(f"Unknown mode: {mode}")


def convert_dlc_tf_to_pytorch(ckpt_path: str, num_keypoints: int = 2,
                              save_path: str = None) -> DLCResNet50:
    """
    Converts a DLC TensorFlow checkpoint to a DLCResNet50 PyTorch model.

    Args:
        ckpt_path: path to TF checkpoint (without .index/.data extension)
        num_keypoints: number of keypoints (2 for BAB)
        save_path: if provided, save the converted model as a .pt file

    Returns:
        DLCResNet50 model with loaded weights in eval mode.
    """
    import tensorflow as tf

    # 1) Read TF checkpoint
    reader = tf.train.load_checkpoint(ckpt_path)
    tf_shapes = reader.get_variable_to_shape_map()
    print(f"TF checkpoint: {len(tf_shapes)} variables")

    # 2) Create PyTorch model
    model = DLCResNet50(num_keypoints=num_keypoints, location_refinement=True)
    pt_state = model.state_dict()
    print(f"PyTorch model: {len(pt_state)} parameters")

    # 3) Build mapping and transfer weights
    mapping = build_tf_to_pytorch_mapping(num_keypoints)
    loaded, skipped, mismatched = 0, 0, 0

    for tf_name, (pt_key, mode) in mapping.items():
        if tf_name not in tf_shapes:
            print(f"  SKIP (not in checkpoint): {tf_name}")
            skipped += 1
            continue
        if pt_key not in pt_state:
            print(f"  SKIP (not in model): {pt_key}")
            skipped += 1
            continue

        tf_val = reader.get_tensor(tf_name)
        pt_val = convert_tf_weight(tf_val, mode)

        if pt_val.shape != pt_state[pt_key].shape:
            print(f"  MISMATCH: {tf_name} {tf_val.shape} -> {pt_val.shape} "
                  f"vs expected {pt_state[pt_key].shape}")
            mismatched += 1
            continue

        pt_state[pt_key] = pt_val
        loaded += 1

    # 4) Load into model
    model.load_state_dict(pt_state)
    model.eval()
    print(f"\nConversion complete: {loaded} loaded, {skipped} skipped, {mismatched} mismatched")

    # 5) Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(),
                     "source_checkpoint": ckpt_path,
                     "num_keypoints": num_keypoints},
                   save_path)
        print(f"Saved PyTorch model: {save_path}")

    return model
