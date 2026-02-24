#!/usr/bin/env python3
"""
Transfer TensorFlow Akita v2 model weights to PyTorch.

This script transfers pretrained weights from a TensorFlow Akita v2 model
(stored in HDF5 format) to a PyTorch implementation. It handles the necessary
weight format conversions and layer mappings between the two frameworks.

Usage:
    python transfer_tf_to_torch.py --target_idx 0 --data_split 0 --organism human --data_name Krietenstein2019_H1hESC
"""

import argparse
import os
import sys

import h5py
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from akita_model.model import SeqNN

# =============================================================================
# Weight Assignment Utilities
# =============================================================================


def assign_conv_weights(h5_file, tf_layer_path, pytorch_conv_layer):
    """
    Assign weights from a TensorFlow 1D convolutional layer to PyTorch.

    Args:
        h5_file (h5py.File): HDF5 file containing TensorFlow weights
        tf_layer_path (str): Path to the TensorFlow layer in the HDF5 file
        pytorch_conv_layer (torch.nn.Conv1d): PyTorch convolutional layer

    Note:
        TensorFlow Conv1D format: [width, in_channels, out_channels]
        PyTorch Conv1D format: [out_channels, in_channels, width]
    """
    # Load TensorFlow weights
    tf_weights = h5_file[tf_layer_path]["kernel:0"][:]

    # Convert to PyTorch tensor and permute dimensions
    pytorch_weights = torch.tensor(tf_weights, dtype=torch.float32)
    pytorch_weights = pytorch_weights.permute(2, 1, 0)  # TF to PyTorch format

    # Verify shape match
    assert (
        pytorch_weights.shape == pytorch_conv_layer.weight.data.shape
    ), f"Shape mismatch: {pytorch_weights.shape} vs {pytorch_conv_layer.weight.data.shape}"

    # Assign weights
    pytorch_conv_layer.weight.data = pytorch_weights
    print(f"  ✓ Assigned: {tf_layer_path}")


def assign_batch_norm_weights(h5_file, tf_layer_path, pytorch_batch_norm_layer):
    """
    Assign batch normalization weights from TensorFlow to PyTorch.

    Args:
        h5_file (h5py.File): HDF5 file containing TensorFlow weights
        tf_layer_path (str): Path to the TensorFlow BatchNorm layer
        pytorch_batch_norm_layer (torch.nn.BatchNorm1d or torch.nn.BatchNorm2d):
            PyTorch batch normalization layer

    Note:
        Maps TensorFlow parameters to PyTorch:
        - beta (TF) → bias (PyTorch)
        - gamma (TF) → weight (PyTorch)
        - moving_mean (TF) → running_mean (PyTorch)
        - moving_variance (TF) → running_var (PyTorch)
    """
    batch_norm_group = h5_file[tf_layer_path]

    # Extract TensorFlow parameters
    beta = batch_norm_group["beta:0"][:]
    gamma = batch_norm_group["gamma:0"][:]
    moving_mean = batch_norm_group["moving_mean:0"][:]
    moving_variance = batch_norm_group["moving_variance:0"][:]

    # Convert to PyTorch tensors
    pytorch_batch_norm_layer.bias.data = torch.tensor(beta, dtype=torch.float32)
    pytorch_batch_norm_layer.weight.data = torch.tensor(gamma, dtype=torch.float32)
    pytorch_batch_norm_layer.running_mean.data = torch.tensor(moving_mean, dtype=torch.float32)
    pytorch_batch_norm_layer.running_var.data = torch.tensor(moving_variance, dtype=torch.float32)

    print(f"  ✓ Assigned: {tf_layer_path}")


def assign_conv2d_weights(h5_file, tf_layer_path, pytorch_conv2d_layer):
    """
    Assign weights from a TensorFlow 2D convolutional layer to PyTorch.

    Args:
        h5_file (h5py.File): HDF5 file containing TensorFlow weights
        tf_layer_path (str): Path to the TensorFlow Conv2D layer
        pytorch_conv2d_layer (torch.nn.Conv2d): PyTorch Conv2D layer

    Note:
        TensorFlow Conv2D format: [height, width, in_channels, out_channels]
        PyTorch Conv2D format: [out_channels, in_channels, height, width]
    """
    # Load TensorFlow weights
    tf_weights = h5_file[tf_layer_path]["kernel:0"][:]

    # Convert to PyTorch tensor and permute dimensions
    pytorch_weights = torch.tensor(tf_weights, dtype=torch.float32)
    pytorch_weights = pytorch_weights.permute(3, 2, 0, 1)  # TF to PyTorch format

    # Verify shape match
    assert (
        pytorch_weights.shape == pytorch_conv2d_layer.weight.data.shape
    ), f"Shape mismatch: {pytorch_weights.shape} vs {pytorch_conv2d_layer.weight.data.shape}"

    # Assign weights
    pytorch_conv2d_layer.weight.data = pytorch_weights
    print(f"  ✓ Assigned: {tf_layer_path}")


def assign_dense_weights(h5_file, tf_layer_path, pytorch_dense_layer):
    """
    Assign weights and biases from a TensorFlow Dense layer to PyTorch.

    Args:
        h5_file (h5py.File): HDF5 file containing TensorFlow weights
        tf_layer_path (str): Path to the TensorFlow Dense layer
        pytorch_dense_layer (torch.nn.Linear): PyTorch linear (dense) layer

    Note:
        TensorFlow Dense format: [in_features, out_features]
        PyTorch Linear format: [out_features, in_features]
    """
    # Load TensorFlow weights and biases
    tf_kernel = h5_file[tf_layer_path]["kernel:0"][:]
    tf_bias = h5_file[tf_layer_path]["bias:0"][:]

    # Convert to PyTorch tensors
    pytorch_weights = torch.tensor(tf_kernel, dtype=torch.float32)
    pytorch_bias = torch.tensor(tf_bias, dtype=torch.float32)

    # Transpose weights to match PyTorch format
    pytorch_weights = pytorch_weights.t()

    # Verify shape match
    assert (
        pytorch_weights.shape == pytorch_dense_layer.weight.data.shape
    ), f"Shape mismatch: {pytorch_weights.shape} vs {pytorch_dense_layer.weight.data.shape}"
    assert (
        pytorch_bias.shape == pytorch_dense_layer.bias.data.shape
    ), f"Shape mismatch: {pytorch_bias.shape} vs {pytorch_dense_layer.bias.data.shape}"

    # Assign weights and biases
    pytorch_dense_layer.weight.data = pytorch_weights
    pytorch_dense_layer.bias.data = pytorch_bias
    print(f"  ✓ Assigned: {tf_layer_path}")


# =============================================================================
# Main Conversion Logic
# =============================================================================


def transfer_weights(model, h5_file, target_idx, organism):
    """
    Transfer all weights from TensorFlow model to PyTorch model.

    Args:
        model (SeqNN): PyTorch model
        h5_file (h5py.File): HDF5 file with TensorFlow weights
        target_idx (int): Target index for final dense layer
        organism (str): Organism type ('mouse' or 'human')
    """
    print("=" * 70)
    print("Transferring Weights")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # TRUNK - 1D Sequence Processing
    # -------------------------------------------------------------------------
    print("\n[1/5] ConvBlock (initial convolution)")
    assign_conv_weights(h5_file, "model_weights/conv1d/conv1d", model.conv_block_1.conv)
    assign_batch_norm_weights(
        h5_file,
        "model_weights/batch_normalization/batch_normalization",
        model.conv_block_1.batch_norm,
    )

    print("\n[2/5] ConvTower (10 layers)")
    conv_tower_mapping = [
        (1, 1, 2),
        (2, 5, 6),
        (3, 9, 10),
        (4, 13, 14),
        (5, 17, 18),
        (6, 21, 22),
        (7, 25, 26),
        (8, 29, 30),
        (9, 33, 34),
        (10, 37, 38),
    ]
    for tf_idx, conv_idx, bn_idx in conv_tower_mapping:
        assign_conv_weights(
            h5_file,
            f"model_weights/conv1d_{tf_idx}/conv1d_{tf_idx}",
            model.conv_tower.conv_tower[conv_idx],
        )
        assign_batch_norm_weights(
            h5_file,
            f"model_weights/batch_normalization_{tf_idx}/batch_normalization_{tf_idx}",
            model.conv_tower.conv_tower[bn_idx],
        )

    print("\n[3/5] Residual Dilated Blocks 1D (11 blocks)")
    residual_1d_blocks = [
        (11, model.residual1d_block1),
        (13, model.residual1d_block2),
        (15, model.residual1d_block3),
        (17, model.residual1d_block4),
        (19, model.residual1d_block5),
        (21, model.residual1d_block6),
        (23, model.residual1d_block7),
        (25, model.residual1d_block8),
        (27, model.residual1d_block9),
        (29, model.residual1d_block10),
        (31, model.residual1d_block11),
    ]
    for tf_start_idx, block in residual_1d_blocks:
        # First conv + batchnorm
        assign_conv_weights(
            h5_file, f"model_weights/conv1d_{tf_start_idx}/conv1d_{tf_start_idx}", block.conv1
        )
        assign_batch_norm_weights(
            h5_file,
            f"model_weights/batch_normalization_{tf_start_idx}/batch_normalization_{tf_start_idx}",
            block.norm1,
        )
        # Second conv + batchnorm
        assign_conv_weights(
            h5_file,
            f"model_weights/conv1d_{tf_start_idx + 1}/conv1d_{tf_start_idx + 1}",
            block.conv2,
        )
        assign_batch_norm_weights(
            h5_file,
            f"model_weights/batch_normalization_{tf_start_idx + 1}/batch_normalization_{tf_start_idx + 1}",
            block.norm2,
        )

    print("\n[4/5] ConvBlockReduce (channel reduction)")
    assign_conv_weights(h5_file, "model_weights/conv1d_33/conv1d_33", model.conv_reduce.layers[1])
    assign_batch_norm_weights(
        h5_file,
        "model_weights/batch_normalization_33/batch_normalization_33",
        model.conv_reduce.layers[2],
    )

    # -------------------------------------------------------------------------
    # HEAD - 2D Contact Matrix Prediction
    # -------------------------------------------------------------------------
    print("\n[5/5] 2D Convolutional Layers")

    print("  Initial Conv2D block:")
    assign_conv2d_weights(h5_file, "model_weights/conv2d/conv2d", model.conv2d_block.block[1])
    assign_batch_norm_weights(
        h5_file,
        "model_weights/batch_normalization_34/batch_normalization_34",
        model.conv2d_block.block[2],
    )

    print("  Residual Dilated Blocks 2D (6 blocks):")
    residual_2d_blocks = [
        (1, 35, model.residual2d_block1),
        (3, 37, model.residual2d_block2),
        (5, 39, model.residual2d_block3),
        (7, 41, model.residual2d_block4),
        (9, 43, model.residual2d_block5),
        (11, 45, model.residual2d_block6),
    ]
    for conv_idx, bn_start_idx, block in residual_2d_blocks:
        # First conv + batchnorm
        assign_conv2d_weights(
            h5_file, f"model_weights/conv2d_{conv_idx}/conv2d_{conv_idx}", block.conv1
        )
        assign_batch_norm_weights(
            h5_file,
            f"model_weights/batch_normalization_{bn_start_idx}/batch_normalization_{bn_start_idx}",
            block.norm1,
        )
        # Second conv + batchnorm
        assign_conv2d_weights(
            h5_file, f"model_weights/conv2d_{conv_idx + 1}/conv2d_{conv_idx + 1}", block.conv2
        )
        assign_batch_norm_weights(
            h5_file,
            f"model_weights/batch_normalization_{bn_start_idx + 1}/batch_normalization_{bn_start_idx + 1}",
            block.norm2,
        )

    print("  Squeeze-and-Excite:")
    assign_batch_norm_weights(
        h5_file,
        "model_weights/squeeze_excite/squeeze_excite/batch_normalization",
        model.squeeze_excite.norm,
    )
    assign_dense_weights(
        h5_file, "model_weights/squeeze_excite/squeeze_excite/dense", model.squeeze_excite.dense1
    )
    assign_dense_weights(
        h5_file, "model_weights/squeeze_excite/squeeze_excite/dense_1", model.squeeze_excite.dense2
    )

    # -------------------------------------------------------------------------
    # Final Dense Layer (Target-specific)
    # -------------------------------------------------------------------------
    print("\n  Final Dense Layer (target-specific):")
    pytorch_dense_layer = model.final.dense

    # Determine correct TensorFlow layer path based on organism
    # Mouse models use: "model_weights/dense_1/dense_1"
    # Human models use: "model_weights/dense/dense"
    if organism == "mouse":
        dense_layer_path = "model_weights/dense_1/dense_1"
    elif organism == "human":
        dense_layer_path = "model_weights/dense/dense"
    else:
        raise ValueError(f"Unknown organism: {organism}")

    print(f"  Using TensorFlow layer: {dense_layer_path}")

    # Verify the layer exists
    if dense_layer_path not in h5_file:
        raise KeyError(f"Dense layer not found in TensorFlow model: {dense_layer_path}")

    # Extract specific target from multi-target TF model
    tf_kernel = h5_file[dense_layer_path]["kernel:0"][:, target_idx]
    tf_kernel = tf_kernel[None, :]  # Reshape to (1, 80)
    tf_bias = h5_file[dense_layer_path]["bias:0"][target_idx]

    # Assign to PyTorch
    pytorch_dense_layer.weight.data = torch.tensor(tf_kernel, dtype=torch.float32)
    pytorch_dense_layer.bias.data = torch.tensor([tf_bias], dtype=torch.float32)
    print(f"  ✓ Assigned final dense layer for target {target_idx}")


def main():
    """Main transfer function."""
    parser = argparse.ArgumentParser(
        description="Transfer TensorFlow Akita v2 weights to PyTorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target_idx", type=int, required=True, help="Target index to extract from TF model"
    )
    parser.add_argument(
        "--data_split", type=int, required=True, help="Data split/fold index (e.g., 0-7)"
    )
    parser.add_argument(
        "--organism",
        type=str,
        required=True,
        choices=["human", "mouse"],
        help="Organism type (determines model index)",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="Krietenstein2019_H1hESC",
        help="Dataset name for output path",
    )
    parser.add_argument(
        "--tf_model_dir",
        type=str,
        default="/project2/fudenber_735/tensorflow_models/akita/v2/models",
        help="Directory containing TensorFlow models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home1/smaruj/pytorch_akita/models/pretrained",
        help="Output directory for PyTorch models",
    )

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("TensorFlow to PyTorch Weight Transfer")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Organism: {args.organism}")
    print(f"Data split: {args.data_split}")
    print(f"Target index: {args.target_idx}")
    print(f"Dataset: {args.data_name}")
    print("=" * 70)
    print()

    # Determine model index based on organism
    organism = args.organism.lower()
    model_idx = 0 if organism == "human" else 1

    # Construct paths
    tf_model_path = f"{args.tf_model_dir}/f{args.data_split}c0/train/model{model_idx}_best.h5"
    output_dir = f"{args.output_dir}/{organism}/{args.data_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/Akita_v2_{organism}_{args.data_name}_model{args.data_split}.pth"

    # Verify TensorFlow model exists
    if not os.path.exists(tf_model_path):
        raise FileNotFoundError(f"TensorFlow model not found: {tf_model_path}")

    print("Loading TensorFlow weights from:")
    print(f"  {tf_model_path}")
    print()

    # Load TensorFlow weights
    h5_file = h5py.File(tf_model_path, "r")

    # Initialize PyTorch model
    print("Initializing PyTorch model...")
    model = SeqNN()
    model = model.to(device)
    model.eval()
    print("✓ Model initialized")
    print()

    # Transfer weights
    transfer_weights(model, h5_file, args.target_idx, organism)

    # Save model
    print()
    print("=" * 70)
    print("Saving Model")
    print("=" * 70)
    torch.save(model.state_dict(), output_file)
    print(f"✓ Model saved to: {output_file}")
    print("=" * 70)
    print()
    print("Transfer complete!")


if __name__ == "__main__":
    main()
