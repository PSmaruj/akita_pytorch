#!/usr/bin/env python3
"""
Transfer TensorFlow Akita model weights to a compatible PyTorch model.

Usage:
    python transfer_tf_to_torch_average.py --data_split 0 --organism mouse
"""
import argparse
import os
import torch
import h5py
from model_v2_compatible import SeqNN


# ------------------------ Weight assignment utilities ------------------------ #

def assign_conv_weights(h5_file, tf_layer_path, pytorch_conv_layer):
    """
    Assign weights from a TensorFlow convolutional layer to a PyTorch convolutional layer.
    
    Parameters:
        h5_file (h5py.File): The HDF5 file containing TensorFlow weights.
        tf_layer_path (str): Path to the TensorFlow layer in the HDF5 file.
        pytorch_conv_layer (torch.nn.Conv1d or torch.nn.Conv2d): The PyTorch convolutional layer.

    """
    # Access TensorFlow kernel weights
    tf_weights = h5_file[tf_layer_path]['kernel:0'][:]
    
    # Convert to PyTorch tensor
    pytorch_weights = torch.tensor(tf_weights, dtype=torch.float32)
    
    # Transpose TensorFlow weights to match PyTorch's format
    pytorch_weights = pytorch_weights.permute(2, 1, 0)
    
    # Ensure the shapes match
    assert pytorch_weights.shape == pytorch_conv_layer.weight.data.shape, \
        f"Shape mismatch: {pytorch_weights.shape} vs {pytorch_conv_layer.weight.data.shape}"
    
    # Assign weights to the PyTorch layer
    pytorch_conv_layer.weight.data = pytorch_weights

    print(f"Assigned weights to PyTorch layer: {pytorch_conv_layer}")
 
    
def assign_batch_norm_weights(h5_file, tf_layer_path, pytorch_batch_norm_layer):
    """
    Assign batch normalization weights from TensorFlow to PyTorch.
    
    Parameters:
        h5_file (h5py.File): The HDF5 file containing TensorFlow weights.
        tf_layer_path (str): Path to the TensorFlow batch normalization layer in the HDF5 file.
        pytorch_batch_norm_layer (torch.nn.BatchNorm1d or torch.nn.BatchNorm2d): The PyTorch batch normalization layer.
    """
    # Access TensorFlow BatchNorm parameters
    batch_norm_group = h5_file[tf_layer_path]

    # Extract TensorFlow parameters
    beta = batch_norm_group["beta:0"][:]  # Bias (beta in TensorFlow)
    gamma = batch_norm_group["gamma:0"][:]  # Scale (gamma in TensorFlow)
    moving_mean = batch_norm_group["moving_mean:0"][:]  # Running mean
    moving_variance = batch_norm_group["moving_variance:0"][:]  # Running variance

    # Convert to PyTorch tensors
    beta_tensor = torch.tensor(beta, dtype=torch.float32)
    gamma_tensor = torch.tensor(gamma, dtype=torch.float32)
    moving_mean_tensor = torch.tensor(moving_mean, dtype=torch.float32)
    moving_variance_tensor = torch.tensor(moving_variance, dtype=torch.float32)

    # Assign values to the PyTorch BatchNorm layer
    pytorch_batch_norm_layer.bias.data = beta_tensor
    pytorch_batch_norm_layer.weight.data = gamma_tensor
    pytorch_batch_norm_layer.running_mean.data = moving_mean_tensor
    pytorch_batch_norm_layer.running_var.data = moving_variance_tensor

    print(f"Assigned batch normalization weights to PyTorch layer: {pytorch_batch_norm_layer}")

    
def assign_conv2d_weights(h5_file, tf_layer_path, pytorch_conv2d_layer):
    """
    Assign weights from a TensorFlow Conv2d layer to a PyTorch Conv2d layer.
    
    Parameters:
        h5_file (h5py.File): The HDF5 file containing TensorFlow weights.
        tf_layer_path (str): Path to the TensorFlow Conv2d layer in the HDF5 file.
        pytorch_conv2d_layer (torch.nn.Conv2d): The PyTorch Conv2d layer.
    """
    # Access TensorFlow Conv2D weights
    tf_weights = h5_file[tf_layer_path]['kernel:0'][:]
    
    # Convert to PyTorch tensor
    pytorch_weights = torch.tensor(tf_weights, dtype=torch.float32)
    
    # Transpose TensorFlow weights to match PyTorch's format
    pytorch_weights = pytorch_weights.permute(3, 2, 0, 1)  # [output_channels, input_channels, filter_height, filter_width]
    
    # Ensure the shapes match
    assert pytorch_weights.shape == pytorch_conv2d_layer.weight.data.shape, \
        f"Shape mismatch: {pytorch_weights.shape} vs {pytorch_conv2d_layer.weight.data.shape}"
    
    # Assign weights to the PyTorch layer
    pytorch_conv2d_layer.weight.data = pytorch_weights

    print(f"Assigned Conv2D weights to PyTorch layer: {pytorch_conv2d_layer}")

    
def assign_dense_weights(h5_file, tf_layer_path, pytorch_dense_layer):
    """
    Assign weights and biases from a TensorFlow Dense layer to a PyTorch Dense layer.
    
    Parameters:
        h5_file (h5py.File): The HDF5 file containing TensorFlow weights.
        tf_layer_path (str): Path to the TensorFlow dense layer in the HDF5 file.
        pytorch_dense_layer (torch.nn.Linear): The PyTorch dense (fully connected) layer.
    """
    # Access TensorFlow Dense layer parameters (weights and biases)
    tf_kernel = h5_file[tf_layer_path]['kernel:0'][:]  # Weights (input_units, output_units)
    tf_bias = h5_file[tf_layer_path]['bias:0'][:]      # Bias (output_units,)
    
    # Convert to PyTorch tensors
    pytorch_weights = torch.tensor(tf_kernel, dtype=torch.float32)
    pytorch_bias = torch.tensor(tf_bias, dtype=torch.float32)
    
    # Transpose TensorFlow weights to match PyTorch's format
    pytorch_weights = pytorch_weights.t()  # (output_units, input_units)
    
    # Ensure the shapes match
    assert pytorch_weights.shape == pytorch_dense_layer.weight.data.shape, \
        f"Shape mismatch: {pytorch_weights.shape} vs {pytorch_dense_layer.weight.data.shape}"
    assert pytorch_bias.shape == pytorch_dense_layer.bias.data.shape, \
        f"Shape mismatch: {pytorch_bias.shape} vs {pytorch_dense_layer.bias.data.shape}"
    
    # Assign weights and biases to the PyTorch layer
    pytorch_dense_layer.weight.data = pytorch_weights
    pytorch_dense_layer.bias.data = pytorch_bias

    print(f"Assigned Dense layer weights and biases to PyTorch layer: {pytorch_dense_layer}")   


# ------------------------ Main conversion logic ------------------------ #

def main(data_split: int, organism: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine model index based on organism
    organism = organism.lower()
    if organism == "human":
        model_idx = 0
    elif organism == "mouse":
        model_idx = 1
    else:
        raise ValueError("organism must be 'human' or 'mouse'")
    
    # Paths
    tf_model_path = f"/project2/fudenber_735/tensorflow_models/akita/v2/models/f{data_split}c0/train/model{model_idx}_best.h5"
    # save_path = f"/scratch1/smaruj/Akita_pytorch_models/tf_transferred/{organism}_models/{data_name}"
    save_path = f"/scratch1/smaruj/Akita_pytorch_models/tf_transferred/random_dense_layer"
    os.makedirs(save_path, exist_ok=True)
    output_file = f"{save_path}/Akita_v2_random_dense_model{data_split}.pth"

    # Load model and weights
    print(f"Loading TensorFlow weights from: {tf_model_path}")
    h5_file = h5py.File(tf_model_path, "r")

    model = SeqNN()
    model = model.to(device)
    model.eval()

    # ------------------------ Assign weights ------------------------ #
    print("Assigning convolutional and normalization layers...")
    
    # TRUNK

    # ConvBlock
    assign_conv_weights(h5_file, "model_weights/conv1d/conv1d", model.conv_block_1.conv)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization/batch_normalization", model.conv_block_1.batch_norm)

    # ConvTower
    # 1
    assign_conv_weights(h5_file, "model_weights/conv1d_1/conv1d_1", model.conv_tower.conv_tower[1])
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_1/batch_normalization_1", model.conv_tower.conv_tower[2])
    # 2
    assign_conv_weights(h5_file, "model_weights/conv1d_2/conv1d_2", model.conv_tower.conv_tower[5])
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_2/batch_normalization_2", model.conv_tower.conv_tower[6])
    # 3
    assign_conv_weights(h5_file, "model_weights/conv1d_3/conv1d_3", model.conv_tower.conv_tower[9])
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_3/batch_normalization_3", model.conv_tower.conv_tower[10])
    # 4
    assign_conv_weights(h5_file, "model_weights/conv1d_4/conv1d_4", model.conv_tower.conv_tower[13])
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_4/batch_normalization_4", model.conv_tower.conv_tower[14])
    # 5
    assign_conv_weights(h5_file, "model_weights/conv1d_5/conv1d_5", model.conv_tower.conv_tower[17])
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_5/batch_normalization_5", model.conv_tower.conv_tower[18])
    # 6
    assign_conv_weights(h5_file, "model_weights/conv1d_6/conv1d_6", model.conv_tower.conv_tower[21])
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_6/batch_normalization_6", model.conv_tower.conv_tower[22])
    # 7
    assign_conv_weights(h5_file, "model_weights/conv1d_7/conv1d_7", model.conv_tower.conv_tower[25])
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_7/batch_normalization_7", model.conv_tower.conv_tower[26])
    # 8
    assign_conv_weights(h5_file, "model_weights/conv1d_8/conv1d_8", model.conv_tower.conv_tower[29])
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_8/batch_normalization_8", model.conv_tower.conv_tower[30])
    # 9
    assign_conv_weights(h5_file, "model_weights/conv1d_9/conv1d_9", model.conv_tower.conv_tower[33])
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_9/batch_normalization_9", model.conv_tower.conv_tower[34])
    # 10
    assign_conv_weights(h5_file, "model_weights/conv1d_10/conv1d_10", model.conv_tower.conv_tower[37])
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_10/batch_normalization_10", model.conv_tower.conv_tower[38])

    # ResidualDilatedBlock

    # 1
    assign_conv_weights(h5_file, "model_weights/conv1d_11/conv1d_11", model.residual1d_block1.conv1)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_11/batch_normalization_11", model.residual1d_block1.norm1)

    assign_conv_weights(h5_file, "model_weights/conv1d_12/conv1d_12", model.residual1d_block1.conv2)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_12/batch_normalization_12", model.residual1d_block1.norm2)

    # 2
    assign_conv_weights(h5_file, "model_weights/conv1d_13/conv1d_13", model.residual1d_block2.conv1)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_13/batch_normalization_13", model.residual1d_block2.norm1)

    assign_conv_weights(h5_file, "model_weights/conv1d_14/conv1d_14", model.residual1d_block2.conv2)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_14/batch_normalization_14", model.residual1d_block2.norm2)

    # 3
    assign_conv_weights(h5_file, "model_weights/conv1d_15/conv1d_15", model.residual1d_block3.conv1)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_15/batch_normalization_15", model.residual1d_block3.norm1)

    assign_conv_weights(h5_file, "model_weights/conv1d_16/conv1d_16", model.residual1d_block3.conv2)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_16/batch_normalization_16", model.residual1d_block3.norm2)

    # 4
    assign_conv_weights(h5_file, "model_weights/conv1d_17/conv1d_17", model.residual1d_block4.conv1)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_17/batch_normalization_17", model.residual1d_block4.norm1)

    assign_conv_weights(h5_file, "model_weights/conv1d_18/conv1d_18", model.residual1d_block4.conv2)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_18/batch_normalization_18", model.residual1d_block4.norm2)

    # 5
    assign_conv_weights(h5_file, "model_weights/conv1d_19/conv1d_19", model.residual1d_block5.conv1)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_19/batch_normalization_19", model.residual1d_block5.norm1)

    assign_conv_weights(h5_file, "model_weights/conv1d_20/conv1d_20", model.residual1d_block5.conv2)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_20/batch_normalization_20", model.residual1d_block5.norm2)

    # 6
    assign_conv_weights(h5_file, "model_weights/conv1d_21/conv1d_21", model.residual1d_block6.conv1)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_21/batch_normalization_21", model.residual1d_block6.norm1)

    assign_conv_weights(h5_file, "model_weights/conv1d_22/conv1d_22", model.residual1d_block6.conv2)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_22/batch_normalization_22", model.residual1d_block6.norm2)

    # 7
    assign_conv_weights(h5_file, "model_weights/conv1d_23/conv1d_23", model.residual1d_block7.conv1)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_23/batch_normalization_23", model.residual1d_block7.norm1)

    assign_conv_weights(h5_file, "model_weights/conv1d_24/conv1d_24", model.residual1d_block7.conv2)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_24/batch_normalization_24", model.residual1d_block7.norm2)

    # 8
    assign_conv_weights(h5_file, "model_weights/conv1d_25/conv1d_25", model.residual1d_block8.conv1)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_25/batch_normalization_25", model.residual1d_block8.norm1)

    assign_conv_weights(h5_file, "model_weights/conv1d_26/conv1d_26", model.residual1d_block8.conv2)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_26/batch_normalization_26", model.residual1d_block8.norm2)

    # 9
    assign_conv_weights(h5_file, "model_weights/conv1d_27/conv1d_27", model.residual1d_block9.conv1)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_27/batch_normalization_27", model.residual1d_block9.norm1)

    assign_conv_weights(h5_file, "model_weights/conv1d_28/conv1d_28", model.residual1d_block9.conv2)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_28/batch_normalization_28", model.residual1d_block9.norm2)

    # 10
    assign_conv_weights(h5_file, "model_weights/conv1d_29/conv1d_29", model.residual1d_block10.conv1)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_29/batch_normalization_29", model.residual1d_block10.norm1)

    assign_conv_weights(h5_file, "model_weights/conv1d_30/conv1d_30", model.residual1d_block10.conv2)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_30/batch_normalization_30", model.residual1d_block10.norm2)

    # 11
    assign_conv_weights(h5_file, "model_weights/conv1d_31/conv1d_31", model.residual1d_block11.conv1)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_31/batch_normalization_31", model.residual1d_block11.norm1)

    assign_conv_weights(h5_file, "model_weights/conv1d_32/conv1d_32", model.residual1d_block11.conv2)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_32/batch_normalization_32", model.residual1d_block11.norm2)

    # ConvBlockReduce
    assign_conv_weights(h5_file, "model_weights/conv1d_33/conv1d_33", model.conv_reduce.layers[1])
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_33/batch_normalization_33", model.conv_reduce.layers[2])

    # HEAD

    assign_conv2d_weights(h5_file, "model_weights/conv2d/conv2d", model.conv2d_block.block[1])
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_34/batch_normalization_34", model.conv2d_block.block[2])

    # ResidualDilatedBlock - 2D

    # 1
    assign_conv2d_weights(h5_file, "model_weights/conv2d_1/conv2d_1", model.residual2d_block1.conv1)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_35/batch_normalization_35", model.residual2d_block1.norm1)

    assign_conv2d_weights(h5_file, "model_weights/conv2d_2/conv2d_2", model.residual2d_block1.conv2)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_36/batch_normalization_36", model.residual2d_block1.norm2)

    # 2
    assign_conv2d_weights(h5_file, "model_weights/conv2d_3/conv2d_3", model.residual2d_block2.conv1)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_37/batch_normalization_37", model.residual2d_block2.norm1)

    assign_conv2d_weights(h5_file, "model_weights/conv2d_4/conv2d_4", model.residual2d_block2.conv2)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_38/batch_normalization_38", model.residual2d_block2.norm2)

    # 3
    assign_conv2d_weights(h5_file, "model_weights/conv2d_5/conv2d_5", model.residual2d_block3.conv1)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_39/batch_normalization_39", model.residual2d_block3.norm1)

    assign_conv2d_weights(h5_file, "model_weights/conv2d_6/conv2d_6", model.residual2d_block3.conv2)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_40/batch_normalization_40", model.residual2d_block3.norm2)

    # 4
    assign_conv2d_weights(h5_file, "model_weights/conv2d_7/conv2d_7", model.residual2d_block4.conv1)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_41/batch_normalization_41", model.residual2d_block4.norm1)

    assign_conv2d_weights(h5_file, "model_weights/conv2d_8/conv2d_8", model.residual2d_block4.conv2)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_42/batch_normalization_42", model.residual2d_block4.norm2)

    # 5
    assign_conv2d_weights(h5_file, "model_weights/conv2d_9/conv2d_9", model.residual2d_block5.conv1)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_43/batch_normalization_43", model.residual2d_block5.norm1)

    assign_conv2d_weights(h5_file, "model_weights/conv2d_10/conv2d_10", model.residual2d_block5.conv2)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_44/batch_normalization_44", model.residual2d_block5.norm2)

    # 6
    assign_conv2d_weights(h5_file, "model_weights/conv2d_11/conv2d_11", model.residual2d_block6.conv1)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_45/batch_normalization_45", model.residual2d_block6.norm1)

    assign_conv2d_weights(h5_file, "model_weights/conv2d_12/conv2d_12", model.residual2d_block6.conv2)
    assign_batch_norm_weights(h5_file, "model_weights/batch_normalization_46/batch_normalization_46", model.residual2d_block6.norm2)

    # squeeze-excite

    assign_batch_norm_weights(h5_file, "model_weights/squeeze_excite/squeeze_excite/batch_normalization", model.squeeze_excite.norm)

    assign_dense_weights(h5_file, "model_weights/squeeze_excite/squeeze_excite/dense", model.squeeze_excite.dense1)

    assign_dense_weights(h5_file, "model_weights/squeeze_excite/squeeze_excite/dense_1", model.squeeze_excite.dense2)

    # Final dense layer (random initialization)
    print("Randomly initializing the final dense layer (no TF weights assigned)...")

    pytorch_dense_layer = model.final.dense

    # Use Xavier (Glorot) uniform initialization for weights and zeros for biases
    torch.nn.init.xavier_uniform_(pytorch_dense_layer.weight)
    torch.nn.init.zeros_(pytorch_dense_layer.bias)

    print(f"Initialized final dense layer randomly: {pytorch_dense_layer}")
    
    # ------------------------ Save model ------------------------ #
    torch.save(model, output_file)
    print(f"\n Model saved to {output_file}\n")


# ------------------------ CLI Entry Point ------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer TensorFlow Akita weights to PyTorch SeqNN model.")
    parser.add_argument("--data_split", type=int, required=True, help="Data split index (e.g. 0–7).")
    parser.add_argument("--organism", type=str, required=True, choices=["human", "mouse"], help="Organism type.")
    args = parser.parse_args()

    main(args.data_split, args.organism)
    