"""
Dataset class for loading Hi-C contact matrices for Akita model training.

This module provides the HiCDataset class for loading preprocessed Hi-C data
consisting of one-hot encoded DNA sequences and their corresponding Hi-C contact
matrices.
"""

import torch
from torch.utils.data import Dataset


class HiCDataset(Dataset):
    """
    PyTorch Dataset for Hi-C contact matrices paired with DNA sequences.

    Loads preprocessed data files containing one-hot encoded (OHE) DNA sequences
    and their corresponding Hi-C contact matrix vectors. The data is loaded into
    memory for efficient access during training.

    Args:
        data_files (list of str): List of file paths to preprocessed .pt data files.
            Each file should contain a list of (ohe_sequence, hic_vector) tuples.

    Attributes:
        data (list): List of (ohe_sequence, hic_vector) tuples loaded from all files.

    Examples:
        >>> train_files = ['train_data_0.pt', 'train_data_1.pt']
        >>> dataset = HiCDataset(data_files=train_files)
        Loading file: train_data_0.pt
        Loading file: train_data_1.pt
        Total sequences loaded: 15000
        >>> seq, hic = dataset[0]
        >>> print(seq.shape)  # Expected: (4, sequence_length)
        >>> print(hic.shape)  # Expected: (contact_matrix_length,)

    Notes:
        - All sequences are expected to have 4 channels (A, C, G, T one-hot encoding)
        - Sequences must be 2D tensors with shape (4, sequence_length)
        - All data is loaded into memory, so ensure sufficient RAM for large datasets
    """

    def __init__(self, data_files):
        self.data = []  # To store all sequences and HiC vectors

        # Load and process the data files
        for file in data_files:
            print(f"Loading file: {file}")
            file_data = torch.load(file, weights_only=True)

            for data in file_data:
                ohe_sequence, hic_vector = data

                # Process the OHE sequence
                ohe_sequence = ohe_sequence.squeeze(0)  # Remove singleton dimension

                # Ensure the sequence has the correct shape
                assert (
                    ohe_sequence.shape[0] == 4
                ), f"Expected 4 channels, but got {ohe_sequence.shape[0]}"
                assert (
                    len(ohe_sequence.shape) == 2
                ), f"Expected 2D shape for sequence, but got {ohe_sequence.shape}"

                # Add processed pair to the data list
                self.data.append((ohe_sequence, hic_vector))

        print(f"Total sequences loaded: {len(self.data)}")

    def __len__(self):
        """Return the total number of sequences in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single data sample.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (ohe_sequence, hic_vector) where:
                - ohe_sequence (torch.Tensor): One-hot encoded DNA sequence
                  with shape (4, sequence_length)
                - hic_vector (torch.Tensor): Hi-C contact matrix vector
        """
        ohe_sequence, hic_vector = self.data[idx]
        return ohe_sequence, hic_vector
