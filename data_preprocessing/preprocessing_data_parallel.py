#!/usr/bin/env python3
"""
Parallel Hi-C data preprocessing for Akita model training.

This script processes Hi-C contact matrices and DNA sequences into PyTorch
tensors suitable for training the Akita model. It handles:
- One-hot encoding of DNA sequences from FASTA files
- Processing Hi-C matrices from cooler files (balancing, filtering, obs/exp)
- Parallel processing across multiple folds
- Gap masking for low-quality genomic regions

The output is saved as .pt files containing (sequence, contact_matrix) pairs.
"""


import argparse
import logging
import os
import random
import re
from multiprocessing import Pool, cpu_count

import cooler
import numpy as np
import pandas as pd
import torch
from astropy.convolution import Gaussian2DKernel, convolve
from cooltools.lib.numutils import (
    adaptive_coarsegrain,
    interp_nan,
    observed_over_expected,
    set_diag,
)
from pyfaidx import Fasta

# =============================================================================
# DNA Sequence Processing
# =============================================================================

def one_hot_encode_sequence(sequence_obj):
    """
    One-hot encode a DNA sequence.

    Args:
        sequence_obj: Sequence object from pyfaidx (or string)

    Returns:
        np.ndarray: One-hot encoded sequence with shape (1, 4, length)
                   Channels are ordered as [A, C, G, T]

    Note:
        Unknown bases (N, etc.) are randomly assigned to A, C, G, or T
    """
    sequence = str(sequence_obj).upper()
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # Convert sequence to integer indices, random base for unknowns
    encoded_sequence = np.array([
        base_to_int.get(base, base_to_int[random.choice("ACGT")])
        for base in sequence
    ])

    # Create one-hot encoding
    one_hot_encoded = np.zeros((4, len(encoded_sequence)), dtype=np.float32)
    one_hot_encoded[encoded_sequence, np.arange(len(encoded_sequence))] = 1

    return np.expand_dims(one_hot_encoded, axis=0)


def extract_coordinates_from_mseq(mseq_str):
    """
    Parse genomic coordinates from string format.

    Args:
        mseq_str (str): Genomic region in format "chr:start-end"
                       Example: "chr1:1000000-2000000"

    Returns:
        tuple: (chrom, start, end)

    Raises:
        ValueError: If format is invalid
    """
    match = re.match(r"(?P<chrom>\w+):(?P<start>\d+)-(?P<end>\d+)", mseq_str)

    if not match:
        raise ValueError(f"Invalid coordinate format: {mseq_str}. "
                        f"Expected format: chr:start-end")

    chrom = match.group('chrom')
    start = int(match.group('start'))
    end = int(match.group('end'))

    return chrom, start, end


# =============================================================================
# Hi-C Matrix Processing
# =============================================================================

def process_hic_matrix(genome_hic_cool, mseq_str, diagonal_offset=2,
                       padding=64, kernel_stddev=1, bin_size=2048, gaps_df=None):
    """
    Process a Hi-C contact matrix for a given genomic region.

    Processing steps:
        1. Load balanced matrix from cooler file
        2. Mask NaN rows/columns (low coverage regions)
        3. Mask gap regions (optional)
        4. Clip diagonal and extreme values
        5. Adaptive coarsegraining based on raw counts
        6. Calculate observed/expected and log transform
        7. Apply padding and interpolate remaining NaNs
        8. Gaussian smoothing

    Args:
        genome_hic_cool (cooler.Cooler): Cooler object for Hi-C data
        mseq_str (str): Genomic region string "chr:start-end"
        diagonal_offset (int): Number of diagonals to mask. Default: 2
        padding (int): Number of bins to crop from edges. Default: 64
        kernel_stddev (float): Std dev for Gaussian smoothing. Default: 1
        bin_size (int): Bin size in base pairs. Default: 2048
        gaps_df (pd.DataFrame): DataFrame with gap regions (chr, start, end).
                               Default: None

    Returns:
        np.ndarray: Processed Hi-C matrix (after padding removal)

    Note:
        Skips regions where >50% of bins are filtered (low quality data)
    """
    # Load balanced Hi-C matrix
    seq_hic_raw = genome_hic_cool.matrix(balance=True).fetch(mseq_str)
    chrom, start, end = extract_coordinates_from_mseq(mseq_str)

    # Identify NaN bins (low coverage)
    seq_hic_nan = np.isnan(seq_hic_raw)
    num_filtered_bins = np.sum(np.sum(seq_hic_nan, axis=0) == len(seq_hic_nan))

    logging.debug(f"Filtered bins in {mseq_str}: {num_filtered_bins}")

    # Quality check: skip if too many bins filtered
    if num_filtered_bins > (0.5 * len(seq_hic_nan)):
        logging.warning(f"Skipping {mseq_str}: >50% bins filtered "
                       f"({num_filtered_bins}/{len(seq_hic_nan)})")
        raise ValueError("Too many filtered bins")

    # Create masks for NaN rows and columns
    row_nan_mask = np.all(seq_hic_nan, axis=1)
    col_nan_mask = np.all(seq_hic_nan, axis=0)

    # Apply NaN masks
    seq_hic_raw[row_nan_mask, :] = np.nan
    seq_hic_raw[:, col_nan_mask] = np.nan

    # Mask gap regions if provided
    if gaps_df is not None:
        gaps_chr = gaps_df[gaps_df['chr'] == chrom]

        for _, gap in gaps_chr.iterrows():
            gap_start = gap['start']
            gap_end = gap['end']

            # Check for overlap with current region
            if (gap_start < end) and (gap_end > start):
                # Convert genomic coordinates to matrix indices
                gap_start_idx = max(gap_start - start, 0) // bin_size
                gap_end_idx = min(gap_end - start, seq_hic_raw.shape[0] * bin_size,
                                 seq_hic_raw.shape[0] * bin_size) // bin_size

                # Update masks for gap regions
                row_nan_mask[gap_start_idx:gap_end_idx] = True
                col_nan_mask[gap_start_idx:gap_end_idx] = True

        # Apply updated masks
        seq_hic_raw[row_nan_mask, :] = np.nan
        seq_hic_raw[:, col_nan_mask] = np.nan

        logging.debug(f"Gap-masked rows: {np.sum(row_nan_mask)}")

    # Clip diagonal values and extreme outliers
    clipval = np.nanmedian(np.diag(seq_hic_raw, diagonal_offset))
    for i in range(-diagonal_offset + 1, diagonal_offset):
        set_diag(seq_hic_raw, clipval, i)

    seq_hic_raw = np.clip(seq_hic_raw, 0, clipval)
    seq_hic_raw[seq_hic_nan] = np.nan

    # Adaptive coarsegraining based on raw counts
    seq_hic_smoothed = adaptive_coarsegrain(
        seq_hic_raw,
        genome_hic_cool.matrix(balance=False).fetch(mseq_str),
        cutoff=2,
        max_levels=8
    )

    seq_hic_nan = np.isnan(seq_hic_smoothed)

    # Calculate observed/expected and log transform
    seq_hic_obsexp = observed_over_expected(seq_hic_smoothed, ~seq_hic_nan)[0]
    log_hic_obsexp = np.log(seq_hic_obsexp)

    # Apply padding (remove edge artifacts)
    if padding > 0:
        log_hic_obsexp = log_hic_obsexp[padding:-padding, padding:-padding]
        row_nan_mask = row_nan_mask[padding:-padding]
        col_nan_mask = col_nan_mask[padding:-padding]

    # Interpolate remaining NaNs
    log_hic_obsexp = interp_nan(log_hic_obsexp)

    # Zero out near-diagonal elements
    for i in range(-diagonal_offset + 1, diagonal_offset):
        set_diag(log_hic_obsexp, 0, i)

    # Apply Gaussian smoothing
    kernel = Gaussian2DKernel(x_stddev=kernel_stddev)
    seq_hic = convolve(log_hic_obsexp, kernel)

    return seq_hic


def upper_triangular_to_vector(matrix, dim=512, diag_offset=2):
    """
    Extract upper triangular portion of matrix, skipping near-diagonal.

    Args:
        matrix (np.ndarray): Square contact matrix
        dim (int): Matrix dimension (should match matrix.shape[0]). Default: 512
        diag_offset (int): Number of diagonals to skip. Default: 2

    Returns:
        np.ndarray: 1D vector of upper triangular elements

    Note:
        The first diag_offset diagonals are excluded as they often contain
        artifacts from Hi-C data processing.
    """
    upper_tri_indices = np.triu_indices(dim, k=diag_offset)
    upper_tri_vector = matrix[upper_tri_indices]

    return upper_tri_vector


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_and_save_dataset(args_tuple):
    """
    Process and save one fold of the dataset.

    This function runs in parallel worker processes. Each worker:
    1. Loads its own file handles (FASTA, cooler)
    2. Processes all regions in the assigned fold
    3. Saves results in batches of 100 samples

    Args:
        args_tuple (tuple): Contains:
            - fold (int): Fold number
            - df (pd.DataFrame): Full dataframe with all regions
            - fasta_file (str): Path to reference genome FASTA
            - cool_file (str): Path to cooler Hi-C file
            - output_dir (str): Directory for output files
            - gaps_df (pd.DataFrame): Gap regions dataframe
            - bin_size (int): Bin size for Hi-C processing

    Returns:
        None (saves .pt files to disk)

    Note:
        Saves every 100 samples to avoid memory issues with large datasets.
    """
    fold, df, fasta_file, cool_file, output_dir, gaps_df, bin_size = args_tuple

    # Load file handles (safe for multiprocessing - each worker has its own)
    genome = Fasta(fasta_file)
    genome_hic_cool = cooler.Cooler(cool_file)

    logging.info(f"[Fold {fold}] Starting processing")

    # Filter to current fold
    df_fold = df[df["fold"] == f"fold{fold}"].reset_index(drop=True)
    logging.info(f"[Fold {fold}] Processing {len(df_fold)} regions")

    data_list = []
    file_count = 0
    os.makedirs(output_dir, exist_ok=True)

    for i, row in enumerate(df_fold.itertuples(index=False)):
        chrom, start, end = row.chrom, row.start, row.end
        mseq_str = f"{chrom}:{start}-{end}"

        if (i + 1) % 10 == 0:
            logging.info(f"[Fold {fold}] Progress: {i+1}/{len(df_fold)}")

        try:
            # Process DNA sequence
            sequence = genome[chrom][start:end]
            ohe_sequence = one_hot_encode_sequence(sequence)

            # Process Hi-C matrix
            matrix = process_hic_matrix(
                genome_hic_cool,
                mseq_str,
                bin_size=2048,
                gaps_df=gaps_df
            )

            # Convert to upper triangular vector
            hic_vector = upper_triangular_to_vector(matrix)

            # Convert to PyTorch tensors
            ohe_tensor = torch.tensor(
                ohe_sequence.squeeze(0),
                dtype=torch.float32
            )
            hic_tensor = torch.tensor(
                hic_vector,
                dtype=torch.float32
            ).unsqueeze(0)

            data_list.append((ohe_tensor, hic_tensor))

            # Save batch every 100 samples or at end
            if (i + 1) % 100 == 0 or i == len(df_fold) - 1:
                output_file = f"{output_dir}/fold{fold}_{file_count}.pt"
                torch.save(data_list, output_file)
                logging.info(f"[Fold {fold}] Saved {len(data_list)} samples "
                           f"to {output_file}")
                data_list = []
                file_count += 1

        except Exception as e:
            logging.error(f"[Fold {fold}] Failed on {mseq_str}: {e}")
            continue

    logging.info(f"[Fold {fold}] Completed")


# =============================================================================
# Main
# =============================================================================

def main():
    """Parse arguments and run parallel preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess Hi-C and DNA sequences for Akita training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--cool_file",
        required=True,
        help="Path to .cool Hi-C file"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save output .pt files"
    )
    parser.add_argument(
        "--fasta_file",
        required=True,
        help="Reference genome FASTA file"
    )
    parser.add_argument(
        "--bed_file",
        required=True,
        help="BED file with regions and fold assignments (chr, start, end, fold)"
    )

    # Optional arguments
    parser.add_argument(
        "--bin_size",
        type=int,
        default=2048,
        help="Bin size for Hi-C processing"
    )
    parser.add_argument(
        "--gaps_file",
        default=None,
        help="BED file of gap regions to mask (chr, start, end)"
    )
    parser.add_argument(
        "--start_fold",
        type=int,
        default=0,
        help="First fold index to process"
    )
    parser.add_argument(
        "--end_fold",
        type=int,
        default=7,
        help="Last fold index to process (inclusive)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(4, cpu_count()),
        help="Number of parallel worker processes"
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logging.info("=" * 70)
    logging.info("Hi-C Data Preprocessing Pipeline")
    logging.info("=" * 70)
    logging.info(f"Cool file: {args.cool_file}")
    logging.info(f"FASTA file: {args.fasta_file}")
    logging.info(f"BED file: {args.bed_file}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Bin size: {args.bin_size}bp")
    logging.info(f"Folds: {args.start_fold} to {args.end_fold}")
    logging.info(f"Workers: {args.num_workers}")
    logging.info("=" * 70)

    # Load input data
    logging.info("Loading BED file...")
    df = pd.read_csv(
        args.bed_file,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "fold"]
    )
    logging.info(f"Loaded {len(df)} regions")

    # Load gaps file if provided
    gaps_df = None
    if args.gaps_file:
        logging.info(f"Loading gaps file: {args.gaps_file}")
        gaps_df = pd.read_csv(
            args.gaps_file,
            sep="\t",
            header=None,
            names=['chr', 'start', 'end']
        )
        logging.info(f"Loaded {len(gaps_df)} gap regions")

    # Prepare arguments for parallel processing
    fold_args = [
        (fold, df, args.fasta_file, args.cool_file, args.output_dir,
         gaps_df, args.bin_size)
        for fold in range(args.start_fold, args.end_fold + 1)
    ]

    # Run parallel processing
    logging.info(f"Starting parallel processing with {args.num_workers} workers...")
    with Pool(processes=args.num_workers) as pool:
        pool.map(generate_and_save_dataset, fold_args)

    logging.info("=" * 70)
    logging.info("All folds processed successfully!")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
