import re
import random
import numpy as np
import pandas as pd
import cooler
import logging
from typing import Optional
from astropy.convolution import Gaussian2DKernel, convolve
from cooltools.lib.numutils import (
    observed_over_expected,
    adaptive_coarsegrain,
    set_diag,
    interp_nan,
)

# ──────────────────────────────────────────────────────────────────────────────
# Sequence utilities
# ──────────────────────────────────────────────────────────────────────────────

def one_hot_encode_sequence(sequence_obj: object) -> np.ndarray:
    """One-hot encode a DNA sequence, randomising ambiguous bases."""
    sequence = str(sequence_obj).upper()
    base_to_int = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.array(
        [base_to_int.get(b, base_to_int[random.choice("ACGT")]) for b in sequence]
    )
    ohe = np.zeros((4, len(encoded)), dtype=np.float32)
    ohe[encoded, np.arange(len(encoded))] = 1
    return np.expand_dims(ohe, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate parsing
# ──────────────────────────────────────────────────────────────────────────────

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
        raise ValueError(f"Invalid coordinate format: {mseq_str}. Expected format: chr:start-end")

    chrom = match.group("chrom")
    start = int(match.group("start"))
    end = int(match.group("end"))

    return chrom, start, end


# =============================================================================
# Hi-C Matrix Processing
# =============================================================================


def process_hic_matrix(
    genome_hic_cool,
    mseq_str,
    diagonal_offset=2,
    padding=64,
    kernel_stddev=1,
    bin_size=2048,
    gaps_df=None,
):
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
        logging.warning(
            f"Seq {mseq_str}: >50% bins filtered ({num_filtered_bins}/{len(seq_hic_nan)})"
        )

    # Create masks for NaN rows and columns
    row_nan_mask = np.all(seq_hic_nan, axis=1)
    col_nan_mask = np.all(seq_hic_nan, axis=0)

    # Apply NaN masks
    seq_hic_raw[row_nan_mask, :] = np.nan
    seq_hic_raw[:, col_nan_mask] = np.nan

    # Mask gap regions if provided
    if gaps_df is not None:
        gaps_chr = gaps_df[gaps_df["chr"] == chrom]

        for _, gap in gaps_chr.iterrows():
            gap_start = gap["start"]
            gap_end = gap["end"]

            # Check for overlap with current region
            if (gap_start < end) and (gap_end > start):
                # Convert genomic coordinates to matrix indices
                gap_start_idx = max(gap_start - start, 0) // bin_size
                gap_end_idx = (
                    min(
                        gap_end - start,
                        seq_hic_raw.shape[0] * bin_size,
                    )
                    // bin_size
                )

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
        seq_hic_raw, genome_hic_cool.matrix(balance=False).fetch(mseq_str), cutoff=2, max_levels=8
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

    # Fill edge/corner NaNs that interp_nan can't handle (no valid neighbors)
    if np.any(np.isnan(log_hic_obsexp)):
        nan_fraction = np.isnan(log_hic_obsexp).mean()
        if nan_fraction > 0.1:
            logging.warning(f"High NaN fraction ({nan_fraction:.1%}) in {mseq_str}, filling with 0")
        log_hic_obsexp = np.nan_to_num(log_hic_obsexp, nan=0.0)

    # Interpolate remaining NaNs
    log_hic_obsexp = interp_nan(log_hic_obsexp)

    # Zero out near-diagonal elements
    for i in range(-diagonal_offset + 1, diagonal_offset):
        set_diag(log_hic_obsexp, 0, i)

    # Apply Gaussian smoothing
    kernel = Gaussian2DKernel(x_stddev=kernel_stddev)
    seq_hic = convolve(log_hic_obsexp, kernel)

    seq_hic[row_nan_mask, :] = np.nan
    seq_hic[:, col_nan_mask] = np.nan

    return seq_hic


# ──────────────────────────────────────────────────────────────────────────────
# Matrix / vector helpers
# ──────────────────────────────────────────────────────────────────────────────

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
