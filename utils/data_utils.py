import random
import re

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from cooltools.lib.numutils import (
    adaptive_coarsegrain,
    interp_nan,
    observed_over_expected,
    set_diag,
)

# ──────────────────────────────────────────────────────────────────────────────
# Sequence utilities
# ──────────────────────────────────────────────────────────────────────────────


def one_hot_encode_sequence(sequence_obj: object) -> np.ndarray:
    """One-hot encode a DNA sequence, randomising ambiguous bases."""
    sequence = str(sequence_obj).upper()
    base_to_int = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.array([base_to_int.get(b, base_to_int[random.choice("ACGT")]) for b in sequence])
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
    Processes Hi-C matrices by applying NaN masking, gap filtering,
    clipping, smoothing, and normalization.
    """
    # 1. Data Loading & Initial Coordinate Extraction
    seq_hic_raw = genome_hic_cool.matrix(balance=True).fetch(mseq_str)
    chrom, start, end = extract_coordinates_from_mseq(mseq_str)

    # 2. Initial NaN Masking
    seq_hic_nan = np.isnan(seq_hic_raw)
    num_filtered_bins = np.sum(np.sum(seq_hic_nan, axis=0) == len(seq_hic_nan))
    print("num_filtered_bins:", num_filtered_bins)

    if num_filtered_bins > (0.5 * len(seq_hic_nan)):
        print(f"More than 50% bins filtered in {mseq_str}. Check Hi-C data quality.")

    row_nan_mask = np.all(seq_hic_nan, axis=1)
    col_nan_mask = np.all(seq_hic_nan, axis=0)

    true_row_indices = np.where(row_nan_mask)[0]
    print(f"Indices of rows with NaNs: {true_row_indices}")

    # Apply the NaN mask earlier in the process to avoid processing NaN-only rows/columns
    seq_hic_raw[row_nan_mask, :] = np.nan
    seq_hic_raw[:, col_nan_mask] = np.nan

    # Check for NaN filtering percentage
    num_filtered_bins = np.sum(np.sum(seq_hic_nan, axis=0) == len(seq_hic_nan))
    print("num_filtered_bins:", num_filtered_bins)

    # 3. Gap Analysis (Update Mask based on genomic gaps)
    if gaps_df is not None:
        gaps_chr = gaps_df[gaps_df["chr"] == chrom]
        for _, gap in gaps_chr.iterrows():
            gap_start = gap["start"]
            gap_end = gap["end"]

            if (gap_start < end) and (gap_end > start):
                gap_start_idx = max(gap_start - start, 0) // bin_size
                gap_end_idx = min(gap_end - start, seq_hic_raw.shape[0]) // bin_size

                row_nan_mask[gap_start_idx:gap_end_idx] = True
                col_nan_mask[gap_start_idx:gap_end_idx] = True

        seq_hic_raw[row_nan_mask, :] = np.nan
        seq_hic_raw[:, col_nan_mask] = np.nan

        true_row_indices = np.where(row_nan_mask)[0]
        print(f"Indices of rows with NaNs: {true_row_indices}")

    # 4. Signal Clipping and Diagonal Handling
    clipval = np.nanmedian(np.diag(seq_hic_raw, diagonal_offset))

    # Neutralize values near the main diagonal
    for i in range(-diagonal_offset + 1, diagonal_offset):
        set_diag(seq_hic_raw, clipval, i)

    seq_hic_raw = np.clip(seq_hic_raw, 0, clipval)
    seq_hic_raw[seq_hic_nan] = np.nan

    # 5. Adaptive Coarsegraining & Normalization
    seq_hic_smoothed = adaptive_coarsegrain(
        seq_hic_raw, genome_hic_cool.matrix(balance=False).fetch(mseq_str), cutoff=2, max_levels=8
    )
    # Observed/Expected calculation
    seq_hic_nan = np.isnan(seq_hic_smoothed)
    seq_hic_obsexp = observed_over_expected(seq_hic_smoothed, ~seq_hic_nan)[0]
    log_hic_obsexp = np.log(seq_hic_obsexp)

    # Apply padding
    if padding > 0:
        log_hic_obsexp = log_hic_obsexp[padding:-padding, padding:-padding]

    log_hic_obsexp = interp_nan(log_hic_obsexp)

    for i in range(-diagonal_offset + 1, diagonal_offset):
        set_diag(log_hic_obsexp, 0, i)

    kernel = Gaussian2DKernel(x_stddev=kernel_stddev)
    seq_hic = convolve(log_hic_obsexp, kernel)

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
