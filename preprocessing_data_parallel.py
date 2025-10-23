#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import torch
import cooler
from cooltools.lib.numutils import observed_over_expected, adaptive_coarsegrain, set_diag, interp_nan
from astropy.convolution import Gaussian2DKernel, convolve
from pyfaidx import Fasta
import logging
import random
import os
from multiprocessing import Pool, cpu_count


# ----------------------------
# Helper Functions
# ----------------------------

def one_hot_encode_sequence(sequence_obj):
    sequence = str(sequence_obj).upper()
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    encoded_sequence = np.array([
        base_to_int.get(base, base_to_int[random.choice("ACGT")]) for base in sequence
    ])

    one_hot_encoded = np.zeros((4, len(encoded_sequence)), dtype=np.float32)
    one_hot_encoded[encoded_sequence, np.arange(len(encoded_sequence))] = 1

    return np.expand_dims(one_hot_encoded, axis=0)


import re

def extract_coordinates_from_mseq(mseq_str):
    # Regular expression to match the format: chrom:start-end
    match = re.match(r"(?P<chrom>\w+):(?P<start>\d+)-(?P<end>\d+)", mseq_str)
    
    if match:
        chrom = match.group('chrom')
        start = int(match.group('start'))
        end = int(match.group('end'))
        return chrom, start, end
    else:
        raise ValueError(f"Invalid mseq_str format: {mseq_str}")


def process_hic_matrix(genome_hic_cool, mseq_str, diagonal_offset=2, padding=64, kernel_stddev=1, bin_size=2048, gaps_df=None):
    seq_hic_raw = genome_hic_cool.matrix(balance=True).fetch(mseq_str)
    
    chrom, start, end = extract_coordinates_from_mseq(mseq_str)
    
    # Check for NaN filtering percentage
    seq_hic_nan = np.isnan(seq_hic_raw)
    num_filtered_bins = np.sum(np.sum(seq_hic_nan, axis=0) == len(seq_hic_nan))
    print("num_filtered_bins:", num_filtered_bins)
    
    if num_filtered_bins > (0.5 * len(seq_hic_nan)):
        print(f"More than 50% bins filtered in {mseq_str}. Check Hi-C data quality.")
    
    ###########
    # Mask for rows/columns full of NaNs
    row_nan_mask = np.all(seq_hic_nan, axis=1)  # Rows with all NaNs
    col_nan_mask = np.all(seq_hic_nan, axis=0)  # Columns with all NaNs
    
    true_row_indices = np.where(row_nan_mask)[0]
    print(f"Indices of rows with NaNs: {true_row_indices}")
    
    # Apply the NaN mask earlier in the process to avoid processing NaN-only rows/columns
    seq_hic_raw[row_nan_mask, :] = np.nan  # Mask entire rows
    seq_hic_raw[:, col_nan_mask] = np.nan  # Mask entire columns
    
    # Check for NaN filtering percentage
    num_filtered_bins = np.sum(np.sum(seq_hic_nan, axis=0) == len(seq_hic_nan))
    print("num_filtered_bins:", num_filtered_bins)
    ###########
    
    # Mask for regions overlapping with gaps
    if gaps_df is not None:
        # Filter gaps_df for the current chromosome
        gaps_chr = gaps_df[gaps_df['chr'] == chrom]
        
        # Iterate through each gap region and mark the corresponding rows and columns as NaN
        for _, gap in gaps_chr.iterrows():
            gap_start = gap['start']
            gap_end = gap['end']
            
            # Check if the gap overlaps with the current region
            if (gap_start < end) and (gap_end > start):
                # Mark rows and columns that fall within the gap range as NaN
                gap_start_idx = max(gap_start - start, 0) // bin_size  # Avoid negative indices
                gap_end_idx = min(gap_end - start, seq_hic_raw.shape[0]) // bin_size # Avoid out of bounds
                
                # Add the affected rows and columns to the NaN mask
                row_nan_mask[gap_start_idx:gap_end_idx] = True
                col_nan_mask[gap_start_idx:gap_end_idx] = True
                
        # Apply the updated NaN mask for gaps
        seq_hic_raw[row_nan_mask, :] = np.nan
        seq_hic_raw[:, col_nan_mask] = np.nan
    
        true_row_indices = np.where(row_nan_mask)[0]
        print(f"Indices of rows with NaNs: {true_row_indices}")
    
    # clip first diagonals and high values
    clipval = np.nanmedian(np.diag(seq_hic_raw, diagonal_offset))
    for i in range(-diagonal_offset+1, diagonal_offset):
        set_diag(seq_hic_raw, clipval, i)
    seq_hic_raw = np.clip(seq_hic_raw, 0, clipval)
    seq_hic_raw[seq_hic_nan] = np.nan
    
    # adaptively coarsegrain based on raw counts
    seq_hic_smoothed = adaptive_coarsegrain(
                            seq_hic_raw,
                            genome_hic_cool.matrix(balance=False).fetch(mseq_str),
                            cutoff=2, max_levels=8)
    seq_hic_nan = np.isnan(seq_hic_smoothed)
    
    # local obs/exp
    seq_hic_obsexp = observed_over_expected(seq_hic_smoothed, ~seq_hic_nan)[0]
    
    log_hic_obsexp = np.log(seq_hic_obsexp)
    
    # Apply padding
    if padding > 0:
        log_hic_obsexp = log_hic_obsexp[padding:-padding, padding:-padding]
        row_nan_mask = row_nan_mask[padding:-padding]
        col_nan_mask = col_nan_mask[padding:-padding]
        
    log_hic_obsexp = interp_nan(log_hic_obsexp)
    for i in range(-diagonal_offset+1, diagonal_offset): set_diag(log_hic_obsexp, 0,i)
    
    kernel = Gaussian2DKernel(x_stddev=kernel_stddev)
    seq_hic = convolve(log_hic_obsexp, kernel)
    
    # Mask NaN-filled rows and columns before returning the result
    seq_hic[row_nan_mask, :] = np.nan  # Mask entire rows with NaNs
    seq_hic[:, col_nan_mask] = np.nan  # Mask entire columns with NaNs
    
    return seq_hic


def upper_triangular_to_vector_skip_diagonals(matrix, dim=512, diag=2):
    
    # Extract the upper triangular part excluding the first two diagonals
    upper_triangular_vector = matrix[np.triu_indices(dim, k=diag)]
    
    return upper_triangular_vector


def generate_and_save_dataset(args_tuple):
    """Process and save one fold — runs in parallel workers."""
    fold, df, fasta_file, cool_file, output_dir, gaps_df = args_tuple

    # Each process loads its own file handles (safe for multiprocessing)
    genome = Fasta(fasta_file)
    genome_hic_cool = cooler.Cooler(cool_file)

    logging.info(f"[Fold {fold}] Starting fold {fold}")
    df_fold = df[df["fold"] == f"fold{fold}"].reset_index(drop=True)

    data_list = []
    file_count = 0
    os.makedirs(output_dir, exist_ok=True)

    for i, row in enumerate(df_fold.itertuples(index=False)):
        chrom, start, end = row.chrom, row.start, row.end
        mseq_str = f"{chrom}:{start}-{end}"
        logging.info(f"[Fold {fold}] Processing {mseq_str} ({i+1}/{len(df_fold)})")

        try:
            sequence = genome[chrom][start:end]
            ohe_sequence = one_hot_encode_sequence(sequence)
            matrix = process_hic_matrix(
                        genome_hic_cool,
                        mseq_str,
                        bin_size=2048,        # or read from user input
                        gaps_df=gaps_df        # load gaps_df beforehand
                    )
            hic_vector = upper_triangular_to_vector_skip_diagonals(matrix)

            ohe_tensor = torch.tensor(ohe_sequence.squeeze(0), dtype=torch.float32)
            hic_tensor = torch.tensor(hic_vector, dtype=torch.float32).unsqueeze(0)

            data_list.append((ohe_tensor, hic_tensor))

            if (i + 1) % 100 == 0 or i == len(df_fold) - 1:
                output_file = f"{output_dir}/fold{fold}_{file_count}.pt"
                torch.save(data_list, output_file)
                logging.info(f"[Fold {fold}] Saved {len(data_list)} samples to {output_file}")
                data_list = []
                file_count += 1

        except Exception as e:
            logging.error(f"[Fold {fold}] Failed on {mseq_str}: {e}")
            continue

    logging.info(f"[Fold {fold}] Completed fold {fold}")


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate preprocessed PyTorch datasets from Hi-C and DNA.")
    parser.add_argument("--cool_file", required=True, help="Path to .cool Hi-C file")
    parser.add_argument("--output_dir", required=True, help="Directory to save output .pt files")
    parser.add_argument("--fasta_file", required=True, help="Reference genome FASTA file")
    parser.add_argument("--bed_file", required=True, help="Path to BED file containing regions and folds")
    parser.add_argument("--bin_size", type=int, default=2048, help="Bin size for Hi-C processing (default=2048)")
    parser.add_argument("--gaps_file", default=None, help="Optional BED file of gaps")
    parser.add_argument("--start_fold", type=int, default=0, help="First fold index (default=0)")
    parser.add_argument("--end_fold", type=int, default=7, help="Last fold index (default=7)")
    parser.add_argument("--num_workers", type=int, default=min(4, cpu_count()),
                    help="Number of folds to process in parallel (default=4 or #CPUs)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Shared dataframe (read once)
    df = pd.read_csv(args.bed_file, sep="\t", header=None, names=["chrom", "start", "end", "fold"])

    gaps_df = pd.read_csv(args.gaps_file, sep="\t", header=0) if args.gaps_file else None
    
    # Prepare argument tuples
    fold_args = [
        (fold, df, args.fasta_file, args.cool_file, args.output_dir, gaps_df)
        for fold in range(args.start_fold, args.end_fold + 1)
    ]

    # Run folds in parallel
    with Pool(processes=args.num_workers) as pool:
        pool.map(generate_and_save_dataset, fold_args)

    logging.info("All folds processed successfully.")
