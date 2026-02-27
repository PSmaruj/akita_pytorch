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
import sys
from multiprocessing import Pool, cpu_count

import cooler
import pandas as pd
import torch
from pyfaidx import Fasta

AKITA_REPO = "/home1/smaruj/pytorch_akita"

sys.path.append(AKITA_REPO)
from utils.data_utils import one_hot_encode_sequence, process_hic_matrix, upper_triangular_to_vector

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
            logging.info(f"[Fold {fold}] Progress: {i + 1}/{len(df_fold)}")

        try:
            # Process DNA sequence
            sequence = genome[chrom][start:end]
            ohe_sequence = one_hot_encode_sequence(sequence)

            # Process Hi-C matrix
            matrix = process_hic_matrix(genome_hic_cool, mseq_str, bin_size=2048, gaps_df=gaps_df)

            # Convert to upper triangular vector
            hic_vector = upper_triangular_to_vector(matrix)

            # Convert to PyTorch tensors
            ohe_tensor = torch.tensor(ohe_sequence.squeeze(0), dtype=torch.float32)
            hic_tensor = torch.tensor(hic_vector, dtype=torch.float32).unsqueeze(0)

            data_list.append((ohe_tensor, hic_tensor))

            # Save batch every 100 samples or at end
            if (i + 1) % 100 == 0 or i == len(df_fold) - 1:
                output_file = f"{output_dir}/fold{fold}_{file_count}.pt"
                torch.save(data_list, output_file)
                logging.info(f"[Fold {fold}] Saved {len(data_list)} samples to {output_file}")
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--cool_file", required=True, help="Path to .cool Hi-C file")
    parser.add_argument("--output_dir", required=True, help="Directory to save output .pt files")
    parser.add_argument("--fasta_file", required=True, help="Reference genome FASTA file")
    parser.add_argument(
        "--bed_file",
        required=True,
        help="BED file with regions and fold assignments (chr, start, end, fold)",
    )

    # Optional arguments
    parser.add_argument("--bin_size", type=int, default=2048, help="Bin size for Hi-C processing")
    parser.add_argument(
        "--gaps_file", default=None, help="BED file of gap regions to mask (chr, start, end)"
    )
    parser.add_argument("--start_fold", type=int, default=0, help="First fold index to process")
    parser.add_argument(
        "--end_fold", type=int, default=7, help="Last fold index to process (inclusive)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(4, cpu_count()),
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
    df = pd.read_csv(args.bed_file, sep="\t", header=None, names=["chrom", "start", "end", "fold"])
    logging.info(f"Loaded {len(df)} regions")

    # Load gaps file if provided
    gaps_df = None
    if args.gaps_file:
        logging.info(f"Loading gaps file: {args.gaps_file}")
        gaps_df = pd.read_csv(args.gaps_file, sep="\t", header=None, names=["chr", "start", "end"])
        logging.info(f"Loaded {len(gaps_df)} gap regions")

    # Prepare arguments for parallel processing
    fold_args = [
        (fold, df, args.fasta_file, args.cool_file, args.output_dir, gaps_df, args.bin_size)
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
