"""
ORCA Model Benchmarking Script
Evaluates ORCA model accuracy (Pearson R, Spearman R, MSE) on a given species/cell type.

ORCA operates at 4 kb resolution on a 2 Mb input window. To compare with Hi-C targets
at 2048 bp resolution, the script:
  1. Extends each Akita-sized test window (1,310,720 bp) to ORCA's 2,000,000 bp input.
  2. Runs the ORCA model to obtain a 500×500 contact map.
  3. Crops the central 262×262 bins (covering the same ~1 Mb as the Akita window).
  4. Upsamples 262×262 → 512×512 via zoom_array to match the 2048 bp target resolution.
  5. Compares the upper-triangle of the prediction with the processed Hi-C target.

Usage:
    python orca_benchmark.py \
        --organism mouse \
        --model_class H1esc_1M \
        --fasta /path/to/genome.fa \
        --cool /path/to/hic.cool \
        --blacklist /path/to/blacklist.bed
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import cooler
import pandas as pd
from pyfaidx import Fasta
from scipy.stats import pearsonr, spearmanr
from cooltools.lib.numutils import zoom_array

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

AKITA_REPO    = "/home1/smaruj/pytorch_akita"
ORCA_REPO     = "/home1/smaruj/orca"
TEST_SETS_DIR = os.path.join(AKITA_REPO, "benchmarking/test_sets")

sys.path.append(AKITA_REPO)
from utils.data_utils import one_hot_encode_sequence, process_hic_matrix, upper_triangular_to_vector

AKITA_LENGTH = 1_310_720   # input length of the Akita test windows
ORCA_LENGTH  = 2_000_000   # ORCA's required input length
ORCA_EXTEND  = (ORCA_LENGTH - AKITA_LENGTH) // 2   # bp to add on each side (344,640)

ORCA_MAP_SIZE    = 500     # ORCA output map dimension
ORCA_CENTER      = 250     # center bin index of the 500×500 map
ORCA_CROP_LO     = 119     # ORCA_CENTER - 131  → start of 262-bin crop
ORCA_CROP_HI     = 381     # ORCA_CENTER + 131  → end of 262-bin crop (262 bins)
TARGET_MAP_SIZE  = 512     # Hi-C target map size after padding removal (2048 bp bins)


# ──────────────────────────────────────────────────────────────────────────────
# ORCA prediction helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_chrom_sizes(fasta: Fasta) -> dict[str, int]:
    """Return a dict of {chrom: length} from a pyfaidx Fasta object."""
    return {name: len(fasta[name]) for name in fasta.keys()}


def extend_to_orca_window(
    chrom: str,
    start: int,
    end: int,
    chrom_sizes: dict[str, int],
) -> tuple[int, int, bool]:
    """
    Extend an Akita window by ORCA_EXTEND bp on each side.
    Returns (orca_start, orca_end, is_valid) where is_valid=False if the
    extended window falls outside the chromosome.
    """
    orca_start = start - ORCA_EXTEND
    orca_end   = end   + ORCA_EXTEND

    chrom_len = chrom_sizes.get(chrom, None)
    if chrom_len is None:
        print(f"  WARNING: chromosome {chrom} not found in FASTA — skipping.")
        return orca_start, orca_end, False
    if orca_start < 0 or orca_end > chrom_len:
        print(
            f"  WARNING: extended window {chrom}:{orca_start}-{orca_end} out of bounds "
            f"(chrom size={chrom_len}) — skipping."
        )
        return orca_start, orca_end, False
    return orca_start, orca_end, True


def predict_orca(
    model: torch.nn.Module,
    sequence: str,
    device: torch.device,
) -> np.ndarray:
    """
    Run ORCA on a 2 Mb sequence and return the cropped, upsampled contact map
    as a numpy array of shape (512, 512), ready for comparison with 2048 bp Hi-C.

    Steps:
      - Encode sequence → (1, 4, 2_000_000) tensor
      - Forward pass → (1, 1, 500, 500) output
      - Crop central 262×262 bins
      - Upsample to 512×512 via zoom_array
    """
    # Pad / trim to exact ORCA input length
    if len(sequence) != ORCA_LENGTH:
        sequence = sequence[:ORCA_LENGTH].ljust(ORCA_LENGTH, "N")

    ohe    = one_hot_encode_sequence(sequence)          # (1, 4, 2_000_000)
    tensor = torch.from_numpy(ohe).to(device)           # keep batch dim from OHE

    with torch.no_grad():
        output = model(tensor)                          # (1, 1, 500, 500)
        
    pred_map = output[0, 0].cpu().numpy()               # (500, 500)

    # Crop central 262×262
    cropped = pred_map[ORCA_CROP_LO:ORCA_CROP_HI, ORCA_CROP_LO:ORCA_CROP_HI]  # (262, 262)

    # Upsample to match 2048 bp Hi-C target resolution
    upsampled = zoom_array(cropped, final_shape=(TARGET_MAP_SIZE, TARGET_MAP_SIZE))

    return upsampled                                    # (512, 512)


# ──────────────────────────────────────────────────────────────────────────────
# Main benchmarking routine
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Resolve paths ─────────────────────────────────────────────────────────
    overlap_table = os.path.join(
        TEST_SETS_DIR, f"benchmark_test_set_{args.organism}.tsv"
    )
    print(f"Organism       : {args.organism}")
    print(f"Model class    : {args.model_class}")
    print(f"Overlap table  : {overlap_table}")

    # ── Load ORCA model ───────────────────────────────────────────────────────
    sys.path.append(ORCA_REPO)
    import orca_models
    ModelClass = getattr(orca_models, args.model_class)
    model = ModelClass()
    model.to(device)
    model.eval()

    # ── Load static data ──────────────────────────────────────────────────────
    overlap_df  = pd.read_csv(overlap_table, sep="\t")
    genome      = Fasta(args.fasta)
    hic         = cooler.Cooler(args.cool)
    chrom_sizes = get_chrom_sizes(genome)

    blacklist_df = None
    if args.blacklist:
        blacklist_df = pd.read_csv(
            args.blacklist,
            sep="\t",
            header=None,
            names=["chr", "start", "end", "fold"],
        )

    # ── Iterate over all test windows ─────────────────────────────────────────
    all_preds, all_targets = [], []
    
    for i, row in enumerate(overlap_df.itertuples(index=False)):
        chrom, start, end = row.chr, row.start, row.end
        region = f"{chrom}:{start}-{end}"
        print(f"[{i}] {region}")

        # Extend window to ORCA input length, skip if out of bounds
        orca_start, orca_end, valid = extend_to_orca_window(
            chrom, start, end, chrom_sizes
        )
        if not valid:
            continue

        # Hi-C target
        hic_mat    = process_hic_matrix(
            hic, region,
            diagonal_offset=2, padding=64, kernel_stddev=1.0,
            bin_size=2048, gaps_df=blacklist_df,
        )
        target_vec = upper_triangular_to_vector(hic_mat, dim=512, diag_offset=2)
        
        # ORCA prediction
        sequence = genome[chrom][orca_start:orca_end].seq.upper()
        pred_map  = predict_orca(model, sequence, device)
        pred_vec = upper_triangular_to_vector(pred_map, dim=512, diag_offset=2)
        
        all_targets.append(target_vec)
        all_preds.append(pred_vec)

    # ── Compute metrics ───────────────────────────────────────────────────────
    preds_flat   = np.array(all_preds).flatten()
    targets_flat = np.array(all_targets).flatten()

    valid      = ~np.isnan(preds_flat) & ~np.isnan(targets_flat)
    preds_v    = preds_flat[valid]
    targets_v  = targets_flat[valid]

    pearson_r  = pearsonr(preds_v, targets_v)[0]
    spearman_r = spearmanr(preds_v, targets_v)[0]
    mse        = float(np.mean((targets_v - preds_v) ** 2))

    print("\n" + "=" * 50)
    print(f"Average Pearson R  : {pearson_r:.6f}")
    print(f"Average Spearman R : {spearman_r:.6f}")
    print(f"MSE                : {mse:.6f}")
    print("=" * 50)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark an ORCA model: report Pearson R, Spearman R, and MSE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--organism",
        required=True,
        choices=["mouse", "human"],
        help="Organism. Determines the test-set TSV.",
    )
    parser.add_argument(
        "--model_class",
        required=True,
        help="ORCA model class name from orca_models, e.g. 'H1esc_1M'.",
    )
    parser.add_argument("--fasta",   required=True, help="Genome FASTA file.")
    parser.add_argument("--cool",    required=True, help="Hi-C .cool file.")
    parser.add_argument(
        "--blacklist",
        default=None,
        help="BED file of blacklisted / gap regions (optional).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())