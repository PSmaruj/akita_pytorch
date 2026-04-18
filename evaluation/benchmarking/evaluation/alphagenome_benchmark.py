"""
AlphaGenome Model Benchmarking Script
Evaluates AlphaGenome model accuracy (Pearson R, Spearman R, MSE) on a given species/cell type.

Usage:
    python alphagenome_benchmark.py \
        --organism mouse \
        --ontology_id EFO:0004038 \
        --fasta /path/to/genome.fa \
        --cool /path/to/hic.cool \
        --n_models 4 \
        --api_key YOUR_API_KEY \
        --blacklist /path/to/blacklist.bed
"""

import argparse
import os
import sys

import cooler
import numpy as np
import pandas as pd
from alphagenome.models import dna_client
from pyfaidx import Fasta
from scipy.stats import pearsonr, spearmanr

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

AKITA_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_SETS_DIR = os.path.join(AKITA_REPO, "evaluation/benchmarking/test_sets")

from utils.data_utils import process_hic_matrix, upper_triangular_to_vector

CROP_BINS = 64  # bins to crop from each side (shared with Akita padding)
BIN_SIZE = 2048  # Hi-C bin size in bp

_ORGANISM_MAP = {
    "mouse": dna_client.Organism.MUS_MUSCULUS,
    "human": dna_client.Organism.HOMO_SAPIENS,
}

_FOLD_VERSION_MAP = {
    0: dna_client.ModelVersion.FOLD_0,
    1: dna_client.ModelVersion.FOLD_1,
    2: dna_client.ModelVersion.FOLD_2,
    3: dna_client.ModelVersion.FOLD_3,
}


# ──────────────────────────────────────────────────────────────────────────────
# Main benchmarking routine
# ──────────────────────────────────────────────────────────────────────────────


def run_benchmark(args: argparse.Namespace) -> None:
    # ── Resolve paths ─────────────────────────────────────────────────────────
    overlap_table = os.path.join(TEST_SETS_DIR, f"benchmark_test_set_{args.organism}.tsv")
    print(f"Organism       : {args.organism}")
    print(f"Ontology ID    : {args.ontology_id}")
    print(f"Overlap table  : {overlap_table}")

    ag_organism = _ORGANISM_MAP[args.organism]

    # ── Load static data ──────────────────────────────────────────────────────
    overlap_df = pd.read_csv(overlap_table, sep="\t")

    # Pre-compute cropped coordinates if not already present
    if "cropped_start" not in overlap_df.columns:
        overlap_df["cropped_start"] = overlap_df["start"] + CROP_BINS * BIN_SIZE
        overlap_df["cropped_end"] = overlap_df["end"] - CROP_BINS * BIN_SIZE

    genome = Fasta(args.fasta)
    hic = cooler.Cooler(args.cool)

    blacklist_df = None
    if args.blacklist:
        blacklist_df = pd.read_csv(
            args.blacklist,
            sep="\t",
            header=None,
            names=["chr", "start", "end", "fold"],
        )

    # ── Iterate over folds / models ───────────────────────────────────────────
    all_preds, all_targets = [], []

    for model_idx in range(args.n_models):
        fold_version = _FOLD_VERSION_MAP[model_idx]
        print(f"\n── Fold {model_idx}")

        dna_model = dna_client.create(args.api_key, model_version=fold_version)

        fold_df = overlap_df[overlap_df["type_alpha"] == f"fold{model_idx}"]

        for i, row in enumerate(fold_df.itertuples(index=False)):
            chrom = row.chr
            start, end = row.start, row.end
            cropped_start = row.cropped_start
            cropped_end = row.cropped_end
            region = f"{chrom}:{start}-{end}"
            print(f"  [{i}] {region}")

            # Hi-C target matrix
            hic_mat = process_hic_matrix(
                hic,
                region,
                diagonal_offset=2,
                padding=64,
                kernel_stddev=1.0,
                bin_size=BIN_SIZE,
                gaps_df=blacklist_df,
            )

            target_vec = upper_triangular_to_vector(hic_mat, dim=512, diag_offset=2)

            # AlphaGenome prediction
            sequence = genome[chrom][cropped_start:cropped_end].seq.upper()
            output = dna_model.predict_sequence(
                organism=ag_organism,
                sequence=sequence,
                requested_outputs=[dna_client.OutputType.CONTACT_MAPS],
                ontology_terms=[args.ontology_id],
            )

            pred_mat = output.contact_maps.values[:, :, 0]
            n = pred_mat.shape[0]
            triu_idx = np.triu_indices(n, k=2)
            pred_vec = pred_mat[triu_idx]

            all_targets.append(target_vec)
            all_preds.append(pred_vec)

    # ── Compute metrics ───────────────────────────────────────────────────────
    preds_flat = np.array(all_preds).flatten()
    targets_flat = np.array(all_targets).flatten()

    valid = ~np.isnan(preds_flat) & ~np.isnan(targets_flat)
    preds_v = preds_flat[valid]
    targets_v = targets_flat[valid]

    pearson_r = pearsonr(preds_v, targets_v)[0]
    spearman_r = spearmanr(preds_v, targets_v)[0]
    mse = float(np.mean((targets_v - preds_v) ** 2))

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
        description="Benchmark AlphaGenome: report Pearson R, Spearman R, and MSE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--organism",
        required=True,
        choices=["mouse", "human"],
        help="Organism. Determines the test-set TSV and AlphaGenome organism enum.",
    )
    parser.add_argument(
        "--ontology_id",
        required=True,
        help="Cell-type ontology ID passed to AlphaGenome, e.g. 'EFO:0004038'.",
    )
    parser.add_argument("--fasta", required=True, help="Genome FASTA file.")
    parser.add_argument("--cool", required=True, help="Hi-C .cool file.")
    parser.add_argument(
        "--n_models",
        type=int,
        default=4,
        help="Number of AlphaGenome fold models to evaluate (0 … n_models-1).",
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("ALPHAGENOME_API_KEY", ""),
        help="AlphaGenome API key. Defaults to $ALPHAGENOME_API_KEY env variable.",
    )
    parser.add_argument(
        "--blacklist",
        default=None,
        help="BED file of blacklisted / gap regions (optional).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
