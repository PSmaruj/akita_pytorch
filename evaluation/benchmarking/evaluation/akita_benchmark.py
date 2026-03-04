"""
Akita Model Benchmarking Script
Evaluates Akita model accuracy (Pearson R, Spearman R, MSE) on a given species/cell type.

Usage:
    python akita_benchmark.py \
        --organism mouse \
        --dataset Hsieh2019_mESC \
        --fasta /project2/fudenber_735/genomes/mm10/mm10.fa \
        --cool /project2/fudenber_735/GEO/Hsieh2019/4DN/mESC_mm10_4DNFILZ1CPT8.mapq_30.2048.cool \
        --n_models 8 \
        --blacklist /path/to/blacklist.bed
"""

import argparse
import os
import sys

import cooler
import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from scipy.stats import pearsonr, spearmanr

# ──────────────────────────────────────────────────────────────────────────────
# Main benchmarking routine
# ──────────────────────────────────────────────────────────────────────────────

AKITA_REPO = "/home1/smaruj/pytorch_akita"
TEST_SETS_DIR = os.path.join(AKITA_REPO, "evaluation/benchmarking/test_sets")
MODELS_DIR = os.path.join(AKITA_REPO, "models/finetuned")

sys.path.append(AKITA_REPO)
from utils.data_utils import one_hot_encode_sequence, process_hic_matrix, upper_triangular_to_vector


def run_benchmark(args: argparse.Namespace) -> None:
    # ── Import model class from the Akita repository ──────────────────────────
    sys.path.append(AKITA_REPO)
    from akita.model import SeqNN  # noqa: F401 (dynamic import)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Resolve paths from organism / dataset ─────────────────────────────────
    overlap_table = os.path.join(TEST_SETS_DIR, f"benchmark_test_set_{args.organism}.tsv")
    model_dir = os.path.join(MODELS_DIR, args.organism, args.dataset, "checkpoints")
    model_prefix = f"Akita_v2_{args.organism}_{args.dataset}"

    print(f"Organism       : {args.organism}")
    print(f"Dataset        : {args.dataset}")
    print(f"Overlap table  : {overlap_table}")
    print(f"Model dir      : {model_dir}")

    # ── Load static data ──────────────────────────────────────────────────────
    overlap_df = pd.read_csv(overlap_table, sep="\t")
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

    # ── Iterate over models ───────────────────────────────────────────────────
    all_preds, all_targets = [], []

    for model_idx in range(args.n_models):
        model_path = os.path.join(
            model_dir,
            f"{model_prefix}_model{model_idx}_finetuned.pth",
        )
        print(f"\n── Model {model_idx}: {model_path}")

        model = SeqNN()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        fold_df = overlap_df[overlap_df["type_akita"] == f"fold{model_idx}"]

        for i, row in enumerate(fold_df.itertuples(index=False)):
            chrom, start, end = row.chr, row.start, row.end
            region = f"{chrom}:{start}-{end}"
            print(f"  [{i}] {region}")

            # One-hot encode input sequence
            ohe = one_hot_encode_sequence(genome[chrom][start:end])
            ohe_tensor = torch.tensor(ohe, dtype=torch.float32)

            # Process Hi-C target matrix
            hic_mat = process_hic_matrix(
                hic,
                region,
                diagonal_offset=2,
                padding=64,
                kernel_stddev=1.0,
                bin_size=2048,
                gaps_df=blacklist_df,
            )

            target_vec = upper_triangular_to_vector(hic_mat, dim=512, diag_offset=2)

            # Model prediction
            with torch.no_grad():
                pred_vec = model(ohe_tensor).squeeze().cpu().numpy()

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
        description="Benchmark an Akita model: report Pearson R, Spearman R, and MSE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--organism",
        required=True,
        choices=["mouse", "human"],
        help="Organism. Determines the test-set TSV and model subdirectory.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g. 'Hsieh2019_mESC'. Used to locate models and build "
        "the model prefix (<organism>/<dataset>/checkpoints/).",
    )
    parser.add_argument("--fasta", required=True, help="Genome FASTA file.")
    parser.add_argument("--cool", required=True, help="Hi-C .cool file.")
    parser.add_argument(
        "--n_models",
        type=int,
        default=8,
        help="Number of cross-validation fold models (0 … n_models-1).",
    )
    parser.add_argument(
        "--blacklist",
        default=None,
        help="BED file of blacklisted / gap regions (optional).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
