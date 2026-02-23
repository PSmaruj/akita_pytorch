"""
build_safe_test_windows.py

For a given species (human or mouse), this script:
  1. Loads the Basenji sequences BED file.
  2. Expands each target window to the full ~1 Mb AlphaGenome input window.
  3. For each of the 4 test folds, identifies test windows that do NOT overlap
     with any training window (preventing data leakage).
  4. Merges the 4 per-fold safe TSVs into a single file.

Outputs (written to --data_dir):
  alphagenome_{species}_safe_windows.tsv       (merged across folds)

Usage:
  python build_safe_test_windows.py --species human
  python build_safe_test_windows.py --species mouse --data_dir /path/to/data
"""

import argparse
from pathlib import Path

import bioframe as bf
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHA_INPUT_SIZE = 512 * 2048  # 1,048,576 bp (~1 Mb)
N_FOLDS = 4
FOLDS = [f"fold{i}" for i in range(N_FOLDS)]

SPECIES_BED = {
    "human": "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/pytorch_akita_benchmarking/sequences_human.bed.gz",
    "mouse": "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/pytorch_akita_benchmarking/sequences_mouse.bed.gz",
}

# Default data root — override with --data_dir
DEFAULT_DATA_DIR = Path("/home1/smaruj/pytorch_akita/benchmarking/test_sets")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_alphagenome_bed(bed_path: Path) -> pd.DataFrame:
    """Load AlphaGenome BED file and compute ~1 Mb input windows."""
    df = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "type"],
        compression="gzip",
    )

    midpoint = (df["start"] + df["end"]) // 2
    df["start"] = (midpoint - ALPHA_INPUT_SIZE // 2).clip(lower=0)
    df["end"] = midpoint + ALPHA_INPUT_SIZE // 2

    return df[["chrom", "start", "end", "type"]]


def get_safe_windows_for_fold(df: pd.DataFrame, test_fold: str) -> pd.DataFrame:
    """
    Return test windows from `test_fold` that have no overlap with any
    training window (all folds except test_fold and its paired valid_fold).

    AlphaGenome uses an 8-fold split where fold N+1 is the validation set
    when fold N is the test set.
    """
    fold_index = int(test_fold.replace("fold", ""))
    valid_fold = f"fold{(fold_index + 1) % N_FOLDS}"

    test_df = df[df["type"] == test_fold].copy()
    train_df = df[~df["type"].isin([test_fold, valid_fold])].copy()

    overlaps = bf.overlap(
        train_df,
        test_df,
        how="inner",
        suffixes=("_train", "_test"),
        cols1=["chrom", "start", "end"],
        cols2=["chrom", "start", "end"],
    )

    # Build a set of test window keys that overlap with training
    overlap_keys = set(
        overlaps[["chrom_test", "start_test", "end_test"]].astype(str).agg("_".join, axis=1)
    )

    test_df["_key"] = test_df[["chrom", "start", "end"]].astype(str).agg("_".join, axis=1)
    safe = test_df[~test_df["_key"].isin(overlap_keys)].drop(columns="_key").copy()
    safe.reset_index(drop=True, inplace=True)

    print(
        f"  {test_fold}: {len(test_df)} total → {len(safe)} safe "
        f"({len(test_df) - len(safe)} removed)"
    )
    return safe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AlphaGenome safe test windows.")
    parser.add_argument(
        "--species",
        required=True,
        choices=["human", "mouse"],
        help="Species to process.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help=("Root data directory. Defaults to " f"{DEFAULT_DATA_DIR}/<species>_cell_types/"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    species = args.species
    data_dir = args.data_dir or DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    bed_path = data_dir / SPECIES_BED[species]
    print(f"Species : {species}")
    print(f"BED file: {bed_path}")
    print(f"Output  : {data_dir}\n")

    # Step 1: load and expand windows
    df = load_alphagenome_bed(bed_path)

    # Step 2: per-fold safe windows
    print("Computing safe test windows per fold:")
    per_fold_dfs = []
    for fold in FOLDS:
        safe = get_safe_windows_for_fold(df, fold)
        per_fold_dfs.append(safe)

    # Step 3: merge across folds
    merged = pd.concat(per_fold_dfs, ignore_index=True)
    merged_path = data_dir / f"alphagenome_{species}_safe_windows.tsv"
    merged.to_csv(merged_path, sep="\t", index=False)

    print(f"\nMerged safe windows: {len(merged)} total")
    print(f"Saved to: {merged_path}")


if __name__ == "__main__":
    main()
