"""
build_benchmark_test_set.py

Overlaps Akita and AlphaGenome test windows to produce a shared benchmark
test set, then applies a species-specific selection strategy:

  Human : keep only windows on chr9 and chr10 (ORCA's test chromosomes),
          so the benchmark is valid for all three models.
  Mouse : randomly sample 500 windows from the full overlap (ORCA is
          human-only, so no chromosome restriction is needed).

Selection logic for the overlap:
  - Keep only Akita windows that overlap AlphaGenome windows from exactly
    one AlphaGenome fold (avoids ambiguous fold assignments).
  - For each such Akita window, pair it with the single closest AlphaGenome
    window by midpoint distance.

Inputs (read from --data_dir):
  alphagenome_{species}_safe_windows.tsv   (output of build_safe_test_windows.py)

Inputs (fixed paths, override with --akita_bed):
  sequences.bed  (Akita V2 BED file for the relevant genome)

Outputs (written to --data_dir):
  benchmark_test_set_{species}.tsv
    columns: chr, start, end, type_alpha, type_akita

Usage:
  python build_benchmark_test_set.py --species human
  python build_benchmark_test_set.py --species mouse
  python build_benchmark_test_set.py --species human \\
      --data_dir /my/data/alphagenome/human_cell_types \\
      --akita_bed /my/data/akita/hg38/sequences.bed
"""

import argparse
from pathlib import Path

import bioframe as bf
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ORCA_TEST_CHROMS = ["chr9", "chr10"]
MOUSE_SAMPLE_N = 500
MOUSE_SAMPLE_SEED = 42

AKITA_BED = {
    "human": "/project2/fudenber_735/tensorflow_models/akita/v2/data/hg38/sequences.bed",
    "mouse": "/project2/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed",
}

DEFAULT_DATA_ROOT = Path("/home1/smaruj/pytorch_akita/benchmarking/test_sets")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_akita_windows(bed_path: Path) -> pd.DataFrame:
    """Load Akita sequences BED file."""
    df = pd.read_csv(bed_path, sep="\t", names=["chr", "start", "end", "type"])
    print(f"  Akita windows loaded : {len(df)}")
    return df


def load_alphagenome_windows(tsv_path: Path) -> pd.DataFrame:
    """Load merged AlphaGenome safe test windows."""
    df = pd.read_csv(tsv_path, sep="\t")
    df.rename(columns={"chrom": "chr"}, inplace=True)
    print(f"  AlphaGenome windows loaded : {len(df)}")
    return df


def compute_overlap(akita_df: pd.DataFrame, alpha_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the pairwise overlap between Akita and AlphaGenome windows,
    keeping only Akita windows that overlap exactly one AlphaGenome fold.
    Each such Akita window is then paired with its single closest
    AlphaGenome window by midpoint distance.
    """
    # --- raw overlap ---
    raw = bf.overlap(
        akita_df,
        alpha_df,
        how="inner",
        suffixes=("_akita", "_alpha"),
        cols1=["chr", "start", "end"],
        cols2=["chr", "start", "end"],
    )
    print(f"  Raw overlap pairs      : {len(raw)}")

    # --- keep Akita windows overlapping exactly one AlphaGenome fold ---
    folds_per_akita = (
        raw.groupby(["chr_akita", "start_akita", "end_akita", "type_akita"])["type_alpha"]
        .nunique()
        .reset_index(name="n_unique_alpha_folds")
    )
    akita_keep = folds_per_akita.query("n_unique_alpha_folds == 1")[
        ["chr_akita", "start_akita", "end_akita", "type_akita"]
    ]
    filtered = raw.merge(
        akita_keep, on=["chr_akita", "start_akita", "end_akita", "type_akita"], how="inner"
    )
    print(
        f"  After single-fold filter : {filtered['chr_akita'].count()} pairs "
        f"({akita_keep.shape[0]} unique Akita windows)"
    )

    # --- for each Akita window, pick the closest AlphaGenome window ---
    filtered = filtered.copy()
    filtered["akita_midpoint"] = filtered["start_akita"] + 0.5 * (
        filtered["end_akita"] - filtered["start_akita"]
    )
    filtered["alpha_midpoint"] = filtered["start_alpha"] + 0.5 * (
        filtered["end_alpha"] - filtered["start_alpha"]
    )
    filtered["midpoint_dist"] = np.abs(filtered["akita_midpoint"] - filtered["alpha_midpoint"])

    df_unique = (
        filtered.sort_values("midpoint_dist")
        .groupby(["chr_akita", "start_akita", "end_akita"])
        .first()
        .reset_index()
    )

    # --- tidy up columns ---
    df_unique = df_unique.rename(
        columns={"chr_akita": "chr", "start_akita": "start", "end_akita": "end"}
    )
    result = df_unique[["chr", "start", "end", "type_alpha", "type_akita"]].copy()
    print(f"  Final unique pairs       : {len(result)}")
    return result


def select_human_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only chr9 and chr10 to match ORCA's test set."""
    selected = df[df["chr"].isin(ORCA_TEST_CHROMS)].copy().reset_index(drop=True)
    print(f"  Human (ORCA chr9+10) windows : {len(selected)}")
    return selected


def select_mouse_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Randomly sample MOUSE_SAMPLE_N windows."""
    n = min(MOUSE_SAMPLE_N, len(df))
    sampled = df.sample(n=n, random_state=MOUSE_SAMPLE_SEED).reset_index(drop=True)
    print(f"  Mouse sampled windows ({n}) : {len(sampled)}")
    return sampled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the shared Akita / AlphaGenome (/ ORCA) benchmark test set."
    )
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
        help=(
            "Directory containing alphagenome_{species}_safe_windows.tsv "
            "and where output will be written. "
            f"Defaults to {DEFAULT_DATA_ROOT}"
        ),
    )
    parser.add_argument(
        "--akita_bed",
        type=Path,
        default=None,
        help="Path to Akita sequences BED file. Defaults to the V2 project path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    species = args.species
    data_dir = args.data_dir or DEFAULT_DATA_ROOT
    akita_bed = args.akita_bed or Path(AKITA_BED[species])

    alphagenome_tsv = data_dir / f"alphagenome_{species}_safe_windows.tsv"
    output_path = data_dir / f"benchmark_test_set_{species}.tsv"

    print(f"Species       : {species}")
    print(f"Akita BED     : {akita_bed}")
    print(f"AlphaGenome   : {alphagenome_tsv}")
    print(f"Output        : {output_path}\n")

    # Load
    print("Loading data...")
    akita_df = load_akita_windows(akita_bed)
    alpha_df = load_alphagenome_windows(alphagenome_tsv)

    # Overlap
    print("\nComputing overlap...")
    overlap_df = compute_overlap(akita_df, alpha_df)

    # Species-specific selection
    print("\nApplying selection strategy...")
    if species == "human":
        final_df = select_human_windows(overlap_df)
    else:
        final_df = select_mouse_windows(overlap_df)

    # Save
    final_df.to_csv(output_path, sep="\t", index=False)
    print(f"\nSaved {len(final_df)} windows to: {output_path}")


if __name__ == "__main__":
    main()
