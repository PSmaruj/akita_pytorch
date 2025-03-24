import numpy as np
import pandas as pd
import torch
import cooler
from cooltools.lib.numutils import observed_over_expected, adaptive_coarsegrain, set_diag, interp_nan
from astropy.convolution import Gaussian2DKernel, convolve
from pyfaidx import Fasta
import logging
import random


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- File Paths ---
FASTA_FILE = "/project/fudenber_735/genomes/mm10/mm10.fa"
BED_FILE = "/project/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed"
COOL_FILE = "/project/fudenber_735/GEO/Hsieh2019/4DN/mESC_mm10_4DNFILZ1CPT8.mapq_30.2048.cool"
OUTPUT_DIR = "/scratch1/smaruj/train_pytorch_akita/mouse"
FOLD = 0

# --- Load Data ---
genome = Fasta(FASTA_FILE)
df = pd.read_csv(BED_FILE, sep="\t", header=None, names=["chrom", "start", "end", "fold"])
df_select_fold = df[df["fold"] == f"fold{FOLD}"].reset_index(drop=True)

genome_hic_cool = cooler.Cooler(COOL_FILE)

# --- Functions ---
import random

def one_hot_encode_sequence(sequence_obj):
    sequence = str(sequence_obj).upper()
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    encoded_sequence = np.array([
        base_to_int.get(base, base_to_int[random.choice("ACGT")]) for base in sequence
    ])

    one_hot_encoded = np.zeros((4, len(encoded_sequence)), dtype=np.float32)
    one_hot_encoded[encoded_sequence, np.arange(len(encoded_sequence))] = 1

    return np.expand_dims(one_hot_encoded, axis=0)


def process_hic_matrix(genome_hic_cool, mseq_str, diagonal_offset=2, padding=64, kernel_stddev=1):
    seq_hic_raw = genome_hic_cool.matrix(balance=True).fetch(mseq_str)
    
    # Check for NaN filtering percentage
    seq_hic_nan = np.isnan(seq_hic_raw)
    num_filtered_bins = np.sum(np.sum(seq_hic_nan, axis=0) == len(seq_hic_nan))
    print("num_filtered_bins:", num_filtered_bins)
    
    if num_filtered_bins > (0.5 * len(seq_hic_nan)):
        print(f"More than 50% bins filtered in {mseq_str}. Check Hi-C data quality.")
        
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
    
    log_hic_obsexp = interp_nan(log_hic_obsexp)
    for i in range(-diagonal_offset+1, diagonal_offset): set_diag(log_hic_obsexp, 0,i)
    
    kernel = Gaussian2DKernel(x_stddev=kernel_stddev)
    seq_hic = convolve(log_hic_obsexp, kernel)
    
    return seq_hic


def upper_triangular_to_vector_skip_diagonals(matrix, dim=512, diag=2):
    
    # Extract the upper triangular part excluding the first two diagonals
    upper_triangular_vector = matrix[np.triu_indices(dim, k=diag)]
    
    return upper_triangular_vector


def generate_and_save_dataset(df, genome, genome_hic_cool, output_dir, fold=0):
    data_list = []
    file_count = 0

    for i, row in enumerate(df.itertuples(index=False)):
        chrom, start, end = row.chrom, row.start, row.end
        mseq_str = f"{chrom}:{start}-{end}"
        
        logging.info(f"Processing {mseq_str}")
        
        sequence = genome[chrom][start:end]
        ohe_sequence = one_hot_encode_sequence(sequence)
        matrix = process_hic_matrix(genome_hic_cool, mseq_str, diagonal_offset=2, padding=64, kernel_stddev=1)
        hic_vector = upper_triangular_to_vector_skip_diagonals(matrix)
        
        ohe_tensor = torch.tensor(ohe_sequence, dtype=torch.float32)
        hic_tensor = torch.tensor(hic_vector, dtype=torch.float32)

        ohe_tensor = ohe_tensor.squeeze(0) # sequence of shape [4, 1048576]
        hic_tensor = hic_tensor.unsqueeze(0) # vector of shape [1, 99681]
        
        data_list.append((ohe_tensor, hic_tensor))

        if (i + 1) % 100 == 0 or i == len(df) - 1:
            output_file = f"{output_dir}/fold{fold}_{file_count}.pt"
            torch.save(data_list, output_file)
            logging.info(f"Saved {len(data_list)} sequences to {output_file}")
            data_list = []
            file_count += 1


# --- Main Execution ---
if __name__ == "__main__":
    generate_and_save_dataset(df_select_fold, genome, genome_hic_cool, OUTPUT_DIR, fold=FOLD)