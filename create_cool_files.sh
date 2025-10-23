#!/bin/bash
#SBATCH --job-name=Hep_hic
#SBATCH --account=fudenber_735           # adjust to your account
#SBATCH --partition=qcb                  # adjust to your partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64               # number of CPUs to use
#SBATCH --mem=0                       # all memory
#SBATCH --time=24:00:00

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_cuda11.8

# Set paths
# ASSEMBLY="mm10"
# CHROMSIZES="/project2/fudenber_735/genomes/mm10/mm10.chrom.sizes.reduced"
ASSEMBLY="hg38"
CHROMSIZES="/project2/fudenber_735/genomes/hg38/hg38.chrom.sizes.reduced"
BIN_SIZE=512
DATA_DIR="/scratch1/smaruj/Akita_pytorch_training_data/human_unprocessed_data/HepG2"
PAIRS_FILE="4DNFIQ4G74OW.pairs.gz"
OUT_PREFIX="${DATA_DIR}/HiC_HepG2.mm10.mapq30"

# --- Step 1. Merge + filter + create 512bp cool file ---
echo "Starting Hi-C processing..."
date

zcat "${DATA_DIR}/${PAIRS_FILE}" \
| pairtools select '(int(mapq1)>=30) and (int(mapq2)>=30)' \
| pairtools sort --nproc ${SLURM_CPUS_PER_TASK} \
> ${OUT_PREFIX}.mapq30.pairs

cooler cload pairs \
    -c1 2 -p1 3 -c2 4 -p2 5 \
    --assembly ${ASSEMBLY} \
    ${CHROMSIZES}:${BIN_SIZE} \
    ${OUT_PREFIX}.mapq30.pairs \
    ${OUT_PREFIX}.${BIN_SIZE}.cool

# --- Step 2. Generate coarser resolutions ---
echo "Generating coarser resolutions..."
cooler coarsen --out ${OUT_PREFIX}.1024.cool -k 2 ${OUT_PREFIX}.512.cool
cooler coarsen --out ${OUT_PREFIX}.2048.cool -k 2 ${OUT_PREFIX}.1024.cool
cooler coarsen --out ${OUT_PREFIX}.4096.cool -k 2 ${OUT_PREFIX}.2048.cool

# --- Step 3. Balance each resolution ---
echo "Balancing matrices..."
cooler balance ${OUT_PREFIX}.512.cool
cooler balance ${OUT_PREFIX}.1024.cool
cooler balance ${OUT_PREFIX}.2048.cool
cooler balance ${OUT_PREFIX}.4096.cool

echo "All done!"
date