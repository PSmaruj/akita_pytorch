#!/bin/bash

#==============================================================================
# Hi-C Data Processing Pipeline
#==============================================================================
# This script processes Hi-C pairs files into multi-resolution cooler files:
#   1. Merges multiple pairs files (if applicable)
#   2. Filters by mapping quality (MAPQ >= 30)
#   3. Creates a 512bp resolution .cool file
#   4. Generates coarser resolutions (1024bp, 2048bp, 4096bp)
#   5. Balances all matrices using ICE normalization
#
# Input: .pairs.gz files (4DN format or similar)
# Output: Balanced .cool files at multiple resolutions
#==============================================================================

#------------------------------------------------------------------------------
# SLURM Configuration
#------------------------------------------------------------------------------
#SBATCH --job-name=create_cool          # Job name
#SBATCH --account=fudenber_735          # Account (adjust to your account)
#SBATCH --partition=qcb                 # Partition (adjust to your partition)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=64              # CPUs per task
#SBATCH --mem=0                         # Memory (0 = use all available)
#SBATCH --time=40:00:00                 # Time limit (HH:MM:SS)
#SBATCH --output=create_cool_%j.out     # Standard output log
#SBATCH --error=create_cool_%j.err      # Standard error log

#------------------------------------------------------------------------------
# Environment Setup
#------------------------------------------------------------------------------
# Activate conda environment with cooler and pairtools
eval "$(conda shell.bash hook)"
conda activate pytorch_hic

#------------------------------------------------------------------------------
# Configuration - MODIFY THESE PATHS FOR YOUR DATA
#------------------------------------------------------------------------------

# Genome assembly and chromosome sizes
ASSEMBLY="mm10"
CHROMSIZES="/project2/fudenber_735/genomes/mm10/mm10.chrom.sizes.reduced"

# For human data, uncomment these lines:
# ASSEMBLY="hg38"
# CHROMSIZES="/project2/fudenber_735/genomes/hg38/hg38.chrom.sizes.reduced"

# Resolution settings
BIN_SIZE=512                            # Base resolution in bp

# Data paths
DATA_DIR="/scratch1/smaruj/Akita_pytorch_training_data/mouse_unprocessed_data/Vian2018_Bcells"

# Input pairs files (can specify multiple files for merging)
PAIRS1="4DNFI27I3P1V.pairs.gz"
PAIRS2="4DNFIFBBAKK4.pairs.gz"

# For single pairs file, use:
# PAIRS_FILE="4DNFI16FU2Y5.pairs.gz"

# Output file prefix
OUT_PREFIX="${DATA_DIR}/HiC_Vian2018_Bcells.${ASSEMBLY}.mapq30"

# Quality filtering
MIN_MAPQ=30                             # Minimum mapping quality

#------------------------------------------------------------------------------
# Pipeline Execution
#------------------------------------------------------------------------------

echo "=========================================="
echo "Hi-C Processing Pipeline Started"
echo "=========================================="
echo "Assembly: ${ASSEMBLY}"
echo "Base resolution: ${BIN_SIZE}bp"
echo "Minimum MAPQ: ${MIN_MAPQ}"
echo "Output prefix: ${OUT_PREFIX}"
echo "Start time: $(date)"
echo ""

#------------------------------------------------------------------------------
# Step 1: Merge, Filter, and Create Base Resolution Cool File
#------------------------------------------------------------------------------
echo "Step 1: Processing pairs files..."
echo "  - Merging input files"
echo "  - Filtering by MAPQ >= ${MIN_MAPQ}"
echo "  - Sorting pairs"

# Merge multiple pairs files (preserves header from first file)
(
  zcat "${DATA_DIR}/${PAIRS1}" | grep '^#'
  zcat "${DATA_DIR}/${PAIRS1}" "${DATA_DIR}/${PAIRS2}" | grep -v '^#'
) \
| pairtools select "(int(mapq1)>=${MIN_MAPQ}) and (int(mapq2)>=${MIN_MAPQ})" \
| pairtools sort --nproc ${SLURM_CPUS_PER_TASK} \
> ${OUT_PREFIX}.mapq${MIN_MAPQ}.pairs

# For single pairs file, use this instead:
# zcat "${DATA_DIR}/${PAIRS_FILE}" \
# | pairtools select "(int(mapq1)>=${MIN_MAPQ}) and (int(mapq2)>=${MIN_MAPQ})" \
# | pairtools sort --nproc ${SLURM_CPUS_PER_TASK} \
# > ${OUT_PREFIX}.mapq${MIN_MAPQ}.pairs

echo " ✓ Filtered pairs file created"

# Create cooler file at base resolution
echo "  - Creating ${BIN_SIZE}bp resolution cooler file"
cooler cload pairs \
    -c1 2 -p1 3 -c2 4 -p2 5 \
    --assembly ${ASSEMBLY} \
    ${CHROMSIZES}:${BIN_SIZE} \
    ${OUT_PREFIX}.mapq${MIN_MAPQ}.pairs \
    ${OUT_PREFIX}.${BIN_SIZE}.cool

echo " ✓ Base resolution cooler created"
echo ""

#------------------------------------------------------------------------------
# Step 2: Generate Coarser Resolutions
#------------------------------------------------------------------------------
echo "Step 2: Generating coarser resolutions..."

cooler coarsen --out ${OUT_PREFIX}.1024.cool -k 2 ${OUT_PREFIX}.512.cool
echo " ✓ 1024bp resolution created"

cooler coarsen --out ${OUT_PREFIX}.2048.cool -k 2 ${OUT_PREFIX}.1024.cool
echo " ✓ 2048bp resolution created"

cooler coarsen --out ${OUT_PREFIX}.4096.cool -k 2 ${OUT_PREFIX}.2048.cool
echo " ✓ 4096bp resolution created"

echo ""

#------------------------------------------------------------------------------
# Step 3: Balance All Resolutions (ICE Normalization)
#------------------------------------------------------------------------------
echo "Step 3: Balancing matrices..."

cooler balance ${OUT_PREFIX}.512.cool
echo " ✓ 512bp balanced"

cooler balance ${OUT_PREFIX}.1024.cool
echo " ✓ 1024bp balanced"

cooler balance ${OUT_PREFIX}.2048.cool
echo " ✓ 2048bp balanced"

cooler balance ${OUT_PREFIX}.4096.cool
echo " ✓ 4096bp balanced"

echo ""

#------------------------------------------------------------------------------
# Completion
#------------------------------------------------------------------------------
echo "=========================================="
echo "Pipeline Completed Successfully!"
echo "=========================================="
echo "Output files created:"
echo "  - ${OUT_PREFIX}.mapq${MIN_MAPQ}.pairs"
echo "  - ${OUT_PREFIX}.512.cool"
echo "  - ${OUT_PREFIX}.1024.cool"
echo "  - ${OUT_PREFIX}.2048.cool"
echo "  - ${OUT_PREFIX}.4096.cool"
echo ""
echo "End time: $(date)"
echo "=========================================="
