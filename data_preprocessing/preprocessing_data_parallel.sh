#!/bin/bash

#==============================================================================
# Parallel Hi-C Data Preprocessing for Akita Training
#==============================================================================
# This script runs the Python preprocessing pipeline to convert Hi-C cooler
# files and DNA sequences into PyTorch tensors for Akita model training.
#
# The script processes data in parallel across multiple folds and handles:
# - One-hot encoding of DNA sequences
# - Hi-C matrix processing (balancing, filtering, obs/exp normalization)
# - Gap region masking
# - Output as .pt files containing (sequence, contact_matrix) pairs
#==============================================================================

#------------------------------------------------------------------------------
# SLURM Configuration
#------------------------------------------------------------------------------
#SBATCH --job-name=preprocess_hic      # Job name
#SBATCH --account=fudenber_735         # Account (adjust to your account)
#SBATCH --partition=qcb                # Partition (adjust to your partition)
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Tasks per node
#SBATCH --cpus-per-task=64             # CPUs per task (for parallel processing)
#SBATCH --mem=0                        # Memory (0 = use all available)
#SBATCH --time=1:00:00                 # Time limit (adjust based on dataset size)
#SBATCH --output=preprocess_%j.out     # Standard output log
#SBATCH --error=preprocess_%j.err      # Standard error log

#------------------------------------------------------------------------------
# Environment Setup
#------------------------------------------------------------------------------
echo "=========================================="
echo "Hi-C Preprocessing Pipeline"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch_hic

#------------------------------------------------------------------------------
# Configuration - MODIFY THESE PATHS FOR YOUR DATA
#------------------------------------------------------------------------------

# Species selection: set to "mouse" or "human"
SPECIES="human"

# Hi-C data
COOL_FILE="/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/Akita_pytorch_training_data/human_cool_files/Krietenstein2019_HFF/HiC_Krietenstein2019_HFF.hg38.mapq30.2048.cool"

# Output directory for processed .pt files
OUTPUT_DIR="/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/Akita_pytorch_training_data/human_training_data/Krietenstein2019_HFF"

# Parallel processing settings
NUM_WORKERS=64                         # Should match or be less than --cpus-per-task
START_FOLD=0                           # First fold to process
END_FOLD=7                             # Last fold to process (inclusive)

# Optional: Set log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL="INFO"

#------------------------------------------------------------------------------
# Species-Specific Paths
#------------------------------------------------------------------------------

if [ "$SPECIES" = "mouse" ]; then
    echo "Species: Mouse (mm10)"
    FASTA_FILE="/project2/fudenber_735/genomes/mm10/mm10.fa"
    BED_FILE="/project2/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed"
    GAPS_FILE="/project2/fudenber_735/backup/DNN_HiC/data_mm10/mm10.blacklist.rep.bed"
    
elif [ "$SPECIES" = "human" ]; then
    echo "Species: Human (hg38)"
    FASTA_FILE="/project2/fudenber_735/genomes/hg38/hg38.fa"
    BED_FILE="/project2/fudenber_735/tensorflow_models/akita/v2/data/hg38/sequences.bed"
    GAPS_FILE="/project2/fudenber_735/backup/DNN_HiC/data_hg38/hg38.blacklist.rep.bed"
    
else
    echo "Error: SPECIES must be 'mouse' or 'human'"
    exit 1
fi

#------------------------------------------------------------------------------
# Display Configuration
#------------------------------------------------------------------------------
echo ""
echo "Configuration:"
echo "  Cool file:      ${COOL_FILE}"
echo "  FASTA file:     ${FASTA_FILE}"
echo "  BED file:       ${BED_FILE}"
echo "  Gaps file:      ${GAPS_FILE}"
echo "  Output dir:     ${OUTPUT_DIR}"
echo "  Workers:        ${NUM_WORKERS}"
echo "  Folds:          ${START_FOLD} to ${END_FOLD}"
echo "  Log level:      ${LOG_LEVEL}"
echo ""

#------------------------------------------------------------------------------
# Validate Input Files
#------------------------------------------------------------------------------
echo "Validating input files..."

if [ ! -f "$COOL_FILE" ]; then
    echo "Error: Cool file not found: ${COOL_FILE}"
    exit 1
fi

if [ ! -f "$FASTA_FILE" ]; then
    echo "Error: FASTA file not found: ${FASTA_FILE}"
    exit 1
fi

if [ ! -f "$BED_FILE" ]; then
    echo "Error: BED file not found: ${BED_FILE}"
    exit 1
fi

if [ ! -f "$GAPS_FILE" ]; then
    echo "Warning: Gaps file not found: ${GAPS_FILE}"
    echo "Continuing without gap masking..."
    GAPS_ARG=""
else
    GAPS_ARG="--gaps_file ${GAPS_FILE}"
fi

echo "✓ Input files validated"
echo ""

#------------------------------------------------------------------------------
# Create Output Directory
#------------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"

#------------------------------------------------------------------------------
# Run Preprocessing Pipeline
#------------------------------------------------------------------------------
echo "=========================================="
echo "Starting preprocessing..."
echo "=========================================="
echo ""

python preprocessing_data_parallel.py \
  --cool_file "$COOL_FILE" \
  --fasta_file "$FASTA_FILE" \
  --bed_file "$BED_FILE" \
  $GAPS_ARG \
  --output_dir "$OUTPUT_DIR" \
  --num_workers "$NUM_WORKERS" \
  --start_fold "$START_FOLD" \
  --end_fold "$END_FOLD" \
  --log_level "$LOG_LEVEL"

# Capture exit code
EXIT_CODE=$?

#------------------------------------------------------------------------------
# Completion Report
#------------------------------------------------------------------------------
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Preprocessing completed successfully!"
    echo ""
    echo "Output files saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Next steps:"
    echo "  1. Verify output files: ls ${OUTPUT_DIR}"
    echo "  2. Check for expected number of .pt files"
    echo "  3. Run training with: train_model.py or finetune_model.py"
else
    echo "✗ Preprocessing failed with exit code: ${EXIT_CODE}"
    echo "Check error log: preprocess_${SLURM_JOB_ID}.err"
fi
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
  