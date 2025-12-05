#!/bin/bash

#==============================================================================
# Train Akita v2 Model from Scratch
#==============================================================================
# This script trains an Akita v2 model from random initialization on a Hi-C
# dataset. The model is trained with early stopping and the best checkpoint
# is saved based on validation loss.
#==============================================================================

#------------------------------------------------------------------------------
# SLURM Configuration
#------------------------------------------------------------------------------
#SBATCH --job-name=train_akita        # Job name
#SBATCH --account=fudenber_735        # Account (adjust to your account)
#SBATCH --partition=qcbgpu            # GPU partition
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=50            # CPUs per task
#SBATCH --gpus-per-node=1             # GPUs per node
#SBATCH --mem=409600MB                # Memory (400GB)
#SBATCH --time=60:00:00               # Time limit
#SBATCH --constraint=l40s             # GPU type constraint
#SBATCH --output=train_%j.out         # Standard output log
#SBATCH --error=train_%j.err          # Standard error log

#------------------------------------------------------------------------------
# Environment Setup
#------------------------------------------------------------------------------
echo "=========================================="
echo "Akita v2 Training from Scratch"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: ${SLURM_JOB_GPUS}"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch_hic

# Verify GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

# Dataset configuration
DATA_DIR="/scratch1/smaruj/Akita_pytorch_training_data/mouse_data/Vian2018_Bcells"
DATASET_NAME="Vian2018_Bcells"
ORGANISM="mouse"                      # mouse or human
MODEL_IDX=0                           # Model/fold index

# Data splits
TEST_FOLD="fold0"                     # Fold for testing (held out)
VAL_FOLD="fold1"                      # Fold for validation

# Training hyperparameters
BATCH_SIZE=4                          # Batch size
EPOCHS=200                            # Maximum number of epochs
LR=0.01                               # Learning rate (5x base rate)
OPTIMIZER="adam"                      # Optimizer: adam or sgd
MOMENTUM=0.98                         # Momentum for SGD (ignored for Adam)
L2_SCALE="1.5e-5"                     # L2 regularization (weight decay)
WEIGHT_CLIPPING=20.0                  # Weight clipping value
EARLY_STOP_PATIENCE=50                # Early stopping patience (epochs)
LOG_INTERVAL=200                      # Logging interval (batches)

# Output paths
OUTPUT_BASE="/scratch1/smaruj/Akita_pytorch_models/trained_from_scratch/${ORGANISM}_models/${DATASET_NAME}"
SAVE_MODEL_PATH="${OUTPUT_BASE}/models/Akita_v2_${ORGANISM}_${DATASET_NAME}_model${MODEL_IDX}_from_scratch.pth"
SAVE_LOSSES_PATH="${OUTPUT_BASE}/losses/Akita_v2_${ORGANISM}_${DATASET_NAME}_model${MODEL_IDX}_from_scratch.csv"


#------------------------------------------------------------------------------
# Display Configuration
#------------------------------------------------------------------------------
echo "Configuration:"
echo "  Data directory:     ${DATA_DIR}"
echo "  Dataset name:       ${DATASET_NAME}"
echo "  Organism:           ${ORGANISM}"
echo "  Model index:        ${MODEL_IDX}"
echo "  Test fold:          ${TEST_FOLD}"
echo "  Validation fold:    ${VAL_FOLD}"
echo ""
echo "Training parameters:"
echo "  Batch size:         ${BATCH_SIZE}"
echo "  Epochs:             ${EPOCHS}"
echo "  Learning rate:      ${LR}"
echo "  Optimizer:          ${OPTIMIZER}"
echo "  Momentum:           ${MOMENTUM}"
echo "  L2 regularization:  ${L2_SCALE}"
echo "  Weight clipping:    ${WEIGHT_CLIPPING}"
echo "  Early stop patience: ${EARLY_STOP_PATIENCE}"
echo "  Log interval:       ${LOG_INTERVAL}"
echo ""
echo "Output paths:"
echo "  Model checkpoint:   ${SAVE_MODEL_PATH}"
echo "  Loss history:       ${SAVE_LOSSES_PATH}"
echo ""


#------------------------------------------------------------------------------
# Validate Input Files
#------------------------------------------------------------------------------
echo "Validating input files..."

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: ${DATA_DIR}"
    exit 1
fi

# Count .pt files
NUM_FILES=$(find "$DATA_DIR" -name "*.pt" | wc -l)
if [ "$NUM_FILES" -eq 0 ]; then
    echo "Error: No .pt files found in ${DATA_DIR}"
    exit 1
fi

echo "✓ Found ${NUM_FILES} .pt files in data directory"
echo ""


#------------------------------------------------------------------------------
# Create Output Directories
#------------------------------------------------------------------------------
mkdir -p "$(dirname "$SAVE_MODEL_PATH")"
mkdir -p "$(dirname "$SAVE_LOSSES_PATH")"


#------------------------------------------------------------------------------
# Run Training
#------------------------------------------------------------------------------
echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""

python train_model.py \
    --data_dir "$DATA_DIR" \
    --test_fold "$TEST_FOLD" \
    --val_fold "$VAL_FOLD" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --optimizer "$OPTIMIZER" \
    --momentum "$MOMENTUM" \
    --l2-scale "$L2_SCALE" \
    --weight-clipping "$WEIGHT_CLIPPING" \
    --early-stop-patience "$EARLY_STOP_PATIENCE" \
    --log-interval "$LOG_INTERVAL" \
    --save-model \
    --save-model-path "$SAVE_MODEL_PATH" \
    --save-losses "$SAVE_LOSSES_PATH"

# Capture exit code
EXIT_CODE=$?


#------------------------------------------------------------------------------
# Completion Report
#------------------------------------------------------------------------------
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo ""
    echo "Output files:"
    echo "  Model checkpoint: ${SAVE_MODEL_PATH}"
    echo "  Loss history:     ${SAVE_LOSSES_PATH}"
    echo ""
    echo "Next steps:"
    echo "  1. Check loss curves to verify convergence"
    echo "  2. Evaluate model: evaluate_model.ipynb"
else
    echo "✗ Training failed with exit code: ${EXIT_CODE}"
    echo "Check error log: train_${SLURM_JOB_ID}.err"
fi
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
