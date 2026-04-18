#!/bin/bash

#==============================================================================
# Fine-tune Akita v2 Model
#==============================================================================
# This script fine-tunes a pretrained Akita v2 model on a new Hi-C dataset.
# The pretrained model (transferred from TensorFlow) is loaded and trained
# with early stopping on the target dataset.
#==============================================================================

#------------------------------------------------------------------------------
# SLURM Configuration
#------------------------------------------------------------------------------
#SBATCH --job-name=finetune_akita     # Job name
#SBATCH --account=fudenber_735        # Account (adjust to your account)
#SBATCH --partition=qcbgpu            # GPU partition
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=50            # CPUs per task
#SBATCH --gpus-per-node=1             # GPUs per node
#SBATCH --mem=409600MB                # Memory (400GB)
#SBATCH --time=27:00:00               # Time limit
#SBATCH --constraint=l40s             # GPU type constraint
#SBATCH --output=finetune_%j.out      # Standard output log
#SBATCH --error=finetune_%j.err       # Standard error log


#------------------------------------------------------------------------------
# Environment Setup
#------------------------------------------------------------------------------
echo "=========================================="
echo "Akita v2 Fine-tuning"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: ${SLURM_JOB_GPUS}"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch_akita

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
DATA_DIR="/path/to/your/training_data/Bonev2017_NPC"  # directory containing processed .pt files
DATA_NAME="Bonev2017_NPC"
ORGANISM="mouse"                      # mouse or human
DATA_SPLIT=0                          # Model/fold index

# Data splits
TEST_FOLD="fold0"                     # Fold for testing (held out)
VAL_FOLD="fold1"                      # Fold for validation

# Training hyperparameters
BATCH_SIZE=4                          # Batch size
EPOCHS=70                             # Maximum number of epochs
LR=0.001                              # Learning rate (5x initial)
OPTIMIZER="adam"                      # Optimizer: adam or sgd
MOMENTUM=0.98                         # Momentum for SGD (ignored for Adam)
L2_SCALE="1.5e-5"                     # L2 regularization (weight decay)
WEIGHT_CLIPPING=10.0                  # Weight clipping value
EARLY_STOP_PATIENCE=5                 # Early stopping patience (epochs)
LOG_INTERVAL=100                      # Logging interval (batches)

#------------------------------------------------------------------------------
# Display Configuration
#------------------------------------------------------------------------------
echo "Configuration:"
echo "  Data directory:     ${DATA_DIR}"
echo "  Dataset name:       ${DATA_NAME}"
echo "  Organism:           ${ORGANISM}"
echo "  Data split:         ${DATA_SPLIT}"
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
# Run Fine-tuning
#------------------------------------------------------------------------------
echo "=========================================="
echo "Starting fine-tuning..."
echo "=========================================="
echo ""

python finetune_model.py \
    --data_dir "$DATA_DIR" \
    --test_fold "$TEST_FOLD" \
    --val_fold "$VAL_FOLD" \
    --data_name "$DATA_NAME" \
    --organism "$ORGANISM" \
    --data-split "$DATA_SPLIT" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --optimizer "$OPTIMIZER" \
    --momentum "$MOMENTUM" \
    --l2-scale "$L2_SCALE" \
    --weight-clipping "$WEIGHT_CLIPPING" \
    --early-stop-patience "$EARLY_STOP_PATIENCE" \
    --log-interval "$LOG_INTERVAL" \
    --save-model

# Capture exit code
EXIT_CODE=$?


#------------------------------------------------------------------------------
# Completion Report
#------------------------------------------------------------------------------
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Fine-tuning completed successfully!"
    echo ""
    echo "Output files saved to the directory specified in finetune_model.py:"
    echo "  - Model checkpoints: .../finetuned/${ORGANISM}_models/${DATA_NAME}/models/"
    echo "  - Loss history:      .../finetuned/${ORGANISM}_models/${DATA_NAME}/losses/"
    echo ""
    echo "Next steps:"
    echo "  1. Check loss curves: analyze_finetuning_loss.ipynb"
    echo "  2. Evaluate model: evaluate_model.ipynb"
else
    echo "✗ Fine-tuning failed with exit code: ${EXIT_CODE}"
    echo "Check error log: finetune_${SLURM_JOB_ID}.err"
fi
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE