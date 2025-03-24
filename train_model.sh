#!/bin/bash

#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=409600MB
#SBATCH --time=14:00:00

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_cuda11.8

# Define training arguments
DATA_DIR="/scratch1/smaruj/train_pytorch_akita/mouse"
TEST_FOLD="fold0"
VAL_FOLD="fold1"
BATCH_SIZE=8
EPOCHS=100
LR=0.001
EARLY_STOP=15
LOG_INTERVAL=200
SAVE_MODEL_PATH="/scratch1/smaruj/train_pytorch_akita/mouse_models/model_0_reference.pt"

python train_model.py \
    --data_dir "$DATA_DIR" \
    --test_fold "$TEST_FOLD" \
    --val_fold "$VAL_FOLD" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --early-stop-patience "$EARLY_STOP" \
    --log-interval "$LOG_INTERVAL" \
    --save-model \
    --save-model-path "$SAVE_MODEL_PATH"