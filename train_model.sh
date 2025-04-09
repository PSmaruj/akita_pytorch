#!/bin/bash

#SBATCH --job-name=mem_opt
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=2
#SBATCH --mem=384000MB
#SBATCH --time=1:00:00
#SBATCH --constraint=a100

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_cuda11.8

# Define training arguments
DATA_DIR="/scratch1/smaruj/train_pytorch_akita/mouse_data/Hsieh2019_mESC_data_local"
TEST_FOLD="fold0"
VAL_FOLD="fold1"
BATCH_SIZE=16
EPOCHS=3
LR=0.0065 # as originally : 0.0065
OPTIMIZER="adam"  # originally sgd
MOMENTUM=0.9       # Only used for SGD, as originally : 0.99575
WEIGHT_CLIPPING=10.0 # as originally : 10.0
EARLY_STOP=12 # as originally
LOG_INTERVAL=100
SAVE_MODEL_PATH="/scratch1/smaruj/train_pytorch_akita/mouse_models/test.pt"
SAVE_LOSSES_PATH="/scratch1/smaruj/train_pytorch_akita/mouse_models_losses/test.csv"

python train_model.py \
    --data_dir "$DATA_DIR" \
    --test_fold "$TEST_FOLD" \
    --val_fold "$VAL_FOLD" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --optimizer "$OPTIMIZER" \
    --momentum "$MOMENTUM" \
    --weight-clipping "$WEIGHT_CLIPPING" \
    --early-stop-patience "$EARLY_STOP" \
    --log-interval "$LOG_INTERVAL" \
    --save-model \
    --save-model-path "$SAVE_MODEL_PATH" \
    --save-losses "$SAVE_LOSSES_PATH"
