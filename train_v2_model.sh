#!/bin/bash

#SBATCH --job-name=fine_tune
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=409600MB
#SBATCH --time=26:00:00
#SBATCH --constraint=l40s

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_cuda11.8

# Define training arguments
DATA_DIR="/scratch1/smaruj/train_pytorch_akita/mouse_data/Hsieh2019_mESC_data_local"
TEST_FOLD="fold0"
VAL_FOLD="fold1"
BATCH_SIZE=4
EPOCHS=70
LR=0.001 # initial learning rate x5
OPTIMIZER="adam"  # originally sgd
MOMENTUM=0.98       # Only used for SGD, as originally : 0.99575
L2_SCALE="1.5e-5"
WEIGHT_CLIPPING=10.0 # as originally : 20.0
EARLY_STOP=5 # originally - 50
LOG_INTERVAL=200
SAVE_MODEL_PATH="/scratch1/smaruj/train_pytorch_akita/mouse_models/model_0_v2_finetuned_shuffled.pt"
SAVE_LOSSES_PATH="/scratch1/smaruj/train_pytorch_akita/mouse_models_losses/model_0_v2_finetuned_shuffled.csv"

python train_v2_model.py \
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
    --early-stop-patience "$EARLY_STOP" \
    --log-interval "$LOG_INTERVAL" \
    --save-model \
    --save-model-path "$SAVE_MODEL_PATH" \
    --save-losses "$SAVE_LOSSES_PATH"
