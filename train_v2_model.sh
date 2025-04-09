#!/bin/bash

#SBATCH --job-name=memory_a100
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=2
#SBATCH --mem=350000MB
#SBATCH --time=32:00:00
#SBATCH --constraint=a100

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_cuda11.8

# Define training arguments
DATA_DIR="/scratch1/smaruj/train_pytorch_akita/mouse_data/Hsieh2019_mESC_data_local"
TEST_FOLD="fold0"
VAL_FOLD="fold1"
BATCH_SIZE=16
EPOCHS=60
LR=0.001 # initial learning rate x5
OPTIMIZER="adam"  # originally sgd
MOMENTUM=0.98       # Only used for SGD, as originally : 0.99575
L2_SCALE="1.5e-5"
WEIGHT_CLIPPING=10.0 # as originally : 20.0
EARLY_STOP=5 # originally - 50
LOG_INTERVAL=100
SAVE_MODEL_PATH="/scratch1/smaruj/train_pytorch_akita/mouse_models/a100_2gpus_memory_opt_finetuning.pt"
SAVE_LOSSES_PATH="/scratch1/smaruj/train_pytorch_akita/mouse_models_losses/a100_2gpus_memory_opt_finetuning.csv"

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
