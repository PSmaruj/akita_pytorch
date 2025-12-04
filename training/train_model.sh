#!/bin/bash

#SBATCH --job-name=Bc_m0
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=409600MB
#SBATCH --time=60:00:00
#SBATCH --constraint=l40s

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_cuda11.8

# Define training arguments
DATA_DIR="/scratch1/smaruj/Akita_pytorch_training_data/mouse_data/Vian2018_Bcells"
TEST_FOLD="fold0"
VAL_FOLD="fold1"
BATCH_SIZE=4
EPOCHS=200
LR=0.01 # initial learning rate x5
OPTIMIZER="adam"  # originally sgd
MOMENTUM=0.98       # Only used for SGD, as originally : 0.99575
L2_SCALE="1.5e-5"
WEIGHT_CLIPPING=20.0 # as originally : 20.0
EARLY_STOP=50 # originally - 50
LOG_INTERVAL=200
SAVE_MODEL_PATH="/scratch1/smaruj/Akita_pytorch_models/finetuned/mouse_models/Vian2018_Bcells/models/Akita_v2_mouse_Vian2018_Bcells_model0_R_from_scratch.pth"
SAVE_LOSSES_PATH="/scratch1/smaruj/Akita_pytorch_models/finetuned/mouse_models/Vian2018_Bcells/losses/Akita_v2_mouse_Vian2018_Bcells_model0_R_from_scratch.csv"

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
