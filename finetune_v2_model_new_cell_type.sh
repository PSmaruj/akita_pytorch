#!/bin/bash

#SBATCH --job-name=ORC_m0
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=409600MB
#SBATCH --time=40:00:00
#SBATCH --constraint=l40s

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_cuda11.8

# Define training arguments
DATA_DIR="/scratch1/smaruj/Akita_pytorch_training_data/mouse_data/Monahan2019_ORC"
TEST_FOLD="fold0"
VAL_FOLD="fold1"
DATA_NAME="Monahan2019_ORC"
ORG="mouse"
DATASPLIT=0
BATCH_SIZE=4 

# --- Stage 1: Head-only fine-tuning ---
HEAD_EPOCHS=15
LR_HEAD=0.002
HEAD_WEIGHT_CLIPPING=10.0
HEAD_L2_SCALE=1e-6
HEAD_EARLY_STOP=8

# --- Stage 2: Full model fine-tuning ---
EPOCHS=70
LR=0.001        # Head LR: LR, Backbone LR: LR*0.1 (handled in script)
WEIGHT_CLIPPING=15.0
L2_SCALE=1.5e-5
EARLY_STOP=20

LOG_INTERVAL=100

# -------------------------
# Run Training
# -------------------------
python finetune_v2_model_new_cell_type.py \
    --data_dir "$DATA_DIR" \
    --test_fold "$TEST_FOLD" \
    --val_fold "$VAL_FOLD" \
    --data_name "$DATA_NAME" \
    --organism "$ORG" \
    --data-split "$DATASPLIT" \
    --batch-size "$BATCH_SIZE" \
    --head_epochs "$HEAD_EPOCHS" \
    --lr_head "$LR_HEAD" \
    --head_weight_clipping "$HEAD_WEIGHT_CLIPPING" \
    --head_l2_scale "$HEAD_L2_SCALE" \
    --head_early_stop "$HEAD_EARLY_STOP" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --weight_clipping "$WEIGHT_CLIPPING" \
    --l2_scale "$L2_SCALE" \
    --early_stop "$EARLY_STOP" \
    --log-interval "$LOG_INTERVAL" \
    --save-model
