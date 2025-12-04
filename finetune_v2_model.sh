#!/bin/bash

#SBATCH --job-name=NPC_m7
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=409600MB
#SBATCH --time=27:00:00
#SBATCH --constraint=l40s

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_cuda11.8

# Define training arguments
DATA_DIR="/scratch1/smaruj/Akita_pytorch_training_data/mouse_data/Bonev2017_NPC"
TEST_FOLD="fold7"
VAL_FOLD="fold0"
DATA_NAME="Bonev2017_NPC"
ORG="mouse"
DATASPLIT=7
BATCH_SIZE=4 
EPOCHS=70
LR=0.001 # initial learning rate x5
OPTIMIZER="adam"  # originally sgd
MOMENTUM=0.98       # Only used for SGD, as originally : 0.99575
L2_SCALE="1.5e-5"
WEIGHT_CLIPPING=10.0 # as originally : 20.0
EARLY_STOP=5 # originally - 50, originally 5 for known cell types fine-tuning
LOG_INTERVAL=100

python finetune_v2_model.py \
    --data_dir "$DATA_DIR" \
    --test_fold "$TEST_FOLD" \
    --val_fold "$VAL_FOLD" \
    --data_name "$DATA_NAME" \
    --organism "$ORG" \
    --data-split "$DATASPLIT" \
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
