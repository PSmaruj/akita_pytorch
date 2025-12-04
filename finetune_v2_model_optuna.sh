#!/bin/bash

#SBATCH --job-name=IM_1_op
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=409600MB
#SBATCH --time=5-00:00:00
#SBATCH --constraint=l40s

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_cuda11.8

# Define Optuna arguments
DATA_DIR="/scratch1/smaruj/Akita_pytorch_training_data/human_data/Rao2014_IMR90"
TEST_FOLD="fold1"
VAL_FOLD="fold2"
DATA_NAME="Rao2014_IMR90"
ORG="human"
DATASPLIT=1
BATCH_SIZE=4
EPOCHS=15  # shorter run for Optuna search
SEED=1
N_TRIALS=20  # number of Optuna trials

python finetune_v2_model_optuna.py \
    --data_dir "$DATA_DIR" \
    --test_fold "$TEST_FOLD" \
    --val_fold "$VAL_FOLD" \
    --data_name "$DATA_NAME" \
    --organism "$ORG" \
    --data_split "$DATASPLIT" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --n-trials "$N_TRIALS"
