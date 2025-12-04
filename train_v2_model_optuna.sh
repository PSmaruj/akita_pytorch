#!/bin/bash

#SBATCH --job-name=Bc_0_opt
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=409600MB
#SBATCH --time=5-00:00:00   # 5 days
#SBATCH --constraint=l40s

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_cuda11.8

# Define training arguments
DATA_DIR="/scratch1/smaruj/Akita_pytorch_training_data/mouse_data/Vian2018_Bcells"
TEST_FOLD="fold0"
VAL_FOLD="fold1"
DATA_NAME="Vian2018_Bcells"
ORG="mouse"
DATASPLIT=0
BATCH_SIZE=4
EPOCHS=15            # per trial, short for Optuna search
N_TRIALS=20           # total Optuna trials
SEED=1

python train_v2_model_optuna.py \
    --data_dir "$DATA_DIR" \
    --test_fold "$TEST_FOLD" \
    --val_fold "$VAL_FOLD" \
    --data_name "$DATA_NAME" \
    --organism "$ORG" \
    --data_split "$DATASPLIT" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --n-trials "$N_TRIALS" \
    --seed "$SEED"
