#!/bin/bash

#SBATCH --job-name=fine_tune
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=409600MB
#SBATCH --time=5-01:00:00
#SBATCH --constraint=l40s

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_cuda11.8

python train_v2_model_optuna.py
