#!/bin/bash
#SBATCH --job-name=ORC_data
#SBATCH --account=fudenber_735
#SBATCH --partition=qcb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0                   # all memory on node
#SBATCH --time=1:00:00

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_cuda11.8

# cooler
COOL_FILE="/scratch1/smaruj/Akita_pytorch_training_data/mouse_unprocessed_data/Monahan2019_ORC/HiC_Monahan2019_ORC.mm10.mapq30.2048.cool"

OUTPUT_DIR="/scratch1/smaruj/Akita_pytorch_training_data/mouse_data/Monahan2019_ORC"

python preprocessing_data_parallel.py \
  --cool_file "$COOL_FILE" \
  --fasta_file /project2/fudenber_735/genomes/mm10/mm10.fa \
  --bed_file /project2/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed \
  --gaps_file /project2/fudenber_735/backup/DNN_HiC/data_mm10/mm10.blacklist.rep.bed \
  --output_dir "$OUTPUT_DIR" \
  --num_workers 64

# python preprocessing_data_parallel.py \
#   --cool_file "$COOL_FILE" \
#   --fasta_file /project2/fudenber_735/genomes/hg38/hg38.fa \
#   --bed_file /project2/fudenber_735/tensorflow_models/akita/v2/data/hg38/sequences.bed \
#   --gaps_file  \
#   --output_dir "$OUTPUT_DIR" \
#   --num_workers 64