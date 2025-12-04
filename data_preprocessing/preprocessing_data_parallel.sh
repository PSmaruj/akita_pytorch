#!/bin/bash
#SBATCH --job-name=nCN_HiC
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
COOL_FILE="/project2/fudenber_735/GEO/bonev_2017_GSE96107/distiller-0.3.1_mm10/results/coolers/HiC_ncx_CN_all.mm10.mapq_30.2048.cool"

OUTPUT_DIR="/scratch1/smaruj/Akita_pytorch_training_data/mouse_data/Bonev2017_ncx_CN"

# MOUSE
python preprocessing_data_parallel.py \
  --cool_file "$COOL_FILE" \
  --fasta_file /project2/fudenber_735/genomes/mm10/mm10.fa \
  --bed_file /project2/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed \
  --gaps_file /project2/fudenber_735/backup/DNN_HiC/data_mm10/mm10.blacklist.rep.bed \
  --output_dir "$OUTPUT_DIR" \
  --num_workers 64

# HUMAN
# python preprocessing_data_parallel.py \
#   --cool_file "$COOL_FILE" \
#   --fasta_file /project2/fudenber_735/genomes/hg38/hg38.fa \
#   --bed_file /project2/fudenber_735/tensorflow_models/akita/v2/data/hg38/sequences.bed \
#   --gaps_file /project2/fudenber_735/backup/DNN_HiC/data_hg38/hg38.blacklist.rep.bed \
#   --output_dir "$OUTPUT_DIR" \
#   --num_workers 64
  