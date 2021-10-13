#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=convdr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=10:00:00
#SBATCH --mem=32000M
#SBATCH --output=job_logs/slurm_output_%A.out

module purge
module load 2020
module load CUDA

# Your job starts in the directory where you call sbatch
cd $HOME/CHEDAR/ConvDR
source activate convdr 

export PYTHONPATH=${PYTHONPATH}:`pwd`

# Activate your environment
#source activate convdr

conda list
# Preprocess CAST-19
python data/preprocess_cast19.py  --car_cbor=datasets/raw/dedup.articles-paragraphs.cbor  --msmarcdatasets/raw/msmarco.tsv  --duplicate_file=datasets/raw/duplicate_list_v1.0.txt  --cast_dir=datasets/raw/cast-19/  --out_data_dir=datasets/cast-lection_dir=datasets/cast-shared
