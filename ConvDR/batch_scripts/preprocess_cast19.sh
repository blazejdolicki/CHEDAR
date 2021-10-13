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
module load 2019
module load Python/3.7.5-foss-2019b
module load Python/3.7.5-foss-2019b
module load CUDA/10.1.243
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module load Anaconda3/2018.12

# Your job starts in the directory where you call sbatch
cd $HOME/CHEDAR/ConvDR

export PYTHONPATH=${PYTHONPATH}:`pwd`
pwd

# Activate your environment
source activate convdr
#conda list

# Preprocess CAST-19
python data/preprocess_cast19.py  --car_cbor=datasets/raw/dedup.articles-paragraphs.cbor  --msmarco_collection=datasets/raw/msmarco.tsv  --duplicate_file=datasets/raw/duplicate_list_v1.0.txt  --cast_dir=datasets/raw/cast-19/  --out_data_dir=datasets/cast-19  --out_collection_dir=datasets/cast-shared
