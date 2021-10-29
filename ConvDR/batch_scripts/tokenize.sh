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

# Activate your environment
source activate convdr

# Tokenize CAST-19
python data/tokenizing.py  --collection=datasets/cast-shared/collection.tsv  --out_data_dir=/project/gpuuva006/team3/cast-tokenized-no_model_type/  --model_name_or_path=checkpoints/ad-hoc-ance-msmarco --model_type=rdot_nll
