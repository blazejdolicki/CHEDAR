#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=convdr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=10:00:00
#SBATCH --mem=16000M
#SBATCH --output=job_logs/slurm_out.out

module purge
module load 2020
module load Python
module load CUDA/11.0.3-GCC-9.3.0
module load Miniconda3

# Your job starts in the directory where you call sbatch
cd $HOME/CHEDAR/ConvDR

export PYTHONPATH=${PYTHONPATH}:`pwd`
pwd

# Activate your environment
source activate convdr
conda info
conda list
