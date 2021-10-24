#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=chedar_trec_results
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=00:15:00
#SBATCH --mem=10000M
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
cd $HOME/trec_eval

./trec_eval -m ndcg_cut.3 -m recip_rank -q ../CHEDAR/ConvDR/datasets/cast-19/qrels.tsv ../CHEDAR/ConvDR/results/cast-19/kd_chedar.trec > ../CHEDAR/ConvDR/results/cast-19/kd_chedar_results.txt
