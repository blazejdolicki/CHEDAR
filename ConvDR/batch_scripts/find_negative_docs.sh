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

# run inference and find negative documents
python drivers/run_convdr_inference.py  --model_path=checkpoints/ad-hoc-ance-msmarco  --eval_file=datasets/cast-19/eval_topics.jsonl  --query=target  --per_gpu_eval_batch_size=8  --ann_data_dir=datasets/cast-19/embeddings  --qrels=datasets/cast-19/qrels.tsv  --processed_data_dir=datasets/cast-19/tokenized  --raw_data_dir=datasets/cast-19   --output_file=results/cast-19/manual_ance.jsonl  --output_trec_file=results/cast-19/manual_ance.trec  --model_type=rdot_nll  --output_query_type=manual  --use_gpu