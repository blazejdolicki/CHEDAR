#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=chedar_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=10:00:00
#SBATCH --mem=60000M
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
NAME="checkpoints/chedar-kd-cast19-m6"
python drivers/run_chedar_train.py  --output_dir=$NAME  \
                                    --model_name_or_path=checkpoints/ad-hoc-ance-msmarco  \
                                    --train_file=datasets/cast-19/SORTED_eval_topics.jsonl  \
                                    --query=no_res  \
                                    --per_gpu_train_batch_size=1  \
                                    --learning_rate=1e-5 \
                                    --log_dir=logs/chedar-kd-cast19-m6   \
                                    --num_train_epochs=8  \
                                    --model_type=rdot_nll  \
                                    --cross_validate  \
                                    --overwrite_output_dir \
                                    --history_encoder_type 6


echo "Running Inference Step"

python drivers/run_chedar_inference.py  --model_path=checkpoints/chedar-kd-cast19-m6 \
                                        --eval_file=datasets/cast-19/SORTED_eval_topics.jsonl \
                                        --query=no_res \
                                        --per_gpu_eval_batch_size=1 \
                                        --cache_dir=../ann_cache_dir \
                                        --ann_data_dir=/project/gpuuva006/CAST19_ANCE_embeddings \
                                        --qrels=datasets/cast-19/qrels.tsv \
                                        --processed_data_dir=/project/gpuuva006/team3/cast-tokenized/ \
                                        --raw_data_dir=datasets/cast-19 \
                                        --output_file=results/cast-19/kd_chedar-train_folds-m6.jsonl \
                                        --output_trec_file=results/cast-19/kd_chedar-train_folds-m6.trec \
                                        --model_type=rdot_nll \
                                        --output_query_type=raw \
                                        --cross_validate \
                                        --use_gpu \
                                        --evaluate_on_train_folds  \
                                        --history_encoder_type 6

cd $HOME/trec_eval

./trec_eval -m ndcg_cut.3 -m recip_rank ../CHEDAR/ConvDR/datasets/cast-19/qrels.tsv ../CHEDAR/ConvDR/results/cast-19/kd_chedar-train_folds-m6.trec > ../CHEDAR/ConvDR/results/cast-19/kd_chedar-train_folds-m6.txt
