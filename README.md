# CHEDAR

This repository contains the code for the paper CHEDAR: Conversation History Embedding for Dense Answer Retrieval.
Our research is built on top of the [CONVDR paper](https://arxiv.org/pdf/2105.04166.pdf) and therefore we use their [respository](https://github.com/thunlp/ConvDR) as base for our approach.
We use the instructions provided by them to attempt to reproduce their results and we provide below their steps along with our modifications. (Some of their commands didn't match the code or missed steps)


*Commands to CHEDAR are at the end of the README.

## ConvDR

This repo contains code and data for SIGIR 2021 paper ["Few-Shot Conversational Dense Retrieval"](https://arxiv.org/pdf/2105.04166.pdf).

## Prerequisites

Install dependencies:

```bash
git clone https://github.com/thunlp/ConvDR.git
cd ConvDR
pip install -r requirements.txt
```

We recommend set `PYTHONPATH` before running the code:

```bash
export PYTHONPATH=${PYTHONPATH}:`pwd`
```

To train ConvDR, we need trained ad hoc dense retrievers. We use [ANCE](https://github.com/microsoft/ANCE) for both tasks. Please downloads those checkpoints here: [TREC CAsT](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Passage_ANCE_FirstP_Checkpoint.zip) and [OR-QuAC](https://data.thunlp.org/convdr/ad-hoc-ance-orquac.cp). For TREC CAsT, we directly use the official model trained on MS MARCO Passage Retrieval task. For OR-QuAC, we initialize the retriever from the official model trained on NQ and TriviaQA, and continue training on OR-QuAC with manually reformulated questions using the ANCE codebase.

The following code downloads those checkpoints and store them in `./checkpoints`.

```bash
mkdir checkpoints
wget https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Passage_ANCE_FirstP_Checkpoint.zip
wget https://data.thunlp.org/convdr/ad-hoc-ance-orquac.cp
unzip Passage_ANCE_FirstP_Checkpoint.zip
mv "Passage ANCE(FirstP) Checkpoint" ad-hoc-ance-msmarco
```

## Data Preparation

By default, we expect raw data to be stored in `./datasets/raw` and processed data to be stored in `./datasets`:

```bash
mkdir datasets
mkdir datasets/raw
```

### TREC CAsT

#### CAsT shared files download

Use the following commands to download the document collection for CAsT-19 & CAsT-20 as well as the MARCO duplicate file:

```bash
cd datasets/raw
wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz 
tar -xf collection.tar.gz
mv collection.tsv msmarco.tsv
wget http://trec-car.cs.unh.edu/datareleases/v2.0/paragraphCorpus.v2.0.tar.xz
tar -xf paragraphCorpus.v2.0.tar.xz
wget http://boston.lti.cs.cmu.edu/Services/treccast19/duplicate_list_v1.0.txt
cd ../../
```

#### CAsT-19 files download

Download necessary files for CAsT-19 and store them into `./datasets/raw/cast-19`:

```bash
mkdir datasets/raw/cast-19
cd datasets/raw/cast-19
wget https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_v1.0.json
wget https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_annotated_resolved_v1.0.tsv
wget https://trec.nist.gov/data/cast/2019qrels.txt
cd ../../
```

#### CAsT preprocessing

Use the scripts `./data/preprocess_cast19`  to preprocess raw CAsT files:

```bash
mkdir datasets/cast-19
mkdir datasets/cast-shared
python data/preprocess_cast19.py  --car_cbor=datasets/raw/dedup.articles-paragraphs.cbor  --msmarco_collection=datasets/raw/msmarco.tsv  --duplicate_file=datasets/raw/duplicate_list_v1.0.txt  --cast_dir=datasets/raw/cast-19/  --out_data_dir=datasets/cast-19  --out_collection_dir=datasets/cast-shared
```

### Generate Document Embeddings

Our code is based on ANCE and we have a similar embedding inference pipeline, where the documents are first tokenized and converted to token ids and then the token ids are used for embedding inference. We create sub-directories `tokenized` and `embeddings` inside `./datasets/cast-shared` and `./datasets/or-quac` to store the tokenized documents and document embeddings, respectively:

```bash
mkdir datasets/cast-shared/tokenized
mkdir datasets/cast-shared/embeddings
```

Run `./data/tokenizing.py` to tokenize documents in parallel:

```bash
# CAsT
python data/tokenizing.py  --collection=datasets/cast-shared/collection.tsv  --out_data_dir=datasets/cast-shared/tokenized  --model_name_or_path=checkpoints/ad-hoc-ance-msmarco --model_type=rdot_nll
```

After tokenization, run `./drivers/gen_passage_embeddings.py` to generate document embeddings:

```bash
# CAsT
python -m torch.distributed.launch --nproc_per_node=$gpu_no python drivers/gen_passage_embeddings.py  --data_dir=datasets/cast-shared/tokenized  --checkpoint=checkpoints/ad-hoc-ance-msmarco  --output_dir=datasets/cast-shared/embeddings  --model_type=rdot_nll
```

Note that we follow the ANCE implementation and this step takes up a lot of memory. To generate all 38M CAsT document embeddings safely, the machine should have at least 200GB memory. It's possible to save memory by generating a part at a time, and we may update the implementation in the future.

## ConvDR Training

Now we are all prepared: we have downloaded & preprocessed data, and we have obtained document embeddings. Simply run `./drivers/run_convdr_train.py` to train a ConvDR using KD (MSE) loss (batch script: `train_basic.sh`):

```bash
# CAsT-19, KD loss only, five-fold cross-validation
python drivers/run_convdr_train.py  --output_dir=checkpoints/convdr-kd-cast19  --model_name_or_path=checkpoints/ad-hoc-ance-msmarco  --train_file=datasets/cast-19/eval_topics.jsonl  --query=no_res  --per_gpu_train_batch_size=4  --learning_rate=1e-5   --log_dir=logs/convdr_kd_cast19  --num_train_epochs=8  --model_type=rdot_nll  --cross_validate
```

To use ranking loss, we need to find negative documents for each query. We use top retrieved negatives documents from the ranking results of **manual** queries. So we need to first perform retrieval using the manual queries (batch script: `find_negative_docs.sh`):

```bash
# CAsT-19
mkdir results/cast-19
python drivers/run_convdr_inference.py  --model_path=checkpoints/ad-hoc-ance-msmarco  --eval_file=datasets/cast-19/eval_topics.jsonl  --query=target  --per_gpu_eval_batch_size=8  --ann_data_dir=datasets/cast-19/embeddings  --qrels=datasets/cast-19/qrels.tsv  --processed_data_dir=datasets/cast-19/tokenized  --raw_data_dir=datasets/cast-19   --output_file=results/cast-19/manual_ance.jsonl  --output_trec_file=results/cast-19/manual_ance.trec  --model_type=rdot_nll  --output_query_type=manual  --use_gpu
```

After the retrieval finishes, we can select negative documents from manual runs and supplement the original training files with them (batch script: `select_neg_docs.sh`):

```bash
# CAsT-19
python data/gen_ranking_data.py  --train=datasets/cast-19/eval_topics.jsonl  --run=results/cast-19/manual_ance.trec  --output=datasets/cast-19/eval_topics.rank.jsonl  --qrels=datasets/cast-19/qrels.tsv  --collection=datasets/cast-shared/collection.tsv  --cast
```

Now we are able to use the ranking loss, with the `--ranking_task` flag on (batch script: `train_ranking.sh`):

```bash
# CAsT-19, Multi-task
python drivers/run_convdr_train.py  --output_dir=checkpoints/convdr-multi-cast19  --model_name_or_path=checkpoints/ad-hoc-ance-msmarco  --train_file=datasets/cast-19/eval_topics.rank.jsonl  --query=no_res  --per_gpu_train_batch_size=4  --learning_rate=1e-5   --log_dir=logs/convdr_multi_cast19  --num_train_epochs=8  --model_type=rdot_nll  --cross_validate  --ranking_task
```

To disable the KD loss, simply set the `--no_mse` flag.

## ConvDR Inference

Run `./drivers/run_convdr_inference.py` to get inference results. `output_file` is the [OpenMatch](https://github.com/thunlp/OpenMatch)-format file for reranking, and `output_trec_file` is the TREC-style run file which can be evaluated by the [trec_eval](https://github.com/usnistgov/trec_eval) tool (batch script: `inference_results.sh`).

```bash
# CAsT-19
python drivers/run_convdr_inference.py  --model_path=checkpoints/convdr-kd-cast19  --eval_file=datasets/cast-19/eval_topics.jsonl  --query=no_res  --per_gpu_eval_batch_size=8  --cache_dir=../ann_cache_dir  --ann_data_dir=datasets/cast-19/embeddings  --qrels=datasets/cast-19/qrels.tsv  --processed_data_dir=datasets/cast-19/tokenized  --raw_data_dir=datasets/cast-19   --output_file=results/cast-19/kd.jsonl  --output_trec_file=results/cast-19/kd.trec  --model_type=rdot_nll  --output_query_type=raw  --use_gpu  --cross_validate
```

The query embedding inference always takes the first GPU. If you set the `--use_gpu` flag (recommended), the retrieval will be performed on the remaining GPUs. The retrieval process consumes a lot of GPU resources. To reduce the resource usage, we split all document embeddings into several blocks, perform searching one-by-one and finally combine the results. If you have enough GPU resources, you can modify the code to perform searching all at once.

### Set up evaluation with `trec_eval`
Follow this steps to obtain evaluation metrics from the TREC-style file.
```
# git clone the aforementioned repository
git clone https://github.com/usnistgov/trec_eval.git
# go to the root dir
cd trec_eval
# set up the repo
make
```
### Run evaluation
Run the following command (batch script: `trec_results.sh`)
```
./trec_eval -m ndcg_cut.3 -m recip_rank ../CHEDAR/ConvDR/datasets/cast-19/qrels.tsv ../CHEDAR/ConvDR/results/cast-19/multi.trec > ../CHEDAR/ConvDR/results/cast-19/multi_results.txt
```

## Download Trained Models

Three trained models can be downloaded with the following link: [CAsT19-KD-CV-Fold1](https://data.thunlp.org/convdr/convdr-kd-cast19-1.zip)

## Results

[Download ConvDR and baseline runs on CAsT](https://drive.google.com/file/d/1F0RwA9sZscUAyE0IyQ7PMrgzNVqDnho5/view?usp=sharing)

## Contact

Please send email to ~~yus17@mails.tsinghua.edu.cn~~ yushi17@foxmail.com.

## CHEDAR 

### LISA

In order to replicate the CHEDAR results, we have provided a list of batch scripts that run the training, inference and compute the trec results in the lisa cluster. These scripts can be found in `CHEDAR/ConvDR/batch_scripts/all_models/`. 
Here each batch script has the 3 steps of the pipeline and each of them correspond to a different model from the following list:

- m1: 2 linear layers + Relu
- m2:  2 linear layers + Sigmoid
- m3: 3 linear layers + ReLU
- m4: 2 linear layers + Relu + Residual (max)
- m5: GRU - not yet
- m6:  2 linear layers + Relu + Dropout
- m7:  3 (linear layers+residual) structure:  linear layer + Relu + Residual (max) + linear + relu+ linear + residual
- m8: LSTM - not yet
- m9:  2 linear layers + Relu + Residual (addition)
- m10:   3 linear layers + Relu + Residual (max)

Note: modify the path of the arguments in the batch scripts to match the files that were generated during the CONVDR steps. 

### Scripts Structure

#### Train
To train a chedar model you can run the `run_chedar_train.py` script with the arguments defined below. 
The CHEDAR variation can be selected using the argument `--history_encoder_type=1` where the number corresponds to the model described in the list above.
Make sure that the files in the paths of arguments `model_name_or_path` and `train_file` match your directory.
'''
python drivers/run_chedar_train.py  --output_dir=checkpoints/chedar-kd-cast19-m1  \
                                    --model_name_or_path=checkpoints/ad-hoc-ance-msmarco  \
                                    --train_file=datasets/cast-19/SORTED_eval_topics.jsonl  \
                                    --query=no_res  \
                                    --per_gpu_train_batch_size=1  \
                                    --learning_rate=1e-5 \
                                    --log_dir=logs/chedar-kd-cast19-m1   \
                                    --num_train_epochs=8  \
                                    --model_type=rdot_nll  \
                                    --cross_validate  \
                                    --overwrite_output_dir \
                                    --history_encoder_type 1

'''

#### Inference  
Similarly as in training the model variation can be defined with the argument `--history_encoder_type=1` and the files at paths from the arguments should match your directory
```
python drivers/run_chedar_inference.py  --model_path=checkpoints/chedar-kd-cast19-m1 \
                                        --eval_file=datasets/cast-19/SORTED_eval_topics.jsonl \
                                        --query=no_res \
                                        --per_gpu_eval_batch_size=1 \
                                        --cache_dir=../ann_cache_dir \
                                        --ann_data_dir=/project/gpuuva006/CAST19_ANCE_embeddings \
                                        --qrels=datasets/cast-19/qrels.tsv \
                                        --processed_data_dir=/project/gpuuva006/team3/cast-tokenized/ \
                                        --raw_data_dir=datasets/cast-19 \
                                        --output_file=results/cast-19/kd_chedar-train_folds-m1.jsonl \
                                        --output_trec_file=results/cast-19/kd_chedar-train_folds-m1.trec \
                                        --model_type=rdot_nll \
                                        --output_query_type=raw \
                                        --cross_validate \
                                        --use_gpu \
                                        --evaluate_on_train_folds  \
                                        --history_encoder_type 1
```

#### Results

If you followed the steps of CONVDR, you should now have a clone of the trec_eval repository, from there you can call the follow command to compute the proper NDCG@3 and MRR.
```
./trec_eval -m ndcg_cut.3 -m recip_rank ../CHEDAR/ConvDR/datasets/cast-19/qrels.tsv ../CHEDAR/ConvDR/results/cast-19/kd_chedar-train_folds-m1.trec > ../CHEDAR/ConvDR/results/cast-19/kd_chedar-train_folds-m1.txt
```
