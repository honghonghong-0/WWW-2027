#!/usr/bin/env bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# UniCRS on Reddit-Movie (same CRSLab split + same Wikidata KG as KBRD/KGSF), WITHOUT the conversation stage:
# recommender turns are a fixed template ("here are some recommendations"), so --use_resp would add no information.
# pre (official Inspired hparams, 5 ep) -> rec (5 ep, prompt encoder from pre/best, no --use_resp); best-valid restore + test dump.
# usage: run_unicrs_reddit.sh <seed> [debug]
set -euo pipefail
SEED=$1; DBG=${2:-}; W=/path/to/base/www27
ROOT=/path/to/base/projects/crs; U=$ROOT/baselines/unicrs; cd $U/src; DS=reddit_wd
PY=/path/to/base/venvs/unicrs/bin/python; export HF_HOME=/path/to/base/.hf; unset HF_HUB_CACHE
while true; do G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>10500{print $1; exit}'); [ -n "$G" ] && break; sleep 120; done
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G; echo "gpu $G $(date)"
TAG=${DBG:+PILOT_}s$SEED; PRE=$U/save/pre_${DS}_$TAG; REC=$U/save/rec_${DS}_$TAG; EP=5; D=""
[ -n "$DBG" ] && { EP=1; D="--debug"; }
[ -f data/$DS/test_data_processed.jsonl ] || (cd data/$DS && $PY process.py)
$PY train_pre.py --dataset $DS --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base --text_encoder roberta-base \
  --num_train_epochs $EP --gradient_accumulation_steps 1 --per_device_train_batch_size 64 --per_device_eval_batch_size 128 --num_warmup_steps 168 \
  --max_length 200 --prompt_max_length 200 --entity_max_length 32 --learning_rate 6e-4 --seed $SEED --output_dir $PRE $D
WWW27_DUMP=$W/results/dumps/unicrs_reddit_$TAG.dump $PY train_rec.py --dataset $DS --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small \
  --text_tokenizer roberta-base --text_encoder roberta-base --n_prefix_rec 10 --prompt_encoder $PRE/best --num_train_epochs $EP \
  --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --num_warmup_steps 33 --context_max_length 200 \
  --prompt_max_length 200 --entity_max_length 32 --learning_rate 1e-4 --seed $SEED --output_dir $REC $D
$PY $REPO/baselines/unicrs_dump2top.py $W/results/dumps/unicrs_reddit_$TAG.dump $W/results/dumps/unicrs_reddit_$TAG.top
$PY $REPO/baselines/reddit_top_eval.py $W/results/dumps/unicrs_reddit_$TAG.top test unicrs_reddit_$TAG
echo "UNICRS_REDDIT_DONE $TAG $(date)"
