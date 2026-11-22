#!/usr/bin/env bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# UniCRS on CRSLab-converted ReDial (original + MS), same split/KG/item pool as KBRD/KGSF. Same protocol as the Reddit row:
# pre (5 ep) -> rec (5 ep), no conversation stage (no --use_resp), best-valid restore + test dump. Official UniCRS ReDial
# hyperparameters (README): pre lr 5e-4 warmup 1389; rec lr 1e-4 warmup 530. Seeds 0-2 sequential on one GPU.
set -uo pipefail
GPU=$1; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
S=/path/to/base; W=$S/00_Research/01_Recommender_Systems/WWW-2027; U=$S/projects/crs/baselines/unicrs
PY=$S/venvs/unicrs/bin/python; export HF_HOME=$S/.hf PIP_CACHE_DIR=$S/.cache/pip TORCH_HOME=$S/.cache/torch WANDB_DISABLED=true; unset HF_HUB_CACHE
cd $U/src
for DS in redialorig_cl redialms_cl; do
  (cd data/$DS && $PY process.py) || { echo "PROCESS_FAIL $DS"; continue; }
  for SEED in 0 1 2; do
    while :; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU); [ "$f" -gt 30000 ] && break; echo "wait gpu free=$f $(date)"; sleep 300; done
    PRE=$U/save/pre_${DS}_s$SEED; REC=$U/save/rec_${DS}_s$SEED; echo "START $DS s$SEED $(date)"
    $PY train_pre.py --dataset $DS --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base --text_encoder roberta-base \
      --num_train_epochs 5 --gradient_accumulation_steps 1 --per_device_train_batch_size 64 --per_device_eval_batch_size 128 --num_warmup_steps 1389 \
      --max_length 200 --prompt_max_length 200 --entity_max_length 32 --learning_rate 5e-4 --seed $SEED --output_dir $PRE || { echo "PRE_FAIL $DS s$SEED"; continue; }
    WWW27_DUMP=$W/results/dumps/unicrs_${DS}_s$SEED.dump $PY train_rec.py --dataset $DS --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small \
      --text_tokenizer roberta-base --text_encoder roberta-base --n_prefix_rec 10 --prompt_encoder $PRE/best --num_train_epochs 5 \
      --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --num_warmup_steps 530 --context_max_length 200 \
      --prompt_max_length 200 --entity_max_length 32 --learning_rate 1e-4 --seed $SEED --output_dir $REC || { echo "REC_FAIL $DS s$SEED"; continue; }
    echo "UNICRS_DONE $DS s$SEED $(date)"
  done
done
echo ALL_DONE
