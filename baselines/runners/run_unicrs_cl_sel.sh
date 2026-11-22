#!/usr/bin/env bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# UniCRS on a CRSLab-converted dataset (same split/KG as KBRD/KGSF). Official UniCRS Inspired hyperparameters (README):
# pre lr 6e-4 warmup 168 ; conv warmup 976 ; rec (inspired_gen) warmup 33. Usage: run_unicrs_cl.sh <dataset> <seed> <gpu-index>
# WWW27 2026-11-17: Inspired rerun. Checkpoints chosen by VALID recall@1+recall@50 (env WWW27_SEL=r1r50) instead of loss; pre 10 epochs, rec 15 epochs.
export WWW27_SEL=r1r50
set -euo pipefail
DS=$1; SEED=$2; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$3
ROOT=/path/to/base/projects/crs; U=$ROOT/baselines/unicrs; cd $U/src
PY=/path/to/base/venvs/unicrs/bin/python; export HF_HOME=/path/to/base/.hf; unset HF_HUB_CACHE
PRE=$U/save/pre_${DS}_sel_s$SEED; CONV=$U/save/conv_${DS}_sel_s$SEED; REC=$U/save/rec_${DS}_sel_s$SEED; GEN=${DS}_gen_sel_s$SEED
(cd data/$DS && $PY process.py)
$PY train_pre.py --dataset $DS --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base --text_encoder roberta-base \
  --num_train_epochs 10 --gradient_accumulation_steps 1 --per_device_train_batch_size 64 --per_device_eval_batch_size 128 --num_warmup_steps 168 \
  --max_length 200 --prompt_max_length 200 --entity_max_length 32 --learning_rate 6e-4 --seed $SEED --output_dir $PRE
(cd data/$DS && $PY process_mask.py)
$PY train_conv.py --dataset $DS --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base --text_encoder roberta-base \
  --n_prefix_conv 20 --prompt_encoder $PRE/best --num_train_epochs 10 --gradient_accumulation_steps 1 --ignore_pad_token_for_loss \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --num_warmup_steps 976 --context_max_length 200 --resp_max_length 183 \
  --prompt_max_length 200 --entity_max_length 32 --learning_rate 1e-4 --seed $SEED --output_dir $CONV
for split in train valid test; do
  $PY infer_conv.py --dataset $DS --split $split --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base \
    --text_encoder roberta-base --n_prefix_conv 20 --prompt_encoder $CONV/best --output_dir $CONV/infer_$split --per_device_eval_batch_size 64 \
    --context_max_length 200 --resp_max_length 183 --prompt_max_length 200 --entity_max_length 32
done
mkdir -p data/$GEN; cp data/$DS/dbpedia_subkg.json data/$DS/entity2id.json data/$DS/item_ids.json data/$DS/relation2id.json data/$GEN/
$PY $REPO/baselines/unicrs_merge_gen.py --raw-dir data/$DS --generated-dir save/$DS --prefix conv_${DS}_sel_s$SEED --output-dir data/$GEN
$PY -c "import sys; sys.path.insert(0, '$ROOT/scripts'); from pathlib import Path; from unicrs_merge_gen import merge_split; merge_split(Path('data/$DS/test_data_processed.jsonl'), Path('save/$DS/conv_${DS}_sel_s${SEED}_test.jsonl'), Path('data/$GEN/test_data_processed.jsonl'))"
$PY train_rec.py --dataset $GEN --tokenizer microsoft/DialoGPT-small --model microsoft/DialoGPT-small --text_tokenizer roberta-base --text_encoder roberta-base \
  --n_prefix_rec 10 --prompt_encoder $PRE/best --num_train_epochs 15 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 1 --num_warmup_steps 33 --context_max_length 200 --prompt_max_length 200 --entity_max_length 32 \
  --learning_rate 1e-4 --seed $SEED --use_resp --output_dir $REC
(cd $U/src && $PY unicrs_eval_cl.py inspired $DS $GEN $REC insp_sel_s$SEED)
echo "UNICRS_SEL_DONE $DS s$SEED"
