#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# KGSF on Reddit-Movie: fix divergence (grad clip 0.1) + lr in {1e-3,3e-4,1e-4}, seed 0; select lr by OUR valid NDCG@10 (reddit_crs_eval). 2026-11-17
W=/path/to/base/www27; cd /path/to/base/projects/crs/baselines/crslab; P=/path/to/base/venvs/unicrs/bin/python
one() { lr=$1; D=$W/results/dumps/kgsftune_lr${lr}_reddit_s0.jsonl; rm -f $D
  OMP_NUM_THREADS=8 WWW27_SEED=0 WWW27_DUMP=$D $P run_crslab.py --config config/crs/kgsf/reddit_www27_tune_lr$lr.yaml --save_system --gpu 1 > $W/logs/kgsftune_lr${lr}.log 2>&1; echo "kgsftune $lr exit $? $(date)" >> $W/logs/kgsftune_queue.log
  for sp in valid test; do CRS_GPU=1 $P $REPO/baselines/reddit_crs_eval.py config/crs/kgsf/reddit_www27_tune_lr$lr.yaml $sp kgsftune_lr${lr}_s0 > $W/logs/reddit_eval_kgsftune_lr${lr}_$sp.log 2>&1; done; echo "EVAL kgsftune $lr done $(date)" >> $W/logs/kgsftune_queue.log; }
one 1e-3 & one 3e-4 & one 1e-4 & wait
