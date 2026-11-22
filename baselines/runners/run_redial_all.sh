#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# ReDial (rec-only) baseline, 3 seeds x 3 datasets. Inspired+TG sequential on GPU1, Reddit seeds parallel on GPU0 then auto-eval.
W=/path/to/base/www27; cd /path/to/base/projects/crs/baselines/crslab
P=/path/to/base/venvs/unicrs/bin/python
small() { for ds in inspired:insp tgredial:tg; do c=${ds%%:*}; t=${ds##*:}; for s in 0 1 2; do D=$W/results/dumps/redial_${t}_s$s.jsonl; rm -f $D
  OMP_NUM_THREADS=8 WWW27_SEED=$s WWW27_DUMP=$D $P run_crslab.py --config config/crs/redial/${c}_www27.yaml --gpu 1 > $W/logs/redial_${t}_s$s.log 2>&1; echo "redial $t s$s exit $? $(date)" >> $W/logs/redial_queue.log; done; done; }
reddit1() { s=$1; D=$W/results/dumps/redial_reddit_s$s.jsonl; rm -f $D
  OMP_NUM_THREADS=8 WWW27_SEED=$s WWW27_DUMP=$D $P run_crslab.py --config config/crs/redial/reddit_www27_s$s.yaml --save_system --gpu 0 > $W/logs/redial_reddit_s$s.log 2>&1; echo "redial reddit s$s exit $? $(date)" >> $W/logs/redial_queue.log
  for sp in valid test; do CRS_GPU=0 $P $REPO/baselines/reddit_crs_eval.py config/crs/redial/reddit_www27_s$s.yaml $sp redial_s$s > $W/logs/reddit_eval_redial_s${s}_$sp.log 2>&1; done; echo "EVAL redial s$s done $(date)" >> $W/logs/redial_queue.log; }
small & reddit1 0 & reddit1 1 & reddit1 2 & wait
