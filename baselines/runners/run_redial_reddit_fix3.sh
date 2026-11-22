#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# ReDial on Reddit-Movie, fixed output, max 3 rec epochs (impatience 1), no per-sample dump during training, seed 0, then eval. Deadline run.
cd /path/to/base/projects/crs/baselines/crslab
OMP_NUM_THREADS=32 WWW27_SEED=0 CUDA_VISIBLE_DEVICES=0 /path/to/base/venvs/unicrs/bin/python run_crslab.py --config config/crs/redial/reddit_www27_fix3_s0.yaml --save_system --gpu 0 > /path/to/base/www27/logs/redialfix3_reddit_s0.log 2>&1
echo "train exit $? $(date)" >> /path/to/base/www27/logs/redialfix3_queue.log
CRS_GPU=0 /path/to/base/venvs/unicrs/bin/python $REPO/baselines/reddit_crs_eval.py config/crs/redial/reddit_www27_fix3_s0.yaml test redialfix3_s0 > /path/to/base/www27/logs/reddit_eval_redialfix3_s0_test.log 2>&1
echo "EVAL done $(date)" >> /path/to/base/www27/logs/redialfix3_queue.log
