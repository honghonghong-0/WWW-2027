#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# usage: run_insp_seeds.sh <kbrd|kgsf> <seed>   Inspired (CRSLab split), rec only, per-sample rank dumps, UUID-pinned H100
m=$1; s=$2
P=/path/to/base/venvs/unicrs/bin/python; W=/path/to/base/www27
cd /path/to/base/projects/crs/baselines/crslab
D=$W/results/dumps/${m}_insp_s${s}.jsonl; rm -f $D
CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 WWW27_SEED=$s WWW27_DUMP=$D $P run_crslab.py --config config/crs/$m/inspired_www27.yaml --gpu 0 > $W/logs/${m}_insp_s${s}.log 2>&1
echo "$m s$s exit $? $(date)" >> $W/logs/insp_queue.log
