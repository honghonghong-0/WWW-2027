#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# usage: run_reddit_seeds.sh <kbrd|kgsf> <seed> <gpu-index>   Reddit-Movie (our split, Wikidata KG), rec only, per-sample rank dumps
m=$1; s=$2; g=$3
P=/path/to/base/venvs/unicrs/bin/python; W=/path/to/base/www27
cd /path/to/base/projects/crs/baselines/crslab
D=$W/results/dumps/${m}_reddit_s${s}.jsonl; rm -f $D
OMP_NUM_THREADS=8 WWW27_SEED=$s WWW27_DUMP=$D $P run_crslab.py --config config/crs/$m/reddit_www27$( [ $s = 0 ] || echo _s$s ).yaml --save_system --gpu $g > $W/logs/${m}_reddit_s${s}.log 2>&1
echo "$m s$s exit $? $(date)" >> $W/logs/reddit_queue.log
