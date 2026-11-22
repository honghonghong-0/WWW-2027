#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# usage: run_tg_seeds.sh <model> <seed> <gpu>   (TG-ReDial, UCCR user split; per-sample rank dumps for offline best-valid selection)
m=$1; s=$2; g=$3
P=/path/to/base/venvs/unicrs/bin/python
if [ $m = uccr ]; then cd /path/to/base/projects/crs/external/UCCR_infer; C=config/crs/uccr/tgredial_www27_recOnly.yaml; else cd /path/to/base/projects/crs/baselines/crslab; C=config/crs/$m/tgredial_www27.yaml; fi
D=/path/to/base/www27/results/dumps/${m}_tg_s${s}.jsonl; rm -f $D
CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 WWW27_SEED=$s WWW27_DUMP=$D $P run_crslab.py --config $C --gpu 0 > /path/to/base/www27/logs/${m}_tg_s${s}_v3.log 2>&1
