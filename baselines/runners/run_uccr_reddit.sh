#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# WWW27: UCCR Reddit seeds 0-2 in parallel on GPU1, each auto-scored on test when done
cd /path/to/base/projects/crs/external/UCCR_infer
for s in 0 1 2; do G=1; [ $s = 0 ] && G=0; (
 D=/path/to/base/www27/results/dumps/uccr_reddit_s$s.jsonl; rm -f $D $D.top
 OMP_NUM_THREADS=8 WWW27_TOPK=1 WWW27_SEED=$s WWW27_DUMP=$D /path/to/base/venvs/unicrs/bin/python run_crslab.py --config config/crs/uccr/reddit_www27_s$s.yaml --gpu $G > /path/to/base/www27/logs/uccr_reddit_s$s.log 2>&1
 echo "s$s exit $? $(date)" >> /path/to/base/www27/logs/uccr_reddit_queue.log
 /path/to/base/venvs/unicrs/bin/python $REPO/baselines/reddit_top_eval.py $D.top test uccr_reddit_s$s /path/to/base/projects/crs/external/UCCR_infer/data/dataset/reddit/nltk/WWW27_id_maps.json >> /path/to/base/www27/logs/uccr_reddit_queue.log 2>&1
) & sleep 20; done; wait
