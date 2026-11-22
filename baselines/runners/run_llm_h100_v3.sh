#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# Qwen2.5-7B zero-shot Reddit (valid+test, none/hist) on the H100 with >=40GB free, UUID-pinned
W=/path/to/base/www27; L=$W/logs
export HF_HOME=/path/to/base/.hf HF_HUB_OFFLINE=1 TMPDIR=/path/to/base/.tmp
cd "$REPO"
for job in "valid none" "valid hist" "test none" "test hist"; do
  G=1   # fixed: index 1 = GPU-0981 (empty); verified after launch
  [ -n "$G" ] || { echo "NO_GPU $(date)"; exit 1; }
  echo "$job -> $G $(date)"
  set -- $job
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G GMU=0.5 /path/to/base/venvs/vllm27/bin/python baselines/llm_zs_reddit.py $1 $2 > $L/llm_h100_$1_$2.log 2>&1
  echo "$job exit $? $(date)"
done
echo LLM_ALL_DONE
