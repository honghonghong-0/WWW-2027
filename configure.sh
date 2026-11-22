#!/usr/bin/env bash
# Replace the path placeholder in all scripts with your own base directory.
#   bash configure.sh /abs/path/to/workspace
# Layout expected under the base directory (created by the data and baseline steps in README.md):
#   <base>/www27/{data,results,logs}     outputs of this repository
#   <base>/datasets/tag_genome_2021/     Tag Genome 2021 release
#   <base>/projects/crs/        CRSLab / UniCRS / UCCR checkouts and their processed data
#   <base>/venvs/<name>/                 Python environments
set -euo pipefail
BASE="${1:?usage: bash configure.sh /abs/path/to/workspace}"
cd "$(dirname "$0")"
grep -rl --exclude-dir=.git "/path/to/base" . | grep -v configure.sh | while read -r f; do
  sed -i.bak "s|/path/to/base|${BASE%/}|g" "$f" && rm -f "$f.bak"
done
mkdir -p "${BASE%/}/www27/results/dumps" "${BASE%/}/www27/logs" "${BASE%/}/www27/data"
echo "configured with base = ${BASE%/}"
