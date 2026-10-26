#!/bin/bash
# 046: the harm-direction control per recipe on one group of cells (CPU work on one dev slot per
# part; the code is scripts/harm_046.py, the per-model instrument scripts/ladder_007.py unchanged).
#   shell/nchc/submit.sh --dev --gpus 1 --time 03:00:00 --name 046-harm-<part> --commit HEAD \
#       -- bash shell/mech/harm_046.sh <part-name> <cells> [--n-boot-sec N]
# <cells> is a comma list of cell ids (scripts/harm_046.py --list-cells). Parts land in
# /work/harrychang08/castor/analysis/046/part_<part-name>.json and are merged with
#   uv run --frozen python scripts/harm_046.py --merge $P/part_*.json --out $PWD/runs/046/harm_per_recipe.json
set -euo pipefail
NAME=$1
CELLS=$2
shift 2
P=/work/harrychang08/castor/analysis/046
mkdir -p $P
uv run --frozen python scripts/harm_046.py --manifest data/manifests/mech_021.json \
    --acts-root /work/harrychang08/castor/activations/021 --cells "$CELLS" \
    --out $P/part_${NAME}.json "$@"
echo HARM-OK
