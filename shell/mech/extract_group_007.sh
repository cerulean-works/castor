#!/bin/bash
# 007: extraction for a group of twins, serially on one GPU (≤ 4 jobs of 007 queued at once).
# Each twin writes its own directory; an existing scores.json with a panel of 6,226 is kept.
#   shell/nchc/submit.sh --dev --gpus 1 --time 02:00:00 --name 007-extract-g1 -- \
#       bash shell/mech/extract_group_007.sh [--fp32 <model-id>] <model-id> ...
set -euo pipefail
W=/work/harrychang08/castor/activations/007
M=data/manifests/mech_007.json
FP32=""
[ "${1:-}" = --fp32 ] && { FP32="$2"; shift 2; }
done_ok() { [ -f "$1/scores.json" ] && uv run --frozen python -c \
    "import json,sys; d=json.load(open('$1/scores.json')); sys.exit(0 if d.get('panel',{}).get('n')==6226 and d.get('git_commit') else 1)"; }
for ID in "$@"; do
    if done_ok $W/$ID; then echo "skip $ID (complete)"; continue; fi
    uv run --frozen python scripts/extract_007.py --models $M --model-id $ID --out $W/$ID
done
if [ -n "$FP32" ]; then
    if done_ok $W-fp32/$FP32; then echo "skip fp32 $FP32"; else
    uv run --frozen python scripts/extract_007.py --models $M --model-id $FP32 --dtype float32 \
        --out $W-fp32/$FP32; fi
fi
echo EXTRACT-OK
