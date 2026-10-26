#!/bin/bash
# 021: 007's extraction (scripts/extract_007.py, unchanged) for a group of new twins,
# serially on one GPU (<= 4 jobs of 021 queued at once). Each twin writes its own directory
# under /work/harrychang08/castor/activations/021/; an existing scores.json with a panel of 6,226
# is kept (skip-if-exists keyed on the extraction's own provenance, never on the path alone).
#   shell/nchc/submit.sh --dev --gpus 1 --time 03:30:00 --name 021-extract-g1 -- \
#       bash shell/mech/extract_group_021.sh <model-id> ...
set -euo pipefail
W=/work/harrychang08/castor/activations/021
M=data/manifests/mech_021.json
done_ok() { [ -f "$1/scores.json" ] && uv run --frozen python -c \
    "import json,sys; d=json.load(open('$1/scores.json')); sys.exit(0 if d.get('panel',{}).get('n')==6226 and d.get('git_commit') else 1)"; }
for ID in "$@"; do
    if done_ok $W/$ID; then echo "skip $ID (complete)"; continue; fi
    uv run --frozen python scripts/extract_007.py --models $M --model-id $ID --out $W/$ID
done
echo EXTRACT-OK
