#!/bin/bash
# 007: ladder, link and secondaries on the 12 twins (CPU work; one dev slot per block).
# selection.json and selection_filtered.json must be committed first (the job reads them from
# its snapshot). One block per job (main | sens | layers | sites), each under dev's 4 h; then
#   uv run --frozen python scripts/ladder_007.py --models x --acts-root x --selection x \
#       --out $PWD/runs/007/ladder.json --merge $P/ladder_{main,sens,layers,sites}.json
#   shell/nchc/submit.sh --dev --gpus 1 --time 04:00:00 --name 007-ladder-<b> -- \
#       bash shell/mech/ladder_007.sh <b>
set -euo pipefail
B=$1
W=/work/harrychang08/castor/activations/007
P=/work/harrychang08/castor/analysis/007
M=data/manifests/mech_007.json
mkdir -p $P
uv run --frozen python scripts/ladder_007.py --models $M --acts-root $W \
    --selection runs/007/selection.json --selection-filtered runs/007/selection_filtered.json \
    --fp32 $W-fp32/dd-dclm-1b-default-dpo-seed1 --out $P/ladder_$B.json --blocks $B
echo LADDER-OK
