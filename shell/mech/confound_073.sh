#!/bin/bash
# 073: within-last-token centring and the refusal-logit control at (13, -1), one set per job
# (scripts/confound_073.py; the GPU only runs the logit lens on stored activations).
#   shell/nchc/submit.sh --dev --gpus 1 --time 02:00:00 --name 073-<set> --commit HEAD \
#       -- bash shell/mech/confound_073.sh <A0|A> [--n-boot N]
#   ... -- bash shell/mech/confound_073.sh smoke     # lens check + a small A part
# Parts land in /work/harrychang08/castor/analysis/073/ and are merged with
#   uv run --frozen python scripts/confound_073.py --merge $P/part_A0.json $P/part_A.json --out $PWD/runs/073/confound.json
set -euo pipefail
P=/work/harrychang08/castor/analysis/073
mkdir -p $P
if [ "$1" = smoke ]; then
    uv run --frozen python scripts/confound_073.py --check-lens dd-dclm-baseline-1b-default-dpo-seed1 \
        --out $P/smoke_lens.json
    uv run --frozen python scripts/confound_073.py --set A --max-models 6 --n-boot 3 \
        --panel-every 4 --out $P/smoke_A.json
else
    SET=$1
    shift
    if [ "${1:-}" = clusters ]; then  # declared addition: bootstrap over last-token groups
        shift
        uv run --frozen python scripts/confound_073.py --set "$SET" --boot clusters \
            --out $P/part_${SET}_clusters.json "$@"
    else
        uv run --frozen python scripts/confound_073.py --set "$SET" --out $P/part_${SET}.json "$@"
    fi
fi
echo CONFOUND-OK
