#!/bin/bash
# 021: the two-site ladder / link on one pair set (CPU work on one dev slot per job; the
# code is scripts/ladder_021.py, the per-model instrument scripts/ladder_007.py unchanged).
#   shell/nchc/submit.sh --dev --gpus 1 --time 04:00:00 --name 021-ladder-<set>-<blocks> \
#       --commit HEAD -- bash shell/mech/ladder_021.sh <set> <blocks>
# <set> is a key of data/manifests/mech_021.json `sets` (A B A0 F S300M S150M); <blocks> is
# main, sites or main,sites. Parts land in /work/harrychang08/castor/analysis/021/ and are merged
# into runs/021/ladder.json with
#   uv run --frozen python scripts/ladder_021.py --merge $P/ladder_*.json --out $PWD/runs/021/ladder.json
set -euo pipefail
S=$1
B=${2:-main,sites}
W=/work/harrychang08/castor/activations/021
P=/work/harrychang08/castor/analysis/021
M=data/manifests/mech_021.json
mkdir -p $P
uv run --frozen python scripts/ladder_021.py --manifest $M --acts-root $W --set $S \
    --site 13,-1 --compare-site 11,-2 --blocks $B --out $P/ladder_${S}_${B//,/-}.json
echo LADDER-OK
