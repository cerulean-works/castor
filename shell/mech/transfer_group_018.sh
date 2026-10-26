#!/bin/bash
# 018: several (target, arm) cells serially on one GPU, so the 60 cells fit in <= 4 jobs of 018
# in Slurm at once (007/022's convention). Each cell is shell/mech/transfer_genjudge_018.sh
# (skip-if-done keyed on provenance); a failed cell does not stop the others, and the job exits
# non-zero if any failed.
#   shell/nchc/submit.sh --gpus 1 --time 2-00:00:00 --name 018-g1 -- \
#       bash shell/mech/transfer_group_018.sh MODEL_ID:ARM MODEL_ID:ARM ...
set -uo pipefail
FAILED=()
for CELL in "$@"; do
    ID=${CELL%%:*}; ARM=${CELL##*:}
    echo "=== $ID $ARM $(date -Is)"
    if bash shell/mech/transfer_genjudge_018.sh "$ID" "$ARM"; then echo "CELL-OK $ID $ARM"
    else echo "CELL-FAILED $ID $ARM"; FAILED+=("$CELL"); fi
done
[ ${#FAILED[@]} -eq 0 ] && { echo TRANSFER-GROUP-OK; exit 0; }
echo "TRANSFER-GROUP-FAILED ${FAILED[*]}"; exit 1
