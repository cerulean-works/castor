#!/bin/bash
# 007: several own/random ablation cells serially on one GPU, so that 15 cells fit in
# ≤ 4 jobs of 007 in the queue at once without a live session to resubmit. Each cell is
# shell/mech/ablate_genjudge_007.sh (skip-if-done keyed on provenance); a failed cell does not
# stop the others, and the job exits non-zero if any failed.
#   shell/nchc/submit.sh --gpus 1 --time 12:00:00 --name 007-abl-g1 -- \
#       bash shell/mech/ablate_group_007.sh MODEL_ID:own MODEL_ID:random ...
set -uo pipefail
FAILED=()
for CELL in "$@"; do
    ID=${CELL%%:*}; KIND=${CELL##*:}
    echo "=== $ID $KIND $(date -Is)"
    if bash shell/mech/ablate_genjudge_007.sh "$ID" "$KIND"; then echo "CELL-OK $ID $KIND"
    else echo "CELL-FAILED $ID $KIND"; FAILED+=("$CELL"); fi
done
[ ${#FAILED[@]} -eq 0 ] && { echo ABLATE-GROUP-OK; exit 0; }
echo "ABLATE-GROUP-FAILED ${FAILED[*]}"; exit 1
