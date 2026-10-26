#!/bin/bash
# 007: re-run one ablation cell after its group job ended, unless that job's log shows it done.
#   shell/nchc/submit.sh --after <group job> ... -- bash shell/mech/ablate_followup_007.sh \
#       <group log> MODEL_ID KIND
set -euo pipefail
LOG=$1; ID=$2; KIND=$3
if grep -q "CELL-OK $ID $KIND" "$LOG"; then echo "done in $LOG: skip"; echo FOLLOWUP-OK; exit 0; fi
bash shell/mech/ablate_genjudge_007.sh "$ID" "$KIND"
echo "CELL-OK $ID $KIND"; echo FOLLOWUP-OK
