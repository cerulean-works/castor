#!/bin/bash
# Evaluate our pretraining runs at one step with the same instruments as the released
# checkpoints: convert <run>/ckpt/step<S>-unsharded to hf_olmo (to_hf.sh), then eval_queue.sh
# (perplexity from model.pt, OLMES-10 from the converted dir).
# <size> (150M, 1B) selects the eval model config and the output dir; it must be the runs' size.
#   bash shell/olmo/eval_ours.sh <size> <step> <run-name>...
set -euo pipefail
SIZE=$1; STEP=$2; shift 2
R=/work/harrychang08/castor/runs/003
OUT=$R/eval/ours-$SIZE
CONFIG=$R/eval/config-$SIZE.yaml
[ -f "$CONFIG" ] || { echo "eval_ours.sh: no $CONFIG" >&2; exit 2; }
REPO="$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"
mkdir -p "$OUT/hf"
LIST="$OUT/list-step$STEP.tsv"; : > "$LIST"
for run in "$@"; do
    src=$R/$run/ckpt/step$STEP-unsharded
    grep -q "\"model\": \"$SIZE\"" "$R/$run/config.yaml.changes.json" \
        || { echo "eval_ours.sh: $run is not a $SIZE run" >&2; exit 2; }
    hf=$OUT/hf/$run-step$STEP
    [ -f "$hf/provenance.json" ] || bash "$REPO/shell/olmo/to_hf.sh" "$src" "$hf"
    printf '%s\t%s\t-\tolmo-ckpt\t%s\n' "${run}__step$STEP" "$hf" "$src/model.pt" >> "$LIST"
done
bash "$REPO/shell/olmo/eval_queue.sh" "$LIST" "$OUT" "$CONFIG"
