#!/bin/bash
# Probe: one same-seed 1B OLMo replicate of 011's run (init 6198, data.seed unset ->
# 6198, OLMo ccabb54e, 8 GPUs) stopped early, as two invocations of 011's unmodified pretrain.sh:
#   025-<P>-s10   --stop_at=10  --save_interval_unsharded=1   (unsharded saves at steps 1..10)
#   025-<P>-l100  --stop_at=100 --save_interval_unsharded=10  (unsharded saves at steps 10, 20, .., 100)
# Both also write OLMo's pre-train step0-unsharded. The two overrides change when OLMo stops and
# saves, not the schedule: max_duration (and so the LR schedule) is 011's. Then each step 0 is
# compared bit for bit with the released step0-seed-default (compare_ckpt.py, as 011).
# Run dirs: /work/harrychang08/castor/runs/003/025-<P>-{s10,l100} (pretrain.sh's fixed root); each
# invocation's output is also kept in /work/harrychang08/castor/runs/025/probe1-logs/<name>.log.
#   bash shell/olmo/probe1_025.sh <P>     (8 GPUs; P = P1, P2, P3)
set -euo pipefail
P=$1
ENVP=/work/harrychang08/castor/third_party/olmo-env
HERE="$(dirname "$(readlink -f "$0")")"
LOGS=/work/harrychang08/castor/runs/025/probe1-logs
mkdir -p "$LOGS"
for spec in "s10 10 1" "l100 100 10"; do
    read -r TAG STOP EVERY <<< "$spec"
    NAME=025-$P-$TAG
    R=/work/harrychang08/castor/runs/003/$NAME
    # a second start on an existing dir would make pretrain.sh resume it: never
    [ ! -e "$R" ] || { echo "probe1_025.sh: $R exists; refusing (a rerun would resume it)" >&2; exit 2; }
    echo "[$(date -Is)] $NAME: stop_at=$STOP save_interval_unsharded=$EVERY on $(hostname)"
    bash "$HERE/pretrain.sh" 1B 6198 - "$NAME" --stop_at="$STOP" --save_interval_unsharded="$EVERY" \
        2>&1 | tee "$LOGS/$NAME.log"
    (unset VIRTUAL_ENV; uv run --frozen --no-sync --project $ENVP python "$HERE/compare_ckpt.py" \
        --ours "$R/ckpt/step0-unsharded/model.pt" --repo allenai/DataDecide-dclm-baseline-1B \
        --revisions step0-seed-default --out "$R/compare-step0.json")
    echo "[$(date -Is)] $NAME done"
done
