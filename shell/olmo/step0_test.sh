#!/bin/bash
# Does OLMo's real train.py path reproduce a released step-0 checkpoint?
# Builds the ladder config (test data subset), runs scripts/train.py on 1 GPU until the
# pre-train checkpoint is written (stop_at=1), then compares it with the released step-0s.
#   bash shell/olmo/step0_test.sh <size> <seed> <repo> <revision...>
set -euo pipefail
SIZE=$1; SEED=$2; REPO=$3; shift 3
ENVP=/work/harrychang08/castor/third_party/olmo-env
OLMO=/work/harrychang08/castor/third_party/OLMo
R=/work/harrychang08/castor/runs/003/step0-test/$SIZE-s$SEED
rm -rf "$R"; mkdir -p "$R"
unset VIRTUAL_ENV
uv run --frozen --no-sync --project $ENVP python shell/olmo/make_config.py --model "$SIZE" --data DCLM-baseline \
    --seed "$SEED" --run-name "step0test-$SIZE-s$SEED" --save-folder "$R/ckpt" --out "$R/config.yaml" \
    --test-existing-only
(cd $OLMO && uv run --frozen --no-sync --project $ENVP torchrun --standalone --nproc-per-node 1 \
    scripts/train.py "$R/config.yaml" --stop_at=1 --data.num_workers=4 --data.persistent_workers=false)
uv run --frozen --no-sync --project $ENVP python shell/olmo/compare_ckpt.py --ours "$R/ckpt/step0-unsharded/model.pt" \
    --repo "$REPO" --revisions "$@" --out "$R/compare.json"
