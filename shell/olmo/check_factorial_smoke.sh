#!/bin/bash
# Stage-2 pre-check on the 1B smoke runs: (1) step-0 weights of each factorial cell equal the
# released step-0 of its *init* seed and differ from the other; (2) the data order (OLMo's
# global_indices.npy) depends only on the data seed: the (init 4, data 6198) cell's indices equal
# those of our 150M seed-6198 run (same mix, same data seed), and the (init 6198, data 4) cell's
# differ. Writes compare-step0.json per run and global_indices.sha256.
set -euo pipefail
ENVP=/work/harrychang08/castor/third_party/olmo-env
R=/work/harrychang08/castor/runs/003
unset VIRTUAL_ENV
for run in smoke-1B-i6198-d4 smoke-1B-i4-d6198; do
    uv run --frozen --no-sync --project $ENVP python shell/olmo/compare_ckpt.py \
        --ours $R/$run/ckpt/step0-unsharded/model.pt --repo allenai/DataDecide-dclm-baseline-1B \
        --revisions step0-seed-default step0-seed-large-aux-2 --out $R/$run/compare-step0.json
done
sha256sum $R/smoke-1B-i6198-d4/ckpt/train_data/global_indices.npy \
    $R/smoke-1B-i4-d6198/ckpt/train_data/global_indices.npy \
    $R/castor-dclm-150M-s6198/ckpt/train_data/global_indices.npy | tee $R/smoke-1B-global_indices.sha256
