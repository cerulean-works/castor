#!/bin/bash
# 035 step 3, identity checks of 011 section 2 for R2 (dev job, no GPU work): step-0 tensors against
# the released step0-seed-default (expect 99/99, max |d| 0) and the data-order sha256 (expect R's
# fc3eb83a...).
#   bash shell/olmo/identity_035.sh
set -euo pipefail
ENVP=/work/harrychang08/castor/third_party/olmo-env
R2=/work/harrychang08/castor/runs/003/castor-dclm-1B-i6198-d6198-r2
unset VIRTUAL_ENV
uv run --frozen --no-sync --project $ENVP python shell/olmo/compare_ckpt.py \
    --ours "$R2/ckpt/step0-unsharded/model.pt" --repo allenai/DataDecide-dclm-baseline-1B \
    --revisions step0-seed-default --out "$R2/compare-step0.json"
sha256sum "$R2/ckpt/train_data/global_indices.npy" | tee "$R2/global_indices.sha256"
echo "identity_035: done"
