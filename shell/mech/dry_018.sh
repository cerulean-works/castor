#!/bin/bash
# 018 dry-run job (1 GPU, dev): fit-set activations of the 12 twins (every token, the
# own-site layers), then the one-target dry run (same-unrot on default/s1, first 50 eval prompts).
#   shell/nchc/submit.sh --dev --gpus 1 --time 02:00:00 --name 018-dry -- bash shell/mech/dry_018.sh
set -euo pipefail
uv run --frozen python scripts/transfer_018.py extract \
    --out-root /work/harrychang08/castor/activations/018
bash shell/mech/transfer_genjudge_018.sh dd-dclm-1b-default-dpo-seed1 same-unrot 50
echo DRY-018-OK
