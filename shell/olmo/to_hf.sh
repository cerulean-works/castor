#!/bin/bash
# Convert one of our OLMo unsharded checkpoints to the hf_olmo format the released DataDecide
# checkpoints use, with OLMo's own hf_olmo/convert_olmo_to_hf.py (Ai2's eval ran on
# "<step>-unsharded-hf" conversions, per the released metrics.json). The converter works in place,
# so it runs on a copy (model.pt + config.yaml) in <out-dir>.
#   bash shell/olmo/to_hf.sh <run>/ckpt/step37500-unsharded <out-dir>
set -euo pipefail
SRC=$1; DST=$2
ENVP=/work/harrychang08/castor/third_party/olmo-env
OLMO=/work/harrychang08/castor/third_party/OLMo
mkdir -p "$DST"
cp "$SRC/model.pt" "$SRC/config.yaml" "$DST/"
unset VIRTUAL_ENV
cd $OLMO
uv run --frozen --no-sync --project $ENVP python hf_olmo/convert_olmo_to_hf.py --checkpoint-dir "$DST" \
    --ignore-olmo-compatibility
echo "{\"source\": \"$SRC\", \"converter\": \"hf_olmo/convert_olmo_to_hf.py@$(git -C $OLMO rev-parse HEAD)\", \"model_pt_sha256\": \"$(sha256sum "$SRC/model.pt" | cut -d' ' -f1)\"}" > "$DST/provenance.json"
