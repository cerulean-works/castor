#!/bin/bash
# 007 own-direction / random-control ablation for one twin (1 GPU): orthogonalize, then the
# unchanged instrument (vLLM generation + judge_v1) on the evaluated panel.
#   shell/nchc/submit.sh --gpus 1 --time 06:00:00 --name 007-abl-<tag> -- \
#       bash shell/mech/ablate_genjudge_007.sh MODEL_ID own|random|own-site
# Checkpoint: /work/harrychang08/castor/models/ablate/007/<MODEL_ID>-<kind>/. Generations and
# labels: /work/harrychang08/castor/runs/<MODEL_ID>-ablate-<kind>/007/. Skip-if-done is keyed on
# provenance: an existing checkpoint is reused only if its provenance names this model and kind
# and (own) the current runs/007/selection.json and the twin's candidates.safetensors (the file
# its direction is read from) by sha256; the generation directory only if its
# ablation_checkpoint.json is that checkpoint's provenance. Anything else is moved aside to
# <dir>.stale-<time> (never deleted) and rebuilt.
# own-site (the twin's own site in selection.json["own"]): checkpoint
# .../models/ablate/022/<MODEL_ID>-ownsite/, outputs .../runs/<MODEL_ID>-ablate-ownsite/022/.
set -euo pipefail
ID=$1; KIND=$2
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
W=/work/harrychang08/castor
P=007; TAG=$KIND
[ "$KIND" = own-site ] && { P=022; TAG=ownsite; }
CKPT=$W/models/ablate/$P/$ID-$TAG
OUT=$W/runs/$ID-ablate-$TAG/$P
SEL=$REPO/runs/007/selection.json
CAND=$W/activations/007/$ID/candidates.safetensors
STAMP=$(date +%Y%m%d-%H%M%S)
if [ -f "$CKPT/provenance.json" ] && ! uv run --frozen python - "$CKPT/provenance.json" "$SEL" "$CAND" "$ID" "$KIND" <<'EOF'
import hashlib, json, sys
prov, sel, cand, mid, kind = sys.argv[1:]
p = json.load(open(prov))
ok = p["source_model_id"] == mid and p["kind"] == kind
if kind != "random":  # the random direction is seeded by the model id and does not use the site
    sha = [hashlib.sha256(open(f, "rb").read()).hexdigest() for f in (sel, cand)]
    ok = ok and [p["selection_sha256"], p.get("candidates_sha256")] == sha
sys.exit(0 if ok else 1)
EOF
then
    echo "stale checkpoint (another selection.json or direction): moving $CKPT aside"
    mv "$CKPT" "$CKPT.stale-$STAMP"
fi
if [ -f "$CKPT/provenance.json" ]; then
    echo "checkpoint exists: $CKPT"
else
    uv run --frozen python scripts/ablate_007.py --models data/manifests/mech_007.json \
        --model-id "$ID" --kind "$KIND" --selection "$SEL" \
        --acts-root "$W/activations/007" --out "$CKPT"
fi
if [ -e "$OUT" ] && ! cmp -s "$CKPT/provenance.json" "$OUT/ablation_checkpoint.json"; then
    echo "generations from another checkpoint: moving $OUT aside"
    mv "$OUT" "$OUT.stale-$STAMP"
fi
mkdir -p "$OUT"
cp "$CKPT/provenance.json" "$OUT/ablation_checkpoint.json"
bash shell/posttrain/gen_judge.sh "$CKPT" - "$ID-ablate-$TAG" "$P"
