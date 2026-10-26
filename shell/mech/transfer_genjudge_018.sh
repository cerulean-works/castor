#!/bin/bash
# 018 cross-twin ablation transfer, one (target, arm) cell on 1 GPU: orthogonalize the target
# with the arm's donor direction (scripts/transfer_018.py ablate), then the unchanged
# instrument (vLLM generation + judge_v1) on the evaluated panel.
#   shell/nchc/submit.sh --gpus 1 --time 06:00:00 --name 018-<tag> -- \
#       bash shell/mech/transfer_genjudge_018.sh MODEL_ID ARM [LIMIT]
# ARM: same-unrot | diff-unrot | diff-rot | diff-mismatch | diff-rot-std.
# Checkpoint: /work/harrychang08/castor/models/ablate/018/<MODEL_ID>-<ARM>/. Generations and
# labels: /work/harrychang08/castor/runs/<MODEL_ID>-ablate-<ARM>/018/ (018-dry with LIMIT, the
# dry run, so a partial panel never sits where the full run goes).
# Skip-if-done is keyed on provenance: a checkpoint is reused only if it names this target and
# arm and the direction it removed is the one transfer_018.py computes now (sha256 of the unit
# vector: donor candidates, selection, Procrustes map); the generation directory only if its
# ablation_checkpoint.json is that checkpoint's provenance. Anything else is moved aside to
# <dir>.stale-<time> (never deleted) and rebuilt.
set -euo pipefail
ID=$1; ARM=$2; LIMIT=${3:-}
W=/work/harrychang08/castor
P=018; [ -n "$LIMIT" ] && P=018-dry
CKPT=$W/models/ablate/018/$ID-$ARM
OUT=$W/runs/$ID-ablate-$ARM/$P
STAMP=$(date +%Y%m%d-%H%M%S)
if [ -f "$CKPT/provenance.json" ]; then
    WANT=$(uv run --frozen python scripts/transfer_018.py direction-sha --target "$ID" --arm "$ARM")
    if ! uv run --frozen python - "$CKPT/provenance.json" "$ID" "$ARM" "$WANT" <<'PY'
import json, sys
prov, mid, arm, want = sys.argv[1:]
p = json.load(open(prov))
sys.exit(0 if (p["source_model_id"], p["kind"], p["direction_sha256"]) == (mid, arm, want) else 1)
PY
    then
        echo "stale checkpoint (another direction): moving $CKPT aside"
        mv "$CKPT" "$CKPT.stale-$STAMP"
    fi
fi
if [ -f "$CKPT/provenance.json" ]; then
    echo "checkpoint exists: $CKPT"
else
    uv run --frozen python scripts/transfer_018.py ablate --target "$ID" --arm "$ARM" --out "$CKPT"
fi
if [ -e "$OUT" ] && ! cmp -s "$CKPT/provenance.json" "$OUT/ablation_checkpoint.json"; then
    echo "generations from another checkpoint: moving $OUT aside"
    mv "$OUT" "$OUT.stale-$STAMP"
fi
mkdir -p "$OUT"
cp "$CKPT/provenance.json" "$OUT/ablation_checkpoint.json"
bash shell/posttrain/gen_judge.sh "$CKPT" - "$ID-ablate-$ARM" "$P" $LIMIT
