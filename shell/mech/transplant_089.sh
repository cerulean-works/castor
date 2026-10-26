#!/bin/bash
# One 089 arm on one GPU: hooked HF generation (scripts/transplant_089.py generate), then every
# judge_v1 stage on its samples.
#   shell/nchc/submit.sh --dev --name 089-<arm> -- bash shell/mech/transplant_089.sh MODEL_ID ARGS...
#   ARGS = --dose K | --donor B [--permute] [--site 11,-2]  [--limit N]  [--prompt-id 091]
# Writes /work/harrychang08/castor/runs/<MODEL_ID>/<089|091>/<arm>/{samples.jsonl,provenance.json,judge_v1/}
# (a --limit N smoke writes <arm>_limit<N>/).
set -euo pipefail
ID=$1; shift
LOG=$(mktemp)
uv run --frozen python scripts/transplant_089.py generate --model-id "$ID" "$@" | tee "$LOG"
OUT=$(python3 -c "import json,sys;l=[x for x in open(sys.argv[1]) if x.startswith('{\"arm\"')];print(json.loads(l[-1])['out'] if l else '')" "$LOG")
if [ -z "$OUT" ]; then
    OUT=$(grep -o 'generation exists: .*' "$LOG" | sed 's|^generation exists: ||')
fi
[ -n "$OUT" ] || { echo "no arm directory in the generate log" >&2; exit 1; }
mkdir -p "$OUT/judge_v1"
if [ -f "$OUT/judge_v1/labels.jsonl" ] && [ "$OUT/judge_v1/labels.jsonl" -nt "$OUT/samples.jsonl" ]; then
    echo "labels exist: $OUT/judge_v1"
else
    bash shell/nchc/judge_stages.sh "$OUT/judge_v1" "$OUT/samples.jsonl"
fi
