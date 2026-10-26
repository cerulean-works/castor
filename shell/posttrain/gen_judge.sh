#!/bin/bash
# vLLM generation on the evaluated panel, then every judge_v1 stage, for one model (1 GPU).
#   shell/nchc/submit.sh --gpus 1 --time 12:00:00 --name 004-gj-<id> -- \
#       bash shell/posttrain/gen_judge.sh MODEL REVISION|- MODEL_ID [PROMPT_ID [LIMIT]]
# Writes /work/harrychang08/castor/runs/<MODEL_ID>/<PROMPT_ID>/{samples,greedy}.jsonl and
# .../judge_v1/{coherence,wildguard,qwen3guard,labels}.jsonl. Skips generation if a
# provenance.json for the full panel (limit null) already exists. LIMIT (first N prompts) is for
# dry runs only and always regenerates.
set -euo pipefail
MODEL=$1; REV=$2; ID=$3; P=${4:-004}; LIMIT=${5:-}
OUT=/work/harrychang08/castor/runs/$ID/$P
REVARG=(); [ "$REV" = - ] || REVARG=(--revision "$REV")
LIMARG=(); [ -z "$LIMIT" ] || LIMARG=(--limit "$LIMIT")
if [ -z "$LIMIT" ] && [ -f "$OUT/provenance.json" ] && grep -q '"limit": null' "$OUT/provenance.json"; then
    echo "generation exists: $OUT"
else
    uv run --frozen python scripts/generate.py --model "$MODEL" "${REVARG[@]}" --model-id "$ID" --prompt-id "$P" "${LIMARG[@]}"
fi
mkdir -p "$OUT/judge_v1"
bash shell/nchc/judge_stages.sh "$OUT/judge_v1" "$OUT/samples.jsonl" "$OUT/greedy.jsonl"
