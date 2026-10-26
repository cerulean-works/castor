#!/bin/bash
# Run every judge_v1 stage over existing generation files (one process per vLLM engine).
#   shell/nchc/submit.sh --dev --name judge -- bash shell/nchc/judge_stages.sh OUT GEN.jsonl [GEN2.jsonl ...]
set -euo pipefail
OUT=${1:?output dir}; shift
for stage in coherence wildguard qwen3guard labels; do
    uv run --frozen python scripts/judge.py --stage "$stage" --generations "$@" --out "$OUT"
done
