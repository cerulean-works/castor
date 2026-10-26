#!/bin/bash
# Judge smoke + throughput: sample released OLMo-2-0425-1B-DPO on the panel, then run
# every judge_v1 stage (one process per vLLM engine) and compose labels.
#   shell/nchc/submit.sh --dev --time 02:00:00 --name judge-smoke -- bash shell/nchc/judge_smoke.sh OUT [N]
set -euo pipefail
OUT=${1:?output dir}; N=${2:-2000}
uv run --frozen python scripts/smoke_generate.py --n "$N" --out "$OUT"
bash shell/nchc/judge_stages.sh "$OUT" "$OUT/generations.jsonl" "$OUT/probes.jsonl"
