#!/bin/bash
# 089 dev smoke: site check, then two short hooked arms (timing) and judging of one.
set -euo pipefail
S=/work/harrychang08/castor/analysis/089/smoke
mkdir -p "$S"
uv run --frozen python scripts/transplant_089.py check-site --out "$S/check_site.json"
for k in 0 1; do
    uv run --frozen python scripts/transplant_089.py generate --model-id dd-dclm-baseline-1b-default-dpo-seed1 \
        --dose $k --limit 64 --out-root "$S/runs"
done
OUT=$S/runs/dd-dclm-baseline-1b-default-dpo-seed1/089/dose_+1
mkdir -p "$OUT/judge_v1"
bash shell/nchc/judge_stages.sh "$OUT/judge_v1" "$OUT/samples.jsonl"
