#!/bin/bash
# Smoke of our-run conversion + validation on the 150M 1b run (scratch output under runs/009/).
set -euo pipefail
S=/work/harrychang08/castor/runs/009/smoke-150M
HF=/work/harrychang08/castor/runs/003/eval/ours-150M/hf/castor-dclm-150M-s1001-step37500
uv run --frozen python scripts/convert_castor_ckpt.py --ref castor/DataDecide-dclm-baseline-150M@s1001 \
  --hf-olmo $HF --run castor-dclm-150M-s1001 --step 37500 --init-seed 1001 --data-seed 1001 \
  --register $S/castor_twins_150M.json --out-root $S/converted
PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/work/harrychang08/castor/third_party/OLMo" uv run --frozen python scripts/validate_conversion.py \
  --model castor/DataDecide-dclm-baseline-150M --revision s1001 --source-dir $HF \
  --converted $S/converted/DataDecide-dclm-baseline-150M/s1001 --out $S/validate.json
