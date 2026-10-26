#!/bin/bash
# 089 site check for the record (scripts/transplant_089.py check-site), with the sampling path's
# attention backend (cuDNN SDPA off) and with 021's (on).
set -euo pipefail
R=/home/harrychang08/castor/runs/089
[ "${1:-}" = cudnn-only ] || uv run --frozen python scripts/transplant_089.py check-site --out $R/check_site.json
uv run --frozen python scripts/transplant_089.py check-site --cudnn-sdp --out $R/check_site_cudnn.json
