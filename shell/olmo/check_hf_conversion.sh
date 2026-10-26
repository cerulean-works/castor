#!/bin/bash
# Our OLMo checkpoint converted to hf_olmo (to_hf.sh) must give the same logits under the
# olmes env's loader (AutoModelForCausalLM + hf_olmo, as oe-eval loads it) as under OLMo itself.
#   bash shell/olmo/check_hf_conversion.sh <ckpt-dir with model.pt> <hf-dir>
set -euo pipefail
SRC=$1; HF=$2
unset VIRTUAL_ENV
[ -f "$HF/provenance.json" ] || bash "$(dirname "$0")/to_hf.sh" "$SRC" "$HF"
uv run --frozen --project /work/harrychang08/castor/third_party/olmes-env python - "$SRC" "$HF" <<'PY'
import sys, torch
import hf_olmo  # noqa: F401  registers OLMoForCausalLM with transformers
from transformers import AutoModelForCausalLM
from olmo.config import ModelConfig
from olmo.model import OLMo
src, hf = sys.argv[1:]
x = torch.randint(0, 50000, (2, 512)).cuda()
m1 = AutoModelForCausalLM.from_pretrained(hf).cuda().eval()
cfg = ModelConfig.load(f"{src}/config.yaml", key="model"); cfg.init_device = "cpu"
m2 = OLMo(cfg); m2.load_state_dict(torch.load(f"{src}/model.pt", map_location="cpu")); m2 = m2.cuda().eval()
with torch.no_grad():
    a, b = m1(x).logits.float(), m2(x).logits.float()
print("dtype", next(m1.parameters()).dtype, "max|dlogit|", (a - b).abs().max().item(), "top1 agree", (a.argmax(-1) == b.argmax(-1)).float().mean().item())
PY
