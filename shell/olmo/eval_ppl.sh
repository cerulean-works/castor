#!/bin/bash
# Held-out perplexity of one checkpoint with OLMo's in-loop evaluator (see eval_ppl.py).
#   bash shell/olmo/eval_ppl.sh <config.yaml> (--hf REPO@REV | --olmo-ckpt model.pt) --out ppl.json
# All paths absolute (the script runs from the OLMo checkout).
set -euo pipefail
CONFIG=$1; shift
ENVP=/work/harrychang08/castor/third_party/olmo-env
OLMO=/work/harrychang08/castor/third_party/OLMo
REPO="$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"
COMMIT=$(cd "$REPO" && uv run --frozen python -c "from castor import provenance; print(provenance.git_commit())")
unset VIRTUAL_ENV
cd $OLMO  # tokenizer identifier in the config is relative to the OLMo checkout
uv run --frozen --no-sync --project $ENVP python "$REPO/shell/olmo/eval_ppl.py" --config "$CONFIG" \
    --git-commit "$COMMIT" "$@"
