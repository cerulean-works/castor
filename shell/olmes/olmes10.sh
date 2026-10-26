#!/bin/bash
# OLMES-10 on one checkpoint, exactly as DataDecide ran it (oe-eval, olmes @ 38af8b6).
# The spec is read off Ai2's released metrics.json (DataDecide-eval-instances sample-evals):
# tasks `<task>:rc::olmes` (5-shot, OLMES few-shot sources), full splits (limit 1e19),
# model_type hf (hf_olmo loader), max_length 2048, batch_size 4. Same command for released
# checkpoints and ours. Aggregation: scripts/olmes10_aggregate.py (castor env).
#   bash shell/olmes/olmes10.sh <hf-repo-or-local-dir> <revision|-> <out-dir>
set -euo pipefail
MODEL=$1; REV=$2; OUT=$3
ENVP=/work/harrychang08/castor/third_party/olmes-env
REPO="$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"
COMMIT=$(cd "$REPO" && uv run --frozen python -c "from castor import provenance; print(provenance.git_commit())")
TASKS="arc_challenge:rc::olmes arc_easy:rc::olmes boolq:rc::olmes csqa:rc::olmes hellaswag:rc::olmes
openbookqa:rc::olmes piqa:rc::olmes socialiqa:rc::olmes winogrande:rc::olmes mmlu:rc::olmes"
REVARG=(); [ "$REV" = "-" ] || REVARG=(--revision "$REV")
mkdir -p "$OUT"
unset VIRTUAL_ENV
# shellcheck disable=SC2086
# oe_eval.launch (the `olmes` CLI) resolves the task aliases and suites, then runs oe_eval.run_eval.
uv run --frozen --project $ENVP python -m oe_eval.launch --model "$MODEL" "${REVARG[@]}" \
    --model-type hf --model-args '{"max_length": 2048}' --batch-size 4 --gpus 1 \
    --limit 10000000000000000000 --task $TASKS --output-dir "$OUT"
cat > "$OUT/provenance.json" <<JSON
{"script": "shell/olmes/olmes10.sh", "model": "$MODEL", "revision": "$REV", "olmes": "38af8b61741b01faeb60e90899addd7e6fa503a0",
 "env": "$ENVP (configs/third_party/olmes.lock)", "git_commit": "$COMMIT", "host": "$(hostname)",
 "slurm_job": "${SLURM_JOB_ID:-none}", "finished": "$(date -Iseconds)"}
JSON
