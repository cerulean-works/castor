#!/bin/bash
# One open-instruct post-training run (unmodified open-instruct @ fc7748ea4 in its own
# locked uv project). Submit through Slurm, 8 GPUs for real runs:
#   shell/nchc/submit.sh --gpus 8 --name sft-olmo2-s1 -- bash shell/posttrain/train.sh \
#       sft configs/posttrain/sft_olmo2_1b.yaml allenai/OLMo-2-0425-1B 1 \
#       /work/harrychang08/castor/models/posttrain/olmo2-1b/sft/seed1 \
#       [--revision SHA] [--set key=value ...] [--ds-config PATH] [--launch-env KEY=VALUE ...]
# Stage sft|dpo picks the entry point; the DeepSpeed config is the stage default
# (castor/posttrain.py OI_DS_CONFIG) unless --ds-config names one. render copies it
# to $OUT/ds_config.json and writes the launcher environment to $OUT/launch_env.sh; both are
# what this script launches with, and provenance.json records them (path, sha256, env). The
# post-training seed index k maps to open-instruct's --seed (k for SFT, 110+k for DPO).
# CASTOR_MAIN_PORT overrides the per-job rendezvous port (several runs inside one job,
# shell/posttrain/colocate.sh); unset, the port is the per-job one.
set -euo pipefail
STAGE=$1; CONFIG=$2; MODEL=$3; K=$4; OUT=$5; shift 5
# absolute: the launch below runs from $OI, and a relative path would land in the job's
# snapshot worktree, deleted when the job ends
[[ $OUT == /* ]] || { echo "OUT must be an absolute path: $OUT" >&2; exit 2; }
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OI=/work/harrychang08/castor/third_party/open-instruct
cd "$REPO"

# castor env: render the run config + provenance (refuses dirty castor code or a modified OI tree)
ARGS_FILE=$(uv run --frozen python scripts/posttrain_run.py render --stage "$STAGE" --config "$CONFIG" \
      --model "$MODEL" --k "$K" --out "$OUT" --job "${SLURM_JOB_ID:-}" "$@" | tail -1)
case $STAGE in
    sft) ENTRY=open_instruct/finetune.py ;;
    dpo) ENTRY=open_instruct/dpo_tune_cache.py ;;
    *) echo "stage must be sft or dpo" >&2; exit 2 ;;
esac
NGPU=$(nvidia-smi -L | wc -l)

# open-instruct env (its own .venv from its own uv.lock); never castor's venv
(
    unset VIRTUAL_ENV; PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "$REPO/.venv" | paste -sd:)
    source "$OUT/launch_env.sh"  # BEAKER_ASSIGNED_CPU_COUNT (tokenization workers), ...
    DS="$OUT/ds_config.json"
    cd "$OI"
    mapfile -t OI_ARGS < "$ARGS_FILE"
    echo "open-instruct $(git rev-parse --short HEAD) | $STAGE | $NGPU GPUs | ds $(sha256sum "$DS" | cut -c1-12) | ${OI_ARGS[*]}"
    uv run --frozen --extra compile --project "$OI" accelerate launch \
        --mixed_precision bf16 --num_processes "$NGPU" --use_deepspeed \
        --main_process_port "${CASTOR_MAIN_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}" \
        --deepspeed_config_file "$DS" --deepspeed_multinode_launcher standard \
        "$ENTRY" "${OI_ARGS[@]}"
)
uv run --frozen python scripts/posttrain_run.py finish --out "$OUT"
