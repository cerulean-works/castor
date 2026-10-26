#!/bin/bash
# Pretrain one DataDecide DCLM-Baseline model with OLMo's unmodified scripts/train.py.
# Config = Ai2's ladder config (make_config.py; only site paths / run name / optional
# data.seed change, listed in config.yaml.changes.json). World size = the GPUs of this job, which
# must be 8 on every segment (Ai2 used 1 node x 8; a resume on another count would change the
# per-device batch and accumulation schedule mid-run). Run length is Ai2's: max_duration 5xC tokens, and Trainer.fit() sets
# stop_at = max_steps + 10 when unset, giving 38,157 steps at 150M and 69,369 at 1B.
# Resume (2-day job chains): if the run's save folder has `latest-unsharded`, training continues
# from it with --load_path (OLMo restores model, optimizer, scheduler and data position).
#   bash shell/olmo/pretrain.sh <size> <seed> <data-seed|-> <run-name> [extra train.py args, smoke only]
set -euo pipefail
SIZE=$1; SEED=$2; DSEED=$3; NAME=$4; shift 4
ENVP=/work/harrychang08/castor/third_party/olmo-env
OLMO=/work/harrychang08/castor/third_party/OLMo
REPO="$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"
R=/work/harrychang08/castor/runs/003/$NAME
# Allocated GPUs: Slurm's CUDA_VISIBLE_DEVICES (nvidia-smi lists the whole node without cgroups).
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then NGPU=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .)
else NGPU=$(nvidia-smi -L | wc -l); fi
[ "$NGPU" -eq 8 ] || { echo "pretrain.sh: world size must be 8 (Ai2's), got $NGPU" >&2; exit 2; }
COMMIT=$(cd "$REPO" && uv run --frozen python -c "from castor import provenance; print(provenance.git_commit())")
mkdir -p "$R"
unset VIRTUAL_ENV
# --no-sync: uv reads flash-attn 2.5.9.post1 as not matching the lock's local-version URL wheel
# (+cu122torch2.3cxx11abifalse) and would reinstall it on every run, racing concurrent jobs.
if [ ! -f "$R/config.yaml" ]; then
    DARG=(); [ "$DSEED" = "-" ] || DARG=(--data-seed "$DSEED")
    uv run --frozen --no-sync --project $ENVP python "$REPO/shell/olmo/make_config.py" --model "$SIZE" \
        --data DCLM-baseline --seed "$SEED" "${DARG[@]}" --run-name "$NAME" \
        --save-folder "$R/ckpt" --out "$R/config.yaml"
fi
# An existing config (resume, or a reused run name) must be the one these arguments would write.
uv run --frozen --no-sync --project $ENVP python - "$R/config.yaml.changes.json" "$SIZE" "$SEED" "$DSEED" "$NAME" <<'PY'
import json, sys
path, size, seed, dseed, name = sys.argv[1:]
c = json.load(open(path))
la, ch = c["ladder_args"], c["changes"]
want_d = f"-> {dseed}" if dseed != "-" else None
got_d = [x for x in ch if x.startswith("data.seed ")]
ok = (la["model"] == size and la["seed"] == int(seed)
      and any(x.endswith(f"-> {name}") for x in ch if x.startswith("run_name "))
      and (got_d == [] if want_d is None else len(got_d) == 1 and got_d[0].endswith(want_d)))
if not ok:
    sys.exit(f"pretrain.sh: {path} (model={la['model']} seed={la['seed']} {got_d}) does not match "
             f"args size={size} seed={seed} data-seed={dseed} name={name}")
PY
# A run already at its final step exits here: resuming it would train past stop_at (OLMo takes a
# batch before checking) and overwrite the final checkpoint. Lets a 2-day chain carry a spare segment.
if [ -e "$R/ckpt/latest-unsharded" ]; then
    AT=$(readlink "$R/ckpt/latest-unsharded" | xargs basename | sed -n 's/^step\([0-9]*\)-unsharded$/\1/p')
    FINAL=$(uv run --frozen --no-sync --project $ENVP python - "$R/config.yaml" <<'PY'
import math, sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
assert c["stop_at"] is None and c["max_duration"].endswith("T"), "pretrain.sh: expects Ai2's token budget"
print(math.ceil(int(c["max_duration"][:-1]) / (c["global_train_batch_size"] * c["model"]["max_sequence_length"])) + 10)
PY
)
    [ -n "$AT" ] || { echo "pretrain.sh: cannot read the step of latest-unsharded" >&2; exit 2; }
    if [ "$AT" -ge "$FINAL" ]; then
        echo "pretrain.sh: $NAME is finished (latest-unsharded = step $AT, final step $FINAL); nothing to do"
        echo "{\"job\": \"${SLURM_JOB_ID:-none}\", \"finished_at\": $AT, \"noop\": \"$(date -Iseconds)\"}" >> "$R/jobs.jsonl"
        exit 0
    fi
fi
LOAD=()
if [ -e "$R/ckpt/latest-unsharded" ]; then
    # Resume. OLMo refuses an existing train_data/ without --save_overwrite (the global indices are
    # rebuilt identically from the seed), and it reopens data-indices/rank*.tsv.gz for writing,
    # which would truncate the previous segment's per-step indices: archive them first.
    # The archive name carries the job id: a segment that crashes before its first new checkpoint
    # resumes from the same step, and its (partial) indices must not land inside the earlier archive.
    if [ -d "$R/ckpt/data-indices" ]; then
        A="$R/ckpt/data-indices.upto-$(readlink "$R/ckpt/latest-unsharded" | xargs basename).job-${SLURM_JOB_ID:-$$}"
        [ ! -e "$A" ] || { echo "pretrain.sh: $A exists" >&2; exit 2; }
        mv "$R/ckpt/data-indices" "$A"
    fi
    LOAD=(--load_path="$R/ckpt/latest-unsharded" --save_overwrite=true)
fi
echo "{\"job\": \"${SLURM_JOB_ID:-none}\", \"host\": \"$(hostname)\", \"gpus\": $NGPU, \"start\": \"$(date -Iseconds)\", \"git_commit\": \"$COMMIT\", \"olmo\": \"$(git -C $OLMO rev-parse HEAD)\", \"config_sha256\": \"$(sha256sum "$R/config.yaml" | cut -d' ' -f1)\", \"resume\": \"${LOAD[*]:-}\", \"extra\": \"$*\"}" >> "$R/jobs.jsonl"
cd $OLMO
uv run --frozen --no-sync --project $ENVP torchrun --standalone --nproc-per-node "$NGPU" scripts/train.py \
    "$R/config.yaml" "${LOAD[@]}" "$@"
echo "{\"job\": \"${SLURM_JOB_ID:-none}\", \"end\": \"$(date -Iseconds)\"}" >> "$R/jobs.jsonl"
