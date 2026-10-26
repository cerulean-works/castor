#!/bin/bash
# Start N post-training commands at once inside ONE Slurm allocation, all on
# the same GPUs (each command is an 8-GPU shell/posttrain/train.sh run), optionally under a
# job-local user-level CUDA MPS daemon. The commands themselves are not changed: copy i only gets
# its own rendezvous port (CASTOR_MAIN_PORT, read by train.sh), its own Triton / Inductor / HF
# datasets cache dirs, and NCCL_DEBUG=INFO (logging only: the transport lines P2P/IPC vs SHM).
# Never CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: every MPS client may use all SMs.
#
#   shell/nchc/submit.sh --gpus 8 --dev -- bash shell/posttrain/colocate.sh <mps 0|1> <logdir> <cmd> <cmd> [<cmd> ...]
#
# Per copy i (1..N): <logdir>/c<i>.log, every line prefixed with a wall-clock epoch time in µs
# ($EPOCHREALTIME at arrival; open-instruct's own step stamps have 1-s resolution), and
# <logdir>/c<i>.rc. Job-level: <logdir>/env.txt (compute mode, MPS server list, node),
# <logdir>/smi.csv (per-GPU memory.used and utilization every second while the copies run).
# A copy still running after COLOCATE_TIMEOUT seconds (default 2400) is killed and recorded as a
# hang (rc 124). Exit status: 0 iff every copy exited 0.
set -uo pipefail
MPS=$1; LOGDIR=$2; shift 2
[ $# -ge 1 ] || { echo "colocate.sh: no commands" >&2; exit 2; }
[[ $LOGDIR == /* ]] || { echo "LOGDIR must be absolute: $LOGDIR" >&2; exit 2; }
N=$#
TIMEOUT=${COLOCATE_TIMEOUT:-2400}
ID=${SLURM_JOB_ID:-$$}
SCRATCH=/work/$USER/castor/tmp/025/colocate-$ID
mkdir -p "$LOGDIR" "$SCRATCH"

cleanup() {
    [ -n "${SMI_PID:-}" ] && kill "$SMI_PID" 2>/dev/null
    if [ "$MPS" = 1 ]; then echo quit | nvidia-cuda-mps-control 2>/dev/null; fi
    rm -rf "$SCRATCH"
}
trap cleanup EXIT

if [ "$MPS" = 1 ]; then
    # job-local pipe/log dirs on node-local disk: only this job's processes find this daemon
    export CUDA_MPS_PIPE_DIRECTORY=${TMPDIR:-/tmp}/mps-$USER-$ID/pipe
    export CUDA_MPS_LOG_DIRECTORY=${TMPDIR:-/tmp}/mps-$USER-$ID/log
    rm -rf "${TMPDIR:-/tmp}/mps-$USER-$ID"; mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
    nvidia-cuda-mps-control -d || { echo "MPS daemon failed to start" >&2; exit 3; }
fi

{
    echo "node $(hostname) job $ID mps $MPS copies $N timeout $TIMEOUT"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    nvidia-smi --query-gpu=index,name,compute_mode --format=csv
    echo "mps control processes:"; pgrep -a -u "$USER" nvidia-cuda-mps || echo "(none)"
} > "$LOGDIR/env.txt" 2>&1
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 1 \
    > "$LOGDIR/smi.csv" 2>&1 &
SMI_PID=$!

stamp() { while IFS= read -r l; do printf '%s %s\n' "$EPOCHREALTIME" "$l"; done; }

PIDS=()
i=0
for CMD in "$@"; do
    i=$((i + 1))
    C=$SCRATCH/c$i
    mkdir -p "$C/triton" "$C/inductor" "$C/hf_datasets"
    (
        export CASTOR_MAIN_PORT=$((29500 + ID % 1000 * 10 + i)) NCCL_DEBUG=INFO \
            TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor HF_DATASETS_CACHE=$C/hf_datasets
        echo "copy $i port $CASTOR_MAIN_PORT start $EPOCHREALTIME"
        echo "cmd: $CMD"
        RC=0; timeout --signal=TERM --kill-after=60 "$TIMEOUT" bash -c "$CMD" 2>&1 || RC=$?
        echo "copy $i rc $RC end $EPOCHREALTIME"
        echo "$RC" > "$LOGDIR/c$i.rc"
    ) 2>&1 | stamp > "$LOGDIR/c$i.log" &
    PIDS+=($!)
done
# MPS server list while the copies run (one server per user; clients attach to it)
if [ "$MPS" = 1 ]; then
    sleep 90; echo "mps server list at +90 s: $(echo get_server_list | nvidia-cuda-mps-control 2>&1)" >> "$LOGDIR/env.txt"
fi
for p in "${PIDS[@]}"; do wait "$p"; done

FAIL=0
for j in $(seq 1 "$N"); do
    rc=$(cat "$LOGDIR/c$j.rc" 2>/dev/null || echo missing)
    echo "copy $j rc $rc"
    [ "$rc" = 0 ] || FAIL=1
done
exit $FAIL
