#!/bin/bash
# Submit any CASTOR command as a Slurm job on the NCHC H200 nodes.
#
#   shell/nchc/submit.sh [options] -- <command ...>
#
# Options:
#   --gpus N        H200s on one node, 1-8 (default 1). Each H200 has 141 GB.
#   --time T        wall time, Slurm format (default 2-00:00:00; partition max)
#   --partition P   default: dev if --dev, else 8gpus (1-8 GPUs, 2 days).
#                   16gpus / 32gpus need >= 8 / >= 16 GPUs, i.e. multi-node, not wired here.
#   --dev           partition dev: 1 GPU-ish smoke tests, 4 h max, usually starts sooner
#   --name NAME     job name (default: first word of the command)
#   --exclude NODES Slurm node list to avoid. Default: the list in shell/nchc/exclude_nodes (the one
#                   place it lives, with the reason for every node).
#                   "none" excludes nothing.
#   --after JOBID   start only after JOBID has ended, whatever its exit state (afterany; job chains)
#   --commit REV    snapshot this commit instead of HEAD; no dirty-tree fallback (a job that submits
#                   jobs, e.g. scripts/grid.py refill, pins the code of every job it starts)
#   --live          run from the live working tree (default: a snapshot, see below)
#   --term-wait     on SIGTERM the batch shell forwards it to the command and waits, instead of
#                   dying at once. Measured on Nano 4 (runs/008/probes/): at a scancel
#                   and at a time limit Slurm sends TERM to the batch shell only (its children
#                   never see it) and SIGKILLs after KillWait (60 s). With this flag a command that
#                   traps TERM (scripts/grid.py's jobs) gets that window for its trap.
#   --use-reserve   may take the last of the 20 job slots per user. Without it a submission is
#                   refused at 19 jobs: that slot is kept for the refill job a scripts/grid.py
#                   job's SIGTERM trap submits, so another submitter cannot stall the grid.
#   --ending-job ID not counted against the 19-slot check: a job that is ending and submits its
#                   successor (scripts/grid.py refill --ended-job) still shows in squeue, and the
#                   refill already counts its slot as free.
#   --dry-run       print the sbatch script instead of submitting
#
# Snapshot: when the tracked code (castor/ scripts/ configs/ shell/ pyproject.toml uv.lock) is
# clean, the job runs from a detached `git worktree` of HEAD under $CASTOR_WORK/tmp/jobs/, with
# PYTHONPATH and UV_PROJECT set so castor imports come from the snapshot but the venv is the
# repo's. The commit a job records is then the code it ran, even if the shared working tree
# changes while the job waits in the queue. data/ is symlinked to the repo's (panel files are
# not tracked). The snapshot is removed when the job ends. A dirty code tree falls back to the
# live tree with a warning (castor.provenance then refuses to record a commit), unless --commit.
# Run from inside a snapshot (a job submitting jobs), REPO is still the shared checkout: logs,
# the venv and new snapshots belong to it, not to the snapshot that is about to be removed.
#
# Examples:
#   shell/nchc/submit.sh --gpus 2 -- bash shell/cache/<script> ...
#   shell/nchc/submit.sh --dev --time 01:00:00 -- uv run --frozen python scripts/<script>.py ...
#
# Limits (sacctmgr, 2026-09-21): 20 running + 20 queued jobs per user; account acd115198.
# Logs: logs/nchc/<name>-<jobid>.out (stdout+stderr). Watch: squeue --me; tail -f the log.
# Node: 8x H200, 112 cores, ~1.9 TB RAM -> this asks 12 cores (site max per GPU) + 200 GB RAM per GPU.

set -euo pipefail

HERE="$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"
# the default exclusion: exactly one non-comment line (castor.grid.read_exclude checks the same;
# a second line would end the #SBATCH block early, a missing one would exclude nothing)
EXCLUDE_FILE="$HERE/shell/nchc/exclude_nodes"
EXCLUDE="$(grep -v '^#' "$EXCLUDE_FILE" | grep . || true)"
[ "$(printf '%s\n' "$EXCLUDE" | grep -c .)" -eq 1 ] || {
    echo "submit.sh: $EXCLUDE_FILE must hold exactly one node list line, has:" >&2
    printf '%s\n' "$EXCLUDE" >&2; exit 2; }
REPO="$(dirname "$(git -C "$HERE" rev-parse --path-format=absolute --git-common-dir)")"
GPUS=1; TIME="2-00:00:00"; PARTITION=""; DEV=0; NAME=""; DRY=0; LIVE=0; AFTER=""; COMMIT_ARG=""; TERMWAIT=0; RESERVE=0; ENDING=""
CODE_PATHS="castor scripts configs shell pyproject.toml uv.lock"

while [ $# -gt 0 ]; do
    case "$1" in
        --gpus)      GPUS="$2"; shift 2 ;;
        --time)      TIME="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --dev)       DEV=1; shift ;;
        --name)      NAME="$2"; shift 2 ;;
        --live)      LIVE=1; shift ;;
        --after)     AFTER="$2"; shift 2 ;;
        --commit)    COMMIT_ARG="$2"; shift 2 ;;
        --exclude)   EXCLUDE="$2"; shift 2 ;;
        --term-wait) TERMWAIT=1; shift ;;
        --use-reserve) RESERVE=1; shift ;;
        --ending-job) ENDING="$2"; shift 2 ;;
        --dry-run)   DRY=1; shift ;;
        --)          shift; break ;;
        -h|--help)   sed -n '2,49p' "$0"; exit 0 ;;
        *)           echo "unknown option: $1 (put the command after --)" >&2; exit 2 ;;
    esac
done
[ $# -gt 0 ] || { echo "no command given (usage: submit.sh [opts] -- <command ...>)" >&2; exit 2; }
[ "$EXCLUDE" = none ] && EXCLUDE=""
[ "$GPUS" -ge 1 ] && [ "$GPUS" -le 8 ] || { echo "--gpus must be 1-8 (single node)" >&2; exit 2; }
if [ "$DRY" -eq 0 ] && [ "$RESERVE" -eq 0 ]; then
    # squeue's own failure (controller timeout) stops the script under set -e: no count means no
    # submission, never a count of 0; `|| true` covers grep's rc 1 on an empty queue only
    QUEUE="$(squeue --me --noheader --format=%i)"
    NJOBS="$(printf '%s' "$QUEUE" | grep -cvxF "${ENDING:-none}" || true)"
    if [ "$NJOBS" -ge 19 ]; then
        echo "submit.sh: $NJOBS of 20 jobs in Slurm; the last slot is kept for a grid trap's refill job (--use-reserve)" >&2
        exit 3
    fi
fi

if [ -z "$PARTITION" ]; then
    if [ "$DEV" -eq 1 ]; then PARTITION=dev; else PARTITION=8gpus; fi
fi
[ "$PARTITION" = dev ] && [ "$TIME" = "2-00:00:00" ] && TIME="04:00:00"
[ -n "$NAME" ] || NAME="$(basename "$1" | sed 's/\.[a-z]*$//')"
[ "$NAME" = python ] || [ "$NAME" = bash ] && [ $# -gt 1 ] && NAME="$(basename "$2" | sed 's/\.[a-z]*$//')"

mkdir -p "$REPO/logs/nchc"
CMD="$(printf '%q ' "$@")"
RUN="$CMD"
if [ "$TERMWAIT" -eq 1 ]; then
    # the command in the background; on TERM the shell forwards it and waits (bash runs a trap
    # only after a foreground child returns; Slurm signals the batch shell only, and a batch
    # shell killed by TERM ends the step at once)
    RUN="$CMD &
CHILD=\$!
trap 'echo \"[\$(date -Is)] SIGTERM: forwarded to the command, waiting\"; kill -TERM \$CHILD; wait \$CHILD' TERM
RC=0; wait \$CHILD || RC=\$?
[ \$RC -eq 0 ] || exit \$RC"
fi

# shellcheck disable=SC2086
DIRTY="$(git -C "$REPO" status --porcelain --untracked-files=all -- $CODE_PATHS)"
if [ -n "$COMMIT_ARG" ]; then
    [ "$LIVE" -eq 0 ] || { echo "--commit and --live exclude each other" >&2; exit 2; }
    COMMIT="$(git -C "$REPO" rev-parse --verify "$COMMIT_ARG^{commit}")"
elif [ "$LIVE" -eq 0 ] && [ -n "$DIRTY" ]; then
    echo "submit.sh: code tree is dirty, running from the live tree (commit first for a snapshot):" >&2
    echo "$DIRTY" | head -5 >&2
    LIVE=1
fi
if [ "$LIVE" -eq 0 ]; then
    WORK="/work/$USER/castor"
    SNAP="$WORK/tmp/jobs/$NAME-$(date +%Y%m%d-%H%M%S)-$$"
    COMMIT="${COMMIT:-$(git -C "$REPO" rev-parse HEAD)}"
    if [ "$DRY" -eq 0 ]; then
        mkdir -p "$WORK/tmp/jobs"
        git -C "$REPO" worktree add -q --detach "$SNAP" "$COMMIT"
        rm -rf "$SNAP/data" && ln -s "$REPO/data" "$SNAP/data"
    fi
    RUNDIR="$SNAP"
    SETUP="export PYTHONPATH=$SNAP\${PYTHONPATH:+:\$PYTHONPATH} UV_PROJECT=$REPO CASTOR_SNAPSHOT=$SNAP
trap 'cd $REPO && git worktree remove --force $SNAP && git worktree prune' EXIT
echo \"git ${COMMIT:0:9} (snapshot $SNAP)\""
else
    RUNDIR="$REPO"
    SETUP="echo \"git \$(git rev-parse --short HEAD)\$(git diff --quiet || echo -dirty) (live tree)\""
fi

SCRIPT="#!/bin/bash
#SBATCH --account=${NCHC_ACCOUNT:-acd115198}
#SBATCH --partition=$PARTITION
#SBATCH --job-name=$NAME
#SBATCH --nodes=1
#SBATCH --gpus-per-node=$GPUS
#SBATCH --cpus-per-task=$((GPUS * 12))
#SBATCH --mem=$((GPUS * 200))G
#SBATCH --time=$TIME
#SBATCH --output=$REPO/logs/nchc/%x-%j.out
#SBATCH --chdir=$REPO
${EXCLUDE:+#SBATCH --exclude=$EXCLUDE}
${AFTER:+#SBATCH --dependency=afterany:$AFTER}

set -eo pipefail
cd $RUNDIR
source shell/nchc/env.sh
echo \"[\$(date -Is)] \$SLURM_JOB_ID on \$(hostname): \$CUDA_VISIBLE_DEVICES\"
$SETUP
echo \"cmd: $CMD\"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
$RUN
echo \"[\$(date -Is)] done\"
"

if [ "$DRY" -eq 1 ]; then
    printf '%s' "$SCRIPT"
else
    printf '%s' "$SCRIPT" | sbatch
    echo "log: $REPO/logs/nchc/$NAME-<jobid>.out"
fi
