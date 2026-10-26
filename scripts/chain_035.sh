#!/bin/bash
# Continue R2's pretraining (run castor-dclm-1B-i6198-d6198-r2) from latest-unsharded as 2-day
# 8-GPU segments submitted one after another. Before every segment submission a `wallet ACD115198`
# reading is appended to notes/pitch/wallet-readings.txt; at <= 155k SU with a segment left to
# submit, nothing is submitted and the script stops for a human.
# Never uses --use-reserve. Seg k+1 (--after seg k, which submit.sh makes afterany) is submitted
# only once seg k is RUNNING, so the chain holds <= 2 of the 20 job slots. Every submission waits
# for a clean code tree (submit.sh snapshots HEAD) and refuses unless shell/olmo/pretrain.sh and
# make_config.py are byte-identical to the first replicate's commit 7b0658c. A queued segment whose
# predecessor crashed starts and resumes from latest-unsharded (afterany). No segment is submitted
# once the final checkpoint exists. Segments are numbered from 2 (seg1 reached step 7500).
#   tmux new-session -d -s castor-035 'bash scripts/chain_035.sh >> logs/nchc/035-chain.log 2>&1'
#   DRY=1 bash scripts/chain_035.sh     # wallet reading printed, seg2's sbatch script printed, nothing submitted or appended
cd /home/harrychang08/castor && source shell/nchc/env.sh
RUN=castor-dclm-1B-i6198-d6198-r2
FINAL=/work/harrychang08/castor/runs/003/$RUN/ckpt/step69369-unsharded
JOBS=logs/nchc/035-chain.jobs
WALLET_LOG=notes/pitch/wallet-readings.txt
FLOOR=155000
say() { echo "[$(date -Is)] $*"; }
state() { squeue -h -j "$1" -o %T 2>/dev/null | head -1; }      # empty once the job is gone
final() { sacct -X -n -j "$1" -o State 2>/dev/null | head -1 | tr -d ' '; }

wallet_ok() {  # $1 = k; appends a reading (unless DRY); returns 1 at or below the floor
    local su
    su=$(wallet ACD115198 2>/dev/null | grep -o 'SU_BALANCE: [-0-9.]*' | grep -o '[-0-9.]*$')
    [ -n "$su" ] || { say "seg$1: no wallet reading; stopping"; return 1; }
    [ -z "${DRY:-}" ] && echo "$(date -Iseconds) SU_BALANCE: $su (035 chain, before seg$1)" >> "$WALLET_LOG"
    say "seg$1: wallet $su SU"
    awk -v s="$su" -v f="$FLOOR" 'BEGIN { exit !(s > f) }' || { say "seg$1: wallet $su <= $FLOOR (D108f stop rule)"; return 1; }
}

JOB=""
submit_seg() {  # $1 = k, $2 = previous job id or ""; sets JOB
    local k=$1 prev=$2 after=() out rc commit
    [ -n "$prev" ] && after=(--after "$prev")
    while true; do
        if ! git diff --quiet 7b0658c HEAD -- shell/olmo/pretrain.sh shell/olmo/make_config.py; then
            say "seg$k: shell/olmo differs from 7b0658c at HEAD $(git rev-parse --short HEAD); refusing"; exit 2
        fi
        if ! commit=$(uv run --frozen python -c "from castor import provenance; print(provenance.git_commit())" 2>/dev/null); then
            say "seg$k: code tree not clean, retry in 10 min"; sleep 600; continue
        fi
        wallet_ok "$k" || exit 3
        out=$(shell/nchc/submit.sh --gpus 8 --time 2-00:00:00 --name "035-r2-seg$k" "${after[@]}" \
              ${DRY:+--dry-run} -- bash shell/olmo/pretrain.sh 1B 6198 - "$RUN" 2>&1); rc=$?
        echo "$out"
        [ -n "${DRY:-}" ] && { say "seg$k: dry run, nothing submitted"; exit 0; }
        if [ $rc -ne 0 ]; then say "seg$k: submit.sh exited $rc, retry in 10 min"; sleep 600; continue; fi
        JOB=$(grep -o 'Submitted batch job [0-9]*' <<< "$out" | grep -o '[0-9]*$')
        [ -n "$JOB" ] || { say "seg$k: no job id in submit.sh output"; exit 2; }
        echo "seg$k $JOB $commit $(date -Is)" >> "$JOBS"
        say "seg$k submitted: job $JOB, snapshot $commit${prev:+, after $prev}"
        return
    done
}

wait_running() {  # $1 = job id; returns 0 when it runs or ended well, 1 otherwise
    local n=0 s f
    while true; do
        s=$(state "$1")
        case "$s" in
            RUNNING) say "job $1 is RUNNING on $(squeue -h -j "$1" -o %N)"; return 0 ;;
            "") f=$(final "$1"); say "job $1 left the queue: $f"
                case "$f" in COMPLETED|TIMEOUT) return 0 ;; *) return 1 ;; esac ;;
            *) n=$((n + 1)); [ $((n % 12)) -eq 0 ] && say "job $1 still $s"; sleep 300 ;;
        esac
    done
}

say "re-arm chain starts (latest-unsharded -> $(readlink /work/harrychang08/castor/runs/003/$RUN/ckpt/latest-unsharded))"
prev=""
for k in 2 3 4 5 6; do
    if [ -e "$FINAL" ]; then
        say "$RUN has its final checkpoint; seg$k not needed"; break
    fi
    submit_seg "$k" "$prev"
    wait_running "$JOB" || { say "seg$k ended in a state this script does not resubmit; stopping"; exit 1; }
    prev=$JOB
done
say "chain script done"
