#!/bin/bash
# Run the perplexity and OLMES-10 instruments over a list of checkpoints, one shard per GPU.
# List lines (TSV): <label> <hf-repo|local-dir> <revision|-> <kind: hf|olmo-ckpt> [olmo model.pt]
#   hf         released hf_olmo checkpoint: ppl via --hf REPO@REV, OLMES via the HF repo/revision
#   olmo-ckpt  our run: ppl via --olmo-ckpt <model.pt>; OLMES via <local hf_olmo dir> (col 2)
# Outputs: <out-root>/<label>/ppl.json and <out-root>/<label>/olmes/ (+ provenance.json).
# Skip-if-done: ppl.json with a git_commit, olmes/provenance.json.
#   bash shell/olmo/eval_queue.sh <list.tsv> <out-root> <config.yaml>
set -uo pipefail
LIST=$1; OUT=$2; CONFIG=$3
REPO="$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"
# Allocated GPUs: Slurm's CUDA_VISIBLE_DEVICES (nvidia-smi lists the whole node without cgroups).
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then GPUS=($(tr ',' ' ' <<< "$CUDA_VISIBLE_DEVICES"))
else mapfile -t GPUS < <(seq 0 $(($(nvidia-smi -L | wc -l) - 1))); fi
NGPU=${#GPUS[@]}
run_one() {
    local label=$1 model=$2 rev=$3 kind=$4 ckpt=${5:-}
    local d="$OUT/$label"; mkdir -p "$d"
    if ! grep -q '"git_commit"' "$d/ppl.json" 2>/dev/null; then
        if [ "$kind" = hf ]; then src=(--hf "$model@$rev"); else src=(--olmo-ckpt "$ckpt"); fi
        bash "$REPO/shell/olmo/eval_ppl.sh" "$CONFIG" "${src[@]}" --out "$d/ppl.json" \
            > "$d/ppl.log" 2>&1 || echo "FAIL ppl $label"
    fi
    if [ ! -f "$d/olmes/provenance.json" ]; then
        for try in 1 2 3; do  # transient HF Hub errors while loading task data
            rm -rf "$d/olmes"
            bash "$REPO/shell/olmes/olmes10.sh" "$model" "$rev" "$d/olmes" > "$d/olmes.log" 2>&1 && break
            echo "retry $try olmes $label"; sleep 30
        done
        [ -f "$d/olmes/provenance.json" ] || echo "FAIL olmes $label"
    fi
    echo "done $label"
}
for g in $(seq 0 $((NGPU - 1))); do
    (
        export CUDA_VISIBLE_DEVICES=${GPUS[$g]}
        awk -v g="$g" -v n="$NGPU" 'NF && $1 !~ /^#/ && (NR - 1) % n == g' "$LIST" |
        while IFS=$'\t' read -r label model rev kind ckpt; do
            run_one "$label" "$model" "$rev" "$kind" "$ckpt"
        done
    ) &
done
wait
echo "queue finished"
