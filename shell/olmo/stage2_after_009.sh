#!/bin/bash
# Run as one job after both stage-2 chains have ended (2 GPUs):
#   1. both runs must be at their final step (ckpt/step69369-unsharded/model.pt); else exit 3;
#   2. the pretraining instruments at step 69,369 (eval_ours.sh: to_hf copy, perplexity, OLMES-10);
#   3. convert each hf_olmo copy with the released twins' converter and register it in
#      data/manifests/castor_twins_1B.json (scripts/convert_castor_ckpt.py);
#   4. validate each conversion against the reference OLMo code (validate_conversion.py thresholds).
#   bash shell/olmo/stage2_after_009.sh
set -euo pipefail
R=/work/harrychang08/castor/runs/003
HF=$R/eval/ours-1B/hf
STEP=69369
declare -A INIT=([castor-init-default-data-aux2]=6198 [castor-init-aux2-data-default]=4)
declare -A DATA=([castor-init-default-data-aux2]=4 [castor-init-aux2-data-default]=6198)
declare -A CELL=([castor-init-default-data-aux2]=init-default-data-aux2
                 [castor-init-aux2-data-default]=init-aux2-data-default)
RUNS=(castor-init-default-data-aux2 castor-init-aux2-data-default)
for run in "${RUNS[@]}"; do
    [ -f "$R/$run/ckpt/step$STEP-unsharded/model.pt" ] \
        || { echo "stage2_after_009: $run has no step $STEP checkpoint" >&2; exit 3; }
done
bash shell/olmo/eval_ours.sh 1B $STEP "${RUNS[@]}"
for run in "${RUNS[@]}"; do  # eval_queue.sh reports failures but exits 0
    d=$R/eval/ours-1B/${run}__step$STEP
    grep -q '"git_commit"' "$d/ppl.json" && [ -f "$d/olmes/provenance.json" ] \
        || { echo "stage2_after_009: evaluation of $run incomplete ($d)" >&2; exit 4; }
done
for run in "${RUNS[@]}"; do
    uv run --frozen python scripts/convert_castor_ckpt.py \
        --ref "castor/DataDecide-dclm-baseline-1B@${CELL[$run]}" --hf-olmo "$HF/$run-step$STEP" \
        --run "$run" --step $STEP --init-seed "${INIT[$run]}" --data-seed "${DATA[$run]}" \
        --register data/manifests/castor_twins_1B.json
    PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/work/harrychang08/castor/third_party/OLMo" \
        uv run --frozen python scripts/validate_conversion.py \
        --model castor/DataDecide-dclm-baseline-1B --revision "${CELL[$run]}" \
        --source-dir "$HF/$run-step$STEP" --out "/work/harrychang08/castor/runs/009/validate/${CELL[$run]}.json"
done
echo "stage2_after_009: done"
