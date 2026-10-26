#!/bin/bash
# Run as one job after R2's pretraining (1 GPU); after_011.sh with RUN, CELL and the validate path
# changed (035). The model is
# stage2_after_009.sh, for one run:
#   1. the run must be at its final step (ckpt/step69369-unsharded/model.pt); else exit 3;
#   2. the pretraining instruments at step 69,369 (eval_ours.sh: to_hf copy, perplexity, OLMES-10);
#   3. convert the hf_olmo copy with the released twins' converter and register it in the live
#      tree's data/manifests/castor_twins_1B.json (absolute path: a relative one would land in the
#      job's snapshot worktree) (scripts/convert_castor_ckpt.py);
#   4. validate the conversion against the reference OLMo code (validate_conversion.py thresholds).
#   bash shell/olmo/after_035.sh
set -euo pipefail
R=/work/harrychang08/castor/runs/003
HF=$R/eval/ours-1B/hf
STEP=69369
RUN=castor-dclm-1B-i6198-d6198-r2
CELL=replicate2-i6198-d6198
MANIFEST=/home/harrychang08/castor/data/manifests/castor_twins_1B.json
[ -f "$R/$RUN/ckpt/step$STEP-unsharded/model.pt" ] \
    || { echo "after_035: $RUN has no step $STEP checkpoint" >&2; exit 3; }
bash shell/olmo/eval_ours.sh 1B $STEP "$RUN"
d=$R/eval/ours-1B/${RUN}__step$STEP  # eval_queue.sh reports failures but exits 0
grep -q '"git_commit"' "$d/ppl.json" && [ -f "$d/olmes/provenance.json" ] \
    || { echo "after_035: evaluation of $RUN incomplete ($d)" >&2; exit 4; }
uv run --frozen python scripts/convert_castor_ckpt.py \
    --ref "castor/DataDecide-dclm-baseline-1B@$CELL" --hf-olmo "$HF/$RUN-step$STEP" \
    --run "$RUN" --step $STEP --init-seed 6198 --data-seed 6198 --register "$MANIFEST"
PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/work/harrychang08/castor/third_party/OLMo" \
    uv run --frozen python scripts/validate_conversion.py \
    --model castor/DataDecide-dclm-baseline-1B --revision "$CELL" \
    --source-dir "$HF/$RUN-step$STEP" --out "/work/harrychang08/castor/runs/035/validate/$CELL.json"
echo "after_035: done"
