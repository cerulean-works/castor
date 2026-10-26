"""The 004 validation table: r_pipe, r_seed and per-source refusal rates (castor.pipeval).

    uv run --frozen python scripts/pipeval_004.py --out runs/posttrain/004/validation.json
    # 010 (same statistics, ours from the ingredient-3 parent, released labels from 004):
    uv run --frozen python scripts/pipeval_004.py --out $PWD/runs/posttrain/010/validation.json \
        --model sft1=olmo2-1b-ingr3-sft-ours-seed1/010 --model sft2=olmo2-1b-ingr3-sft-ours-seed2/010 \
        --model dpo1=olmo2-1b-ingr3-dpo-ours-seed1/010 --model dpo2=olmo2-1b-ingr3-dpo-ours-seed2/010

Reads <runs>/<model-id>/<prompt-id>/judge_v1/labels.jsonl for our SFT/DPO
seeds 1 and 2 and the released SFT / DPO / Instruct, sampled responses only (greedy excluded).
Prompt id 004 unless a --model override names another (pipeval.model_map); our four twins must
share one lineage and prompt run, and every model's labels one instrument (pipeval.check_*).
Pre-registered metric: WildGuard's own refusal call on every response (`wg_vulnerable_incl`;
a parse failure is excluded). Secondary (reported, not gated): refusal among coherent
responses (`wg_vulnerable`; incoherent and judge errors excluded).
Gated: DPO (ours seed 1 vs released DPO). Reported, not gated: SFT vs released SFT; Instruct
refusal rates for context.
"""

import argparse
import json
from pathlib import Path

from castor import evalset, pipeval, provenance


def load(model: tuple[str, str], field: str) -> dict:
    return pipeval.per_prompt(pipeval.refusals_from_labels(pipeval.load_rows(model), field))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    pipeval.add_model_arg(ap)
    a = ap.parse_args()
    models = pipeval.model_map(pipeval.MODELS, a.model)
    commit = provenance.git_commit()
    panel = evalset.load_eval()
    eval_manifest = json.loads(evalset.EVAL_MANIFEST.read_text())["eval"]
    instrument = pipeval.check_instrument(models, eval_manifest["sha256"])
    res = {}
    for tag, field in (("", "wg_vulnerable_incl"), ("coherent_", "wg_vulnerable")):
        pp = {k: load(v, field) for k, v in models.items()}
        res[f"{tag}dpo" + ("_gated" if not tag else "")] = pipeval.validate(
            pp["dpo1"], pp["dpo2"], pp["dpo_ai2"], panel, a.n_boot)
        res[f"{tag}sft_reported"] = pipeval.validate(
            pp["sft1"], pp["sft2"], pp["sft_ai2"], panel, a.n_boot)
        res[f"{tag}instruct_refusal_by_source"] = {
            s: pipeval.refusal_rate(pp["instruct_ai2"], [r["id"] for r in panel if r["source"] == s])
            for s in sorted({r["source"] for r in panel})}
    res.update({
        "metric": "wg_vulnerable_incl (pre-registered); coherent_* = wg_vulnerable (D015, secondary)",
        "models": {k: "/".join(v) for k, v in models.items()}, "eval_manifest": eval_manifest,
        "instrument": instrument,
        "git_commit": commit, "command": provenance.command(),
    })
    res["verdict"] = res["dpo_gated"]["verdict"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res["dpo_gated"][k] for k in ("r_pipe", "r_seed", "diff", "verdict")}))


if __name__ == "__main__":
    main()
