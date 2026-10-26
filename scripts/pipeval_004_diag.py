"""Diagnostics for the 004 validation (not pre-registered; run after the gated result).

    uv run --frozen python scripts/pipeval_004_diag.py --out runs/posttrain/004/diagnostics.json
    # 010: the same --model KEY=MODEL_ID/PROMPT_ID overrides as scripts/pipeval_004.py

Reads <runs>/<model-id>/<prompt-id>/judge_v1/labels.jsonl, with the same checks as
scripts/pipeval_004.py (one lineage for our twins, one instrument for all models).

On harmful prompts, pre-registered metric (WildGuard refusal call, `wg_vulnerable_incl`):
  - split-half reliability of each model's per-prompt P(non-refusal): samples 0-9 vs 10-19,
    Pearson, Spearman-Brown stepped up to K = 20 (the ceiling any between-model r can reach);
  - the full matrix of between-model Pearson r over the 7 models;
  - r_pipe and r_seed per panel source;
  - the symmetric check with ours seed 2 in place of seed 1.
"""

import argparse
import json
from pathlib import Path

from castor import evalset, pipeval, provenance

FIELD = "wg_vulnerable_incl"


def split_half(rows: list[dict], ids: list[str], k: int = 20) -> dict:
    halves = [pipeval.per_prompt(pipeval.refusals_from_labels(
        [r for r in rows if 0 <= r["sample"] < k and (r["sample"] < k // 2) == first], FIELD))
        for first in (True, False)]
    r = pipeval.pearson(pipeval.vector(halves[0], ids), pipeval.vector(halves[1], ids))
    return {"r_half": r, "spearman_brown": 2 * r / (1 + r)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    pipeval.add_model_arg(ap)
    a = ap.parse_args()
    models = pipeval.model_map(pipeval.MODELS, a.model)
    commit = provenance.git_commit()
    panel = evalset.load_eval()
    instrument = pipeval.check_instrument(
        models, json.loads(evalset.EVAL_MANIFEST.read_text())["eval"]["sha256"])
    harmful = [r["id"] for r in panel if r["is_harmful"]]
    rows = {k: pipeval.load_rows(v) for k, v in models.items()}
    pp = {k: pipeval.per_prompt(pipeval.refusals_from_labels(v, FIELD)) for k, v in rows.items()}
    vec = {k: pipeval.vector(v, harmful) for k, v in pp.items()}

    res = {"reliability": {k: split_half(rows[k], harmful) for k in models},
           "r_matrix": {f"{x}|{y}": pipeval.pearson(vec[x], vec[y])
                        for i, x in enumerate(models) for y in list(models)[i + 1:]},
           "by_source": {}}
    for s in sorted({r["source"] for r in panel if r["is_harmful"]}):
        ids = [r["id"] for r in panel if r["is_harmful"] and r["source"] == s]
        v = {k: pipeval.vector(pp[k], ids) for k in ("sft1", "sft2", "sft_ai2",
                                                    "dpo1", "dpo2", "dpo_ai2")}
        res["by_source"][s] = {
            "n": len(ids),
            "dpo_r_pipe": pipeval.pearson(v["dpo1"], v["dpo_ai2"]),
            "dpo_r_seed": pipeval.pearson(v["dpo1"], v["dpo2"]),
            "sft_r_pipe": pipeval.pearson(v["sft1"], v["sft_ai2"]),
            "sft_r_seed": pipeval.pearson(v["sft1"], v["sft2"])}
    res["symmetric_seed2"] = {
        "dpo_r_pipe": pipeval.pearson(vec["dpo2"], vec["dpo_ai2"]),
        "sft_r_pipe": pipeval.pearson(vec["sft2"], vec["sft_ai2"])}
    res |= {"field": FIELD, "n_harmful": len(harmful),
            "models": {k: "/".join(v) for k, v in models.items()}, "instrument": instrument,
            "git_commit": commit, "command": provenance.command()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
