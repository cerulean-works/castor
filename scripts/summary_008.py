"""Derived readings of `runs/008/grid.json` (no new statistic).

    uv run --frozen python scripts/summary_008.py --grid runs/008/grid.json \
        --out runs/008/summary.json --tsv runs/008/per_recipe.tsv

- per_recipe.tsv: the per-recipe table of the primary (adversarial-harmful, judge_v1 primary
  outcome): MZ / U / R ceiling-corrected r with prompt-bootstrap CIs (R per recipe = its models
  against every other recipe's), MZ − U corrected with its CI, U − R corrected, and the recipe's
  mean vulnerability (raw, mean of its models).
- secondary items, pre-registered for the 24 non-main recipes and post hoc for the main one:
  the per-recipe values' mean over the 24 non-main recipes beside the main recipe's value, for
  (1) the template residual and (2) each source. The pooled values are grid.json's.
- mean correlation with olmo2-1b-dpo-ai2 over the main recipe's six grid twins and
  over all 150 models (raw and corrected).
- recipe means of mean vulnerability: min, max, SD.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from castor import provenance

MAIN = "dclm-baseline"
REFERENCE = "olmo2-1b-dpo-ai2"


def recipe_vulnerability(per_model: dict, models: dict) -> dict[str, float]:
    out: dict[str, list] = {}
    for mid, st in per_model.items():
        if mid in models:  # the reference is not a grid model
            out.setdefault(models[mid]["recipe"], []).append(st["mean_vulnerability"])
    return {r: float(np.mean(v)) for r, v in sorted(out.items())}


def non_main(per_recipe: dict, key: str = "MZ_minus_U_corrected") -> dict:
    v = [d[key] for r, d in per_recipe.items() if r != MAIN]
    return {"mean_non_main": float(np.mean(v)), "n_non_main": len(v),
            "min_non_main": float(min(v)), "max_non_main": float(max(v)),
            "main": per_recipe[MAIN][key]}


def reference(per_model: dict, models: dict) -> dict:
    def block(ids):
        return {"n": len(ids), "r": float(np.mean([per_model[i]["r"] for i in ids])),
                "r_corrected": float(np.mean([per_model[i]["r_corrected"] for i in ids]))}
    main = [m for m in per_model if models[m]["recipe"] == MAIN]
    return {"main_recipe": block(main), "all": block(list(per_model)),
            "min_r_corrected": float(min(v["r_corrected"] for v in per_model.values())),
            "max_r_corrected": float(max(v["r_corrected"] for v in per_model.values()))}


def table(g: dict) -> list[dict]:
    p = g["adv_harmful"]["primary"]
    lv, ci = p["levels"]["per_recipe"], p["ci_prompts"]["per_recipe"]
    vul = recipe_vulnerability(p["per_model"], g["models"])
    rows = []
    for r in sorted(lv, key=lambda x: -lv[x]["MZ_minus_U_corrected"]):
        d, c = lv[r], ci[r]
        row = {"recipe": r}
        for t in ("MZ", "U", "R"):
            row |= {f"r{t}": d[t]["r_corrected"], f"r{t}_lo": c[f"{t}_r_corrected"]["lo"],
                    f"r{t}_hi": c[f"{t}_r_corrected"]["hi"]}
        row |= {"gap": d["MZ_minus_U_corrected"], "gap_lo": c["MZ_minus_U_corrected"]["lo"],
                "gap_hi": c["MZ_minus_U_corrected"]["hi"], "U_minus_R": d["U_minus_R_corrected"],
                "mean_vulnerability": vul[r]}
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tsv", type=Path, required=True)
    a = ap.parse_args()
    g = json.loads(a.grid.read_text())
    s, p = g["secondary_D042"], g["adv_harmful"]["primary"]
    rows = table(g)
    with open(a.tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}
                    for r in rows)
    vul = recipe_vulnerability(p["per_model"], g["models"])
    ref_pm = {m: v for m, v in s["reference_olmo2_1b_dpo"]["per_model"].items() if m != REFERENCE}
    res = {
        "grid": str(a.grid), "tsv": str(a.tsv),
        "gap_per_recipe": {"min": min(r["gap"] for r in rows), "max": max(r["gap"] for r in rows),
                           "n_gap_lo_gt_0": sum(r["gap_lo"] > 0 for r in rows)},
        "D042_non_main": {k: non_main(s[k]["levels"]["per_recipe"]) for k in
                          ("template_residual", "source_wildjailbreak", "source_harmbench_humanjb")},
        "reference": reference(ref_pm, g["models"]),
        "recipe_vulnerability": {"min": min(vul.values()), "max": max(vul.values()),
                                 "argmin": min(vul, key=vul.get), "argmax": max(vul, key=vul.get),
                                 "sd": float(np.std(list(vul.values()), ddof=1))},
        "git_commit": provenance.git_commit(), "command": provenance.command(),
        "script": "scripts/summary_008.py"}
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
