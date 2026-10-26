"""The grid's nested variance shares on the audit-corrected levels, beside the raw shares. Pure
CPU; reads runs/008/grid.json and runs/008/audit_correct.json only.

    uv run --frozen python scripts/corrected_shares_033.py --out /abs/path/runs/033/corrected_shares.json

Estimator: castor.gridstats.nested_variance on per-model mean vulnerability (adversarial
harmful, primary); interval of each share: the recipe bootstrap of castor.seedpower.components
with no sampling term (same three shares; 1,000 resamples of the 25 recipes, seed 0).
Variants: raw (asserted equal to grid.json's shares and runs/045's recipe-share interval),
per_source (of record) and per_group_source (descriptive only). The audit's own uncertainty in
the factors is not propagated (audit_correct.json keeps no per-draw levels); the interval is
over recipes only.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from castor import gridstats, provenance, seedpower
from castor.reliability import ModelInfo

VARIANTS = ("raw", "per_source", "per_group_source")


def level_array(models: dict, level: dict) -> np.ndarray:
    """y[a, b, n] in recipe / pretraining-seed / post-training-seed order (as 045's level_array)."""
    recipes = sorted({v["recipe"] for v in models.values()})
    y = []
    for r in recipes:
        bs = sorted({v["pretrain_seed"] for v in models.values() if v["recipe"] == r})
        y.append([[level[m] for m, v in sorted(models.items(), key=lambda kv: kv[1]["posttrain_seed_k"])
                   if v["recipe"] == r and v["pretrain_seed"] == b] for b in bs])
    return np.array(y, dtype=float)


def shares_block(models: dict, level: dict, n_boot: int, seed: int) -> dict:
    infos = {m: ModelInfo(v["recipe"], v["pretrain_seed"], str(v["posttrain_seed_k"]))
             for m, v in models.items()}
    nv = gridstats.nested_variance(level, infos)
    y = level_array(models, level)
    comp = seedpower.components(y, 0.0)
    keys = {"recipe": "recipe", "pretrain_seed": "pretrain_seed", "posttrain_seed": "posttrain_true"}
    for k, ck in keys.items():  # the two code paths give the same point shares
        assert abs(nv["shares"][k] - comp["shares"][ck]) < 1e-12, (k, nv["shares"][k], comp["shares"][ck])
    draws = seedpower.recipe_bootstrap(lambda ix: seedpower.components(y[ix], 0.0), y.shape[0], n_boot, seed)
    return {"shares": {k: float(v) for k, v in nv["shares"].items()},
            "shares_ci": {k: seedpower.ci([d["shares"][ck] for d in draws], nv["shares"][k])
                          for k, ck in keys.items()},
            "components": {k: float(v) for k, v in nv["components"].items()},
            "mean_level": float(y.mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=Path, default=Path("runs/008/grid.json"))
    ap.add_argument("--audit", type=Path, default=Path("runs/008/audit_correct.json"))
    ap.add_argument("--comp045", type=Path, default=Path("runs/045/components.json"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    commit, t0 = provenance.git_commit(), time.time()
    g, au = json.loads(a.grid.read_text()), json.loads(a.audit.read_text())
    models = g["models"]
    pm = g["adv_harmful"]["primary"]["per_model"]
    levels = {"raw": {m: pm[m]["mean_vulnerability"] for m in models}}
    for v in VARIANTS[1:]:
        levels[v] = {m: au["levels"][v]["model"][m]["point"] for m in models}
    for m in models:  # the audit's raw block is grid.json's
        assert au["levels"]["raw"]["model"][m] == levels["raw"][m], m
    out = {v: shares_block(models, levels[v], a.n_boot, a.seed) for v in VARIANTS}
    gs = g["adv_harmful"]["primary"]["variance"]["shares"]
    for k in gs:
        assert abs(out["raw"]["shares"][k] - gs[k]) < 1e-12, (k, out["raw"]["shares"][k], gs[k])
    c45 = json.loads(a.comp045.read_text())["primary"]["shares_ci"]["recipe"]
    r = out["raw"]["shares_ci"]["recipe"]
    assert abs(r["lo"] - c45["lo"]) < 1e-12 and abs(r["hi"] - c45["hi"]) < 1e-12, (r, c45)
    assert au["levels"]["of_record"] == "per_source"
    res = {"of_record": au["levels"]["of_record"], "variants": out,
           "move_of_record_minus_raw": {k: out["per_source"]["shares"][k] - out["raw"]["shares"][k]
                                        for k in gs},
           "subset": "adv_harmful", "outcome": "primary", "n_models": len(models),
           "n_boot": a.n_boot, "seed": a.seed, "grid": str(a.grid), "audit": str(a.audit),
           "audit_git_commit": au["git_commit"], "grid_git_commit": g.get("git_commit"),
           "script": "scripts/corrected_shares_033.py", "git_commit": commit,
           "command": provenance.command(), "seconds": round(time.time() - t0, 2)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({v: {k: round(x, 4) for k, x in out[v]["shares"].items()} for v in VARIANTS}))


if __name__ == "__main__":
    main()
