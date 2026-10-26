"""Every pair of the recipe grid's 150 models with its raw and ceiling-corrected per-prompt
correlation (adversarial-harmful prompts, WildGuard refusal), for the pair-distribution figure.
Pure CPU; reads the judge_v1 labels grid.json names.

    uv run --frozen python scripts/pair_dist_053.py --grid runs/008/grid.json \
        --out /abs/path/runs/053/pairs.json

The per-pair values are recomputed with the library of record (castor.gridstats: halves ->
reliabilities -> corr_matrix -> r / sqrt(rel_a rel_b), pair_masks), exactly as
scripts/analyze_008.py computes the grid's levels, and checked against grid.json before the
file is written: every model's reliability (|d| < 1e-9), the mean of the per-pair corrected r
within each recipe for MZ / U / R against levels.per_recipe (|d| < 1e-9), and the mean over all
cross-recipe pairs against levels.R_overall. runs/044/prompts.json is not used: it rounds P to 4
decimals and drops one undefined prompt, so it cannot reproduce grid.json to 1e-9.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from castor import evalset, gridstats, provenance, reliability, sanityb
from castor.judge.common import read_jsonl

SUBSET, OUTCOME, FIELD = sanityb.PRIMARY_SUBSET, "primary", sanityb.PRIMARY_FIELD
TOL = 1e-9


def load(grid_path: Path) -> tuple[dict, list[str], list[reliability.ModelInfo], dict]:
    g = json.loads(grid_path.read_text())
    panel = evalset.load_eval()
    all_ids = [r["id"] for r in panel]
    ids = sanityb.subset_ids(panel, SUBSET)
    pos = {pid: i for i, pid in enumerate(all_ids)}
    idx = np.array([pos[p] for p in ids])
    names = sorted(g["models"])
    infos = [reliability.ModelInfo(g["models"][m]["recipe"], g["models"][m]["pretrain_seed"],
                                   str(g["models"][m]["posttrain_seed_k"])) for m in names]
    ys = {}
    for m in names:
        rows = list(read_jsonl(Path(g["models"][m]["labels"])))
        y, counts = sanityb.outcome_matrix(rows, all_ids, FIELD)
        assert counts["missing"] == 0, (m, counts)
        ys[m] = y[idx]
    return g, names, infos, ys


def pairs(g: dict, names: list[str], infos: list[reliability.ModelInfo], ys: dict) -> dict:
    P, Po, Pe = gridstats.halves(ys, names)
    rel = gridstats.reliabilities(Po, Pe)
    ref = g[SUBSET][OUTCOME]
    for m, r in zip(names, rel):
        assert abs(r - ref["reliability"][m]) < TOL, (m, r, ref["reliability"][m])
    r = gridstats.corr_matrix(P)
    with np.errstate(invalid="ignore"):
        rc = r / np.sqrt(np.outer(rel, rel))
    rc[~np.outer(rel > 0, rel > 0)] = np.nan
    masks = gridstats.pair_masks(infos)
    rows = []
    for lvl in reliability.LEVELS:
        for i, j in zip(*np.nonzero(masks[lvl])):
            rows.append({"a": names[i], "b": names[j], "level": lvl,
                         "recipe_a": infos[i].recipe, "recipe_b": infos[j].recipe,
                         "r": float(r[i, j]), "r_corrected": float(rc[i, j])})
    # the checks against grid.json: per recipe (MZ, U: both models in it; R: exactly one) and R overall
    lv = ref["levels"]
    checks = {}
    for q, d in lv["per_recipe"].items():
        for lvl in reliability.LEVELS:
            sel = [x for x in rows if x["level"] == lvl
                   and ((x["recipe_a"] == q) != (x["recipe_b"] == q) if lvl == "R"
                        else x["recipe_a"] == q == x["recipe_b"])]
            assert len(sel) == d[lvl]["n_pairs"], (q, lvl, len(sel), d[lvl]["n_pairs"])
            for f in ("r", "r_corrected"):
                mine = float(np.nanmean([x[f] for x in sel]))
                assert abs(mine - d[lvl][f]) < TOL, (q, lvl, f, mine, d[lvl][f])
            checks[f"{q}:{lvl}"] = {"n_pairs": len(sel), "max_abs_diff": max(
                abs(float(np.nanmean([x[f] for x in sel])) - d[lvl][f]) for f in ("r", "r_corrected"))}
    cross = [x for x in rows if x["level"] == "R"]
    assert len(cross) == lv["R_overall"]["n_pairs"]
    for f in ("r", "r_corrected"):
        mine = float(np.nanmean([x[f] for x in cross]))
        assert abs(mine - lv["R_overall"][f]) < TOL, ("R_overall", f, mine, lv["R_overall"][f])
    n = {lvl: int(masks[lvl].sum()) for lvl in reliability.LEVELS}
    assert sum(n.values()) == len(names) * (len(names) - 1) // 2 == len(rows)
    return {"n_models": len(names), "subset": SUBSET, "outcome": OUTCOME, "field": FIELD,
            "n_prompts": int(P.shape[1]), "n_pairs": n, "correction": "r / sqrt(rel_a rel_b)",
            "levels_checked": {"per_recipe": checks,
                               "R_overall": {k: lv["R_overall"][k] for k in ("n_pairs", "r", "r_corrected")},
                               "tolerance": TOL},
            "pairs": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=Path, default=Path("runs/008/grid.json"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (submit.sh snapshots are deleted)"
    commit, cmd, t0 = provenance.git_commit(), provenance.command(), time.time()
    g, names, infos, ys = load(a.grid)
    out = pairs(g, names, infos, ys)
    out.update({"grid": str(a.grid), "grid_git_commit": g["git_commit"],
                "grid_sha256": hashlib.sha256(a.grid.read_bytes()).hexdigest(),
                "script": "scripts/pair_dist_053.py", "git_commit": commit, "command": cmd,
                "seconds": time.time() - t0})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=0, default=float))
    print(json.dumps({"n_pairs": out["n_pairs"], "seconds": out["seconds"]}))


if __name__ == "__main__":
    main()
