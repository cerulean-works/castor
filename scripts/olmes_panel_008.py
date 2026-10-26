"""Descriptive panel: recipe mean vulnerability vs OLMES-10.

    uv run --frozen python scripts/olmes_panel_008.py --grid runs/008/grid.json \
        --olmes /work/harrychang08/castor/data/datadecide_evals/datadecide_1B_final_table.csv \
        --out runs/008/olmes_panel.json

Per base (75): mean vulnerability on adversarial-harmful prompts (judge_v1 primary outcome, raw)
averaged over its post-training seeds 1 and 2 (`grid.json` per_model), and the base's released
OLMES-10 macro average (DataDecide-eval-results@9919b5a0, column default / large aux 2 / large aux
3). Per recipe (25): the mean of both. Pearson and Spearman r across recipes and across bases.
Descriptive only: no prediction, no rule.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from castor import gridstats, provenance

SEED_COLUMN = {"default": "default", "large-aux-2": "large aux 2", "large-aux-3": "large aux 3"}


def olmes_table(path: Path) -> dict[tuple[str, str], float]:
    """(recipe slug, pretraining seed) -> OLMES-10 macro average."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["task"] == "olmes_10_macro_avg":
                slug = gridstats.OLMES_NAMES[row["data"]]
                for seed, col in SEED_COLUMN.items():
                    out[(slug, seed)] = float(row[col])
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def panel(per_model: dict, models: dict, olmes: dict) -> dict:
    """per_model: model id -> {mean_vulnerability}; models: model id -> {recipe, pretrain_seed}."""
    base_v = defaultdict(list)
    for mid, st in per_model.items():
        if mid not in models:  # the released reference is not a grid model
            continue
        m = models[mid]
        base_v[(m["recipe"], m["pretrain_seed"])].append(st["mean_vulnerability"])
    bases = sorted(base_v)
    bv = np.array([np.mean(base_v[b]) for b in bases])
    bo = np.array([olmes[b] for b in bases])
    recipes = sorted({b[0] for b in bases})
    rv = np.array([bv[[i for i, b in enumerate(bases) if b[0] == r]].mean() for r in recipes])
    ro = np.array([bo[[i for i, b in enumerate(bases) if b[0] == r]].mean() for r in recipes])
    return {
        "per_recipe": {r: {"mean_vulnerability": float(v), "olmes10": float(o)}
                       for r, v, o in zip(recipes, rv, ro)},
        "per_base": {f"{r}|{s}": {"mean_vulnerability": float(v), "olmes10": float(o)}
                     for (r, s), v, o in zip(bases, bv, bo)},
        "recipes": {"n": len(recipes), "pearson": float(np.corrcoef(rv, ro)[0, 1]),
                    "spearman": spearman(rv, ro)},
        "bases": {"n": len(bases), "pearson": float(np.corrcoef(bv, bo)[0, 1]),
                  "spearman": spearman(bv, bo)},
        "vulnerability_sd_recipe_means": float(rv.std(ddof=1)),
        "olmes10_sd_recipe_means": float(ro.std(ddof=1))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--olmes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    g = json.loads(a.grid.read_text())
    res = panel(g["adv_harmful"]["primary"]["per_model"], g["models"], olmes_table(a.olmes))
    res |= {"grid": str(a.grid), "olmes": str(a.olmes), "outcome": "adv_harmful / primary (raw)",
            "git_commit": provenance.git_commit(), "command": provenance.command(),
            "script": "scripts/olmes_panel_008.py"}
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("recipes", "bases")}, indent=1))


if __name__ == "__main__":
    main()
