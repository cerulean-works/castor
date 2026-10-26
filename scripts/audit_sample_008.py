"""Draw 008's recipe-stratified supplement to the human audit and write blinded labelling sheets.

    uv run --frozen python scripts/audit_sample_008.py --manifest data/manifests/grid_008.json \
        --n 400 --seed 2008 --annotators harry coauthor --out /work/harrychang08/castor/runs/008/audit

Population: every judge_v1 row (T = 1 samples and greedy) of the 150 grid models (post-training
seeds 1 and 2 of every base, the main recipe from stage A0) on the two adversarial-harmful
sources (WildJailbreak, HarmBench HumanJailbreaks). Design: `castor.audit` unchanged (40 % judge
disagreements, PARTIAL / INVALID x3, simple random within strata), with the stratum's first key
= the recipe instead of the model, so the budget spreads equally over the 25 recipes x 2 sources
(8 items per cell). Writes, under --out (on /work: sheets hold harmful text):
sheet_<annotator>.csv, key.jsonl (as audit_sample.py, plus `model_id` = the real model and
`recipe`), manifest.json. Score with scripts/audit_score.py.

--recipes R [R ...] keeps only those recipes' models (080's audit: the main recipe's six swap
models, `--manifest data/manifests/grid_080.json --recipes dclm-baseline --n 200 --seed 2080`).
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from castor import audit, grid, provenance, sanityb
from castor.judge.common import read_jsonl, write_jsonl

PANEL = Path(__file__).resolve().parent.parent / "data" / "panel" / "panel_v1.jsonl"
GRID_SEEDS = (1, 2)  # as scripts/analyze_008.py
SOURCES = ("wildjailbreak", "harmbench_humanjb")
KEY_FIELDS = ("item_id", "model_id", "recipe", "model", "prompt_id", "sample", "source", "stratum",
              "pool", "weight", "status", "flag", "coherent", "coherence_rule", "llm_invalid",
              "wg_label", "wg_partial", "q3g_label", "q3g_partial", "refusal_agree")


def grid_models(g: grid.Grid, runs: Path) -> dict[str, tuple[str, Path]]:
    """model id -> (recipe, its 008 run dir) for the cells with a GRID_SEED."""
    return {c.model_id: (c.base.recipe, c.gen_dir(runs)) for c in g.cells() if c.k in GRID_SEEDS}


def generation_files(models: dict[str, tuple[str, Path]]) -> dict[str, list[str]]:
    """model id -> the generation files the draw reads (its text join, `audit.read_generations`)."""
    return {mid: [str(run / f) for f in audit.GEN_FILES] for mid, (_, run) in models.items()}


def population(models: dict[str, tuple[str, Path]], adv_ids: set[str], source: dict) -> list[dict]:
    """Rows on the adversarial-harmful prompts of SOURCES, `model` replaced by the recipe (the
    stratum key) and the real model kept as `model_id`."""
    rows = []
    for mid, (recipe, run) in sorted(models.items()):
        for r in read_jsonl(run / "judge_v1" / "labels.jsonl"):
            if r["prompt_id"] in adv_ids and source[r["prompt_id"]] in SOURCES:
                rows.append({**r, "model_id": mid, "model": recipe, "recipe": recipe,
                             "source": source[r["prompt_id"]],
                             "flag": bool(r["wg_partial"] or r["llm_invalid"])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2008)
    ap.add_argument("--annotators", nargs="+", default=["harry", "coauthor"])
    ap.add_argument("--runs", type=Path, default=grid.RUNS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recipes", nargs="+")
    a = ap.parse_args()
    commit, cmd = provenance.git_commit(), provenance.command()

    panel = list(read_jsonl(PANEL))
    source = {r["id"]: r["source"] for r in panel}
    adv = set(sanityb.subset_ids(panel, sanityb.PRIMARY_SUBSET))
    models = grid_models(grid.load(a.manifest), a.runs)
    if a.recipes:
        models = {m: v for m, v in models.items() if v[0] in a.recipes}
        assert {v[0] for v in models.values()} == set(a.recipes), "a recipe has no model"
    rows = population(models, adv, source)
    drawn = audit.draw(rows, a.n, seed=a.seed)
    rng = random.Random(a.seed + 1)
    rng.shuffle(drawn)
    ids = rng.sample(range(16 ** 6), len(drawn))
    need = {(d["model_id"], d["prompt_id"], d["sample"]) for d in drawn}
    files = generation_files({m: models[m] for m in sorted({d["model_id"] for d in drawn})})
    gens = audit.read_generations(files, need)
    a.out.mkdir(parents=True, exist_ok=True)
    for d, i in zip(drawn, ids):
        d["item_id"] = f"{i:06x}"
    write_jsonl(a.out / "key.jsonl", [{k: d.get(k) for k in KEY_FIELDS} for d in drawn])
    for ann in a.annotators:
        audit.write_sheet(a.out / f"sheet_{ann}.csv", [
            (d["item_id"], gens[(d["model_id"], d["prompt_id"], d["sample"])]["prompt"],
             gens[(d["model_id"], d["prompt_id"], d["sample"])]["response"]) for d in drawn])
    manifest = {
        "n": len(drawn), "seed": a.seed, "frac_disagree": audit.FRAC_DISAGREE,
        "flag_weight": audit.FLAG_WEIGHT, "manifest": str(a.manifest), "sources": SOURCES,
        "n_models": len(models), "recipes": a.recipes, "annotators": a.annotators, "population": len(rows),
        "pool_counts": dict(Counter(d["pool"] for d in drawn)),
        "status_counts": dict(Counter(d["status"] for d in drawn)),
        "flag_count": sum(bool(d["flag"]) for d in drawn),
        "per_recipe_source": {f"{r}|{s}": n for (r, s), n in sorted(
            Counter((d["recipe"], d["source"]) for d in drawn).items())},
        "per_model": dict(Counter(d["model_id"] for d in drawn)),
        "generations_by_model": files,
        "text_key": ["model_id", "prompt_id", "sample"],
        "git_commit": commit, "command": cmd, "script": "scripts/audit_sample_008.py",
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("n", "n_models", "population", "pool_counts",
                                               "status_counts", "flag_count")}))


if __name__ == "__main__":
    main()
