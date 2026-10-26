"""080 adopted twins: the 2x2 (same / different base x same / different SFT safety set) on the
main recipe, the pre-registered prediction, the base jackknife and the per-model levels.
CPU only; reads judge_v1 labels.

    uv run --frozen python scripts/adopted_080.py --out /abs/path/runs/080 \
        [--n-boot 1000] [--seed 0] [--workers 6]

Tulu arm: 008's A0 cell (`ladder.anchor_cells`, 3 bases x seeds 1-4, labels under 008).
Swap arm: `data/manifests/grid_080.json` (variant swap080, labels under 080); the main
recipe's 6 models form the 2x2; the 6 cross-recipe queries enter only an exploratory block
(same levels on their own recipe, Tulu = 008's seeds 1-2) and the levels table.
Subset adversarial-harmful x `wg_vulnerable`, K = 20; Pearson over prompts of per-prompt P,
split-half corrected (006/008); prompt bootstrap 1,000 draws, seed 0 (008's).

Writes summary.json (every number with a `where` path, command, commit) and pairs.tsv.

--posthoc (093, D129) also writes posthoc.json: two reads added after the data, not
pre-registered: (A) the main-recipe 2x2 on each adversarial-harmful source separately
(WildJailbreak, HarmBench HumanJailbreaks), stratified prompt bootstrap; (B) 008's MZ / U / R
ladder (`gridstats.levels`) on the three recipes the swap arm covers, swap arm beside the Tulu
models at the same cells (and at 008's own cells), one prompt bootstrap.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from castor import adopted, evalset, grid, ladder, pipeval, provenance, reliability, sanityb
from castor.judge.common import read_jsonl

FIELD, SUBSET = sanityb.PRIMARY_FIELD, sanityb.PRIMARY_SUBSET
GRID_080 = grid.ROOT / "data" / "manifests" / "grid_080.json"
GRID_JSON = grid.ROOT / "runs" / "008" / "grid.json"
SIGNAL_A0 = ladder.ANCHOR_SIGNAL
MAIN = sanityb.RECIPE
ANALYZE = partial(reliability.twin_correlations, level_of=adopted.level_of,
                  levels=adopted.LEVELS)


def _matrix(args: tuple[str, list[str]]) -> tuple[np.ndarray, dict]:
    path, ids = args
    y, c = sanityb.outcome_matrix(read_jsonl(path), ids, FIELD)
    assert c["missing"] == 0, f"{path}: {c['missing']} missing label cells"
    return y, c


def models() -> dict[str, tuple[reliability.ModelInfo, str]]:
    """model id -> (080 ModelInfo, label prompt id) for both arms, every recipe."""
    out = {c.model_id: (adopted.info(MAIN, c.pretrain_seed, adopted.TULU, c.k), "008")
           for c in ladder.anchor_cells()}
    g = grid.load(GRID_080)
    assert g.variant == "swap080" and g.prompt_id == "080"
    for c in g.cells():
        out[c.model_id] = (adopted.info(c.base.recipe, c.base.pretrain_seed, adopted.SWAP, c.k),
                           "080")
    gm = json.loads(GRID_JSON.read_text())["models"]
    query_recipes = {c.base.recipe for c in g.cells()} - {MAIN}
    for m, v in gm.items():
        if v["recipe"] in query_recipes:
            out[m] = (adopted.info(v["recipe"], v["pretrain_seed"], adopted.TULU,
                                   v["posttrain_seed_k"]), "008")
    return out


def block(ys: dict, info: dict, n_boot: int, seed: int) -> dict:
    tw = ANALYZE(ys, info)
    return {"levels": tw["levels"], "reliability": tw["reliability"], "pairs": tw["pairs"],
            "point": adopted.summary(tw),
            "ci_prompts": reliability.bootstrap_prompts(ys, info, n_boot, seed,
                                                        stat=adopted.summary, analyze=ANALYZE),
            "jackknife_bases": reliability.jackknife_bases(ys, info, stat=adopted.summary,
                                                           analyze=ANALYZE)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--posthoc", action="store_true",
                    help="also write posthoc.json (093: reads added after the data)")
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (submit.sh snapshots are deleted)"
    commit, cmd, t0 = provenance.git_commit(), provenance.command(), time.time()
    ids = sanityb.subset_ids(evalset.load_eval(), SUBSET)
    mods = models()
    names = sorted(mods)
    eval_sha = json.loads(evalset.EVAL_MANIFEST.read_text())["eval"]["sha256"]
    instrument = pipeval.check_instrument({m: (m, mods[m][1]) for m in names}, eval_sha)
    paths = {m: sanityb.labels_path(m, mods[m][1]) for m in names}
    with ProcessPoolExecutor(a.workers) as ex:
        got = list(ex.map(_matrix, [(str(paths[m]), ids) for m in names]))
    ys = {m: y for m, (y, _) in zip(names, got, strict=True)}
    counts = {m: c for m, (_, c) in zip(names, got, strict=True)}
    info = {m: mods[m][0] for m in names}

    # instrument check: the Tulu A0 levels must equal runs/008/signal_a0.json's
    a0 = json.loads(SIGNAL_A0.read_text())[SUBSET]["primary"]
    main_ys = {m: ys[m] for m in names if info[m].recipe == MAIN}
    main = block(main_ys, info, a.n_boot, a.seed)
    for lvl in ("MZ", "U"):
        assert abs(main["levels"][lvl]["r_corrected"] - a0["levels"][lvl]["r_corrected"]) < 1e-12
    for m in main_ys:
        if adopted.arm(info[m]) == adopted.TULU:
            assert abs(float(np.nanmean(reliability.per_prompt_p(ys[m])))
                       - a0["per_model"][m]["mean_vulnerability"]) < 1e-12, m

    gj = json.loads(GRID_JSON.read_text())
    lv = gj[SUBSET]["primary"]["levels"]
    r_u = lv["U_r_corrected"]
    ci = main["ci_prompts"]
    est, lo, hi = (ci["r_adopted_corrected"][k] for k in ("estimate", "lo", "hi"))
    rule = adopted.reading(est, lo, r_u, main["point"]["r_swapU_corrected"])

    explore = {}
    for rec in sorted({i.recipe for i in info.values()} - {MAIN}):
        sub = {m: ys[m] for m in names if info[m].recipe == rec}
        b = block(sub, info, a.n_boot, a.seed)
        explore[rec] = {k: b[k] for k in ("levels", "ci_prompts")} | {
            "U_of_record_008": lv["per_recipe"][rec]["U"]["r_corrected"],
            "MZ_of_record_008": lv["per_recipe"][rec]["MZ"]["r_corrected"]}

    rel = {m: reliability.split_half(ys[m])["reliability"] for m in names}
    levels_tbl = {m: adopted.level_row(reliability.per_prompt_p(ys[m]), rel[m])
                  | {"arm": adopted.arm(info[m]), "recipe": info[m].recipe,
                     "pretrain_seed": info[m].pretrain_seed,
                     "posttrain_seed": info[m].posttrain_seed, "labels": str(paths[m]),
                     "label_counts": counts[m]} for m in names}

    def arm_mean(recipe, arm, key):
        v = [r[key] for r in levels_tbl.values() if r["recipe"] == recipe and r["arm"] == arm]
        return float(np.mean(v)) if v else float("nan")

    S = "summary.json"
    run = f"scripts/adopted_080.py @ {commit} ({cmd})"
    where = {"primary": f"{S} primary.estimate / lo / hi; {run}",
             "jackknife_se_adopted": f"{S} primary.jackknife_se_bases; {run}",
             "r_U_of_record_008": f"{S} primary.r_U_of_record_008 (runs/008/grid.json "
                                  f"{SUBSET}.primary.levels.U_r_corrected); {run}",
             "reading": f"{S} primary.reading; {run}",
             "levels_by_arm": f"{S} levels_by_arm.<recipe>.<arm>; {run}",
             "levels_by_model": f"{S} levels_by_model.<model>; {run}",
             "ceiling_flags": f"{S} ceiling_flags; {run}",
             "exploratory_query_recipes": f"{S} exploratory_query_recipes.<recipe>.ci_prompts; "
                                          f"{run}"}
    for k in main["ci_prompts"]:
        where[k] = (f"{S} main.ci_prompts.{k} (estimate, lo, hi; jackknife SE at "
                    f"main.jackknife_bases.{k}.se); {run}")
    summary = {
        "not_preregistered": ["adopted_minus_cross_corrected", "exploratory_query_recipes"],
        "where": where,
        "subset": SUBSET, "field": FIELD, "n_prompts": len(ids), "n_boot": a.n_boot,
        "seed": a.seed, "models": {m: {"arm": adopted.arm(info[m]), "recipe": info[m].recipe,
                                       "pretrain_seed": info[m].pretrain_seed,
                                       "posttrain_seed": info[m].posttrain_seed}
                                   for m in names},
        "primary": {
            "metric": "mean corrected adopted-twin correlation, main recipe, 24 pairs",
            "estimate": est, "lo": lo, "hi": hi,
            "n_pairs": main["levels"]["adopted"]["n_pairs"],
            "jackknife_se_bases": main["jackknife_bases"]["r_adopted_corrected"]["se"],
            "r_U_of_record_008": r_u,
            "r_U_of_record_008_ci_prompts": gj[SUBSET]["primary"]["ci_prompts"]["U_r_corrected"],
            "r_U_main_recipe_008": lv["per_recipe"][MAIN]["U"]["r_corrected"],
            "r_U_A0_signal": a0["levels"]["U"]["r_corrected"],
            "MZ_minus_U_of_record_008": lv["MZ_minus_U_corrected"],
            "reading": rule,
            "where": f"{S} primary; ci from main.ci_prompts.r_adopted_corrected"},
        "main": {k: main[k] for k in ("levels", "point", "ci_prompts", "jackknife_bases",
                                      "reliability")},
        "levels_by_model": levels_tbl,
        "levels_by_arm": {rec: {arm: {k: arm_mean(rec, arm, k)
                                      for k in ("mean_vulnerability", "reliability")}
                                for arm in (adopted.TULU, adopted.SWAP)}
                          for rec in sorted({i.recipe for i in info.values()})},
        "ceiling_flags": sorted(m for m, r in levels_tbl.items() if r["ceiling_flag"]),
        "exploratory_query_recipes": {
            "note": "not pre-registered: the same levels on each cross-recipe query's own "
                    "recipe (Tulu = 008 seeds 1-2, swap = seed 1; no swapMZ)",
            **explore},
        "instrument": instrument,
        "instrument_check": "Tulu A0 MZ / U corrected and per-model means equal "
                            "runs/008/signal_a0.json to 1e-12",
        "script": "scripts/adopted_080.py", "git_commit": commit, "command": cmd,
        "signal_a0": str(SIGNAL_A0.relative_to(grid.ROOT)),
        "grid_json": str(GRID_JSON.relative_to(grid.ROOT)),
        "grid_json_git_commit": gj["git_commit"], "seconds": time.time() - t0}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / S).write_text(json.dumps(summary, indent=1, default=float))
    with open(a.out / "pairs.tsv", "w") as f:
        f.write("recipe\tlevel\ta\tb\tr\tr_corrected\n")
        f.writelines(f"{MAIN}\t{p['level']}\t{p['a']}\t{p['b']}\t{p['r']}\t{p['r_corrected']}\n"
                     for p in main["pairs"])
    if a.posthoc:
        posthoc(a, ys, info, ids, names, commit, cmd, main)
    print(json.dumps({"primary": summary["primary"], "point": main["point"],
                      "ceiling_flags": summary["ceiling_flags"],
                      "seconds": summary["seconds"]}, indent=1, default=float))


def source_groups(ids: list[str]) -> dict[str, list[int]]:
    """Row indices of each adversarial-harmful source, in the eval set's source names."""
    src = {r["id"]: r["source"] for r in evalset.load_eval()}
    out: dict[str, list[int]] = {}
    for i, pid in enumerate(ids):
        out.setdefault(src[pid], []).append(i)
    return {k: out[k] for k in SOURCES}


SOURCES = ("wildjailbreak", "harmbench_humanjb")


def posthoc(a, ys: dict, info: dict, ids: list[str], names: list[str], commit: str, cmd: str,
            main: dict) -> None:
    P = "posthoc.json"
    run = f"scripts/adopted_080.py --posthoc @ {commit} ({cmd})"
    t0 = time.time()
    # (A) the main-recipe 2x2 per eval source; the whole subset must give summary.json's points
    main_ys = {m: ys[m] for m in names if info[m].recipe == MAIN}
    groups = source_groups(ids)
    assert sorted(i for g in groups.values() for i in g) == list(range(len(ids)))
    full = adopted.summary(ANALYZE(main_ys, info))
    for k, v in full.items():
        assert abs(v - main["point"][k]) < 1e-12, k
    split = adopted.source_split(main_ys, info, groups, a.n_boot, a.seed)
    # (B) 008's ladder on the swap arm's three recipes
    arms = adopted.ladder_arms(info)
    assert [len(v) for v in arms.values()] == [12, 12, 18]
    lad = adopted.ladder(ys, arms, a.n_boot, a.seed)
    gj = json.loads(GRID_JSON.read_text())[SUBSET]["primary"]["levels"]["per_recipe"]
    ci = lad["ci_prompts"]
    for q in {i.recipe for i in arms["tulu_grid008"].values()}:  # within-recipe U and MZ = 008's
        for lvl in ("U", "MZ"):
            assert abs(ci[f"tulu_grid008.{q}.{lvl}_r_corrected"]["estimate"]
                       - gj[q][lvl]["r_corrected"]) < 1e-12, (q, lvl)
    for lvl, k in (("swapU", "U"), ("swapMZ", "MZ")):  # the swap arm's main recipe = 080's 2x2
        assert abs(ci[f"swap.{MAIN}.{k}_r_corrected"]["estimate"]
                   - main["point"][f"r_{lvl}_corrected"]) < 1e-12, lvl
    out = {
        "added_after_data": True,
        "note": "093 (D129): two exploratory reads added after 080's data; no prediction, no "
                "rule; none moves a registered number. Corrected Pearson over prompts of "
                "per-prompt P (wg_vulnerable, K = 20), split-half reliabilities recomputed on "
                "every draw; percentile 95 % intervals.",
        "A_source_split": {
            "what": "080's main-recipe levels (castor.adopted.LEVELS, the same 18 models and pairs "
                    "as summary.json main) on each adversarial-harmful source separately; "
                    "reliabilities within the source; prompts resampled within each source "
                    "(stratified, one draw for both sources); '<b>_minus_<a>.' = b - a",
            "sources": list(SOURCES), "n_boot": a.n_boot, "seed": a.seed, **split,
            "where": f"{P} A_source_split.ci_prompts.<source>.<key> (estimate, lo, hi); {run}"},
        "B_ladder": {
            "what": "008's per-recipe MZ / U / R and their means over recipes (castor.gridstats."
                    "levels) on the three recipes the swap arm covers, each arm on its own "
                    "models; MZ exists on the main recipe only in swap and tulu_matched; "
                    "seed_over_recipe = (MZ - U) / (U - R), means over recipes, NaN on a draw "
                    "with U - R <= 0; one prompt bootstrap, the same rows for every arm; "
                    "'<arm>_minus_tulu_matched.' = arm - tulu_matched",
            "n_boot": a.n_boot, "seed": a.seed, **lad,
            "checks": "tulu_grid008 per-recipe U and MZ equal runs/008/grid.json per_recipe to "
                      "1e-12; swap main-recipe U and MZ equal summary.json main.point to 1e-12",
            "where": f"{P} B_ladder.ci_prompts.<arm>.<key> (estimate, lo, hi); {run}"},
        "script": "scripts/adopted_080.py --posthoc", "git_commit": commit, "command": cmd,
        "seconds": time.time() - t0}
    (a.out / P).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
