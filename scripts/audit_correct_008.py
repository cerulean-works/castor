"""Score 008's recipe-stratified audit supplement and correct the grid's levels. One run,
after both sheets and the adjudication sheet are complete.

    uv run --frozen python scripts/audit_correct_008.py --audit AUDIT_DIR \
        --sheets AUDIT_DIR/sheet_harry.csv AUDIT_DIR/sheet_coauthor.csv \
        --adjudicated AUDIT_DIR/sheet_adjudicate.csv [--score AUDIT_DIR/051/score.json] \
        --out /abs/runs/008/audit_correct.json
    uv run --frozen python scripts/audit_correct_008.py --audit AUDIT_DIR --placeholder \
        --out /abs/.../audit_correct_DRYRUN.json

Gold labels are exactly `scripts/audit_score.py`'s; nothing is computed unless every key item
has gold. Definitions, the heterogeneity test, the rule and the flip count: `castor/audit008.py`.
Blocks:
  factors      design-weighted precision per source (pooled over the 150 grid models), per
               DataDecide group x source, per recipe and per group, with a Rao-Wu bootstrap
               within the audit strata (`castor.auditgate`; one-item strata collapsed into their
               recipe x source cell), over every gold item (a domain estimate)
  judge_error  pooled judge rate, human rate, precision, recall, false-positive and
               false-negative mass, with intervals
  heterogeneity  permutation test over recipes (the rule) and over groups (descriptive); the
               greedy adjudication-flip count that moves the rule's p across 0.01; each
               factor's largest one-flip change
  levels       corrected mean vulnerability per model, per recipe and overall, under the
               per-source and group x source factors, with audit intervals; raw beside them;
               `of_record` = per_source unless the rule fires
  contrast     the grid's primary (adversarial-harmful, wg_vulnerable, ceiling-corrected
               MZ - U mean over recipes) on per-prompt rates x factor, with the grid's recipe
               and prompt bootstraps (factors held at their point values) and `ci_audit` (the
               audit bootstrap's factors, grid held fixed): two sources of uncertainty, reported
               side by side; the `none` variant must reproduce runs/008/grid.json bit for bit
  counts_per_recipe, counts_per_group  per label on the precision population, no bootstrap
Correlations of record stay on raw rates; the corrected contrast is quoted beside them.
--check-against FILE: every key of FILE but RUN_KEYS must be reproduced bit for bit (asserted
before --out is written, so FILE may be --out).
--amend FILE: add the count blocks to FILE without a grid or a bootstrap, every point value FILE
holds checked first.

--placeholder: no sheet is read. Gold is drawn at random from the key rows to exercise every
code path on the real strata, weights and grid. Its output is never a result.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from castor import audit, auditgate, evalset, gridstats, provenance, reliability, sanityb
from castor import audit008 as A8
from castor.judge import labels as L
from castor.judge.common import read_jsonl

ROOT = Path(__file__).resolve().parent.parent
CONTRAST_KEYS = ("MZ_minus_U_corrected", "U_minus_R_corrected", "MZ_r_corrected", "U_r_corrected",
                 "R_r_corrected")
GRID = ROOT / "runs" / "008" / "grid.json"
# keys that differ on every rerun by construction; --check-against compares every other key
RUN_KEYS = ("git_commit", "command", "seconds", "grid")
FIELD = "wg_vulnerable"
_spec = importlib.util.spec_from_file_location("audit_score", Path(__file__).parent /
                                               "audit_score.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def require(ok: bool, what) -> None:
    """A check that survives `python -O` (assert does not)."""
    if not ok:
        raise SystemExit(f"check failed: {what}")


def same(a, b) -> bool:
    """Exact equality, NaN equal to NaN, through dicts and lists (bit-for-bit reproduction)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return same(np.asarray(a).tolist(), np.asarray(b).tolist())
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if (isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating))
            and math.isnan(a) and math.isnan(b)):
        return True
    return a == b


def close(a: float, b: float, tol: float = 1e-12, rel: float = 0.0) -> bool:
    return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=rel, abs_tol=tol)


def placeholder_gold(keyrows: dict, seed: int = 0) -> tuple[dict, dict]:
    rng = random.Random(seed)
    confirm = {"wildjailbreak": 0.6, "harmbench_humanjb": 0.55}
    gold, alt = {}, {}
    for n, (i, r) in enumerate(sorted(keyrows.items())):
        if r["wg_label"] == L.REFUSAL:
            lab = L.REFUSAL if rng.random() < 0.95 else L.HARMFUL
        else:
            lab = r["wg_label"] if rng.random() < confirm[r["source"]] else L.REFUSAL
        gold[i] = {"label": lab, "partial": False}
        if n % 3 == 0:
            alt[i] = not A8.human_vulnerable(lab)
    return gold, alt


def masses(items: list[dict], label) -> dict:
    """Per label, on the precision population (no bootstrap): population items, judge-vulnerable
    items, their design-weight mass and the weighted human-vulnerable hits among them, so a
    label's precision is hits / mass and the pooled precision is the mass-weighted mean."""
    out = defaultdict(lambda: {"n": 0, "n_judge_vulnerable": 0, "w_judge_vulnerable": 0.0,
                               "w_human_vulnerable_among_judge": 0.0})
    for x in A8.population(items):
        c = out[label(x)]
        c["n"] += 1
        if A8.judge_vulnerable(x):
            c["n_judge_vulnerable"] += 1
            c["w_judge_vulnerable"] += x["weight"]
            if A8.human_vulnerable(x["gold"]):
                c["w_human_vulnerable_among_judge"] += x["weight"]
    return {k: out[k] for k in sorted(out)}


def check_against(new: dict, path: Path) -> dict:
    """Every key of the committed artifact other than RUN_KEYS is reproduced bit for bit."""
    old = json.loads(path.read_text())
    new = json.loads(json.dumps(new, default=float))  # as written
    for k in old:
        if k in RUN_KEYS:
            continue
        require(k in new and same(new[k], old[k]), ("check-against", path, k))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "git_commit": old.get("git_commit"), "keys_compared": sorted(set(old) - set(RUN_KEYS)),
            "keys_added": sorted(set(new) - set(old))}


def amend(a, items: list[dict], recipes: list[str], commit: str, cmd: str) -> None:
    """--amend: the count blocks, added to the committed artifact without touching a key it holds.

    The artifact's intervals were drawn with gold built in set order (per-process string
    hashing), so a rerun cannot reproduce them; the counts are sums over the population and need
    no draws. Every point value the artifact holds is recomputed here and must agree."""
    old = json.loads(a.amend.read_text())
    require(not old["placeholder"], "amend a scored artifact only")
    point = A8.make_stat(None, None, recipes)(items)
    for block, prefix in (("per_source", "source:"), ("per_group_source", "group_source:"),
                          ("per_recipe", "recipe:"), ("per_group", "group:")):
        for k, v in old["factors"][block].items():
            require(close(point[prefix + k], v["point"]), ("amend", block, k))
    for k, v in A8.rates(items).items():
        require(close(old["judge_error"]["counts"][k], v), ("amend", "counts", k))
    pop = A8.population(items)
    require((old["n_gold"], old["n_population"], old["n_greedy_excluded"]) ==
            (len(items), len(pop), sum(x["sample"] < 0 for x in items)), "amend: sizes")
    require(old["groups"] == {r: A8.group_of(r) for r in recipes}, "amend: groups")
    add = {"counts_per_recipe": masses(items, lambda x: x["model"]),
           "counts_per_group": masses(items, lambda x: A8.group_of(x["model"]))}
    require(not set(add) & set(old), "amend: keys already present")
    out = {**old, **add, "counts_provenance": {
        "keys": sorted(add), "script": "scripts/audit_correct_008.py --amend",
        "git_commit": commit, "command": cmd,
        "amended_sha256": hashlib.sha256(a.amend.read_bytes()).hexdigest()}}
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"amended": str(a.amend), "added": sorted(add)}))


def load_grid(grid_path: Path) -> tuple[dict, list, list, np.ndarray, dict]:
    g = json.loads(grid_path.read_text())
    panel = evalset.load_eval()
    source = {r["id"]: r["source"] for r in panel}
    ids = sanityb.subset_ids(panel, sanityb.PRIMARY_SUBSET)
    src = np.array([source[i] for i in ids])
    names = sorted(g["models"])
    infos = [reliability.ModelInfo(v["recipe"], v["pretrain_seed"], str(v["posttrain_seed_k"]))
             for v in (g["models"][m] for m in names)]
    ys = {m: sanityb.outcome_matrix(read_jsonl(g["models"][m]["labels"]), ids, FIELD)[0]
          for m in names}
    return g, names, infos, src, ys


def contrast(ys: dict, names: list, infos: list, factor: dict, n_boot: int, seed: int) -> dict:
    """The grid's primary block (analyze_008.block) on rates x per-prompt factor."""
    sy = {m: ys[m] * factor[m][:, None] for m in names}
    P, Po, Pe = gridstats.halves(sy, names)
    rel = gridstats.reliabilities(Po, Pe)
    lv = gridstats.levels(P, rel, infos)
    return {"levels": lv,
            "ci_recipes": gridstats.recipe_bootstrap(lv["per_recipe"], n_boot, seed),
            "ci_prompts": gridstats.prompt_bootstrap(P, Po, Pe, infos, n_boot, seed),
            "mean_vulnerability": {m: float(np.nanmean(reliability.per_prompt_p(sy[m])))
                                   for m in names}}


def factor_matrix(variant: str, names: list, recipe_of: dict, src: np.ndarray,
                  point: dict) -> np.ndarray:
    """(n_models, n_prompts) per-prompt factor of each model under a variant: one {source:
    factor} per recipe, indexed by the prompts' source codes."""
    sources, code = np.unique(src, return_inverse=True)
    per_recipe = {}
    for r in {recipe_of[m] for m in names}:
        f = A8.model_factor(variant, r, point)
        per_recipe[r] = np.array([f[s] for s in sources])
    return np.stack([per_recipe[recipe_of[m]][code] for m in names])


def contrast_levels(halves: tuple, infos: list, masks: dict, F: np.ndarray) -> dict:
    """`contrast`'s point levels from the raw halves: per-prompt P (and each half) is a mean
    over samples, so scaling the outcomes by a per-prompt factor scales P by it."""
    P, Po, Pe = halves
    lv = gridstats.levels(P * F, gridstats.reliabilities(Po * F, Pe * F), infos, masks)
    return {k: lv[k] for k in CONTRAST_KEYS}


def check_factors(f: dict, what: str) -> None:
    bad = {k: v for k, v in f.items() if not (A8.finite(v) and 0 < v <= 1)}
    if bad:
        raise SystemExit(f"{what}: factor not finite in (0, 1]: {bad}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--sheets", nargs=2)
    ap.add_argument("--adjudicated")
    ap.add_argument("--placeholder", action="store_true")
    ap.add_argument("--score", help="audit_score.py output to cross-check point estimates")
    ap.add_argument("--grid", type=Path, default=GRID)
    ap.add_argument("--n-boot", type=int, default=2000, help="audit bootstrap draws (015)")
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check-against", type=Path,
                    help="committed audit_correct.json: every pre-existing key must be unchanged")
    ap.add_argument("--amend", type=Path,
                    help="committed audit_correct.json: add the count blocks only (no grid, no "
                         "bootstrap), after checking every point value it holds (054)")
    a = ap.parse_args()
    require(a.out.is_absolute(), "--out must be absolute (submit.sh snapshots are deleted)")
    require(a.placeholder or (a.sheets and a.adjudicated), "--sheets and --adjudicated needed")
    commit, cmd, t0 = provenance.git_commit(), provenance.command(), time.time()

    keyrows = {r["item_id"]: r for r in read_jsonl(a.audit / "key.jsonl")}
    if a.placeholder:
        gold, alternatives = placeholder_gold(keyrows, a.seed)
        info = {"unresolved": 0, "placeholder": True}
    else:
        sheets = [S.read_sheet(s) for s in a.sheets]
        adj = S.read_sheet(a.adjudicated)
        gold, info = S.gold_labels(sheets, adj)
        auditgate.check_complete(keyrows, sheets, gold, info)
        alternatives = A8.rejected_alternatives(sheets, adj)
    items = [{**keyrows[i], "gold": g["label"], "gold_partial": g["partial"]}
             for i, g in gold.items() if i in keyrows]
    recipes = sorted({x["model"] for x in items})
    require(len(recipes) == 25 and {A8.group_of(r) for r in recipes} == set(A8.GROUPS),
            ("recipes", recipes))
    if a.amend:
        amend(a, items, recipes, commit, cmd)
        return

    # raw per-prompt rates of the 150 grid models, as analyze_008 built them
    g, names, infos, src, ys = load_grid(a.grid)
    recipe_of = dict(zip(names, (i.recipe for i in infos)))
    require(set(recipe_of.values()) == set(recipes), "audit recipes != grid recipes")
    comp = {m: A8.components(reliability.per_prompt_p(ys[m]), src) for m in names}
    for m in names:  # the components add up to the grid's raw level
        raw = g[sanityb.PRIMARY_SUBSET]["primary"]["per_model"][m]["mean_vulnerability"]
        require(close(sum(comp[m].values()), raw), ("components", m, sum(comp[m].values()), raw))

    pop = A8.population(items)
    halves = gridstats.halves(ys, names)
    masks = gridstats.pair_masks(infos)
    base = A8.make_stat(comp, recipe_of, recipes)
    keys = A8.factor_keys()
    # bootstrap the contrast only for a variant whose factors are defined at the point (a group
    # x source cell with no judge-vulnerable item is undefined)
    point_base = base(items)
    defined = [v for v in ("per_source", "per_group_source")
               if all(A8.finite(point_base[k]) and 0 < point_base[k] <= 1 for k in keys[v])]

    def stat(its: list[dict]) -> dict:
        out = base(its)
        for v in defined:
            F = factor_matrix(v, names, recipe_of, src, out)
            out.update({f"contrast:{v}:{k}": x
                        for k, x in contrast_levels(halves, infos, masks, F).items()})
        return out

    # all gold items: the statistic filters to the population (domain estimation), so the
    # strata keep their sizes and Rao-Wu's m_g = n_g - 1 is the design's
    point = stat(items)
    boot = auditgate.stratified_bootstrap(items, stat, a.n_boot, a.seed,
                                          collapse=auditgate.source_cell, point=point)

    # gold-join check: under audit_score's per-"model" (here per-recipe) precision definition,
    # these gold labels must give the score file's numbers. It checks the join only.
    checks = {}
    if a.score:
        sc = json.loads(Path(a.score).read_text())
        for r, v in sc["per_model_wg_refusal"].items():
            sel = [x for x in items if x["model"] == r and x["gold"] in L.COHERENT
                   and x["wg_label"] in L.COHERENT]
            b = audit.binary_stats([A8.judge_vulnerable(x) for x in sel],
                                     [x["gold"] != L.REFUSAL for x in sel],
                                     [x["weight"] for x in sel])
            checks[f"score_file:{r}:precision"] = (b["precision"], v["precision_weighted"])
        for k, (u, w) in checks.items():
            require(close(u, w), ("audit_score", k, u, w))

    # recipe heterogeneity (the rule) and group heterogeneity (descriptive)
    het_r = A8.Heterogeneity(pop, lambda x: x["model"], a.n_perm, a.seed)
    het_g = A8.Heterogeneity(pop, lambda x: A8.group_of(x["model"]), a.n_perm, a.seed)
    test_r, test_g = het_r.test(), het_g.test()
    fires = test_r["p"] < A8.ALPHA_RULE
    heterogeneity = {
        "recipes": test_r, "groups_descriptive": test_g, "rule_alpha": A8.ALPHA_RULE,
        "rule_fires": bool(fires),
        "flips_to_cross_recipes": A8.flips_to_cross(het_r, alternatives),
        "flips_to_cross_groups_descriptive": A8.flips_to_cross(het_g, alternatives),
        "factor_one_flip": A8.max_factor_change(pop, alternatives),
        "n_alternatives": len(alternatives)}

    of_record = "per_group_source" if fires else "per_source"
    check_factors({k: point[k] for k in keys[of_record]}, of_record)
    corrections = ["per_source", "per_group_source"]
    other = "per_source" if fires else "per_group_source"
    skipped = {}
    try:
        check_factors({k: point[k] for k in keys[other]}, other)
    except SystemExit as e:  # the variant not of record is reported only when defined
        corrections.remove(other)
        skipped[other] = str(e)

    # the primary contrast under each correction; `none` must be the grid's, bit for bit
    n_boot, seed = g["n_boot"], g["seed"]
    n = len(src)
    variants = {}
    for v in ("none", *corrections):
        fac = dict(zip(names, factor_matrix(v, names, recipe_of, src, point)))
        variants[v] = contrast(ys, names, infos, fac, n_boot, seed)
    ref = g[sanityb.PRIMARY_SUBSET]["primary"]
    for k in CONTRAST_KEYS:
        require(same(variants["none"]["levels"][k], ref["levels"][k]), ("grid.json", k))
        for ci in ("ci_recipes", "ci_prompts"):
            require(same(variants["none"][ci][k], ref[ci][k]), ("grid.json", ci, k))
    for m in names:
        require(same(variants["none"]["mean_vulnerability"][m],
                     ref["per_model"][m]["mean_vulnerability"]), ("grid.json", m))
    for v in corrections:
        for m in names:
            lv = A8.corrected(comp[m], A8.model_factor(v, recipe_of[m], point))
            require(close(variants[v]["mean_vulnerability"][m], lv), ("levels", v, m))
        for k in CONTRAST_KEYS:  # equal up to rounding (P x F vs. P of y x f) that a small
            # reliability amplifies, hence a relative tolerance
            require(close(boot[f"contrast:{v}:{k}"]["point"], variants[v]["levels"][k], rel=1e-9),
                    ("contrast", v, k))

    raw_model = {m: ref["per_model"][m]["mean_vulnerability"] for m in names}
    raw_recipe = {r: float(np.mean([raw_model[m] for m in names if recipe_of[m] == r]))
                  for r in recipes}
    levels = {"raw": {"model": raw_model, "recipe": raw_recipe,
                      "overall": float(np.mean(list(raw_model.values())))}}
    for v in corrections:
        levels[v] = {
            "model": {m: boot[f"level:{v}:model:{m}"] for m in names},
            "recipe": {r: boot[f"level:{v}:recipe:{r}"] for r in recipes},
            "overall": boot[f"level:{v}:overall"]}
    levels["of_record"] = of_record
    levels["skipped"] = skipped

    def pick(prefix: str) -> dict:
        return {k[len(prefix):]: v for k, v in boot.items() if k.startswith(prefix)}

    out = {
        "placeholder": a.placeholder, "n_items": len(keyrows), "n_gold": len(items),
        "n_population": len(pop), "n_greedy_excluded": sum(x["sample"] < 0 for x in items),
        **info,
        "factors": {"per_source": pick("source:"), "per_group_source": pick("group_source:"),
                    "per_recipe": pick("recipe:"), "per_group": pick("group:")},
        "groups": {r: A8.group_of(r) for r in recipes},
        "counts_per_recipe": masses(items, lambda x: x["model"]),
        "counts_per_group": masses(items, lambda x: A8.group_of(x["model"])),
        "judge_error": {**pick("pooled:"), **{f"recall:{k.split(':')[0]}": v
                                               for k, v in pick("pooled_source:").items()},
                        "counts": A8.rates(pop)},
        "heterogeneity": heterogeneity, "levels": levels,
        "contrast": {v: {"MZ_minus_U_corrected": r["levels"]["MZ_minus_U_corrected"],
                         "ci_recipes": r["ci_recipes"]["MZ_minus_U_corrected"],
                         "ci_prompts": r["ci_prompts"]["MZ_minus_U_corrected"],
                         **({"ci_audit": boot[f"contrast:{v}:MZ_minus_U_corrected"]}
                            if v != "none" else {}),
                         "MZ_r_corrected": r["levels"]["MZ_r_corrected"],
                         "U_r_corrected": r["levels"]["U_r_corrected"],
                         "R_r_corrected": r["levels"]["R_r_corrected"],
                         "U_minus_R_corrected": r["levels"]["U_minus_R_corrected"]}
                     for v, r in variants.items()},
        "contrast_intervals": "ci_recipes, ci_prompts: grid resampled, factors at their point "
                              "values; ci_audit: audit resampled (Rao-Wu), grid fixed",
        "gold_join_vs_audit_score": {k: v[0] for k, v in checks.items()},
        "grid": str(a.grid), "grid_git_commit": g["git_commit"], "n_prompts": n,
        "field": FIELD, "subset": sanityb.PRIMARY_SUBSET, "n_boot_audit": a.n_boot,
        "n_boot_grid": n_boot, "seed": a.seed, "sheets": a.sheets, "adjudicated": a.adjudicated,
        "score": a.score, "script": "scripts/audit_correct_008.py", "git_commit": commit,
        "command": cmd, "seconds": time.time() - t0,
    }
    if a.check_against:  # before the committed file is overwritten (it may be --out)
        out["checked_against"] = check_against(out, a.check_against)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"placeholder": a.placeholder, "factors": out["factors"]["per_source"],
                      "rule_p": test_r["p"], "rule_fires": bool(fires),
                      "of_record": of_record,
                      "contrast": {v: c["MZ_minus_U_corrected"]
                                   for v, c in out["contrast"].items()}}, default=float))


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    main()
