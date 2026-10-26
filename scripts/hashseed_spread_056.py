"""Monte Carlo spread of banked audit intervals over the Python hash seed.

At OLD_REV, `audit_score.gold_labels` and `audit008.rejected_alternatives` built their output in
`set(a) & set(b)` order, which depends on per-process string hashing. That order feeds the Rao-Wu
bootstrap (`auditgate.stratified_bootstrap`), the permutation blocks (`audit008.Heterogeneity`)
and the greedy flip count's tie-breaking, so the banked intervals of targets 015, 023 and 051 are
one draw of the hash seed. This script recomputes each artifact's resampled quantities as its
script does, under hash seeds 0..19 with the OLD_REV functions, and once in sorted order (the
current code: it must equal the `_ordered` artifact the real script writes, which checks that the
functions below mirror the scripts).

    # one target, children run in parallel, each under its own PYTHONHASHSEED
    uv run --frozen python scripts/hashseed_spread_056.py spread --target 015 --workers 4 \
        --out /abs/runs/056/spread_015.json
    # 051 needs the grid's label files: a 1-GPU --dev job (CPU work, more cores and memory)
    # merge the parts into one spread table
    uv run --frozen python scripts/hashseed_spread_056.py merge --out /abs/runs/056/spread.json \
        runs/056/spread_015.json runs/056/spread_023.json runs/056/spread_051.json
    (from a snapshot of a commit that holds the parts and the `_ordered` artifacts)

Per key: the banked point / lo / hi; over the hash seeds the range of each endpoint, its width,
and the largest distance from the banked endpoint; whether every seed's point equals the banked
point bit for bit; the sorted run's interval. Points are order-free sums; only the draws move.
The order-dependent scalars (permutation p, flip counts) are reported per key and never enter
the `all_interval_points_equal_banked` flag.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import types
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
from castor import provenance

OLD_REV = "ff41defbdbda67284747e80429426cd5339ccf0e"  # last commit with hash-ordered output
SEEDS = list(range(20))
ARTIFACTS = {  # artifact of record (repo copy) per target
    "015": ROOT / "runs/006/audit/015/report_015.json",
    "023": ROOT / "runs/006/023/result.json",
    "051": ROOT / "runs/008/audit_correct.json",
}
HEADLINE = {  # headline intervals per target (flattened keys)
    "015": ["bootstrap/adv_twins_source:wildjailbreak:precision",
            "bootstrap/adv_twins_source:harmbench_humanjb:precision",
            "bootstrap/wg_overall", "bootstrap/wg_worst_model", "bootstrap/wg_twins_minus_refs"],
    "023": ["primary/ci/gap_human", "primary/ci/gap_judge",
            "primary/ci/gap_judge_minus_human"],
    "051": ["factors/per_source/wildjailbreak", "factors/per_source/harmbench_humanjb",
            "judge_error/precision", "judge_error/recall", "levels/per_source/overall",
            "contrast/per_source/ci_audit", "heterogeneity/recipes/p",
            "heterogeneity/groups_descriptive/p"],
}
SCALARS = {  # non-interval quantities that depend on item order (compared as points)
    "051": ["heterogeneity/recipes/p", "heterogeneity/recipes/T",
            "heterogeneity/groups_descriptive/p", "heterogeneity/groups_descriptive/T",
            "heterogeneity/flips_to_cross_recipes/n_flips",
            "heterogeneity/flips_to_cross_groups_descriptive/n_flips"],
}


# ---- old code -------------------------------------------------------------------------------


def git_module(name: str, path: str, rev: str = OLD_REV) -> types.ModuleType:
    """A module from a committed revision of the repo (the pre-fix code, verbatim)."""
    src = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout
    mod = types.ModuleType(name)
    mod.__file__ = str(ROOT / path)
    exec(compile(src, f"{rev[:7]}:{path}", "exec"), mod.__dict__)  # noqa: S102 (our own committed code)
    return mod


def gold_fns(order: str):
    """(gold_labels, rejected_alternatives) of the fixed code ('sorted') or of OLD_REV ('old')."""
    if order == "sorted":
        from castor import audit008 as A8
        return provenance.load_script("audit_score").gold_labels, A8.rejected_alternatives
    return (git_module("audit_score_old", "scripts/audit_score.py").gold_labels,
            git_module("audit008_old", "castor/audit008.py").rejected_alternatives)


def flatten(d, prefix: str = "") -> dict:
    """Interval leaves (dicts with lo and hi) -> {point, lo, hi}, keyed by their path."""
    out = {}
    if isinstance(d, dict):
        if "lo" in d and "hi" in d:
            pt = d.get("point", d.get("estimate"))
            return {prefix: {"point": pt, "lo": d["lo"], "hi": d["hi"]}}
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}/{k}" if prefix else str(k)))
    return out


def eq(a, b) -> bool:
    """Bit-for-bit equality, NaN equal to NaN."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def absdiff(a, b):
    """|a - b| for two numbers, None when either side is missing (a None flip count, say)."""
    if isinstance(a, int | float) and isinstance(b, int | float):
        return abs(a - b)
    return None


def get(d: dict, path: str):
    for k in path.split("/"):
        d = d[k]
    return d


# ---- one run per target (mirrors the scripts' lines; the sorted run checks the mirror) ------


def run_015(order: str) -> dict:
    R = provenance.load_script("audit_report_015")
    from castor import auditgate, evalset, sanityb
    from castor.judge.common import read_jsonl
    art = json.loads(ARTIFACTS["015"].read_text())
    gold_labels, _ = gold_fns(order)
    sheets = [R.S.read_sheet(s) for s in art["sheets"]]
    adj = R.S.read_sheet(art["adjudicated"])
    gold, info = gold_labels(sheets, adj)
    keyrows = {r["item_id"]: r for r in read_jsonl(Path(art["sheets"][0]).parent / "key.jsonl")}
    R.check_complete(keyrows, sheets, gold, info)
    items = [{**keyrows[i], "gold": g["label"], "gold_partial": g["partial"]}
             for i, g in gold.items() if i in keyrows]
    judges = ["wg"] + (["q3g"] if any(x.get("q3g_label") for x in items) else [])
    models = sorted({x["model"] for x in items})
    adv = set(sanityb.subset_ids(evalset.load_eval(), sanityb.PRIMARY_SUBSET))
    sources = sorted({x["source"] for x in items if x["prompt_id"] in adv})
    stat = R.make_stat(models, judges, adv, sources)
    point = stat(items)
    boot = auditgate.stratified_bootstrap(items, stat, art["n_boot"], art["seed"],
                                          collapse=R.cell, nan_low=R.is_low, point=point)
    return {"bootstrap": boot}


def run_023(order: str) -> dict:
    H = provenance.load_script("human_twin_023")
    from castor.judge.common import read_jsonl
    art = json.loads(ARTIFACTS["023"].read_text())
    gold_labels, _ = gold_fns(order)
    d = Path(art["sheets"][0]).parent
    manifest = json.loads((d / "manifest.json").read_text())
    keyrows = {r["item_id"]: r for r in read_jsonl(d / "key.jsonl")}
    read_sheet = H._load("audit_score").read_sheet
    sheets = [read_sheet(s) for s in art["sheets"]]
    adj = read_sheet(art["adjudicated"])
    gold, info = gold_labels(sheets, adj)
    assert info["unresolved"] == 0 and set(gold) == set(keyrows)
    items = [{**keyrows[i], "gold": g["label"], "gold_partial": g["partial"], "weight": 1.0}
             for i, g in gold.items()]
    prompts, n = manifest["prompts"], art["n_boot"]
    out = {"primary": H._contrast(items, prompts, incl=False, n_boot=n),
           "secondary_incoherent_included": H._contrast(items, prompts, incl=True, n_boot=n)}
    out["secondary_per_source"] = {}
    for src in H.humantwin.SOURCES:
        in_src = {x["prompt_id"] for x in items if x["source"] == src}
        out["secondary_per_source"][src] = H._contrast(
            [x for x in items if x["source"] == src], [p for p in prompts if p in in_src],
            incl=False, n_boot=n)
    out["per_annotator"] = {name: H._contrast(H.humantwin.with_labels(items, sh), prompts,
                                              incl=False, n_boot=n)
                            for name, sh in zip(H.ANNOTATORS, sheets, strict=True)}
    return out


def run_051(order: str) -> dict:
    AC = provenance.load_script("audit_correct_008")
    from castor import audit008 as A8
    from castor import auditgate, gridstats, reliability
    from castor.judge.common import read_jsonl
    art = json.loads(ARTIFACTS["051"].read_text())
    gold_labels, rejected_alternatives = gold_fns(order)
    sheets = [AC.S.read_sheet(s) for s in art["sheets"]]
    adj = AC.S.read_sheet(art["adjudicated"])
    gold, info = gold_labels(sheets, adj)
    keyrows = {r["item_id"]: r for r in read_jsonl(Path(art["sheets"][0]).parent / "key.jsonl")}
    auditgate.check_complete(keyrows, sheets, gold, info)
    alternatives = rejected_alternatives(sheets, adj)
    items = [{**keyrows[i], "gold": g["label"], "gold_partial": g["partial"]}
             for i, g in gold.items() if i in keyrows]
    recipes = sorted({x["model"] for x in items})
    _, names, infos, src, ys = AC.load_grid(AC.GRID)
    recipe_of = dict(zip(names, (i.recipe for i in infos)))
    comp = {m: A8.components(reliability.per_prompt_p(ys[m]), src) for m in names}
    pop = A8.population(items)
    halves = gridstats.halves(ys, names)
    masks = gridstats.pair_masks(infos)
    base = A8.make_stat(comp, recipe_of, recipes)
    keys = A8.factor_keys()
    point_base = base(items)
    defined = [v for v in ("per_source", "per_group_source")
               if all(A8.finite(point_base[k]) and 0 < point_base[k] <= 1 for k in keys[v])]

    def stat(its: list[dict]) -> dict:
        out = base(its)
        for v in defined:
            F = AC.factor_matrix(v, names, recipe_of, src, out)
            out.update({f"contrast:{v}:{k}": x
                        for k, x in AC.contrast_levels(halves, infos, masks, F).items()})
        return out

    point = stat(items)
    boot = auditgate.stratified_bootstrap(items, stat, art["n_boot_audit"], art["seed"],
                                          collapse=auditgate.source_cell, point=point)
    het_r = A8.Heterogeneity(pop, lambda x: x["model"], art["heterogeneity"]["recipes"]["n_perm"],
                             art["seed"])
    het_g = A8.Heterogeneity(pop, lambda x: A8.group_of(x["model"]),
                             art["heterogeneity"]["groups_descriptive"]["n_perm"], art["seed"])

    def pick(prefix: str) -> dict:
        return {k[len(prefix):]: v for k, v in boot.items() if k.startswith(prefix)}

    corrections = [v for v in ("per_source", "per_group_source") if v in art["levels"]]
    return {
        "factors": {"per_source": pick("source:"), "per_group_source": pick("group_source:"),
                    "per_recipe": pick("recipe:"), "per_group": pick("group:")},
        "judge_error": {**pick("pooled:"), **{f"recall:{k.split(':')[0]}": v
                                              for k, v in pick("pooled_source:").items()}},
        "levels": {v: {"model": {m: boot[f"level:{v}:model:{m}"] for m in names},
                       "recipe": {r: boot[f"level:{v}:recipe:{r}"] for r in recipes},
                       "overall": boot[f"level:{v}:overall"]} for v in corrections},
        "contrast": {v: {"ci_audit": boot[f"contrast:{v}:MZ_minus_U_corrected"]}
                     for v in corrections},
        "heterogeneity": {
            "recipes": het_r.test(), "groups_descriptive": het_g.test(),
            "flips_to_cross_recipes": A8.flips_to_cross(het_r, alternatives),
            "flips_to_cross_groups_descriptive": A8.flips_to_cross(het_g, alternatives)},
    }


RUNS = {"015": run_015, "023": run_023, "051": run_051}


def extract(target: str, res: dict) -> dict:
    """Flattened intervals plus the target's order-dependent scalars."""
    flat = flatten(res)
    for k in SCALARS.get(target, []):
        flat[k] = {"point": get(res, k)}
    return flat


# ---- driver ---------------------------------------------------------------------------------


def cmd_one(a) -> None:
    t0 = time.time()
    res = RUNS[a.target](a.order)
    out = {"target": a.target, "order": a.order,
           "hash_seed": os.environ.get("PYTHONHASHSEED"), "values": extract(a.target, res),
           "seconds": time.time() - t0}
    Path(a.out).write_text(json.dumps(out, default=float))


def summarise(banked: dict, runs: dict[int, dict], ordered: dict) -> dict:
    rows = {}
    for k, b in banked.items():
        r = {"banked": b, "points_equal_banked": all(eq(runs[s][k]["point"], b["point"])
                                                     for s in runs),
             "sorted": ordered[k], "sorted_point_equal_banked": eq(ordered[k]["point"], b["point"])}
        if "lo" in b:
            pts = [absdiff(runs[s][k]["point"], b["point"]) for s in runs]
            pts.append(absdiff(ordered[k]["point"], b["point"]))
            r["point_max_abs_from_banked"] = None if None in pts else max(pts)
            for e in ("lo", "hi"):
                v = np.array([np.nan if runs[s][k][e] is None else runs[s][k][e]
                              for s in runs], float)
                bk = np.nan if b[e] is None else float(b[e])
                od = np.nan if ordered[k][e] is None else float(ordered[k][e])
                fin = v[np.isfinite(v)]  # an undefined draw (NaN endpoint) is counted, not ranged
                if fin.size == 0 or not np.isfinite(bk):
                    r[e] = {"n_nan": int(len(v) - fin.size), "min": None, "max": None,
                            "width": None, "max_abs_from_banked": None,
                            "banked_within_range": None, "sorted_within_range": None,
                            "sorted_minus_banked": None}
                    continue
                r[e] = {"n_nan": int(len(v) - fin.size),
                        "min": float(np.min(fin)), "max": float(np.max(fin)),
                        "width": float(np.max(fin) - np.min(fin)),
                        "max_abs_from_banked": float(np.max(np.abs(fin - bk))),
                        "banked_within_range": bool(np.min(fin) <= bk <= np.max(fin)),
                        "sorted_within_range": bool(np.min(fin) <= od <= np.max(fin)),
                        "sorted_minus_banked": None if not np.isfinite(od) else float(od - bk)}
        else:  # order-dependent scalar (permutation p, T, flip count)
            v = [runs[s][k]["point"] for s in runs]
            r["values"] = {"min": min(v, key=lambda x: (x is None, x)),
                           "max": max(v, key=lambda x: (x is not None, x)),
                           "n_distinct": len(set(v))}
        rows[k] = r
    return rows


AGGREGATES = ("n_keys", "n_intervals", "n_nan_endpoints", "max_abs_endpoint_from_banked",
              "max_endpoint_range_width", "all_interval_points_equal_banked",
              "interval_points_not_equal")


def aggregate(rows: dict) -> dict:
    """The spread table's headline numbers from the per-key rows. Intervals only: the
    order-dependent scalars (permutation p, flip counts) move by construction and are reported
    per key and in `order_scalars`, so they never enter the equality flag. A NaN endpoint is
    counted, never silently dropped by `max`."""
    ivals = [r for r in rows.values() if "lo" in r]
    ends = [r[e] for r in ivals for e in ("lo", "hi")]

    def n_nan(x: dict) -> int:  # a part written before this field: NaN propagated into min
        return x["n_nan"] if "n_nan" in x else int(x["min"] is None or not math.isfinite(x["min"]))

    def finite(xs: list) -> list:
        return [x for x in xs if isinstance(x, int | float) and math.isfinite(x)]

    return {"n_keys": len(rows), "n_intervals": len(ivals),
            "n_nan_endpoints": sum(n_nan(x) for x in ends),
            "max_abs_endpoint_from_banked": max(finite([x["max_abs_from_banked"] for x in ends]),
                                                default=None),
            "max_endpoint_range_width": max(finite([x["width"] for x in ends]), default=None),
            "all_interval_points_equal_banked": all(
                r["points_equal_banked"] and r["sorted_point_equal_banked"] for r in ivals),
            # each interval point that is not bit-identical, with its size, so last-bit noise
            # (051: ci_audit under one BLAS thread) is told apart from a moved point
            "interval_points_not_equal": [
                {"key": k, "max_abs_diff": r.get("point_max_abs_from_banked",
                                                 absdiff(r["sorted"]["point"], r["banked"]["point"]))}
                for k, r in rows.items() if "lo" in r
                and not (r["points_equal_banked"] and r["sorted_point_equal_banked"])]}


def run_children(target: str, workers: int) -> tuple[dict[int, dict], dict]:
    """Per-seed values (old order) and the sorted run, each from its own child process. Scratch
    files live in a temporary directory outside the repo and go with it; the first failing child
    cancels the ones not yet started."""
    with tempfile.TemporaryDirectory(prefix=f"castor_056_{target}_") as tmp:
        def child(seed: int | None) -> dict:
            order = "sorted" if seed is None else "old"
            f = Path(tmp) / f"{order}_{seed}.json"
            env = {**os.environ, "PYTHONHASHSEED": str(0 if seed is None else seed)}
            subprocess.run([sys.executable, __file__, "one", "--target", target, "--order",
                            order, "--out", str(f)], env=env, check=True, cwd=ROOT)
            return json.loads(f.read_text())

        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(child, s) for s in [*SEEDS, None]]
            wait(futs, return_when=FIRST_EXCEPTION)
            for f in futs:
                f.cancel()
            results = [f.result() for f in futs]  # raises the first child's error
    return {s: r["values"] for s, r in zip(SEEDS, results[:-1])}, results[-1]["values"]


def cmd_spread(a) -> None:
    t0 = time.time()
    commit = provenance.git_commit()
    art = json.loads(ARTIFACTS[a.target].read_text())
    runs, ordered = run_children(a.target, a.workers)
    keys = set(ordered)
    banked = extract_banked(a.target, art)
    missing = sorted(keys - set(banked))
    assert not missing, f"keys computed here but absent from the artifact: {missing[:5]}"
    dropped = sorted(set(banked) - keys)
    assert not dropped, f"intervals of record the mirror does not compute: {dropped[:5]}"
    assert all(set(r) == keys for r in runs.values()), "runs differ in their keys"
    rows = summarise(banked, runs, ordered)
    out = {
        "target": a.target, "artifact": str(ARTIFACTS[a.target].relative_to(ROOT)),
        "artifact_git_commit": art["git_commit"], "hash_seeds": SEEDS, "old_rev": OLD_REV,
        **aggregate(rows),
        "headline": {k: rows[k] for k in HEADLINE[a.target]},
        "per_key": rows, "sorted_values": ordered,
        "per_seed": {str(s): {k: runs[s][k] for k in HEADLINE[a.target]} for s in SEEDS},
        "script": "scripts/hashseed_spread_056.py", "git_commit": commit,
        "command": provenance.command(), "seconds": time.time() - t0,
    }
    Path(a.out).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: out[k] for k in ("target", *AGGREGATES)}))


def extract_banked(target: str, art: dict) -> dict:
    if target == "015":
        flat = flatten({"bootstrap": art["bootstrap"]})
    elif target == "023":
        flat = flatten({k: art[k] for k in ("primary", "secondary_incoherent_included",
                                            "secondary_per_source", "per_annotator")})
    else:
        flat = flatten({k: art[k] for k in ("factors", "judge_error", "levels")})
        flat.update(flatten({"contrast": {v: {"ci_audit": c["ci_audit"]}
                                          for v, c in art["contrast"].items() if "ci_audit" in c}}))
    for k in SCALARS.get(target, []):
        flat[k] = {"point": get(art, k)}
    return flat


def check_ordered(target: str, ordered_artifact: Path, spread: dict) -> dict:
    """The `_ordered` artifact (real script, fixed code) against this script's sorted run (every
    interval and scalar) and against the banked artifact (every point: interval points and the
    permutation test's observed T). Exact comparison; each mismatch is listed with its size, so
    a last-bit difference (float sums in another order, another BLAS thread count) is told
    apart from a real one. Permutation p values and flip counts are Monte Carlo quantities that
    move with the order by construction: listed under `order_scalars`, not as points."""
    new = extract_banked(target, json.loads(ordered_artifact.read_text()))
    sv = spread["sorted_values"]
    mirror = []
    for k in sv:
        if k not in new or new[k].keys() != sv[k].keys():
            mirror.append({"key": k, "max_abs_diff": None})
        elif not all(eq(new[k][e], sv[k][e]) for e in sv[k]):
            diffs = [absdiff(new[k][e], sv[k][e]) for e in sv[k]]
            mirror.append({"key": k, "max_abs_diff": None if None in diffs else max(diffs)})
    banked = extract_banked(target, json.loads(ARTIFACTS[target].read_text()))
    moving = [k for k in SCALARS.get(target, []) if not k.endswith("/T")]
    pdiff = [{"key": k, "banked": banked[k]["point"], "ordered": new[k]["point"],
              "abs_diff": absdiff(new[k]["point"], banked[k]["point"])}
             for k in banked if k not in moving and not eq(new[k]["point"], banked[k]["point"])]
    return {"ordered_artifact": str(ordered_artifact.relative_to(ROOT)),
            "n_compared": len(sv), "mirror_mismatches": mirror,
            "points_not_bit_identical": pdiff,
            "order_scalars": {k: {"banked": banked[k]["point"], "ordered": new[k]["point"]}
                              for k in moving}}


def cmd_merge(a) -> None:
    parts = {}
    for p in a.parts:  # repo-relative: merge runs on committed parts and artifacts
        s = json.loads((ROOT / p).read_text())
        t = s["target"]
        ordered = ARTIFACTS[t].with_name(ARTIFACTS[t].stem + "_ordered.json")
        # the aggregates are recomputed from the part's rows, so a part written before the
        # interval-only flag (056 review) merges to the same table as one written after
        parts[t] = {**{k: s[k] for k in ("artifact", "artifact_git_commit", "headline",
                                         "git_commit", "command")},
                    **aggregate(s["per_key"]), "source": p}
        parts[t]["ordered_check"] = check_ordered(t, ordered, s) if ordered.exists() else None
    out = {"targets": parts, "hash_seeds": SEEDS, "old_rev": OLD_REV,
           "script": "scripts/hashseed_spread_056.py merge", "git_commit": provenance.git_commit(),
           "command": provenance.command()}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({t: {k: v[k] for k in AGGREGATES[2:]}
                      | {"ordered_check": v["ordered_check"] and {
                          k: len(v["ordered_check"][k]) for k in ("mirror_mismatches",
                                                                  "points_not_bit_identical")}}
                      for t, v in parts.items()}, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("one")
    p.add_argument("--target", choices=sorted(RUNS), required=True)
    p.add_argument("--order", choices=("old", "sorted"), required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("spread")
    p.add_argument("--target", choices=sorted(RUNS), required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    p = sub.add_parser("merge")
    p.add_argument("--out", required=True)
    p.add_argument("parts", nargs="+", help="spread files, repo-relative")
    a = ap.parse_args()
    assert Path(a.out).is_absolute(), "--out must be absolute (D036)"
    {"one": cmd_one, "spread": cmd_spread, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    main()
