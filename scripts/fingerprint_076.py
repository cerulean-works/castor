"""Identify a held-out model's pretraining run among 75, and its data recipe among 25, from its
per-prompt vulnerability profile (definitions in castor/fingerprint.py). CPU only; reads the 150
grid models' judge_v1 labels.

    uv run --frozen python scripts/fingerprint_076.py --out /abs/path/runs/076 \
        [--n-boot 1000] [--n-sub 200] [--seed 0] [--workers 4] [--queries MANIFEST]

Models: `runs/008/grid.json` → `models` (25 recipes × 3 pretraining seeds × post-training seeds
{1, 2}). Subset adversarial-harmful × `wg_vulnerable` (primary outcome, K = 20); prompts
undefined in any grid model are dropped. Bases in sorted pretraining-seed order:
base 1 `default`, base 2 `large-aux-2`, base 3 `large-aux-3`.

Writes summary.json (pre-registered accuracies with intervals, prompt-count curves,
confusions, secondaries, provenance), queries.json (per-query detail) and fingerprint_curve.tsv
(appendix figure data: accuracy against prompt count, run and recipe).

--queries MANIFEST (080): the manifest's models (another fine-tune of grid bases) are the
queries, the 75 seed-1 runs the references; prompts undefined in any grid or query model are
dropped. Writes only fingerprint_queries.json and fingerprint_curve.tsv under --out.
"""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from castor import evalset, fingerprint, grid, provenance, reliability, sanityb
from castor.judge.common import read_jsonl

FIELD, SUBSET = "wg_vulnerable", sanityb.PRIMARY_SUBSET
GRID_JSON = grid.ROOT / "runs" / "008" / "grid.json"
GRID_SEEDS = (1, 2)
COUNTS = (3212, 1000, 300, 100, 50, 20, 10)  # pre-registered; 3212 = all kept prompts


def _profile(args: tuple[str, list[str]]) -> np.ndarray:
    path, ids = args
    y, c = sanityb.outcome_matrix(read_jsonl(path), ids, FIELD)
    assert c["missing"] == 0, f"{path}: {c['missing']} missing label cells"
    return reliability.per_prompt_p(y)


def load(paths: dict[str, Path], ids: list[str], workers: int) -> dict[str, np.ndarray]:
    names = sorted(paths)
    with ProcessPoolExecutor(workers) as ex:
        ps = list(ex.map(_profile, [(str(paths[m]), ids) for m in names]))
    return dict(zip(names, ps, strict=True))


def block(res: dict, labels_q: list[str], labels_r: list[str], truth: list[int],
          n_boot: int, seed: int, key_q=None, key_r=None) -> dict:
    """Accuracy with the pre-registered bootstrap and the exact interval, plus the failures
    (query → assigned reference) and margin statistics."""
    ok = res["correct"]
    k, n = int(ok.sum()), len(ok)
    fails = []
    for i in np.flatnonzero(~ok):
        a = int(res["assigned"][i])
        fails.append({"query": labels_q[i], "truth": labels_r[truth[i]],
                      "assigned": labels_r[a] if a >= 0 else None,
                      "rank_of_truth": int(res["rank"][i]),
                      "c_truth": float(res["c_true"][i]),
                      "c_assigned": float(res["C"][i, a]) if a >= 0 else None}
                     | ({"same_group": key_r(labels_r[a]) == key_q(labels_q[i])}
                        if key_q and a >= 0 else {}))
    m = res["margin"]
    return {"n_queries": n, "n_correct": k, "accuracy": k / n,
            "ci_boot_queries": fingerprint.boot_ci(ok, n_boot, seed),
            "ci_exact": fingerprint.clopper_pearson(k, n),
            "chance": 1 / len(labels_r),
            "c_true": {"mean": float(res["c_true"].mean()), "min": float(res["c_true"].min())},
            "c_best_impostor": {"mean": float(res["c_impostor"].mean()),
                                "max": float(res["c_impostor"].max())},
            "margin_true_minus_impostor": {"mean": float(m.mean()), "min": float(m.min()),
                                           "n_negative": int((m < 0).sum())},
            "rank_of_truth": dict(sorted(Counter(int(r) for r in res["rank"]).items())),
            "failures": fails}


def recipe_test(P: dict, recipes: list[str], bases: list[str], held: str, n_boot: int,
                seed: int, counts=None, n_sub: int = 0) -> tuple[dict, list]:
    """References = each recipe's mean profile over the bases other than `held` (both seeds);
    queries = the held base's models."""
    ref_bases = [b for b in bases if b != held]
    R = np.vstack([np.mean([P[(r, b, s)] for b in ref_bases for s in GRID_SEEDS], axis=0)
                   for r in recipes])
    qk = [(r, held, s) for r in recipes for s in GRID_SEEDS]
    Q = np.vstack([P[k] for k in qk])
    truth = [recipes.index(k[0]) for k in qk]
    res = fingerprint.identify(Q, R, truth)
    out = block(res, ["/".join(map(str, k)) for k in qk], recipes, truth, n_boot, seed)
    out |= {"held_out_base": held, "reference_bases": ref_bases}
    curve = fingerprint.prompt_curve(Q, R, truth, counts, n_sub, seed) if counts else []
    return out, curve


def queries(a, prof: dict, names: list[str], gm: dict, keep: np.ndarray, ids: list[str],
            runs: list[tuple], commit: str, t0: float) -> None:
    """Each query model's run among the 75 seed-1 references, all prompts and the curve."""
    g = grid.load(a.queries)
    qcells = g.cells()
    qprof = load({c.model_id: c.gen_dir() / "judge_v1" / "labels.jsonl" for c in qcells},
                 ids, a.workers)
    qnames = [c.model_id for c in qcells]
    keep = keep & np.isfinite(np.vstack([qprof[m] for m in qnames])).all(axis=0)
    dropped = [i for i, k in zip(ids, keep, strict=True) if not k]
    key = {(gm[m]["recipe"], gm[m]["pretrain_seed"]): m for m in names
           if gm[m]["posttrain_seed_k"] == 1}
    R = np.vstack([prof[key[rb]][keep] for rb in runs])
    Q = np.vstack([qprof[m][keep] for m in qnames])
    truth = [runs.index((c.base.recipe, c.base.pretrain_seed)) for c in qcells]
    n_kept = int(keep.sum())
    counts = [min(c, n_kept) for c in COUNTS]
    lbl = [f"{r}/{b}" for r, b in runs]
    res = fingerprint.identify(Q, R, truth)
    out = block(res, qnames, lbl, truth, a.n_boot, a.seed)
    curve = fingerprint.prompt_curve(Q, R, truth, counts, a.n_sub, a.seed)
    rows = [{"query": m, "truth": lbl[truth[i]], "assigned": lbl[res["assigned"][i]] if res["assigned"][i] >= 0 else None,
             "correct": bool(res["correct"][i]), "rank_of_truth": int(res["rank"][i]),
             "c_truth": float(res["c_true"][i]), "c_best_impostor": float(res["c_impostor"][i]),
             "best_impostor": lbl[res["impostor"][i]],
             "best_impostor_recipe": runs[res["impostor"][i]][0],
             "impostor_same_recipe": runs[res["impostor"][i]][0] == runs[truth[i]][0]}
            for i, m in enumerate(qnames)]
    c50 = next(c for c in curve if c["n_prompts"] == 50)
    summary = {
        "subset": SUBSET, "field": FIELD, "queries_manifest": str(a.queries),
        "n_prompts_subset": len(ids), "n_prompts_kept": n_kept, "dropped_undefined": dropped,
        "counts": counts, "references": "75 runs at post-training seed 1 (runs/008/grid.json)",
        "identification": out, "curve": curve, "queries": rows,
        "prediction": {"run accuracy all prompts >= 0.80": bool(out["accuracy"] >= 0.80),
                       "values": {"all": out["accuracy"], "50_mean": c50["mean"]}},
        "script": "scripts/fingerprint_076.py", "git_commit": commit,
        "command": provenance.command(), "n_boot": a.n_boot, "n_sub": a.n_sub, "seed": a.seed,
        "grid_json": str(GRID_JSON.relative_to(grid.ROOT)), "seconds": time.time() - t0}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "fingerprint_queries.json").write_text(json.dumps(summary, indent=1, default=float))
    with open(a.out / "fingerprint_curve.tsv", "w") as f:
        f.write("test\tdirection\tn_prompts\tmean\tlo\thi\tmin\tchance\n")
        f.writelines(f"run\tref_seed1_query_{g.variant}\t{c['n_prompts']}\t{c['mean']}\t"
                     f"{c['lo']}\t{c['hi']}\t{c['min']}\t{1 / len(runs)}\n" for c in curve)
    print(json.dumps({"identification": {k: out[k] for k in ("n_correct", "n_queries",
                                                             "ci_exact")},
                      "curve_50": c50, "n_prompts_kept": n_kept}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-sub", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--queries", type=Path)
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (submit.sh snapshots are deleted)"
    commit, t0 = provenance.git_commit(), time.time()
    panel = evalset.load_eval()
    ids = sanityb.subset_ids(panel, SUBSET)

    gj = json.loads(GRID_JSON.read_text())
    gm = gj["models"]
    assert len(gm) == 150 and {v["posttrain_seed_k"] for v in gm.values()} == set(GRID_SEEDS)
    prof = load({m: Path(v["labels"]) for m, v in gm.items()}, ids, a.workers)
    t_load = time.time() - t0

    # instrument check: each model's mean level must equal runs/008/grid.json's
    pm = gj[SUBSET]["primary"]["per_model"]
    for m, p in prof.items():
        assert abs(float(np.nanmean(p)) - pm[m]["mean_vulnerability"]) < 1e-12, m

    names = sorted(gm)
    Pall = np.vstack([prof[m] for m in names])
    keep = np.isfinite(Pall).all(axis=0)
    dropped = [i for i, k in zip(ids, keep, strict=True) if not k]
    key = {(gm[m]["recipe"], gm[m]["pretrain_seed"], gm[m]["posttrain_seed_k"]): m for m in names}
    P = {k: prof[m][keep] for k, m in key.items()}
    recipes = sorted({k[0] for k in key})
    bases = sorted({k[1] for k in key})
    assert len(recipes) == 25 and len(bases) == 3 and len(P) == 150
    runs = [(r, b) for r in recipes for b in bases]
    if a.queries:
        return queries(a, prof, names, gm, keep, ids, runs, commit, t0)
    n_kept = int(keep.sum())
    counts = [min(c, n_kept) for c in COUNTS]

    # ---- run identification (75 runs), both directions ------------------------------------
    run_lbl = [f"{r}/{b}" for r, b in runs]
    run_blocks, run_curves, qrows = {}, {}, []
    for ref_s, q_s in ((1, 2), (2, 1)):
        R = np.vstack([P[(r, b, ref_s)] for r, b in runs])
        Q = np.vstack([P[(r, b, q_s)] for r, b in runs])
        truth = list(range(len(runs)))
        res = fingerprint.identify(Q, R, truth)
        d = f"ref_seed{ref_s}_query_seed{q_s}"
        run_blocks[d] = block(res, run_lbl, run_lbl, truth, a.n_boot, a.seed,
                              key_q=lambda s: s.split("/")[0], key_r=lambda s: s.split("/")[0])
        run_curves[d] = fingerprint.prompt_curve(Q, R, truth, counts, a.n_sub, a.seed)
        qrows += [{"test": "run", "direction": d, "query": run_lbl[i],
                   "assigned": run_lbl[res["assigned"][i]], "c_truth": float(res["c_true"][i]),
                   "c_best_impostor": float(res["c_impostor"][i]),
                   "best_impostor": run_lbl[res["impostor"][i]],
                   "impostor_same_recipe":
                       runs[res["impostor"][i]][0] == runs[i][0]} for i in truth]
    # where the best impostor sits: same recipe (another pretraining seed) or another recipe
    imp_same = {d: float(np.mean([q["impostor_same_recipe"] for q in qrows
                                  if q["direction"] == d])) for d in run_blocks}

    # ---- recipe identification (25 recipes) from an unseen run ----------------------------
    held = bases[2]
    rec, rec_curve = recipe_test(P, recipes, bases, held, a.n_boot, a.seed, counts, a.n_sub)
    rotations = {b: recipe_test(P, recipes, bases, b, a.n_boot, a.seed)[0] for b in bases}
    rot_pooled = sum(v["n_correct"] for v in rotations.values()) / \
        sum(v["n_queries"] for v in rotations.values())

    prim = run_blocks["ref_seed1_query_seed2"]
    c50 = {d: next(c for c in run_curves[d] if c["n_prompts"] == 50) for d in run_curves}
    prov = {"script": "scripts/fingerprint_076.py", "git_commit": commit,
            "command": provenance.command(), "n_boot": a.n_boot, "n_sub": a.n_sub,
            "seed": a.seed, "grid_json": str(GRID_JSON.relative_to(grid.ROOT)),
            "grid_json_git_commit": gj["git_commit"], "seconds_load": t_load}
    summary = {
        "subset": SUBSET, "field": FIELD, "n_prompts_subset": len(ids),
        "n_prompts_kept": n_kept, "dropped_undefined": dropped,
        "bases_order": bases, "counts": counts,
        "primary": {"metric": "run identification accuracy, all prompts, ref seed 1 → query "
                              "seed 2", "accuracy": prim["accuracy"],
                    "ci_boot_queries": prim["ci_boot_queries"], "ci_exact": prim["ci_exact"]},
        "run": run_blocks, "run_curve": run_curves,
        "run_best_impostor_same_recipe_share": imp_same,
        "recipe": rec, "recipe_curve": rec_curve,
        "recipe_rotations_secondary": {"per_held_base": rotations,
                                       "pooled_accuracy": rot_pooled},
        "prediction": {
            "run accuracy all prompts >= 0.95": bool(prim["accuracy"] >= 0.95),
            "run accuracy 50 prompts (mean over subsets) >= 0.80":
                bool(c50["ref_seed1_query_seed2"]["mean"] >= 0.80),
            "recipe accuracy from an unseen run < 0.5": bool(rec["accuracy"] < 0.5),
            "recipe accuracy above chance (1/25)": bool(rec["accuracy"] > 1 / 25),
            "values": {"run_all": prim["accuracy"],
                       "run_50": c50["ref_seed1_query_seed2"]["mean"],
                       "recipe": rec["accuracy"]}},
        **prov, "seconds": time.time() - t0}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (a.out / "queries.json").write_text(json.dumps(
        {"rows": qrows, "recipe": rec["failures"], **prov}, indent=1, default=float))
    with open(a.out / "fingerprint_curve.tsv", "w") as f:
        f.write("test\tdirection\tn_prompts\tmean\tlo\thi\tchance\n")
        for d, cs in run_curves.items():
            f.writelines(f"run\t{d}\t{c['n_prompts']}\t{c['mean']}\t{c['lo']}\t{c['hi']}"
                        f"\t{1 / len(runs)}\n" for c in cs)
        f.writelines(f"recipe\theld_{held}\t{c['n_prompts']}\t{c['mean']}\t{c['lo']}\t{c['hi']}"
                    f"\t{1 / len(recipes)}\n" for c in rec_curve)
    print(json.dumps({"primary": summary["primary"], "prediction": summary["prediction"],
                      "seconds": summary["seconds"]}, indent=1))


if __name__ == "__main__":
    main()
