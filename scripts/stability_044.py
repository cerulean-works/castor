"""Seed-stable, ancestry-independent and universal jailbreaks over the grid's judge labels
(castor/stability.py for the definitions). Pure CPU; reads judge_v1 labels only.

    uv run --frozen python scripts/stability_044.py --out /abs/path/runs/044 [--n-boot 1000]

Models: the 150 grid models of record (`runs/008/grid.json` → `models`: 25 recipes × 3 pretraining
seeds × post-training seeds {1, 2}); the ladder's 150M / 300M twins and the main recipe's A0 1B
twins (`ladder.anchor_cells`, seeds 1–4) as the per-size secondary; the 12 A0 twins also reproduce
runs/041/seed_stable.json (asserted). Subset adversarial-harmful × `wg_vulnerable` (the primary
outcome, K = 20); templates = HarmBench human jailbreaks (behaviour::template).

Writes prompts.json (per prompt: P per model, stability counts per level), templates.json
(per template: share per model, ranks), summary.json (the pre-registered statistics with
prompt / template bootstrap intervals, secondaries, provenance) and two figure tsvs."""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from castor import evalset, figdata, grid, ladder, provenance, reliability, sanityb, stability
from castor.judge.common import read_jsonl

FIELD, SUBSET, HJ = "wg_vulnerable", sanityb.PRIMARY_SUBSET, "harmbench_humanjb"
GRID_JSON = grid.ROOT / "runs" / "008" / "grid.json"
SEED_STABLE_041 = grid.ROOT / "runs" / "041" / "seed_stable.json"
GRID_SEEDS = (1, 2)  # = analyze_008.GRID_SEEDS: every recipe gets the same two seeds
TOP = 10


def _matrix(args: tuple[str, list[str]]) -> np.ndarray:
    path, ids = args
    y, c = sanityb.outcome_matrix(read_jsonl(path), ids, FIELD)
    assert c["missing"] == 0, f"{path}: {c['missing']} missing label cells"
    return y


def load(paths: dict[str, Path], ids: list[str], workers: int) -> dict[str, np.ndarray]:
    names = sorted(paths)
    with ProcessPoolExecutor(workers) as ex:
        ys = list(ex.map(_matrix, [(str(paths[m]), ids) for m in names]))
    return dict(zip(names, ys, strict=True))


def ci_mean(v: np.ndarray, n_boot: int, seed: int) -> list[float]:
    """Prompt-bootstrap interval of a mean over prompts."""
    return stability.percentile([v[i].mean() for i in stability.boot_indices(len(v), n_boot,
                                                                              seed)])


def prompt_block(P: np.ndarray, info: list, n_boot: int, seed: int, universal: dict) -> dict:
    """(i) and (ii) on finite P (n_models, n_prompts). `universal` = {name: row indices}: the
    groups of models read as "all recipes"."""
    n = P.shape[1]
    out = {"n_models": P.shape[0], "n_prompts": n, "levels": {}, "pairs": {}}
    tiers = {"seed_stable (within base)": stability.groups_of(info, "base"),
             "ancestry_independent (within recipe)": stability.groups_of(info, "recipe")}
    tiers |= {f"universal ({k})": {"all": v} for k, v in universal.items()}
    for name, gs in tiers.items():
        g = stability.group_stable(P, gs)
        ev, no = g["every_frac"], g["none_frac"]
        out["levels"][name] = {
            "n_groups": g["n_groups"], "models_per_group": g["models_per_group"],
            "share_every": float(ev.mean()),
            "ci_every": ci_mean(ev, n_boot, seed),
            "share_none": float(no.mean()),
            "ci_none": ci_mean(no, n_boot, seed),
            "per_group_share_every": {k: v["share_every"] for k, v in g["per_group"].items()},
            "_n_every": g["n_every"], "_n_none": g["n_none"]}
    c = stability.pair_counts(P, info)
    for lvl in stability.LEVELS:
        if c[lvl]["n_pairs"] == 0:
            continue
        pt = stability.pair_rates(c[lvl])
        draws = [stability.pair_rates(c[lvl], i)
                 for i in stability.boot_indices(n, n_boot, seed)]
        out["pairs"][lvl] = {"n_pairs": int(c[lvl]["n_pairs"]), **pt,
                             **{f"ci_{k}": stability.percentile([d[k] for d in draws])
                                for k in pt},
                             "_flip": c[lvl]["flip"]}
    return out


def template_block(ys: dict[str, np.ndarray], names: list[str], info: list, col: np.ndarray,
                   tnames: list[str], n_boot: int, seed: int, check: bool) -> dict:
    sh = [stability.template_shares(ys[m], col, len(tnames)) for m in names]
    S, E, O = (np.vstack([s[k] for s in sh]) for k in ("full", "even", "odd"))
    assert np.isfinite(S).all() and np.isfinite(E).all() and np.isfinite(O).all()
    masks = stability.level_masks(info)
    pt = stability.template_levels(S, E, O, masks)
    if check:  # the vectorised levels equal reliability.twin_correlations with the template measure
        tw = reliability.twin_correlations({m: ys[m] for m in names},
                                           dict(zip(names, info, strict=True)), "spearman",
                                           measure=stability.template_measure(col, len(tnames)))
        for lvl in stability.LEVELS:
            if masks[lvl].any():
                for k, key in ((f"r_{lvl}", "r"), (f"r_{lvl}_corrected", "r_corrected")):
                    assert abs(pt[k] - tw["levels"][lvl][key]) < 1e-9, (k, pt[k], tw["levels"])
    draws = [stability.template_levels(S, E, O, masks, i)
             for i in stability.boot_indices(len(tnames), n_boot, seed)]
    rel = reliability.spearman_brown(stability._rowcorr(stability.rank_rows(E),
                                                        stability.rank_rows(O)))
    lv = {k: {"value": v, "ci_templates": stability.percentile([d[k] for d in draws])}
          for k, v in pt.items() if np.isfinite(v)}
    if masks["R"].any():  # secondary (not pre-registered): U − R under the same template draws
        k = ("r_U_corrected", "r_R_corrected")
        lv["U_minus_R_corrected"] = {
            "value": pt[k[0]] - pt[k[1]],
            "ci_templates": stability.percentile([d[k[0]] - d[k[1]] for d in draws])}
    return {"n_templates": len(tnames), "levels": lv, "S": S, "masks": masks,
            "reliability": {"min": float(rel.min()), "mean": float(rel.mean()),
                            "max": float(rel.max())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (submit.sh snapshots are deleted)"
    commit, t0 = provenance.git_commit(), time.time()
    panel = evalset.load_eval()
    ids = sanityb.subset_ids(panel, SUBSET)
    row = {r["id"]: r for r in panel}
    src = [row[i]["source"] for i in ids]

    gj = json.loads(GRID_JSON.read_text())
    gm = gj["models"]
    assert len(gm) == 150 and {v["posttrain_seed_k"] for v in gm.values()} == set(GRID_SEEDS)
    paths = {m: Path(v["labels"]) for m, v in gm.items()}
    a0 = {c.model_id: c for c in ladder.anchor_cells()}
    sizes = {z: {c.model_id: c for c in ladder.cells((z,))} for z in ladder.SIZES}
    paths |= {m: sanityb.labels_path(m, ladder.ANCHOR_PROMPT_ID) for m in a0}
    for cs in sizes.values():
        paths |= {m: sanityb.labels_path(m, ladder.PROMPT_ID) for m in cs}
    ys = load(paths, ids, a.workers)
    t_load = time.time() - t0

    # runs/041's seed-stable shares from the same labels and rule (the instrument check)
    rg = figdata.prompt_range({m: reliability.per_prompt_p(ys[m]) for m in a0})
    s041 = figdata.stable_shares(rg["min"], rg["max"])
    ref = json.loads(SEED_STABLE_041.read_text())["shares"]
    assert (s041["n_every"], s041["n_none"], s041["n"]) == \
        (ref["n_every"], ref["n_none"], ref["n"]), (s041, ref)

    # ---- the grid -------------------------------------------------------------------------
    names = sorted(gm)
    info = [reliability.ModelInfo(gm[m]["recipe"], gm[m]["pretrain_seed"],
                                  str(gm[m]["posttrain_seed_k"])) for m in names]
    Pall = np.vstack([reliability.per_prompt_p(ys[m]) for m in names])
    keep = np.isfinite(Pall).all(axis=0)
    P = Pall[:, keep]
    kept = [i for i, k in zip(ids, keep, strict=True) if k]
    seed1 = [j for j, i in enumerate(info) if i.posttrain_seed == "1"]
    grid_p = prompt_block(P, info, a.n_boot, a.seed,
                          {"seed-1 models": seed1, "all 150 models": list(range(len(info)))})

    hj = [j for j, s in enumerate(src) if s == HJ]
    tnames, col = stability.template_index([row[ids[j]]["source_id"].split("::")[1]
                                            for j in hj])
    hj_ys = {m: ys[m][hj] for m in names}
    grid_t = template_block(hj_ys, names, info, col, tnames, a.n_boot, a.seed, check=True)

    # per-template table and the top-10 lists (names only)
    S, mk = grid_t.pop("S"), grid_t.pop("masks")
    absdiff = {lvl: np.array([np.abs(S[x] - S[y]) for x, y in zip(*np.nonzero(mk[lvl]),
                                                                    strict=True)]).mean(axis=0)
               for lvl in stability.LEVELS}
    tmin, tmean, tmax = S.min(axis=0), S.mean(axis=0), S.max(axis=0)
    trows = [{"template": t, "n_prompts": int((col == j).sum()), "mean": float(tmean[j]),
              "min": float(tmin[j]), "max": float(tmax[j]),
              **{f"mean_absdiff_{lvl}": float(absdiff[lvl][j]) for lvl in stability.LEVELS}}
             for j, t in enumerate(tnames)]
    top_stable = sorted(trows, key=lambda r: (-r["min"], -r["mean"], r["template"]))[:TOP]
    top_flip = sorted(trows, key=lambda r: (-r["mean_absdiff_MZ"], r["template"]))[:TOP]

    # ---- the ladder sizes (main recipe), two seeds as the grid and all four ------------------
    by_size = {}
    for z, cs in {**sizes, "1B": a0}.items():
        for ks in (GRID_SEEDS, (1, 2, 3, 4)):
            ms = sorted(m for m, c in cs.items() if c.k in ks)
            inf = [cs[m].info for m in ms]
            Pz = np.vstack([reliability.per_prompt_p(ys[m]) for m in ms])
            kz = np.isfinite(Pz).all(axis=0)
            blk = prompt_block(Pz[:, kz], inf, a.n_boot, a.seed, {})
            tb = template_block({m: ys[m][hj] for m in ms}, ms, inf, col, tnames,
                                a.n_boot, a.seed, check=True)
            tb.pop("S"), tb.pop("masks")
            for v in blk["levels"].values():
                v.pop("_n_every"), v.pop("_n_none"), v.pop("per_group_share_every")
            for v in blk["pairs"].values():
                v.pop("_flip")
            by_size[f"{z} seeds {''.join(map(str, ks))}"] = {
                "models": ms, "n_dropped_undefined": int((~kz).sum()), "prompts": blk,
                "templates": tb}

    # ---- per-prompt table --------------------------------------------------------------------
    lv = grid_p["levels"]
    short = {"seed_stable (within base)": "base", "ancestry_independent (within recipe)": "recipe",
             "universal (seed-1 models)": "universal_seed1",
             "universal (all 150 models)": "universal_all"}
    L = {short[k]: v for k, v in lv.items()}
    flip_mz = grid_p["pairs"]["MZ"]["_flip"]
    prompts = {"models": names, "subset": SUBSET, "field": FIELD,
               "dropped_undefined": [i for i, k in zip(ids, keep, strict=True) if not k],
               "rule": f"stable-broken: every model of the group P >= {stability.EVERY}; "
                       f"stable-refused: every model P <= {stability.NONE} (041)",
               "rows": [{"prompt_id": pid, "source": row[pid]["source"],
                         "template": row[pid]["source_id"].split("::")[1]
                         if row[pid]["source"] == HJ else None,
                         "P": [round(float(x), 4) for x in P[:, j]],
                         **{f"n_every_{k}": int(v["_n_every"][j]) for k, v in L.items()},
                         **{f"n_none_{k}": int(v["_n_none"][j]) for k, v in L.items()},
                         "n_flip_MZ": int(flip_mz[j])}
                        for j, pid in enumerate(kept)]}
    for v in lv.values():
        v.pop("_n_every"), v.pop("_n_none")
    for v in grid_p["pairs"].values():
        v.pop("_flip")
    templates = {"models": names, "templates": tnames,
                 "share": [[round(float(x), 4) for x in r] for r in S], "rows": trows}

    # ---- the pre-registered readings (descriptive; no rule) ---------------------------------
    e = {k: v["share_every"] for k, v in lv.items()}
    sb, ai, un = (e["seed_stable (within base)"], e["ancestry_independent (within recipe)"],
                  e["universal (seed-1 models)"])
    tl = grid_t["levels"]
    prov = {"script": "scripts/stability_044.py", "git_commit": commit,
            "command": provenance.command(), "n_boot": a.n_boot, "seed": a.seed,
            "grid_json": str(GRID_JSON.relative_to(grid.ROOT)),
            "grid_json_git_commit": gj["git_commit"], "seconds_load": t_load}
    summary = {
        "subset": SUBSET, "field": FIELD, "n_prompts_subset": len(ids),
        "n_dropped_undefined": int((~keep).sum()), "rule_041": [stability.EVERY, stability.NONE],
        "check_041": {"models": sorted(a0), "shares": s041, "equals": str(SEED_STABLE_041)},
        "grid": {"prompts": grid_p, "templates": grid_t,
                 "top10_stable_templates": top_stable, "top10_flipping_templates": top_flip},
        "by_size_016": by_size,
        "universal_prompts_by_source": {
            k: dict(sorted(Counter(r["source"] for r in prompts["rows"]
                                   if r[f"n_every_{k}"] == 1).items()))
            for k in ("universal_seed1", "universal_all")},
        "prediction": {
            "seed_stable > ancestry_independent > universal (share_every)":
                bool(sb > ai > un), "values": [sb, ai, un],
            "template ranks agree more within recipe (U) than between recipes (R)":
                bool(tl["r_U_corrected"]["value"] > tl["r_R_corrected"]["value"])},
        **prov, "seconds": time.time() - t0}
    a.out.mkdir(parents=True, exist_ok=True)
    for name, obj in (("prompts", prompts), ("templates", templates), ("summary", summary)):
        (a.out / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
    # figure data: stability shares by level; template rank agreement by pair type
    with open(a.out / "stability_levels.tsv", "w") as f:
        f.write("level\tlabel\tx\tevery\tevery_lo\tevery_hi\tnone\tnone_lo\tnone_hi\n")
        for x, (k, lab) in enumerate((("seed_stable (within base)", "within a base"),
                                      ("ancestry_independent (within recipe)", "within a recipe"),
                                      ("universal (seed-1 models)", "across recipes")), 1):
            v = lv[k]
            f.write(f"{k.split(' ')[0]}\t{lab}\t{x}\t{v['share_every']}\t{v['ci_every'][0]}\t"
                    f"{v['ci_every'][1]}\t{v['share_none']}\t{v['ci_none'][0]}\t"
                    f"{v['ci_none'][1]}\n")
    with open(a.out / "template_rank.tsv", "w") as f:
        f.write("pair\tlabel\tx\tr\tlo\thi\tflip\tflip_lo\tflip_hi\n")
        for x, (lvl, lab) in enumerate((("MZ", "same base"), ("U", "same recipe"),
                                        ("R", "other recipe")), 1):
            t, pr = tl[f"r_{lvl}_corrected"], grid_p["pairs"][lvl]
            f.write(f"{lvl}\t{lab}\t{x}\t{t['value']}\t{t['ci_templates'][0]}\t"
                    f"{t['ci_templates'][1]}\t{pr['flip']}\t{pr['ci_flip'][0]}\t"
                    f"{pr['ci_flip'][1]}\n")
    print(json.dumps({"prediction": summary["prediction"], "seconds": summary["seconds"]},
                     default=float))


if __name__ == "__main__":
    main()
