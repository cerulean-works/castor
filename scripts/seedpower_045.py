"""How many seeds a safety evaluation needs, from the recipe grid (castor/seedpower.py's module
docstring holds the definitions). Pure CPU; reads runs/008/grid.json and the grid's
judge_v1 labels (for σ²_sampling only).

    uv run --frozen python scripts/seedpower_045.py --out /abs/path/runs/045 \
        [--grid runs/008/grid.json] [--n-boot 1000] [--seed 0]

Primary: adv_harmful × primary (wg_vulnerable). Secondary (descriptive): benign × primary and
adv_harmful × q3g. Writes components.json, se_table.json, flips.json, prompt_reliability.json,
se_design.tsv and flip_curve.tsv (appendix figure data) under --out.
"""

import argparse
import hashlib
import json
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

from castor import evalset, provenance, reliability, sanityb, seedpower
from castor.judge.common import read_jsonl

CELLS = {"primary": ("adv_harmful", "primary"), "benign": ("benign", "primary"),
         "q3g": ("adv_harmful", "q3g")}
FIELDS = {"primary": "wg_vulnerable", "q3g": "q3g_vulnerable"}
BINS = (0.0, 0.01, 0.02, 0.04, 0.08, 1.0)  # |true difference| bins of the flip curve


def level_array(g: dict, subset: str, outcome: str) -> tuple[np.ndarray, list, list, list]:
    """y[a, b, n] of grid.json per-model mean vulnerability, with recipes, bases and the model
    ids in [a][b][n] order."""
    pm = g[subset][outcome]["per_model"]
    recipes = sorted({v["recipe"] for v in g["models"].values()})
    ids, y, bases = [], [], []
    for r in recipes:
        bs = sorted({v["pretrain_seed"] for v in g["models"].values() if v["recipe"] == r})
        bases.append(bs)
        row_ids = [[m for m, v in sorted(g["models"].items(), key=lambda kv: kv[1]["posttrain_seed_k"])
                    if v["recipe"] == r and v["pretrain_seed"] == b] for b in bs]
        ids.append(row_ids)
        y.append([[pm[m]["mean_vulnerability"] for m in row] for row in row_ids])
    return np.array(y, dtype=float), recipes, bases, ids


def sampling_block(g: dict, cells: dict) -> dict:
    """Per cell: per-model σ²_sampling (reliability and binomial), the level recomputed from
    labels (checked against grid.json), and their means over models."""
    panel = evalset.load_eval()
    all_ids = [r["id"] for r in panel]
    pos = {pid: i for i, pid in enumerate(all_ids)}
    idx = {s: np.array([pos[p] for p in sanityb.subset_ids(panel, s)])
           for s in {s for s, _ in cells.values()}}
    out = {c: {"per_model": {}} for c in cells}
    for m, v in sorted(g["models"].items()):
        rows = list(read_jsonl(Path(v["labels"])))
        mats = {o: sanityb.outcome_matrix(rows, all_ids, FIELDS[o])[0]
                for o in {o for _, o in cells.values()}}
        for c, (s, o) in cells.items():
            y = mats[o][idx[s]]
            P = reliability.per_prompt_p(y)
            rel = reliability.split_half(y)["reliability"]
            level = float(np.nanmean(P))
            ref = g[s][o]["per_model"][m]
            assert abs(level - ref["mean_vulnerability"]) < 1e-12, (m, c, level)
            assert abs(rel - ref["reliability"]) < 1e-9, (m, c, rel)
            out[c]["per_model"][m] = {"level": level, "reliability": rel,
                                      "var_sampling": seedpower.sampling_variance(P, rel),
                                      "var_binomial": seedpower.binomial_sampling_variance(y)}
    for c in cells:
        pmv = out[c]["per_model"].values()
        out[c]["mean_var_sampling"] = float(np.mean([d["var_sampling"] for d in pmv]))
        out[c]["mean_var_binomial"] = float(np.mean([d["var_binomial"] for d in pmv]))
    return out


def analyse(y: np.ndarray, samp: float, n_boot: int, seed: int) -> dict:
    a = y.shape[0]
    comp = seedpower.components(y, samp)

    def comp_of(ix):
        return seedpower.components(y[ix], samp)

    draws = seedpower.recipe_bootstrap(comp_of, a, n_boot, seed)
    keys = ("recipe", "pretrain_seed", "posttrain_seed", "posttrain_true", "sampling")
    comp_ci = {k: seedpower.ci([d[k] for d in draws], comp[k]) for k in keys}
    share_ci = {k: seedpower.ci([d["shares"][k] for d in draws], comp["shares"][k])
                for k in comp["shares"]}
    diff = comp["pretrain_seed"] - comp["posttrain_seed"]
    comp_out = {"components": {k: comp[k] for k in keys}, "ci": comp_ci,
                "shares": comp["shares"], "shares_ci": share_ci,
                "sd": {k: float(np.sqrt(max(comp[k], 0.0))) for k in keys},
                "pre_minus_post": seedpower.ci([d["pretrain_seed"] - d["posttrain_seed"]
                                                for d in draws], diff),
                "pre_gt_post": bool(diff > 0),
                "design": {"a_recipes": a, "b_bases": y.shape[1], "n_posttrain": y.shape[2]}}

    se = seedpower.se_table(comp)
    se_ci = {k: seedpower.ci([seedpower.se_table(d)[k] for d in draws], v) for k, v in se.items()}
    ext_pre, ext_post = tuple(range(1, 21)), (1, 2, 4, 8)
    se_out = {"unit": "rate (1 point = 0.01)", "se": se, "ci": se_ci,
              "smallest_le_1pt": seedpower.smallest_design(comp, seedpower.POINT),
              "smallest_le_2pt": seedpower.smallest_design(comp, 2 * seedpower.POINT),
              "extrapolated": {
                  "n_pre": list(ext_pre), "n_post": list(ext_post),
                  "smallest_le_1pt": seedpower.smallest_design(comp, seedpower.POINT,
                                                               ext_pre, ext_post),
                  "smallest_le_2pt": seedpower.smallest_design(comp, 2 * seedpower.POINT,
                                                               ext_pre, ext_post),
                  "se_1pt_floor_n_post_inf": {p: float(np.sqrt(max(comp["pretrain_seed"], 0) / p))
                                              for p in ext_pre}},
              "single_seed_se_ci": se_ci["1x1"]}

    pairs = seedpower.flip_rates(y)
    flips = seedpower.summarise_flips(pairs)
    fdraws = seedpower.recipe_bootstrap(
        lambda ix: seedpower.summarise_flips(seedpower.flip_rates(y, recipes=ix)), a, n_boot, seed)
    v1 = max(comp["pretrain_seed"], 0) + max(comp["posttrain_seed"], 0)
    d = np.array([p["abs_delta"] for p in pairs])
    f = np.array([p["flip"] for p in pairs])
    curve = []
    for lo, hi in pairwise(BINS):
        sel = (d >= lo) & (d < hi)
        mid = float(np.mean(d[sel])) if sel.any() else (lo + hi) / 2
        curve.append({"lo": lo, "hi": hi, "n_pairs": int(sel.sum()),
                      "flip_empirical": float(np.mean(f[sel])) if sel.any() else float("nan"),
                      "mean_abs_delta": mid, "flip_model": seedpower.model_flip(mid, v1)})
    flips_out = {**flips,
                 "ci": {k: seedpower.ci([dd[k] for dd in fdraws], flips[k])
                        for k in ("overall", "ge_2pt")},
                 "model_based": {"var_single_draw": v1,
                                 "overall": float(np.mean([seedpower.model_flip(x, v1)
                                                           for x in d])),
                                 "ge_2pt": float(np.mean([seedpower.model_flip(x, v1)
                                                          for x in d[d >= 0.02]]))},
                 "true_difference": "all-seed (6-model) mean difference",
                 "curve": curve,
                 "pairs": pairs}
    return {"components": comp_out, "se_table": se_out, "flips": flips_out}


def prompt_reliability(g: dict, subset: str, outcome: str, recipes: list, n_boot: int,
                       seed: int) -> dict:
    per = g[subset][outcome]["levels"]["per_recipe"]
    designs = {f"{p}x{q}": (p, q) for p in (1, 2, 3) for q in (1, 2, 4)}
    rows = {}
    for r in recipes:
        r_u, r_mz, rel = per[r]["U"]["r"], per[r]["MZ"]["r"], per[r]["mean_reliability"]
        rows[r] = {"r_U": r_u, "r_MZ": r_mz, "rel": rel,
                   "noise_base": r_mz - r_u, "noise_post": rel - r_mz, "noise_sampling": 1 - rel,
                   "g": {k: seedpower.g_coefficient(r_u, r_mz, p, q)
                         for k, (p, q) in designs.items()}}
    a = len(recipes)

    def mean_of(ix, fn):
        return float(np.mean([fn(rows[recipes[i]]) for i in ix]))

    out = {"m": {}, "designs": {}}
    for m, (p, q) in seedpower.M_DESIGNS.items():
        k = f"{p}x{q}"
        pt = mean_of(range(a), lambda d, k=k: d["g"][k])
        out["m"][str(m)] = {"design": k, **seedpower.ci(seedpower.recipe_bootstrap(
            lambda ix, k=k: mean_of(ix, lambda d: d["g"][k]), a, n_boot, seed), pt)}
    for k in designs:
        out["designs"][k] = mean_of(range(a), lambda d, k=k: d["g"][k])
    for key in ("r_U", "r_MZ", "rel", "noise_base", "noise_post", "noise_sampling"):
        out[f"mean_{key}"] = mean_of(range(a), lambda d, key=key: d[key])
    out["per_recipe"] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grid", type=Path, default=Path("runs/008/grid.json"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute"
    commit, t0 = provenance.git_commit(), time.time()
    raw = a.grid.read_bytes()
    g = json.loads(raw)
    assert g["complete"] and g["n_models"] == 150, "grid incomplete"
    prov = {"script": "scripts/seedpower_045.py", "git_commit": commit,
            "command": provenance.command(), "grid": str(a.grid),
            "grid_sha256": hashlib.sha256(raw).hexdigest(), "grid_git_commit": g["git_commit"],
            "n_boot": a.n_boot, "seed": a.seed}

    samp = sampling_block(g, CELLS)
    res = {k: {} for k in ("components", "se_table", "flips", "prompt_reliability")}
    for c, (s, o) in CELLS.items():
        y, recipes, bases, ids = level_array(g, s, o)
        assert y.shape == (25, 3, 2), y.shape
        an = analyse(y, samp[c]["mean_var_sampling"], a.n_boot, a.seed)
        an["components"]["sampling"] = {k: v for k, v in samp[c].items()}
        an["components"]["layout"] = {"recipes": recipes, "bases": bases, "model_ids": ids}
        an["flips"]["recipes"] = recipes
        for k in ("components", "se_table", "flips"):
            res[k][c] = {"subset": s, "outcome": o, **an[k]}
        res["prompt_reliability"][c] = {"subset": s, "outcome": o, **prompt_reliability(
            g, s, o, recipes, a.n_boot, a.seed)}

    a.out.mkdir(parents=True, exist_ok=True)
    prov["seconds"] = round(time.time() - t0, 1)
    for k, v in res.items():
        (a.out / f"{k}.json").write_text(json.dumps({"provenance": prov, **v}, indent=1) + "\n")
    p = res["se_table"]["primary"]
    with open(a.out / "se_design.tsv", "w") as fh:
        fh.write("n_pre\tn_post\tn_models\tse\tse_lo\tse_hi\n")
        for k, v in p["se"].items():
            n_pre, n_post = map(int, k.split("x"))
            fh.write(f"{n_pre}\t{n_post}\t{n_pre * n_post}\t{v:.6f}\t"
                     f"{p['ci'][k]['lo']:.6f}\t{p['ci'][k]['hi']:.6f}\n")
    with open(a.out / "flip_curve.tsv", "w") as fh:
        fh.write("lo\thi\tn_pairs\tmean_abs_delta\tflip_empirical\tflip_model\n")
        fh.writelines(f"{b['lo']}\t{b['hi']}\t{b['n_pairs']}\t{b['mean_abs_delta']:.6f}\t"
                     f"{b['flip_empirical']:.6f}\t{b['flip_model']:.6f}\n" for b in res["flips"]["primary"]["curve"])
    print(json.dumps({c: {"components": res["components"][c]["components"],
                          "smallest_1pt": res["se_table"][c]["smallest_le_1pt"],
                          "flip": res["flips"][c]["overall"]} for c in CELLS}, indent=1))


if __name__ == "__main__":
    main()
