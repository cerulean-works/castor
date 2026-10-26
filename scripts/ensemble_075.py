"""Refuse-if-any ensembles across pretraining runs (castor/ensemble.py for the estimators).
Pure CPU; reads the 150 grid models' judge_v1 labels only.

    uv run --frozen python scripts/ensemble_075.py --out /abs/path/runs/075 [--n-draws 1000]

Per data recipe of the grid (25), k = 2 (pre-registered):
  single     base 1, post-training seed 1
  unrelated  base 1 and base 2, post-training seed 1 (two pretraining runs)
  MZ         base 1 at post-training seeds 1 and 2 (the control: two models, one run)
Bases are numbered in DataDecide's seed order (default, large-aux-2, large-aux-3) = 1, 2, 3.
Secondary: the full set (all 3 unrelated pairs at seed 1, all 3 MZ pairs; reference = the mean
of the two members as single models) and k = 3 unrelated (bases 1-3, seed 1; no MZ control at
k = 3 exists on the grid). Subsets: adversarial-harmful (vulnerability, reliably-vulnerable
share) and benign (over-refusal), outcome wg_vulnerable (008's coherence filter).

Before anything new, 044's within-base and within-recipe stable-broken shares are recomputed from
the same labels and asserted equal to runs/044/summary.json to 1e-9.

Writes summary.json (pooled numbers, recipe-bootstrap intervals, prediction, provenance),
per_recipe.tsv and tradeoff.tsv (the figure data)."""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from castor import ensemble, evalset, grid, provenance, reliability, sanityb, stability
from castor.judge.common import read_jsonl

FIELD = "wg_vulnerable"
SUBSETS = ("adv_harmful", "benign")
GRID_JSON = grid.ROOT / "runs" / "008" / "grid.json"
SUMMARY_044 = grid.ROOT / "runs" / "044" / "summary.json"
BASES = ("default", "large-aux-2", "large-aux-3")  # DataDecide seed order = base 1, 2, 3
L044 = {"base": "seed_stable (within base)", "recipe": "ancestry_independent (within recipe)"}


def _matrices(args: tuple[str, dict[str, list[str]]]) -> dict[str, np.ndarray]:
    path, ids = args
    rows = list(read_jsonl(path))  # an iterator: read once, used by every subset
    out = {}
    for s, i in ids.items():
        y, c = sanityb.outcome_matrix(rows, i, FIELD)
        assert c["missing"] == 0, f"{path}: {c['missing']} missing label cells ({s})"
        out[s] = y
    return out


def arms(recipe: str) -> dict[str, list[tuple[str, int]]]:
    """Ensemble kind -> members as (base, post-training seed); the designated arms first."""
    b1, b2, b3 = BASES
    out = {"single": [(b1, 1)], "unrelated": [(b1, 1), (b2, 1)], "mz": [(b1, 1), (b1, 2)],
           "unrelated_k3": [(b1, 1), (b2, 1), (b3, 1)]}
    for x, y in ((b1, b2), (b1, b3), (b2, b3)):
        out[f"full_unrelated:{x}+{y}"] = [(x, 1), (y, 1)]
    for x in BASES:
        out[f"full_mz:{x}"] = [(x, 1), (x, 2)]
    for x in BASES:
        for k in (1, 2):
            out[f"single:{x}/{k}"] = [(x, k)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (submit.sh snapshots are deleted)"
    commit, t0 = provenance.git_commit(), time.time()
    panel = evalset.load_eval()
    ids = {s: sanityb.subset_ids(panel, s) for s in SUBSETS}

    gj = json.loads(GRID_JSON.read_text())
    gm = gj["models"]
    assert len(gm) == 150
    names = sorted(gm)
    with ProcessPoolExecutor(a.workers) as ex:
        mats = list(ex.map(_matrices, [(gm[m]["labels"], ids) for m in names]))
    ys = {s: dict(zip(names, (x[s] for x in mats), strict=True)) for s in SUBSETS}
    del mats
    t_load = time.time() - t0

    # drop prompts undefined in any grid model (044's rule), per subset
    keep, P = {}, {}
    for s in SUBSETS:
        Pall = np.vstack([reliability.per_prompt_p(ys[s][m]) for m in names])
        keep[s] = np.isfinite(Pall).all(axis=0)
        P[s] = Pall[:, keep[s]]
        ys[s] = {m: y[keep[s]] for m, y in ys[s].items()}

    # ---- 044's reproduction (asserted before anything new) ----------------------------------
    info = [reliability.ModelInfo(gm[m]["recipe"], gm[m]["pretrain_seed"],
                                  str(gm[m]["posttrain_seed_k"])) for m in names]
    ref = json.loads(SUMMARY_044.read_text())["grid"]["prompts"]
    check = {}
    for by, key in L044.items():
        got = float(stability.group_stable(P["adv_harmful"], stability.groups_of(info, by))
                    ["every_frac"].mean())
        want = ref["levels"][key]["share_every"]
        assert abs(got - want) < 1e-9, (by, got, want)
        check[key] = {"recomputed": got, "044": want}
    assert P["adv_harmful"].shape[1] == ref["n_prompts"], (P["adv_harmful"].shape, ref["n_prompts"])

    # ---- per-model draws (common random numbers), per-recipe ensembles ------------------------
    idx = {m: j for j, m in enumerate(names)}
    by_cell = {(gm[m]["recipe"], gm[m]["pretrain_seed"], gm[m]["posttrain_seed_k"]): m
               for m in names}
    recipes = sorted({gm[m]["recipe"] for m in names})
    per = {}  # recipe -> kind -> subset -> {resampled, product}
    for r in recipes:
        members = {c: by_cell[(r, *c)] for c in {c for v in arms(r).values() for c in v}}
        draws = {}
        for c, m in members.items():
            rng = ensemble.model_rng(a.seed, idx[m])
            draws[c] = {s: ensemble.draw_outcomes(ys[s][m], a.n_draws, rng) for s in SUBSETS}
        per[r] = {}
        for kind, mem in arms(r).items():
            per[r][kind] = {"models": [members[c] for c in mem]}
            for s in SUBSETS:
                pr = ensemble.ensemble_p_resampled([draws[c][s] for c in mem])
                pp = ensemble.ensemble_p_product([ys[s][members[c]] for c in mem])
                per[r][kind][s] = {"resampled": ensemble.readout(pr),
                                   "product": ensemble.readout(pp)}

    # ---- pooled readings ---------------------------------------------------------------------
    def col(kind: str, s: str, stat: str, est: str = "resampled") -> np.ndarray:
        return np.array([per[r][kind][s][est][stat] for r in recipes])

    def full(prefix: str, s: str, stat: str, est: str = "resampled") -> np.ndarray:
        """Per recipe: the mean over the full set's three pairs of (ensemble, member-mean)."""
        ens, ref_ = [], []
        for r in recipes:
            ks = [k for k in per[r] if k.startswith(prefix)]
            ens.append(np.mean([per[r][k][s][est][stat] for k in ks]))
            ref_.append(np.mean([np.mean([per[r][f"single:{b}/{k}"][s][est][stat]
                                          for b, k in arms(r)[kk]]) for kk in ks]))
        return np.array(ens), np.array(ref_)

    def boot(arrays: dict, f) -> list:
        return ensemble.recipe_bootstrap(arrays, f, a.n_boot, a.seed)

    def mean_ci(v: np.ndarray) -> dict:
        return {"value": float(v.mean()), "ci_recipes": boot({"v": v}, lambda v: v.mean())}

    def block(single, unrel, mz) -> dict:
        """Reductions (single - ensemble), their ratio unrelated / MZ, with recipe intervals."""
        arr = {"single": single, "unrel": unrel, "mz": mz}
        return {"single": mean_ci(single), "unrelated": mean_ci(unrel), "mz": mean_ci(mz),
                "reduction_unrelated": mean_ci(single - unrel),
                "reduction_mz": mean_ci(single - mz),
                "ratio_unrelated_over_mz": {
                    "value": ensemble.ratio_of_reductions(single, unrel, mz),
                    "ci_recipes": boot(arr, ensemble.ratio_of_reductions)},
                "unrelated_minus_mz": mean_ci(unrel - mz)}

    def costs(single, unrel, mz) -> dict:
        """Over-refusal = 1 - mean P_ens on benign; cost = ensemble - single."""
        o = {k: 1 - v for k, v in (("single", single), ("unrel", unrel), ("mz", mz))}
        return {"single": mean_ci(o["single"]), "unrelated": mean_ci(o["unrel"]),
                "mz": mean_ci(o["mz"]), "cost_unrelated": mean_ci(o["unrel"] - o["single"]),
                "cost_mz": mean_ci(o["mz"] - o["single"]),
                "cost_unrelated_minus_mz": mean_ci(o["unrel"] - o["mz"]),
                "cost_ratio_unrelated_over_mz": {
                    "value": ensemble.ratio_of_reductions(single, unrel, mz),
                    "ci_recipes": boot({"single": single, "unrel": unrel, "mz": mz},
                                       ensemble.ratio_of_reductions)}}

    pooled = {}
    for est in ("resampled", "product"):
        adv = {stat: block(col("single", "adv_harmful", stat, est),
                           col("unrelated", "adv_harmful", stat, est),
                           col("mz", "adv_harmful", stat, est)) for stat in ("reliable", "mean")}
        ben = costs(col("single", "benign", "mean", est), col("unrelated", "benign", "mean", est),
                    col("mz", "benign", "mean", est))
        pooled[est] = {"adv_harmful": adv, "benign_over_refusal": ben}

    # secondaries: the full set (member-mean reference) and k = 3 unrelated
    secondary = {}
    for stat in ("reliable", "mean"):
        eu, ru = full("full_unrelated:", "adv_harmful", stat)
        em, rm = full("full_mz:", "adv_harmful", stat)
        secondary[f"full_set_adv_{stat}"] = {
            "unrelated": mean_ci(eu), "unrelated_reference": mean_ci(ru),
            "mz": mean_ci(em), "mz_reference": mean_ci(rm),
            "reduction_unrelated": mean_ci(ru - eu), "reduction_mz": mean_ci(rm - em),
            "ratio_unrelated_over_mz": {
                "value": float((ru - eu).mean() / (rm - em).mean()),
                "ci_recipes": boot({"a": ru - eu, "b": rm - em},
                                   lambda a, b: float(a.mean() / b.mean()))}}
    eu, ru = full("full_unrelated:", "benign", "mean")
    em, rm = full("full_mz:", "benign", "mean")
    secondary["full_set_benign_over_refusal"] = {
        "cost_unrelated": mean_ci(ru - eu), "cost_mz": mean_ci(rm - em),
        "cost_unrelated_minus_mz": mean_ci((ru - eu) - (rm - em))}
    s1 = col("single", "adv_harmful", "reliable")
    k3 = col("unrelated_k3", "adv_harmful", "reliable")
    secondary["unrelated_k3"] = {
        "adv_reliable": mean_ci(k3), "reduction_reliable": mean_ci(s1 - k3),
        "adv_mean": mean_ci(col("unrelated_k3", "adv_harmful", "mean")),
        "benign_over_refusal": mean_ci(1 - col("unrelated_k3", "benign", "mean")),
        "cost": mean_ci(col("single", "benign", "mean") - col("unrelated_k3", "benign", "mean")),
        "note": "no MZ control at k = 3 on the grid (two post-training seeds per base)"}

    # agreement of the two estimators (the resampled estimate converges to the product rule)
    est_gap = {s: float(max(abs(per[r][k][s]["resampled"][t] - per[r][k][s]["product"][t])
                            for r in recipes for k in per[r] for t in ("mean",)))
               for s in SUBSETS}

    # ---- the pre-registered prediction (descriptive; no go/kill rule) -----------------------
    pr = pooled["resampled"]
    ratio = pr["adv_harmful"]["reliable"]["ratio_unrelated_over_mz"]
    bo = pr["benign_over_refusal"]
    prediction = {
        "ratio of reliably-vulnerable reductions (unrelated / MZ) >= 1.5 with recipe CI > 1":
            bool(ratio["value"] >= 1.5 and ratio["ci_recipes"][0] > 1),
        "ratio": ratio,
        "unrelated over-refusal > MZ over-refusal":
            bool(bo["unrelated"]["value"] > bo["mz"]["value"]),
        "over_refusal_unrelated_minus_mz": bo["cost_unrelated_minus_mz"]}

    summary = {
        "design": {"k": 2, "bases": dict(zip(("1", "2", "3"), BASES, strict=True)),
                   "single": "base 1 / seed 1", "unrelated": "bases 1+2 / seed 1",
                   "mz": "base 1 / seeds 1+2", "rule": f"reliably vulnerable: P_ens >= {ensemble.EVERY}",
                   "field": FIELD, "n_draws": a.n_draws},
        "n_prompts": {s: int(keep[s].sum()) for s in SUBSETS},
        "dropped_undefined": {s: [i for i, k in zip(ids[s], keep[s], strict=True) if not k]
                              for s in SUBSETS},
        "check_044": check, "estimator_max_abs_gap_mean": est_gap,
        "pooled": pooled, "secondary": secondary, "prediction": prediction,
        "recipes": recipes,
        "per_recipe": {r: {k: v for k, v in per[r].items() if not k.startswith("single:")}
                       for r in recipes},
        "script": "scripts/ensemble_075.py", "git_commit": commit,
        "command": provenance.command(), "n_boot": a.n_boot, "seed": a.seed,
        "grid_json": str(GRID_JSON.relative_to(grid.ROOT)),
        "grid_json_git_commit": gj["git_commit"], "seconds_load": t_load,
        "seconds": time.time() - t0}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    with open(a.out / "per_recipe.tsv", "w") as f:
        cols = [(k, s, t) for k in ("single", "unrelated", "mz", "unrelated_k3")
                for s, t in (("adv_harmful", "reliable"), ("adv_harmful", "mean"),
                             ("benign", "mean"))]
        f.write("recipe\t" + "\t".join(
            f"{k}_{'over_refusal' if s == 'benign' else 'adv_' + t}" for k, s, t in cols) + "\n")
        for r in recipes:
            vals = [per[r][k][s]["resampled"][t] for k, s, t in cols]
            vals = [1 - v if s == "benign" else v for v, (_, s, _) in zip(vals, cols, strict=True)]
            f.write(r + "\t" + "\t".join(f"{v:.6f}" for v in vals) + "\n")
    with open(a.out / "tradeoff.tsv", "w") as f:
        # one mark per recipe per ensemble kind: x = over-refusal cost, y = reduction of the
        # reliably-vulnerable share, both relative to the single model (base 1, seed 1)
        f.write("kind\trecipe\tx_cost\ty_reduction\ty_mean_reduction\n")
        for kind, lab in (("mz", "pair-mz"), ("unrelated", "pair-unrelated")):
            for r in recipes:
                s, e = per[r]["single"], per[r][kind]
                f.write(f"{lab}\t{r}\t{s['benign']['resampled']['mean'] - e['benign']['resampled']['mean']:.6f}\t"
                        f"{s['adv_harmful']['resampled']['reliable'] - e['adv_harmful']['resampled']['reliable']:.6f}\t"
                        f"{s['adv_harmful']['resampled']['mean'] - e['adv_harmful']['resampled']['mean']:.6f}\n")
    print(json.dumps({"check_044": check, "prediction": prediction, "seconds": summary["seconds"]},
                     default=float))


if __name__ == "__main__":
    main()
