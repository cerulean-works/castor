"""Statistics of the 25-recipe grid: per-recipe ceiling-corrected twin correlations, the pre-registered primary contrast and its two CIs, and the secondaries.

Inputs are per-model outcome matrices Y[m] (n_prompts, K) as in castor.reliability, all on one
prompt order, and ModelInfo(recipe, pretrain_seed, posttrain_seed) per model. Per-prompt
P = mean over coherent samples; P_odd / P_even = the same over the odd / even sample columns.

Definitions (fixed before any grid data existed):
  - pair correlations are Pearson over prompts where both P are defined (castor.reliability.corr,
    computed here for all pairs at once by `corr_matrix`); corrected = r / sqrt(rel_a rel_b)
    with rel = Spearman–Brown split-half reliability (castor.reliability.split_half);
  - per recipe r: MZ = mean over its same-base pairs, U = mean over its different-base pairs,
    R = mean over pairs with exactly one model in r (raw and corrected, per-pair correction);
  - primary = mean over recipes of (MZ_corr − U_corr); recipe bootstrap = recipes resampled with
    replacement (per-recipe values held fixed); prompt bootstrap = prompts resampled, every
    correlation and reliability recomputed; percentile 95 % intervals;
  - "overall" R = mean over all cross-recipe pairs.
Recipes with one post-training seed have no MZ pair: their MZ and MZ − U are undefined (NaN)
and every mean over recipes skips them, so r_MZ and MZ − U average over the recipes with two
seeds (9) while U, R and U − R average over all 25. The recipe bootstrap then resamples within
two strata, recipes with and without MZ pairs, so each draw keeps 9 and 16 (with every recipe at
two seeds it is the plain bootstrap). The strata, n_recipes_MZ and the prediction's recipe set
come from the design (a recipe has MZ pairs), never from whether its corrected MZ is finite: a
two-seed recipe whose MZ is undefined (reliability ≤ 0 or NaN) stays in its stratum, drops out
of the nanmeans, and is counted in n_recipes_MZ_undefined. The nested variance of mean
vulnerability is computed on a balanced design: with one seed per base its "pretrain_seed"
component is the base-level variance, which also holds post-training-seed variance (an upper
bound; `posttrain_seed` NaN).
"""

from itertools import combinations

import numpy as np

from castor import reliability
from castor.reliability import ModelInfo

# DataDecide-eval-results@9919b5a0 recipe name -> HF repo recipe slug
OLMES_NAMES = {
    "C4": "c4", "DCLM-Baseline": "dclm-baseline",
    "DCLM-Baseline (QC 10%)": "dclm-baseline-qc-10p",
    "DCLM-Baseline (QC 20%)": "dclm-baseline-qc-20p",
    "DCLM-Baseline (QC 7%, FW2)": "dclm-baseline-qc-7p-fw2",
    "DCLM-Baseline (QC 7%, FW3)": "dclm-baseline-qc-7p-fw3",
    "DCLM-Baseline (QC FW 10%)": "dclm-baseline-qc-fw-10p",
    "DCLM-Baseline (QC FW 3%)": "dclm-baseline-qc-fw-3p",
    "DCLM-Baseline 25% / Dolma 75%": "dclm-baseline-25p-dolma1.7-75p",
    "DCLM-Baseline 50% / Dolma 50%": "dclm-baseline-50p-dolma1.7-50p",
    "DCLM-Baseline 75% / Dolma 25%": "dclm-baseline-75p-dolma1.7-25p",
    "Dolma1.6++": "dolma1_6plus", "Dolma1.7": "dolma1_7",
    "Dolma1.7 (no Flan)": "dolma1_7-no-flan", "Dolma1.7 (no Reddit)": "dolma1_7-no-reddit",
    "Dolma1.7 (no code)": "dolma1_7-no-code", "Dolma1.7 (no math, code)": "dolma1_7-no-math-code",
    "Falcon": "falcon", "Falcon+CC": "falcon-and-cc", "Falcon+CC (QC 10%)": "falcon-and-cc-qc-10p",
    "Falcon+CC (QC 20%)": "falcon-and-cc-qc-20p",
    "Falcon+CC (QC Orig 10%)": "falcon-and-cc-qc-orig-10p",
    "Falcon+CC (QC Tulu 10%)": "falcon-and-cc-qc-tulu-10p",
    "FineWeb-Edu": "fineweb-edu", "FineWeb-Pro": "fineweb-pro",
}
TOP_FRACTION = 0.10


# ---- vectorised correlations ----------------------------------------------------------------


def corr_matrix(X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
    """Pearson correlation of every row of X with every row of Z (default X) over the columns
    where both are finite: pairwise-complete, equal to castor.reliability.corr per pair
    (NaN if < 3 common columns or a constant row on them)."""
    Z = X if Z is None else Z
    mx, mz = np.isfinite(X), np.isfinite(Z)
    x0, z0 = np.where(mx, X, 0.0), np.where(mz, Z, 0.0)
    mxf, mzf = mx.astype(float), mz.astype(float)
    n = mxf @ mzf.T
    sx, sz = x0 @ mzf.T, mxf @ z0.T
    sxx, szz = (x0 ** 2) @ mzf.T, mxf @ (z0 ** 2).T
    sxz = x0 @ z0.T
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxz - sx * sz / n
        vx = sxx - sx ** 2 / n
        vz = szz - sz ** 2 / n
        r = cov / np.sqrt(vx * vz)
    tiny = 1e-12 * np.maximum(n, 1)
    r[(n < 3) | (vx <= tiny) | (vz <= tiny)] = np.nan
    return np.clip(r, -1.0, 1.0)


def rowwise_corr(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """corr(X[i], Z[i]) for each row i (pairwise-complete)."""
    m = np.isfinite(X) & np.isfinite(Z)
    n = m.sum(1)
    x, z = np.where(m, X, 0.0), np.where(m, Z, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, mz = x.sum(1) / n, z.sum(1) / n
        dx, dz = np.where(m, x - mx[:, None], 0), np.where(m, z - mz[:, None], 0)
        r = (dx * dz).sum(1) / np.sqrt((dx ** 2).sum(1) * (dz ** 2).sum(1))
    r[n < 3] = np.nan
    return r


def halves(ys: dict[str, np.ndarray], names: list[str]) -> tuple[np.ndarray, ...]:
    """(P, P_odd, P_even), each (n_models, n_prompts), in `names` order. P_odd uses sample
    columns 0, 2, … as castor.reliability.split_half does."""
    P = np.stack([reliability.per_prompt_p(ys[m]) for m in names])
    Po = np.stack([reliability.per_prompt_p(ys[m][:, 0::2]) for m in names])
    Pe = np.stack([reliability.per_prompt_p(ys[m][:, 1::2]) for m in names])
    return P, Po, Pe


def reliabilities(Po: np.ndarray, Pe: np.ndarray) -> np.ndarray:
    return np.array([reliability.spearman_brown(r) for r in rowwise_corr(Po, Pe)])


# ---- levels ---------------------------------------------------------------------------------


def pair_masks(infos: list[ModelInfo]) -> dict[str, np.ndarray]:
    """Upper-triangle boolean masks (n, n) of the MZ / U / R pairs."""
    n = len(infos)
    out = {lvl: np.zeros((n, n), bool) for lvl in reliability.LEVELS}
    for i, j in combinations(range(n), 2):
        lvl = reliability.pair_level(infos[i], infos[j])
        if lvl is not None:
            out[lvl][i, j] = True
    return out


def levels(P: np.ndarray, rel: np.ndarray, infos: list[ModelInfo],
           masks: dict | None = None) -> dict:
    """Per recipe MZ / U / R (raw and corrected), their means over recipes, the primary
    contrast and the overall cross-recipe level."""
    masks = pair_masks(infos) if masks is None else masks
    r = corr_matrix(P)
    with np.errstate(invalid="ignore"):
        rc = r / np.sqrt(np.outer(rel, rel))
    rc[~(np.outer(rel > 0, rel > 0))] = np.nan
    recipes = sorted({i.recipe for i in infos})
    rec = np.array([i.recipe for i in infos])
    per = {}
    for q in recipes:
        inq = rec == q
        both, one = np.outer(inq, inq), np.logical_xor.outer(inq, inq)
        d = {}
        for lvl, sel in (("MZ", masks["MZ"] & both), ("U", masks["U"] & both),
                         ("R", masks["R"] & one)):
            d[lvl] = {"n_pairs": int(sel.sum()),
                      "r": _nanmean(r[sel]), "r_corrected": _nanmean(rc[sel])}
        d["MZ_minus_U_corrected"] = d["MZ"]["r_corrected"] - d["U"]["r_corrected"]
        d["U_minus_R_corrected"] = d["U"]["r_corrected"] - d["R"]["r_corrected"]
        d["MZ_minus_U"] = d["MZ"]["r"] - d["U"]["r"]
        d["U_minus_R"] = d["U"]["r"] - d["R"]["r"]
        d["mean_reliability"] = _nanmean(rel[inq])
        per[q] = d
    summ = summarise(per)
    summ["R_overall"] = {"n_pairs": int(masks["R"].sum()), "r": _nanmean(r[masks["R"]]),
                         "r_corrected": _nanmean(rc[masks["R"]])}
    return {"per_recipe": per, **summ}


def summarise(per: dict, recipes: list[str] | None = None) -> dict:
    """Means over recipes (a list with repeats = one recipe-bootstrap draw)."""
    recipes = sorted(per) if recipes is None else recipes
    keys = ("MZ_minus_U_corrected", "U_minus_R_corrected", "MZ_minus_U", "U_minus_R")
    out = {k: _nanmean([per[q][k] for q in recipes]) for k in keys}
    for lvl in reliability.LEVELS:
        for f in ("r", "r_corrected"):
            out[f"{lvl}_{f}"] = _nanmean([per[q][lvl][f] for q in recipes])
    out["recipe_minus_seed_effect"] = out["U_minus_R_corrected"] - out["MZ_minus_U_corrected"]
    out["n_recipes"] = len(set(recipes))
    mz = {q for q in recipes if per[q]["MZ"]["n_pairs"] > 0}
    out["n_recipes_MZ"] = len(mz)
    out["n_recipes_MZ_undefined"] = len({q for q in mz
                                         if not np.isfinite(per[q]["MZ_minus_U_corrected"])})
    return out


def seeds_per_recipe(infos) -> dict[str, set[str]]:
    """recipe -> the post-training seeds present among `infos` (ModelInfo values)."""
    out: dict[str, set[str]] = {}
    for i in infos:
        out.setdefault(i.recipe, set()).add(i.posttrain_seed)
    return out


def _nanmean(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def _ci(draws: list[dict], point: dict, alpha: float = 0.05) -> dict:
    out = {}
    for k, v in point.items():
        if not isinstance(v, float):
            continue
        d = np.array([x[k] for x in draws], dtype=float)
        d = d[np.isfinite(d)]
        lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2]) if len(d) else (np.nan, np.nan)
        out[k] = {"estimate": v, "lo": float(lo), "hi": float(hi), "n_draws": len(d)}
    return out


def recipe_bootstrap(per: dict, n_boot: int, seed: int) -> dict:
    """Recipes resampled with replacement within two strata: with MZ pairs and without (see
    module doc); one stratum = the plain recipe bootstrap."""
    rng = np.random.default_rng(seed)
    has = sorted(q for q in per if per[q]["MZ"]["n_pairs"] > 0)
    rest = sorted(q for q in per if q not in set(has))
    point = summarise(per)
    draws = []
    for _ in range(n_boot):
        pick = [has[i] for i in rng.integers(0, len(has), len(has))] if has else []
        pick += [rest[i] for i in rng.integers(0, len(rest), len(rest))] if rest else []
        draws.append(summarise(per, pick))
    return _ci(draws, point)


def prompt_bootstrap(P: np.ndarray, Po: np.ndarray, Pe: np.ndarray, infos: list[ModelInfo],
                     n_boot: int, seed: int) -> dict:
    """Prompts resampled (the same columns for every model); reliabilities recomputed."""
    rng = np.random.default_rng(seed)
    masks = pair_masks(infos)
    full = levels(P, reliabilities(Po, Pe), infos, masks)
    point = {k: v for k, v in full.items() if k != "per_recipe" and k != "R_overall"}
    draws, per_recipe = [], []
    n = P.shape[1]
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        lv = levels(P[:, idx], reliabilities(Po[:, idx], Pe[:, idx]), infos, masks)
        per_recipe.append(lv.pop("per_recipe"))
        lv.pop("R_overall")
        draws.append(lv)
    out = _ci(draws, point)
    # per-recipe intervals for the figure (the point estimate's recipes, so n_boot = 0 gives the
    # same keys with empty intervals, as _ci does)
    out["per_recipe"] = {}
    for q in full["per_recipe"]:
        pq = {f"{lvl}_{f}": [d[q][lvl][f] for d in per_recipe]
              for lvl in reliability.LEVELS for f in ("r", "r_corrected")}
        pq["MZ_minus_U_corrected"] = [d[q]["MZ_minus_U_corrected"] for d in per_recipe]
        out["per_recipe"][q] = {k: _q(v) for k, v in pq.items()}
    return out


def _q(v) -> dict:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {"lo": float(np.quantile(v, 0.025)), "hi": float(np.quantile(v, 0.975))} if len(v) \
        else {"lo": float("nan"), "hi": float("nan")}


def prediction(lv: dict, var: dict) -> dict:
    """The three parts of the pre-registered prediction, read off (reported, no gate)."""
    per = {q: d for q, d in lv["per_recipe"].items() if d["MZ"]["n_pairs"] > 0}
    # recipes with two post-training seeds only; an undefined (NaN) level is not an order
    order = {q: bool(d["MZ"]["r_corrected"] > d["U"]["r_corrected"] > d["R"]["r_corrected"])
             for q, d in per.items()}
    return {"n_recipes_MZ_gt_U_gt_R": sum(order.values()), "n_recipes": len(per),
            "n_recipes_MZ_undefined": sum(not np.isfinite(d["MZ"]["r_corrected"])
                                          for d in per.values()),
            "most_recipes": sum(order.values()) > len(per) / 2, "per_recipe": order,
            "recipe_effect_ge_seed_effect": bool(lv["U_minus_R_corrected"]
                                                 >= lv["MZ_minus_U_corrected"]),
            "recipe_var_gt_seed_var": bool(var["components"]["recipe"]
                                           > var["components"]["pretrain_seed"])}


# ---- secondaries ----------------------------------------------------------------------------


def nested_variance(means: dict[str, float], infos: dict[str, ModelInfo]) -> dict:
    """Balanced nested random-effects ANOVA of per-model mean vulnerability: recipe >
    pretraining seed (base) > post-training seed. Method-of-moments components (Searle,
    Casella & McCulloch 1992, ch. 4); negative estimates are reported and truncated at 0 in the
    shares."""
    cells: dict[str, dict[str, list[float]]] = {}
    for m, v in means.items():
        i = infos[m]
        cells.setdefault(i.recipe, {}).setdefault(i.pretrain_seed, []).append(v)
    a = len(cells)
    bs = {len(s) for s in cells.values()}
    ns = {len(v) for s in cells.values() for v in s.values()}
    assert len(bs) == 1 and len(ns) == 1, "nested ANOVA needs a balanced design"
    b, n = bs.pop(), ns.pop()
    assert a > 1 and b > 1, "nested ANOVA needs ≥ 2 recipes and ≥ 2 bases per recipe"
    y = np.array([[cells[q][s] for s in sorted(cells[q])] for q in sorted(cells)])  # a,b,n
    gm = y.mean()
    rm, bm = y.mean(axis=(1, 2)), y.mean(axis=2)
    ms_r = b * n * np.sum((rm - gm) ** 2) / (a - 1)
    ms_b = n * np.sum((bm - rm[:, None]) ** 2) / (a * (b - 1))
    ms_e = np.sum((y - bm[..., None]) ** 2) / (a * b * (n - 1)) if n > 1 else float("nan")
    comp = {"recipe": (ms_r - ms_b) / (b * n),
            # n = 1: the base level also holds post-training-seed variance (module doc)
            "pretrain_seed": (ms_b - ms_e) / n if n > 1 else ms_b,
            "posttrain_seed": ms_e}
    pos = {k: max(v, 0.0) for k, v in comp.items() if np.isfinite(v)}
    tot = sum(pos.values())
    return {"a_recipes": a, "b_bases": b, "n_posttrain": n,
            "mean_squares": {"recipe": ms_r, "pretrain_seed": ms_b, "posttrain_seed": ms_e},
            "components": comp, "shares": {k: v / tot for k, v in pos.items()},
            "sd_recipe_means": float(rm.std(ddof=1)),
            "mean_within_recipe_sd_of_bases": float(bm.std(axis=1, ddof=1).mean())}


def additive_residual(p: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """p − least-squares fit of intercept + row effect + column effect (behaviour + template)
    over the finite entries; NaN stays NaN. Unbalanced and incomplete designs are fine
    (the fitted values are unique even though the dummy coding is rank-deficient)."""
    ok = np.isfinite(p)
    ur, ri = np.unique(rows, return_inverse=True)
    uc, ci = np.unique(cols, return_inverse=True)
    D = np.zeros((len(p), len(ur) + len(uc)))
    D[np.arange(len(p)), ri] = 1
    D[np.arange(len(p)), len(ur) + ci] = 1
    beta, *_ = np.linalg.lstsq(D[ok], p[ok], rcond=None)
    out = np.full(len(p), np.nan)
    out[ok] = p[ok] - D[ok] @ beta
    return out


def residualise(P: np.ndarray, Po: np.ndarray, Pe: np.ndarray, rows: np.ndarray,
                cols: np.ndarray) -> tuple[np.ndarray, ...]:
    """Each model's P, P_odd and P_even residualised separately (the ceiling of the
    residual is the split-half reliability of the residualised halves)."""
    f = lambda M: np.stack([additive_residual(x, rows, cols) for x in M])
    return f(P), f(Po), f(Pe)


def top_decile_transfer(P: np.ndarray, Po: np.ndarray, Pe: np.ndarray, infos: list[ModelInfo],
                        frac: float = TOP_FRACTION, seed: int = 0) -> dict:
    """For each ordered pair (A, B): S = the top `frac` of prompts by P_A,odd (ties
    broken by one fixed random order of prompts), then mean P_A,even over S (A's own ceiling,
    free of selection bias) and mean P_B over S; transfer = mean P_B / mean P_A,even.
    Per recipe (A in the recipe) and level (MZ, U: B in the recipe; R: B in another), the mean
    over ordered pairs of both means and of the ratio."""
    n_models, n = P.shape
    k = int(np.ceil(frac * n))
    tie = np.random.default_rng(seed).permutation(n)
    rec = np.array([i.recipe for i in infos])
    lvl = np.full((n_models, n_models), "", dtype=object)
    for i, j in combinations(range(n_models), 2):
        lvl[i, j] = lvl[j, i] = reliability.pair_level(infos[i], infos[j]) or ""
    acc: dict[str, dict[str, list]] = {}
    for a in range(n_models):
        score = np.where(np.isfinite(Po[a]), Po[a], -np.inf)
        top = np.lexsort((tie, -score))[:k]
        own = np.nanmean(Pe[a, top])
        other = np.nanmean(P[:, top], axis=1)
        d = acc.setdefault(rec[a], {})
        for b in range(n_models):
            if lvl[a, b]:
                e = d.setdefault(lvl[a, b], {"own": [], "other": [], "ratio": []})
                e["own"].append(own)
                e["other"].append(other[b])
                e["ratio"].append(other[b] / own if own > 0 else np.nan)
    per = {q: {L: {"n_ordered_pairs": len(v["own"]), "mean_P_A_even": _nanmean(v["own"]),
                   "mean_P_B": _nanmean(v["other"]), "transfer": _nanmean(v["ratio"])}
               for L, v in d.items()} for q, d in acc.items()}
    mean = {L: {f: _nanmean([per[q][L][f] for q in per if L in per[q]])
                for f in ("mean_P_A_even", "mean_P_B", "transfer")} for L in reliability.LEVELS}
    return {"k_prompts": k, "per_recipe": per, "mean_over_recipes": mean}


def reference_correlation(P: np.ndarray, rel: np.ndarray, p_ref: np.ndarray, rel_ref: float,
                          names: list[str]) -> dict:
    """Descriptive: each model against a released reference model."""
    r = corr_matrix(P, p_ref[None, :])[:, 0]
    return {m: {"r": float(r[i]), "r_corrected": float(r[i] / np.sqrt(rel[i] * rel_ref))
                if rel[i] > 0 and rel_ref > 0 else float("nan")}
            for i, m in enumerate(names)}


def ratio_bootstrap(per: dict, n_boot: int, seed: int, num: str = "MZ_minus_U_corrected",
                    den: str = "U_minus_R_corrected", alpha: float = 0.05) -> dict:
    """The ratio of two means over recipes (default: mean MZ − U over mean U − R,
    the seed-over-recipe ratio) with `recipe_bootstrap`'s resampling (same strata, same rng
    calls, so each draw is the same recipe pick and `numerator`/`denominator` reproduce its
    intervals); percentile interval of the ratio of the drawn means. Also the interval of the
    difference num − den, and the fraction of draws whose denominator mean is ≤ 0."""
    rng = np.random.default_rng(seed)
    has = sorted(q for q in per if per[q]["MZ"]["n_pairs"] > 0)
    rest = sorted(q for q in per if q not in set(has))

    def means(recipes: list[str]) -> tuple[float, float]:
        return (_nanmean([per[q][num] for q in recipes]), _nanmean([per[q][den] for q in recipes]))

    n0, d0 = means(sorted(per))
    draws = []
    for _ in range(n_boot):
        pick = [has[i] for i in rng.integers(0, len(has), len(has))] if has else []
        pick += [rest[i] for i in rng.integers(0, len(rest), len(rest))] if rest else []
        draws.append(means(pick))
    n, d = np.array(draws).T

    def ci(x: np.ndarray, est: float) -> dict:
        x = x[np.isfinite(x)]
        lo, hi = np.quantile(x, [alpha / 2, 1 - alpha / 2]) if len(x) else (np.nan, np.nan)
        return {"estimate": float(est), "lo": float(lo), "hi": float(hi), "n_draws": len(x)}

    pos = d > 0
    return {"ratio": ci(np.where(pos, n / np.where(pos, d, 1.0), np.nan),
                        n0 / d0 if d0 > 0 else float("nan")),
            "frac_den_nonpositive": float(np.mean(~pos)),
            "difference": ci(n - d, n0 - d0), "numerator": ci(n, n0), "denominator": ci(d, d0)}
