"""Adopted twins (080): the same pretraining run post-trained on a different SFT safety set.

Two arms on the main recipe's three bases: Tulu (the reference SFT mixture, post-training seeds
1-4, 008's A0 cell) and swap (the swap mixture, seeds 1-2). A model's arm rides in its
`ModelInfo.posttrain_seed` as "<arm>:<k>". Pair levels:

    same base     Tulu x swap -> adopted     swap x swap -> swapMZ     Tulu x Tulu -> MZ
    other base    Tulu x swap -> cross       swap x swap -> swapU      Tulu x Tulu -> U

Correlations, reliabilities, the prompt bootstrap and the base jackknife are
`castor.reliability`'s, unchanged; this module only names the levels and the reading rule.
"""

import numpy as np

from castor import gridstats
from castor.reliability import ModelInfo, twin_correlations

TULU, SWAP = "tulu", "swap"
LEVELS = ("adopted", "swapMZ", "swapU", "cross", "MZ", "U")
CEILING = 0.9  # mean P above this is flagged as a reason a correlation may be compressed


def info(recipe: str, pretrain_seed: str, arm: str, k: int) -> ModelInfo:
    assert arm in (TULU, SWAP), arm
    return ModelInfo(recipe, pretrain_seed, f"{arm}:{k}")


def arm(i: ModelInfo) -> str:
    return i.posttrain_seed.split(":", 1)[0]


def level_of(a: ModelInfo, b: ModelInfo) -> str | None:
    """The 080 level of a pair on one recipe; None for the same model or two recipes."""
    if a.recipe != b.recipe or a == b:
        return None
    arms = {arm(a), arm(b)}
    same = a.pretrain_seed == b.pretrain_seed
    if arms == {TULU, SWAP}:
        return "adopted" if same else "cross"
    if arms == {SWAP}:
        return "swapMZ" if same else "swapU"
    return "MZ" if same else "U"


def summary(res: dict) -> dict[str, float]:
    """The bootstrapped quantities: raw and corrected r per level, and the contrasts."""
    lv = res["levels"]
    out = {}
    for lvl in LEVELS:
        out[f"r_{lvl}"] = lv[lvl]["r"]
        out[f"r_{lvl}_corrected"] = lv[lvl]["r_corrected"]
    c = {k: lv[k]["r_corrected"] for k in LEVELS}
    out["swapMZ_minus_swapU_corrected"] = c["swapMZ"] - c["swapU"]
    out["MZ_minus_U_corrected"] = c["MZ"] - c["U"]
    out["adopted_minus_U_corrected"] = c["adopted"] - c["U"]
    out["MZ_minus_adopted_corrected"] = c["MZ"] - c["adopted"]
    # not pre-registered (080 note): what ancestry still adds across the two safety sets
    out["adopted_minus_cross_corrected"] = c["adopted"] - c["cross"]
    return out


def reading(estimate: float, lo: float, r_u: float, swap_u: float) -> dict:
    """The pre-registered reading rule (D125): above r_u (008's U of record) with the interval
    excluding it -> the run outweighs the safety data; between the swap arm's own U and r_u ->
    the run's residual partly survives; at or below the swap U -> scope "within a fixed safety
    recipe". The prediction holds only if the point is above r_u and the lower bound is too."""
    if estimate > r_u:
        band = "above r_U"
    elif estimate > swap_u:
        band = "between swap U and r_U"
    else:
        band = "at or below swap U"
    return {"band": band, "prediction_holds": bool(estimate > r_u and lo > r_u),
            "point_above_r_U": bool(estimate > r_u), "interval_excludes_r_U": bool(lo > r_u)}


def level_row(p: np.ndarray, rel: float) -> dict:
    """One model's level: mean P over defined prompts, its split-half reliability, the flag."""
    m = float(np.nanmean(p))
    return {"mean_vulnerability": m, "reliability": float(rel),
            "n_prompts_defined": int(np.isfinite(p).sum()), "ceiling_flag": bool(m > CEILING)}


# ---- 093: two reads added after the data (not pre-registered; D129) --------------------------


def _ci(draws: list[dict], point: dict, alpha: float = 0.05) -> dict:
    out = {}
    for k, v in point.items():
        d = np.array([x[k] for x in draws], dtype=float)
        n_all = len(d)
        d = d[np.isfinite(d)]
        lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2]) if len(d) else (np.nan, np.nan)
        out[k] = {"estimate": float(v), "lo": float(lo), "hi": float(hi), "n_draws": len(d),
                  "n_draws_undefined": n_all - len(d)}
    return out


def source_split(ys: dict[str, np.ndarray], info: dict, groups: dict[str, np.ndarray],
                 n_boot: int = 1000, seed: int = 0) -> dict:
    """The 080 levels on each group of prompt rows (reliabilities recomputed within the group),
    and each group minus the first one. Prompts are resampled within each group (stratified), one
    draw serving every group, so the group differences have intervals too."""
    names = list(groups)

    def stats(idx: dict[str, np.ndarray]) -> dict:
        per = {g: summary(twin_correlations({m: y[idx[g]] for m, y in ys.items()}, info,
                                            level_of=level_of, levels=LEVELS)) for g in names}
        out = {f"{g}.{k}": v for g in names for k, v in per[g].items()}
        for g in names[1:]:
            out.update({f"{g}_minus_{names[0]}.{k}": per[g][k] - per[names[0]][k]
                        for k in per[g]})
        return out

    rows = {g: np.asarray(r, dtype=int) for g, r in groups.items()}
    point = stats(rows)
    rng = np.random.default_rng(seed)
    draws = [stats({g: r[rng.integers(0, len(r), len(r))] for g, r in rows.items()})
             for _ in range(n_boot)]
    return {"n_prompts": {g: len(r) for g, r in rows.items()}, "ci_prompts": _ci(draws, point)}


def ladder_arms(info: dict) -> dict[str, dict]:
    """(B)'s arms on the swap arm's recipes: swap; Tulu at the swap arm's cells (recipe, base,
    post-training seed k); Tulu at 008's cells (seeds 1-2 on every recipe)."""
    swap = {m: i for m, i in info.items() if arm(i) == SWAP}
    cells = {(i.recipe, i.pretrain_seed, i.posttrain_seed.split(":")[1]) for i in swap.values()}
    tulu = {m: i for m, i in info.items() if arm(i) == TULU}
    recipes = {c[0] for c in cells}
    matched = {m: i for m, i in tulu.items()
               if (i.recipe, i.pretrain_seed, i.posttrain_seed.split(":")[1]) in cells}
    grid008 = {m: i for m, i in tulu.items()
               if i.recipe in recipes and i.posttrain_seed in ("tulu:1", "tulu:2")}
    assert len(matched) == len(swap), "a swap cell has no Tulu model"
    return {"tulu_matched": matched, "swap": swap, "tulu_grid008": grid008}


def ladder(ys: dict[str, np.ndarray], arms: dict[str, dict], n_boot: int = 1000,
           seed: int = 0) -> dict:
    """008's per-recipe MZ / U / R (`gridstats.levels`, the grid's definitions) computed for each
    arm on its own models, plus the seed-over-recipe ratio and each arm's difference from the
    first. `arms` maps an arm name to {model id: ModelInfo}; one prompt bootstrap (the same rows
    for every arm and model) gives every interval."""
    names = list(arms)
    mats = {}
    for a in names:
        ms = sorted(arms[a])
        mats[a] = (gridstats.halves(ys, ms), [arms[a][m] for m in ms])
    masks = {a: gridstats.pair_masks(mats[a][1]) for a in names}
    keys = ("MZ_r_corrected", "U_r_corrected", "R_r_corrected", "MZ_minus_U_corrected",
            "U_minus_R_corrected")

    def stats(idx: np.ndarray | None) -> tuple[dict, dict]:
        out, per = {}, {}
        for a in names:
            (P, Po, Pe), infos = mats[a]
            if idx is not None:
                P, Po, Pe = P[:, idx], Po[:, idx], Pe[:, idx]
            lv = gridstats.levels(P, gridstats.reliabilities(Po, Pe), infos, masks[a])
            out.update({f"{a}.{k}": lv[k] for k in keys})
            num, den = lv["MZ_minus_U_corrected"], lv["U_minus_R_corrected"]
            out[f"{a}.seed_over_recipe"] = num / den if den > 0 else float("nan")
            for q, d in lv["per_recipe"].items():
                for lvl in ("MZ", "U", "R"):
                    out[f"{a}.{q}.{lvl}_r_corrected"] = d[lvl]["r_corrected"]
                out[f"{a}.{q}.U_minus_R_corrected"] = d["U_minus_R_corrected"]
            per[a] = lv["per_recipe"]
        for a in names[1:]:
            for k in ("U_minus_R_corrected", "MZ_minus_U_corrected"):
                out[f"{a}_minus_{names[0]}.{k}"] = out[f"{a}.{k}"] - out[f"{names[0]}.{k}"]
        return out, per

    point, per = stats(None)
    n = next(iter(mats.values()))[0][0].shape[1]
    rng = np.random.default_rng(seed)
    draws = [stats(rng.integers(0, n, n))[0] for _ in range(n_boot)]
    return {"models": {a: sorted(arms[a]) for a in names},
            "pairs": {a: {q: {lvl: per[a][q][lvl]["n_pairs"] for lvl in ("MZ", "U", "R")}
                          for q in per[a]} for a in names},
            "ci_prompts": _ci(draws, point)}
