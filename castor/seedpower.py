"""How many seeds a safety evaluation needs: variance components of a recipe's
vulnerability level, the standard error of a level under n_pre × n_post seeds, the sign-flip rate
of single-seed recipe comparisons, and the reliability of per-prompt vulnerability averaged over
twins. Pure functions on arrays; scripts/seedpower_045.py feeds them the grid.

Definitions (pre-registered):
  - levels y[a, b, n] = mean vulnerability (mean over prompts of per-prompt P) of the model of
    recipe a, pretraining seed b, post-training seed n; the grid is 25 × 3 × 2;
  - components: balanced nested random-effects ANOVA, method of moments (Searle, Casella &
    McCulloch 1992, ch. 4), the estimator of castor.gridstats.nested_variance; σ²_post(pre) is the
    within-base mean square, so it holds the K = 20 sampling noise of the level. σ²_sampling is
    that noise, from the split-half reliability: a prompt's error variance is (1 − rel)·Var_p(P)
    (classical test theory), and the level's is its mean over prompts / n_prompts (prompts'
    samples are independent). The binomial plug-in Σ p̂(1 − p̂)/(k − 1) / n² is the cross-check.
    σ²_post,true = σ²_post(pre) − σ²_sampling. Negative estimates are reported and set to 0
    wherever they enter a standard error;
  - SE of a recipe's level with n_pre pretraining seeds × n_post post-training seeds each, at
    K = 20 (the instrument): SE² = σ²_pre/n_pre + σ²_post(pre)/(n_pre·n_post). "1 point" = 0.01
    on the rate scale. The smallest design = fewest models (n_pre·n_post), ties broken by fewer
    pretraining seeds (the expensive kind);
  - sign flip: for every unordered recipe pair and every single-seed draw (one model of each
    recipe, (b·n)² draws), the draw flips when its level difference has the opposite sign to the
    all-seed mean difference; an exact tie counts 1/2. Overall = mean over pairs of the per-pair
    rate; also over pairs with |true difference| ≥ 0.02. Model-based companion: a single-draw
    difference ~ N(Δ, 2σ²_single), so P(flip | Δ) = Φ(−|Δ| / sqrt(2σ²_single)), 0.5 at Δ = 0;
  - per-prompt reliability of a recipe's per-prompt vulnerability averaged over n_pre × n_post
    twins, as an estimate of the recipe's expected per-prompt profile: the generalisability
    coefficient of the nested design, r_U / (r_U + (r_MZ − r_U)/n_pre + (1 − r_MZ)/(n_pre·n_post))
    with raw (uncorrected) single-model correlations r_U and r_MZ. With n_post = 1 it is
    Spearman–Brown on r_U (m = n_pre distinct pretraining seeds); m = 1, 2, 3, 6 are the designs
    1×1, 2×1, 3×1, 3×2. The 1 − r_MZ part splits into sampling (1 − rel) and post-training seed
    (rel − r_MZ), rel = mean split-half reliability of the recipe's models;
  - intervals: recipe bootstrap (recipes resampled with replacement, 1,000 draws, seed 0); a
    flip draw skips pairs of two copies of one recipe.
"""

import math
from itertools import combinations

import numpy as np

POINT = 0.01
DESIGN_PRE = (1, 2, 3)
DESIGN_POST = (1, 2, 4)
M_DESIGNS = {1: (1, 1), 2: (2, 1), 3: (3, 1), 6: (3, 2)}


# ---- (i) components -------------------------------------------------------------------------


def anova(y: np.ndarray) -> dict:
    """Nested ANOVA of y[a, b, n] (a recipes, b bases each, n post-training seeds each)."""
    y = np.asarray(y, dtype=float)
    a, b, n = y.shape
    assert a > 1 and b > 1 and n > 1, "needs ≥ 2 at every level"
    gm = y.mean()
    rm, bm = y.mean(axis=(1, 2)), y.mean(axis=2)
    ms_r = b * n * np.sum((rm - gm) ** 2) / (a - 1)
    ms_b = n * np.sum((bm - rm[:, None]) ** 2) / (a * (b - 1))
    ms_e = np.sum((y - bm[..., None]) ** 2) / (a * b * (n - 1))
    return {"recipe": float((ms_r - ms_b) / (b * n)), "pretrain_seed": float((ms_b - ms_e) / n),
            "posttrain_seed": float(ms_e)}


def sampling_variance(P: np.ndarray, rel: float) -> float:
    """Variance of one model's level from its K samples: (1 − rel)·Var_p(P) / n_prompts, over
    the prompts where P is defined (module doc)."""
    p = np.asarray(P, dtype=float)
    p = p[np.isfinite(p)]
    return float((1.0 - rel) * p.var(ddof=1) / len(p))


def binomial_sampling_variance(y: np.ndarray) -> float:
    """Plug-in Σ_i p̂_i(1 − p̂_i)/(k_i − 1) / n² over prompts with ≥ 2 usable samples (k_i usable
    samples of prompt i; prompts with one sample add nothing and are counted in n)."""
    y = np.asarray(y, dtype=float)
    k = np.sum(~np.isnan(y), axis=1)
    ok = k > 0
    s = np.nansum(y, axis=1)
    p = np.where(ok, s / np.maximum(k, 1), np.nan)
    v = np.where(k > 1, p * (1 - p) / np.maximum(k - 1, 1), 0.0)
    n = int(ok.sum())
    return float(v[ok].sum() / n ** 2)


def components(y: np.ndarray, sampling: float) -> dict:
    """ANOVA components plus σ²_sampling, σ²_post,true and the shares (negatives → 0)."""
    c = anova(y)
    c["sampling"] = float(sampling)
    c["posttrain_true"] = c["posttrain_seed"] - c["sampling"]
    parts = {k: max(c[k], 0.0) for k in ("recipe", "pretrain_seed", "posttrain_true", "sampling")}
    tot = sum(parts.values())
    c["shares"] = {k: v / tot for k, v in parts.items()}
    return c


# ---- (ii) standard error of a recipe's level ------------------------------------------------


def se_level(comp: dict, n_pre: int, n_post: int) -> float:
    v_pre, v_post = max(comp["pretrain_seed"], 0.0), max(comp["posttrain_seed"], 0.0)
    return float(np.sqrt(v_pre / n_pre + v_post / (n_pre * n_post)))


def se_table(comp: dict, pres=DESIGN_PRE, posts=DESIGN_POST) -> dict:
    return {f"{p}x{q}": se_level(comp, p, q) for p in pres for q in posts}


def smallest_design(comp: dict, target: float, pres=DESIGN_PRE, posts=DESIGN_POST):
    """(n_pre, n_post) with SE ≤ target and the fewest models, ties to fewer pretraining seeds;
    None if no design in the set reaches it."""
    ok = [(p * q, p, q) for p in pres for q in posts if se_level(comp, p, q) <= target]
    if not ok:
        return None
    _, p, q = min(ok)
    return [p, q]


# ---- (iii) sign flips -----------------------------------------------------------------------


def pair_flip(xa: np.ndarray, xb: np.ndarray, delta: float) -> float:
    """Share of single draws (one level of each) whose difference has the opposite sign to
    `delta` (the true difference a − b); ties count 1/2. delta == 0 gives NaN."""
    if delta == 0:
        return float("nan")
    d = np.subtract.outer(np.ravel(xa), np.ravel(xb))
    s = np.sign(delta)
    return float(np.mean((np.sign(d) == -s) + 0.5 * (d == 0)))


def flip_rates(y: np.ndarray, truth: np.ndarray | None = None, recipes=None) -> list[dict]:
    """Per unordered pair of `recipes` (indices into y; default all, a list with repeats = one
    bootstrap draw; pairs of one recipe with itself are skipped): the true difference (default
    the all-seed mean difference) and the single-draw flip rate."""
    y = np.asarray(y, dtype=float)
    a = y.shape[0]
    flat = y.reshape(a, -1)
    truth = flat.mean(axis=1) if truth is None else np.asarray(truth, dtype=float)
    idx = list(range(a)) if recipes is None else list(recipes)
    out = []
    for i, j in combinations(range(len(idx)), 2):
        ri, rj = idx[i], idx[j]
        if ri == rj:
            continue
        delta = float(truth[ri] - truth[rj])
        out.append({"a": ri, "b": rj, "delta": delta, "abs_delta": abs(delta),
                    "flip": pair_flip(flat[ri], flat[rj], delta)})
    return out


def summarise_flips(pairs: list[dict], min_delta: float = 2 * POINT) -> dict:
    f = np.array([p["flip"] for p in pairs])
    d = np.array([p["abs_delta"] for p in pairs])
    big = d >= min_delta
    return {"overall": float(np.nanmean(f)), "n_pairs": len(pairs),
            "ge_2pt": float(np.nanmean(f[big])) if big.any() else float("nan"),
            "n_pairs_ge_2pt": int(big.sum())}


def model_flip(delta: float, var_single: float) -> float:
    """P(sign flip) of one single-draw comparison under normal noise of variance 2·var_single."""
    if var_single <= 0:
        return 0.0 if delta != 0 else 0.5
    return 0.5 * math.erfc(abs(delta) / math.sqrt(2 * var_single) / math.sqrt(2))  # Φ(−|Δ|/sd)


# ---- (iv) per-prompt reliability ------------------------------------------------------------


def g_coefficient(r_u: float, r_mz: float, n_pre: int, n_post: int) -> float:
    """Reliability of the n_pre × n_post twin mean as a measure of the recipe's per-prompt
    profile (module doc). n_post = 1 → Spearman–Brown(r_U, n_pre)."""
    return float(r_u / (r_u + (r_mz - r_u) / n_pre + (1 - r_mz) / (n_pre * n_post)))


# ---- bootstrap ------------------------------------------------------------------------------


def recipe_bootstrap(fn, a: int, n_boot: int = 1000, seed: int = 0) -> list:
    """fn(recipe index array) for n_boot resamples of the a recipes with replacement."""
    rng = np.random.default_rng(seed)
    return [fn(rng.integers(0, a, a)) for _ in range(n_boot)]


def ci(draws, point: float, alpha: float = 0.05) -> dict:
    v = np.asarray([d for d in draws if d is not None and np.isfinite(d)], dtype=float)
    lo, hi = np.quantile(v, [alpha / 2, 1 - alpha / 2]) if len(v) else (np.nan, np.nan)
    return {"estimate": float(point), "lo": float(lo), "hi": float(hi), "n_draws": len(v)}
