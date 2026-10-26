"""Attack leaderboards under reseeding: how often two jailbreak templates' ASR order flips
between two models, by pair kind (MZ twins, unrelated runs of one recipe, cross-recipe pairs).
Pure functions on template shares; scripts/leaderboard_074.py feeds them the grid.

Pre-registered definitions:
  - a template's ASR on one model = its success share, castor.stability.template_shares (mean P
    over the template's prompts, P = mean WildGuard `wg_vulnerable` over coherent samples);
    S[m, t] holds model m's shares;
  - a template pair (i < j; 91 templates -> 4,095 pairs) has sign s_m = sign(S[m, i] - S[m, j]) on
    model m. Between models a and b the pair *flips* when s_a = -s_b != 0 (the order on a is the
    reverse of the order on b); an exact tie on one model and not the other counts 1/2, a tie
    on both counts 0. A cell = (model pair, template pair);
  - ASR-gap bins (points; one point = 0.01): [0, 2), [2, 5), [5, 10), [10, 20), [20, 100], on the
    absolute gap of the pair's mean shares over the recipe's six models (MZ and U pairs) and,
    for cross-recipe pairs (reference only), over the two recipes' twelve models. The mean
    contains the two compared models;
  - pooled flip rate of a bin = flipping cells / cells, summed over the pair kind's model pairs
    (MZ 3, U 12 per recipe; R 36 per recipe pair) and the bin's template pairs;
  - top-k change between two models: top-1 = the templates at the maximum share (ties shared),
    changed with probability 1 - |M_a ∩ M_b| / (|M_a| |M_b|) (uniform tie-break); top-5 = the
    five highest shares, changed when the two sets differ (a tie at the fifth place is counted
    in `n_tied_top5` and broken by the higher mean share over the recipe(s), then by name);
  - per-recipe Spearman: mean over a recipe's pairs of a kind of the raw and the reliability-
    corrected Spearman of template shares, castor.stability.template_levels on the recipe's six
    models; the mean over recipes equals the grid level (equal pair counts per recipe);
  - intervals: 95 % percentile bootstrap, 1,000 draws, seed 0, over data recipes for pooled
    rates (within-recipe counts weighted by each recipe's draw count; a cross-recipe pair of
    recipes weighted by the product of the two counts, pairs of a recipe with itself skipped),
    over templates for per-recipe rates (a template pair weighted by the product
    of its two templates' draw counts; the pair's bin is fixed by the full-sample means).
Secondary (labelled, not pre-registered): the reversal rate binned by the gap *observed on one
model* (ordered pairs: a shows template i ahead by a gap in the bin; b reverses it; b's tie
counts 1/2; a's ties are excluded), which is the quantity an attack paper's single-checkpoint
comparison reads."""

from collections.abc import Sequence
from itertools import combinations

import numpy as np

POINT = 0.01
EDGES = (0.0, 0.02, 0.05, 0.10, 0.20)  # bin b = [EDGES[b], EDGES[b + 1]); the last is open
BIN_LABELS = ("0-2", "2-5", "5-10", "10-20", ">20")
PRIMARY_BIN = 1  # 2-5 points
TOP5 = 5


def template_pairs(n_t: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n_t, 1)


def gap_bins(mean_share: np.ndarray, iu: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Bin index of each template pair from |mean_i - mean_j|."""
    gap = np.abs(mean_share[iu[0]] - mean_share[iu[1]])
    return np.searchsorted(np.asarray(EDGES), gap, side="right") - 1


def signs(S: np.ndarray, iu: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """(n_models, n_t) shares -> (n_models, n_pairs) order signs in {-1, 0, 1}."""
    return np.sign(S[:, iu[0]] - S[:, iu[1]]).astype(np.int8)


def flip(sa: np.ndarray, sb: np.ndarray) -> np.ndarray:
    """Per cell: 1 when the orders are reversed, 1/2 when exactly one model ties, else 0."""
    p = sa.astype(np.int16) * sb
    one_tie = (sa == 0) ^ (sb == 0)
    return (p < 0) + 0.5 * one_tie


def bin_sums(F: np.ndarray, bins: np.ndarray, n_bins: int = len(EDGES)) -> tuple[np.ndarray,
                                                                                   np.ndarray]:
    """F (n_model_pairs, n_tp) flips, bins (n_tp,) -> per bin (flip sum, cells)."""
    F = np.atleast_2d(F)
    f = np.bincount(bins, weights=F.sum(axis=0), minlength=n_bins)
    c = np.bincount(bins, minlength=n_bins) * F.shape[0]
    return f, c.astype(float)


def observed_gap_sums(S: np.ndarray, sg: np.ndarray, pairs: Sequence[tuple[int, int]],
                      iu: tuple[np.ndarray, np.ndarray], n_bins: int = len(EDGES)):
    """Secondary: ordered pairs (a, b) and (b, a) of each pair; cells binned by the gap observed
    on the first model, a's ties excluded; reversal on the second, its tie 1/2."""
    f, c = np.zeros(n_bins), np.zeros(n_bins)
    for x, y in pairs:
        for a, b in ((x, y), (y, x)):
            ok = sg[a] != 0
            bins = gap_bins(S[a], iu)[ok]
            rev = (sg[b][ok] == -sg[a][ok]) + 0.5 * (sg[b][ok] == 0)
            f += np.bincount(bins, weights=rev, minlength=n_bins)
            c += np.bincount(bins, minlength=n_bins)
    return f, c


# ---- top-k ---------------------------------------------------------------------------------


def top1_change(sa: np.ndarray, sb: np.ndarray) -> float:
    """P(the top-1 template differs) under a uniform tie-break among each model's maxima."""
    ma, mb = sa == sa.max(), sb == sb.max()
    return float(1.0 - (ma & mb).sum() / (ma.sum() * mb.sum()))


def top_set(s: np.ndarray, k: int, tiebreak: np.ndarray) -> tuple[frozenset, bool]:
    """The k highest shares (ties broken by `tiebreak` descending, then index); and whether a tie
    straddles the k-th place."""
    order = np.lexsort((np.arange(len(s)), -tiebreak, -s))
    tied = bool(s[order[k - 1]] == s[order[k]]) if len(s) > k else False
    return frozenset(order[:k].tolist()), tied


def top5_change(sa: np.ndarray, sb: np.ndarray, tiebreak: np.ndarray) -> tuple[float, float, int]:
    """(changed 0/1, overlap share |A ∩ B| / 5, number of the two models with a tie at 5th)."""
    a, ta = top_set(sa, TOP5, tiebreak)
    b, tb = top_set(sb, TOP5, tiebreak)
    return float(a != b), len(a & b) / TOP5, int(ta) + int(tb)


# ---- recipe structure ----------------------------------------------------------------------


def within_pairs(rows: Sequence[int], bases: Sequence[str]) -> dict[str, list[tuple[int, int]]]:
    """Model pairs inside one recipe: MZ (same base) and U (other base)."""
    out = {"MZ": [], "U": []}
    for x, y in combinations(range(len(rows)), 2):
        out["MZ" if bases[x] == bases[y] else "U"].append((rows[x], rows[y]))
    return out


def ratio(f: np.ndarray, c: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, f / np.maximum(c, 1e-300), np.nan)


def pooled_within(f: np.ndarray, c: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """f, c (n_recipes, ...) -> pooled rate with recipe weights w (draw counts; all ones if None)."""
    w = np.ones(f.shape[0]) if w is None else np.asarray(w, dtype=float)
    return ratio(np.tensordot(w, f, 1), np.tensordot(w, c, 1))


def pooled_cross(f: np.ndarray, c: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """f, c (n_recipes, n_recipes, ...) symmetric with zero diagonal -> pooled rate over recipe
    pairs weighted by w_r w_s (pairs of one recipe with itself skipped: zero diagonal)."""
    n = f.shape[0]
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    W = np.outer(w, w)
    np.fill_diagonal(W, 0.0)
    return ratio(np.tensordot(W, f, 2), np.tensordot(W, c, 2))


def recipe_draws(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) draw counts of recipes resampled with replacement."""
    rng = np.random.default_rng(seed)
    return np.vstack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)])


def template_pair_weights(n_t: int, iu: tuple[np.ndarray, np.ndarray], n_boot: int,
                          seed: int) -> np.ndarray:
    """(n_boot, n_pairs): w_i w_j for templates resampled with replacement."""
    rng = np.random.default_rng(seed)
    W = np.vstack([np.bincount(rng.integers(0, n_t, n_t), minlength=n_t) for _ in range(n_boot)])
    return (W[:, iu[0]] * W[:, iu[1]]).astype(float)


def percentile(draws, alpha: float = 0.05) -> list[float]:
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if not len(d):
        return [float("nan"), float("nan")]
    return [float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))]
