"""Which jailbreaks break a model whatever its seeds. Pure functions over per-prompt
vulnerability P (castor.reliability.per_prompt_p).

Stability rule (castor.figdata.stable_shares): a prompt is *stable-broken* in a
group of models if every model's P >= EVERY (0.9), *stable-refused* if no model's P exceeds NONE
(0.1). Groups: a base (its post-training seeds; "seed-stable"), a recipe (all its bases and seeds;
"ancestry-independent"), all recipes ("universal").

Pairs: a prompt *flips* in a pair when one model breaks it (P >= EVERY) and the other does not
(P < EVERY); it is *co-broken* when both break it; the *strong* flip is P >= EVERY on one and
P <= NONE on the other. Pair levels are reliability.pair_level's: MZ (same base), U (same
recipe, other pretraining seed), R (other recipe). Per-prompt pair counts come from group counts
(a group with h breakers among n models holds h(n - h) flipping and h(h - 1)/2 co-broken pairs),
so a level's rate is a mean over prompts and bootstraps by resampling prompts.

Templates (HarmBench human jailbreaks, prompt = behaviour::template): a template's success share
in one model is the mean P over its prompts; its reliability is the sample split-half
(even / odd sample columns -> shares -> Spearman across templates -> Spearman–Brown); pair rank
agreement is the Spearman correlation of two models' shares, corrected by the two
reliabilities (Spearman disattenuation), level = mean over pairs (reliability.twin_correlations
with a template measure; `template_levels` is its vectorised twin, checked equal)."""

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from castor import figdata, reliability

EVERY, NONE = 0.9, 0.1  # figdata.stable_shares defaults
LEVELS = reliability.LEVELS


# ---- groups: stable classes ----------------------------------------------------------------


def groups_of(info: Sequence[reliability.ModelInfo], by: str) -> dict[str, list[int]]:
    """Row indices of `info` grouped by "base" (recipe, pretraining seed), "recipe" or "all"."""
    key = {"base": lambda i: f"{i.recipe}/{i.pretrain_seed}", "recipe": lambda i: i.recipe,
           "all": lambda i: "all"}[by]
    out: dict[str, list[int]] = {}
    for j, i in enumerate(info):
        out.setdefault(key(i), []).append(j)
    return dict(sorted(out.items()))


def group_stable(P: np.ndarray, groups: dict[str, list[int]]) -> dict:
    """P (n_models, n_prompts), finite. Per group, the stable shares (figdata.stable_shares on the
    group's min and max), and per prompt the share of groups in which it is stable-broken /
    stable-refused (`every_frac`, `none_frac`: their means are the mean group shares)."""
    P = np.asarray(P, dtype=float)
    assert np.isfinite(P).all(), "group_stable takes finite P (drop undefined prompts first)"
    per, ev, no = {}, [], []
    for g, rows in groups.items():
        lo, hi = P[rows].min(axis=0), P[rows].max(axis=0)
        per[g] = figdata.stable_shares(lo, hi, EVERY, NONE) | {"n_models": len(rows)}
        ev.append(lo >= EVERY)
        no.append(hi <= NONE)
    ev, no = np.array(ev), np.array(no)
    return {"per_group": per, "n_groups": len(groups),
            "models_per_group": sorted({len(r) for r in groups.values()}),
            "every_frac": ev.mean(axis=0), "none_frac": no.mean(axis=0),
            "n_every": ev.sum(axis=0), "n_none": no.sum(axis=0)}


# ---- pairs: flips and co-breaks ------------------------------------------------------------


def _pairs_within(flag_a: np.ndarray, flag_b: np.ndarray | None, rows: list[int]) -> np.ndarray:
    """Per prompt, unordered pairs inside `rows`: with flag_b None, pairs where both are
    flagged (h(h-1)/2); else pairs with one model flag_a and the other flag_b, flag_a and flag_b
    disjoint (h_a h_b) or equal-complement (h(n-h))."""
    ha = flag_a[rows].sum(axis=0)
    if flag_b is None:
        return ha * (ha - 1) / 2
    return ha * flag_b[rows].sum(axis=0)


def pair_counts(P: np.ndarray, info: Sequence[reliability.ModelInfo]) -> dict:
    """Per level (MZ / U / R) and per prompt: `flip` (one >= EVERY, the other < EVERY),
    `strong` (one >= EVERY, the other <= NONE), `both` (both >= EVERY) pair counts, and the
    level's number of pairs. MZ = within-base sums, U = within-recipe minus MZ, R = all minus
    within-recipe."""
    P = np.asarray(P, dtype=float)
    assert np.isfinite(P).all(), "pair_counts takes finite P"
    hi, lo = P >= EVERY, P <= NONE
    nb = ~hi
    tiers = {}
    for by in ("base", "recipe", "all"):
        gs = groups_of(info, by).values()
        tiers[by] = {
            "flip": sum(_pairs_within(hi, nb, r) for r in gs),
            "strong": sum(_pairs_within(hi, lo, r) for r in gs),
            "both": sum(_pairs_within(hi, None, r) for r in gs),
            "n_pairs": sum(len(r) * (len(r) - 1) // 2 for r in gs)}
    out = {}
    for lvl, (a, b) in {"MZ": ("base", None), "U": ("recipe", "base"),
                        "R": ("all", "recipe")}.items():
        out[lvl] = {k: tiers[a][k] - (tiers[b][k] if b else 0)
                    for k in ("flip", "strong", "both", "n_pairs")}
    return out


def pair_rates(c: dict, idx: np.ndarray | None = None) -> dict:
    """One level's counts -> rates over the prompts `idx` (all if None): flip and strong-flip
    shares and co-break share (per pair, the share of prompts; mean over pairs), and transfer =
    P(the other model breaks | one model breaks) over ordered pairs = 2 both / (2 both + flip)."""
    sel = (lambda v: v) if idx is None else (lambda v: v[idx])
    n = c["n_pairs"]
    f, s, b = sel(c["flip"]), sel(c["strong"]), sel(c["both"])
    m = len(f)
    if n == 0 or m == 0:
        return dict.fromkeys(("flip", "strong_flip", "cobreak", "transfer"), float("nan"))
    den = 2 * b.sum() + f.sum()
    return {"flip": float(f.sum() / (n * m)), "strong_flip": float(s.sum() / (n * m)),
            "cobreak": float(b.sum() / (n * m)),
            "transfer": float(2 * b.sum() / den) if den else float("nan")}


# ---- templates -----------------------------------------------------------------------------


def template_index(templates: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """Sorted template names and each prompt's column into them."""
    names = sorted(set(templates))
    pos = {t: i for i, t in enumerate(names)}
    return names, np.array([pos[t] for t in templates])


def _shares(p: np.ndarray, col: np.ndarray, n_t: int) -> np.ndarray:
    """Mean of the finite p per template (NaN if a template has none)."""
    ok = np.isfinite(p)
    s = np.bincount(col[ok], weights=p[ok], minlength=n_t)
    c = np.bincount(col[ok], minlength=n_t)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def template_shares(y: np.ndarray, col: np.ndarray, n_t: int) -> dict:
    """One model's (n_prompts, K) outcome rows -> template success shares from all samples and
    from the even / odd sample columns."""
    y = np.asarray(y, dtype=float)
    return {"full": _shares(reliability.per_prompt_p(y), col, n_t),
            "even": _shares(reliability.per_prompt_p(y[:, 0::2]), col, n_t),
            "odd": _shares(reliability.per_prompt_p(y[:, 1::2]), col, n_t)}


def template_measure(col: np.ndarray, n_t: int):
    """A `measure` for reliability.twin_correlations: (template shares, Spearman split-half
    reliability), so twin_correlations(..., method="spearman") is the twin ladder on templates."""
    def measure(y: np.ndarray, method: str) -> tuple[np.ndarray, float]:
        s = template_shares(y, col, n_t)
        r = reliability.corr(s["even"], s["odd"], method)
        return s["full"], reliability.spearman_brown(r)
    return measure


def rank_rows(m: np.ndarray) -> np.ndarray:
    """Average ranks per row (ties share the mean rank), as reliability._rank."""
    return np.vstack([reliability._rank(r) for r in m])


def _rowcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))


def level_masks(info: Sequence[reliability.ModelInfo]) -> dict[str, np.ndarray]:
    """Upper-triangle boolean masks (n_models, n_models) of each level's pairs."""
    n = len(info)
    out = {lvl: np.zeros((n, n), dtype=bool) for lvl in LEVELS}
    for a, b in combinations(range(n), 2):
        lvl = reliability.pair_level(info[a], info[b])
        if lvl is not None:
            out[lvl][a, b] = True
    return out


def template_levels(S: np.ndarray, E: np.ndarray, O: np.ndarray, masks: dict[str, np.ndarray],
                    idx: np.ndarray | None = None) -> dict:
    """Shares (n_models, n_templates) from all / even / odd samples -> per level the mean raw
    and corrected Spearman over its pairs, on the templates `idx` (a bootstrap draw; all if None).
    Templates must be finite in every model (the caller drops the others)."""
    if idx is not None:
        S, E, O = S[:, idx], E[:, idx], O[:, idx]
    rS = rank_rows(S)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(rS)
    rel = reliability.spearman_brown(_rowcorr(rank_rows(E), rank_rows(O)))
    rel = np.where(rel > 0, rel, np.nan)
    Cc = C / np.sqrt(np.outer(rel, rel))
    out = {}
    for lvl, mk in masks.items():
        out[f"r_{lvl}"] = float(np.nanmean(C[mk])) if mk.any() else float("nan")
        out[f"r_{lvl}_corrected"] = float(np.nanmean(Cc[mk])) if mk.any() else float("nan")
    return out


# ---- bootstrap -----------------------------------------------------------------------------


def percentile(draws: Sequence[float], alpha: float = 0.05) -> list[float]:
    d = np.asarray([x for x in draws if np.isfinite(x)], dtype=float)
    if not len(d):
        return [float("nan"), float("nan")]
    return [float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))]


def boot_indices(n: int, n_boot: int, seed: int) -> list[np.ndarray]:
    """The draws of one resampling stream (rows with replacement), fixed by `seed`."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(n_boot)]
