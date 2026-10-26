"""Audit gate reporting: the gate tier, fragility of the design-weighted accuracy to single-item
flips, and a bootstrap over items within the audit strata.

Every accuracy here is the design-weighted one of `audit_score.agreement` (weights N_h / n_h) on
the same scorable items (human and judge both coherent), so the point estimates equal the
scorer's.
"""

import math
import random
from collections import defaultdict
from collections.abc import Callable, Collection, Sequence

import numpy as np

GATE, TIER1_FLOOR, MODEL_FLOOR = 0.85, 0.80, 0.75
# Weighted accuracies are float sums of non-integer design weights (N_h / n_h), so an accuracy
# exactly at a threshold can come out 1 ulp below it. The pre-registered rules read ">=", so
# "at the threshold" is decided explicitly: within EPS counts as at or above.
EPS = 1e-9


def at_or_above(acc: float, threshold: float) -> bool:
    return acc >= threshold - EPS


def tier(overall_w: float, worst_model_w: float) -> str:
    """GO; tier 1 = overall in [0.80, 0.85) and no model < 0.75; tier 2 otherwise
    (overall < 0.80, any model < 0.75, or an undefined accuracy)."""
    if math.isnan(overall_w) or math.isnan(worst_model_w):
        return "KILL, tier 2"
    model_ok = at_or_above(worst_model_w, MODEL_FLOOR)
    if at_or_above(overall_w, GATE) and model_ok:
        return "GO"
    if at_or_above(overall_w, TIER1_FLOOR) and model_ok:
        return "KILL of the 0.85 gate, tier 1"
    return "KILL, tier 2"


def weighted_accuracy(correct: Sequence[bool], w: Sequence[float]) -> float:
    tot = float(sum(w))
    return sum(wi for c, wi in zip(correct, w) if c) / tot if tot else float("nan")


def flips_to_cross(correct: Sequence[bool], w: Sequence[float], threshold: float) -> dict:
    """Fewest single-item label flips that move the weighted accuracy across `threshold`.

    Above (acc at or above threshold, `at_or_above`): correct items turned wrong, largest
    weight first, until acc is below. Below: wrong items turned correct, largest weight first,
    until acc is at or above. Largest-weight-first gives the minimum count. n_flips is None if
    no number of flips crosses (all items already on the far side)."""
    tot = float(sum(w))
    acc = weighted_accuracy(correct, w)
    down = at_or_above(acc, threshold)
    pool = sorted((wi for c, wi in zip(correct, w) if c == down), reverse=True)
    num, k = acc * tot, None
    for i, wi in enumerate(pool, 1):
        num += -wi if down else wi
        if at_or_above(num / tot, threshold) != down:
            k = i
            break
    med = float(np.median(w)) if len(w) else float("nan")
    return {"threshold": threshold, "accuracy": acc, "direction": "down" if down else "up",
            "n_flips": k, "margin": acc - threshold,
            "delta_per_flip_median_weight": med / tot if tot else float("nan"),
            "delta_per_flip_max_weight": (max(w) / tot) if tot else float("nan")}


def bootstrap_groups(items: list[dict], collapse: Callable[[dict], str] | None = None
                     ) -> list[list[dict]]:
    """Resampling groups: the audit strata (`item['stratum']`), with one-item strata collapsed.

    A one-item stratum carries no within-stratum variance information, so it is merged
    (the collapsed-stratum estimator; Wolter 2007, *Introduction to Variance Estimation*):
    one-item strata sharing `collapse(item)` (e.g. the (model, source) cell) pool together; a
    pool still of one item joins the smallest stratum with the same collapse key, else the
    smallest group overall (a pool met before any group exists waits until all pools are placed;
    if no group exists at all, the lone singletons form one group). Collapsing overstates variance
    slightly (conservative)."""
    by = defaultdict(list)
    for x in items:
        by[x["stratum"]].append(x)
    groups = {("s", h): g for h, g in by.items() if len(g) > 1}
    pools = defaultdict(list)
    for g in by.values():
        if len(g) == 1:
            pools[collapse(g[0]) if collapse else ""].append(g[0])
    waiting = []
    for key, p in sorted(pools.items()):
        if len(p) > 1:
            groups[("c", key)] = p
            continue
        same = [k for k, g in groups.items()
                if k[0] == "s" and collapse and collapse(g[0]) == key]
        cand = same or list(groups)
        if not cand:
            waiting += p
            continue
        k = min(cand, key=lambda k: (len(groups[k]), k))
        groups[k] = groups[k] + p
    if waiting and not groups:
        if len(waiting) < 2:
            raise ValueError("bootstrap needs at least two items")
        groups[("c", "")] = waiting
    elif waiting:
        k = min(groups, key=lambda k: (len(groups[k]), k))
        groups[k] = groups[k] + waiting
    return [groups[k] for k in sorted(groups)]


def stratified_bootstrap(items: list[dict], stat: Callable[[list[dict]], dict],
                         n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
                         collapse: Callable[[dict], str] | None = None,
                         nan_low: Collection[str] | Callable[[str], bool] = (),
                         point: dict | None = None) -> dict:
    """Rao-Wu rescaled bootstrap within the audit strata (Rao & Wu 1988, JASA 83:231; Rao, Wu
    & Yue 1992, Survey Methodology 18:209), percentile interval per key of `stat`.

    Each group (a stratum, one-item strata collapsed, `bootstrap_groups`) of n_g items is
    resampled m_g = n_g - 1 times with replacement; item i drawn r_i times gets weight
    w_i * (1 - lam_i + lam_i * n_g / (n_g - 1) * r_i), lam_i = sqrt(1 - f_i), f_i = 1 / w_i the
    stratum sampling fraction n_h / N_h. This removes the naive bootstrap's (n_h - 1) / n_h
    variance shrinkage, which is large for the audit's 2-5-item strata, and applies the finite
    population correction (a fully sampled stratum has no variance). `stat` must read the
    weights (`item['weight']`); every item appears once per draw.

    Undefined (NaN) draws are not dropped. Keys matching `nan_low` (accuracies read against a
    floor, where undefined fails the rule as in `tier`) count them as -inf. For other keys the
    interval is NaN if any draw is undefined; `frac_undefined` is reported for every key."""
    groups = [(g, [math.sqrt(max(0.0, 1.0 - 1.0 / x["weight"])) for x in g])
              for g in bootstrap_groups(items, collapse)]
    is_low = nan_low if callable(nan_low) else (lambda k, s=frozenset(nan_low): k in s)
    rng = random.Random(seed)
    point = stat(items) if point is None else point
    draws = defaultdict(list)
    for _ in range(n_boot):
        s = []
        for g, lams in groups:
            n = len(g)
            r = [0] * n
            for _ in range(n - 1):
                r[rng.randrange(n)] += 1
            for x, ri, lam in zip(g, r, lams):
                s.append({**x, "weight": x["weight"] * (1 - lam + lam * n / (n - 1) * ri)})
        for k, v in stat(s).items():
            draws[k].append(v)
    out = {}
    for k, v in point.items():
        d = np.array(draws[k], dtype=float)
        undef = np.isnan(d)
        if is_low(k):
            d = np.where(undef, -np.inf, d)
        ok = len(d) and not np.isnan(d).any()
        lo, hi = (np.quantile(d, [alpha / 2, 1 - alpha / 2], method="inverted_cdf") if ok
                  else (np.nan, np.nan))
        out[k] = {"point": v, "lo": float(lo), "hi": float(hi),
                  "frac_undefined": float(undef.mean()) if len(d) else float("nan")}
    return out


def source_cell(x: dict) -> str:
    """Collapse key for one-item strata: the (model, source) cell. `model` is the stratum's
    first key (a model or, in the recipe-stratified supplement, a recipe)."""
    return f"{x['model']}|{x['source']}"


def check_complete(keyrows: dict, sheets: list[dict], gold: dict, info: dict) -> None:
    """Stop unless every key item has gold (no blanks, no unadjudicated disagreement)."""
    unknown = set().union(*sheets) - set(keyrows)
    blank = [sorted(set(keyrows) - set(s)) for s in sheets]
    missing = set(keyrows) - set(gold)
    if unknown or info["unresolved"] or any(blank) or missing:
        raise SystemExit(
            f"audit incomplete, nothing computed (D043c blind order): {info['unresolved']} "
            f"disagreements not adjudicated; unlabelled items per sheet "
            f"{[len(b) for b in blank]}; {len(missing)} of {len(keyrows)} items without gold; "
            f"{len(unknown)} sheet items not in key.jsonl")
