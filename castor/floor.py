"""The nondeterminism floor, rung 2 of the ladder: per-prompt twin correlations between the
post-trained twins of released `seed-default` and of our same-seed replicate (init 6198, data
order 6198, our hardware), beside the MZ and U levels recomputed by the same code.

Bases (DCLM-Baseline 1B, step 69,369; post-training seeds k = 1..4 on each):
    default, large-aux-2, large-aux-3   released
    replicate                           castor-dclm-1B-i6198-d6198
Pair groups (every model pair, all k × k'):
    MZ_006      same released base (18 pairs)
    U_006       two released bases (48 pairs)
    rung2       default × replicate (the pre-registered primary, 16 pairs)
    MZ_rep      within the replicate (6 pairs; descriptive)
    rep_other   replicate × large-aux-2 / large-aux-3 (32 pairs; descriptive: a different seed
                across hardware, the cross-hardware analogue of U_006)
Correlations, reliabilities and the ceiling correction are castor.reliability's; only the
grouping of pairs is new. Group values are means over pairs of per-pair corrected (and raw)
Pearson r.

Pre-registered prediction: rung2 lies strictly between r_U and r_MZ, nearer r_U; falsified if
the prompt-bootstrap CI of rung2 covers r_U or r_MZ, or the point lies nearer r_MZ. The seed
jackknife is leave-one-post-trained-model-out over the 8 models of the rung-2 pair (with two
bases a base jackknife is undefined).
"""

from collections.abc import Callable

import numpy as np

from castor import reliability, sanityb

REPLICATE = "replicate"
DEFAULT = "default"
SEEDS_011 = (*sanityb.PRETRAIN_SEEDS, REPLICATE)  # the closed set of bases
GROUPS = ("MZ_006", "U_006", "rung2", "MZ_rep", "rep_other")


def _check_labels(a: str, b: str, seeds: tuple[str, ...]) -> None:
    """The pair groups are closed: a label outside `seeds` (e.g. the manifest's revision
    `replicate2-i6198-d6198` passed where the base label `replicate2` belongs) raises rather
    than falling silently into the released-base groups."""
    unknown = {a, b} - set(seeds)
    if unknown:
        raise ValueError(f"unknown pretraining-seed label(s) {sorted(unknown)}; "
                         f"the bases are {seeds}")


def group_of(a: str, b: str) -> str:
    """Pair group of two pretraining seeds (base labels), closed over `SEEDS_011`."""
    _check_labels(a, b, SEEDS_011)
    if a == b:
        return "MZ_rep" if a == REPLICATE else "MZ_006"
    if REPLICATE not in (a, b):
        return "U_006"
    return "rung2" if DEFAULT in (a, b) else "rep_other"


def _group_means(pairs: list[dict], info: dict[str, reliability.ModelInfo],
                 groups: tuple[str, ...], grouper: Callable[[str, str], str]
                 ) -> tuple[dict[str, float], dict[str, list[dict]]]:
    """Per group: pair count `n_<g>`, mean raw r `r_<g>` and mean corrected r `r_<g>_corrected`
    (nanmean over the group's twin pairs; NaN for an empty group). Returns the means and the
    pair rows by group. Shared by `summary` and `summary_035` (only the grouping differs)."""
    by: dict[str, list[dict]] = {g: [] for g in groups}
    for row in pairs:
        by[grouper(info[row["a"]].pretrain_seed, info[row["b"]].pretrain_seed)].append(row)
    out: dict[str, float] = {}
    for g, rows in by.items():
        out[f"n_{g}"] = len(rows)
        out[f"r_{g}"] = float(np.nanmean([x["r"] for x in rows])) if rows else float("nan")
        out[f"r_{g}_corrected"] = (float(np.nanmean([x["r_corrected"] for x in rows]))
                                   if rows else float("nan"))
    return out, by


def summary(res: dict, info: dict[str, reliability.ModelInfo]) -> dict[str, float]:
    """Means over pairs per group (raw and corrected) and the contrasts rung2 − U_006,
    MZ_006 − rung2, and rung2's position in the U → MZ gap."""
    out, _ = _group_means(res["pairs"], info, GROUPS, group_of)
    for sfx in ("", "_corrected"):
        mz, u, r2 = out[f"r_MZ_006{sfx}"], out[f"r_U_006{sfx}"], out[f"r_rung2{sfx}"]
        out[f"rung2_minus_U{sfx}"] = r2 - u
        out[f"MZ_minus_rung2{sfx}"] = mz - r2
        out[f"rung2_gap_fraction{sfx}"] = (r2 - u) / (mz - u) if mz != u else float("nan")
        # descriptive, not pre-registered: rung 2 against the replicate's own different-seed
        # pairs (same hardware crossing and configuration on the replicate side)
        out[f"rung2_minus_rep_other{sfx}"] = r2 - out[f"r_rep_other{sfx}"]
    out["mean_reliability"] = float(np.nanmean(list(res["reliability"].values())))
    return out


def stat_for(info: dict[str, reliability.ModelInfo]) -> Callable[[dict], dict]:
    return lambda res: summary(res, info)


def jackknife_models(ys: dict[str, np.ndarray], info: dict[str, reliability.ModelInfo],
                     drop: list[str], z: float = 1.959964,
                     stat: Callable[[dict], dict] | None = None) -> dict:
    """Leave-one-model-out jackknife over `drop`: SE and a normal CI (estimate ± z SE) per
    summary quantity. `stat` defaults to `summary`."""
    stat = stat_for(info) if stat is None else stat
    point = stat(reliability.twin_correlations(ys, info))
    reps = [stat(reliability.twin_correlations({m: y for m, y in ys.items() if m != d}, info))
            for d in drop]
    out = {}
    for k, v in point.items():
        d = np.array([r[k] for r in reps], dtype=float)
        d = d[np.isfinite(d)]
        se = (float(np.sqrt((len(d) - 1) / len(d) * np.sum((d - d.mean()) ** 2)))
              if len(d) > 1 else float("nan"))
        out[k] = {"estimate": v, "se": se, "lo": v - z * se, "hi": v + z * se, "n_reps": len(d)}
    return out


def prediction(r2: float, lo: float, hi: float, r_u: float, r_mz: float) -> dict:
    """The frozen rule: strictly between, nearer r_U, CI covering neither reference."""
    between = r_u < r2 < r_mz
    nearer_u = abs(r2 - r_u) < abs(r2 - r_mz)
    covers_u, covers_mz = lo <= r_u <= hi, lo <= r_mz <= hi
    return {"r_rung2": r2, "ci": [lo, hi], "r_U": r_u, "r_MZ": r_mz,
            "strictly_between": between, "nearer_r_U": nearer_u,
            "ci_covers_r_U": covers_u, "ci_covers_r_MZ": covers_mz,
            "holds": bool(between and nearer_u and not covers_u and not covers_mz)}


# ---- per-base-pair table --------------------------------------------------------------------


def base_pair_key(a: str, b: str) -> str:
    """One key per unordered pair of bases (pretraining-seed labels), 'x|y' in sorted order;
    'x|x' for the within-base (MZ) pairs."""
    return "|".join(sorted((a, b)))


def by_base_pair(pairs: list[dict], info: dict[str, reliability.ModelInfo],
                 grouper: Callable[[str, str], str] = group_of) -> dict[str, dict]:
    """Mean raw and corrected r per base pair, with the min and max of the corrected r over its
    twin pairs (16 across bases, 6 within), and the pair group, from the `pairs` rows (model ids
    a, b, r, r_corrected) that floor_011.py writes. Descriptive: no interval (the prompt CI
    carries no base-pair sampling). `grouper` names the group of a base pair."""
    by: dict[str, list[dict]] = {}
    for row in pairs:
        by.setdefault(base_pair_key(info[row["a"]].pretrain_seed,
                                    info[row["b"]].pretrain_seed), []).append(row)
    out = {}
    for key, rows in sorted(by.items()):
        a, b = key.split("|")
        rc = np.array([x["r_corrected"] for x in rows], dtype=float)
        out[key] = {"group": grouper(a, b), "n_pairs": len(rows),
                    "r": float(np.nanmean([x["r"] for x in rows])),
                    "r_corrected": float(np.nanmean(rc)),
                    "min_corrected": float(np.nanmin(rc)), "max_corrected": float(np.nanmax(rc))}
    return out


# ---- a second same-seed replicate R2 ---------------------------------------------------------
# A second same-seed run on our stack, `replicate2` (castor-dclm-1B-i6198-d6198-r2), beside R
# (`replicate`) and the released D (`default`). Three same-seed base pairs: R-D (different
# stack), R-R2 (same stack), R2-D (different stack, a replicate of R-D). Rung 2 = the mean of the
# three pair means (48 twin pairs). Pre-registered prediction: r(R-R2) within ± 0.02 of the
# earlier r(R-D) and the prompt-bootstrap CI of r(R-R2) − r(R-D) covering 0. Branch: a positive
# difference whose CI excludes 0 adds two labelled rows (2a same stack = R-R2 = `rung2_RR2`, 2b
# different stack = mean of R-D and R2-D = `rung2_diffstack`) beside the three-pair mean;
# otherwise one row. Both rows are summary keys, so each carries the prompt-bootstrap CI and the
# jackknife SE whichever branch the data select.

REPLICATE2 = "replicate2"
SAME_SEED = (DEFAULT, REPLICATE, REPLICATE2)
SEEDS_035 = (*sanityb.PRETRAIN_SEEDS, REPLICATE, REPLICATE2)  # the closed set with R2
RUNG2_PAIRS_035 = ("rung2_RD", "rung2_RR2", "rung2_R2D")
GROUPS_035 = ("MZ_006", "U_006", *RUNG2_PAIRS_035, "MZ_rep", "MZ_R2", "rep_other", "R2_other")
TOL_035 = 0.02  # pre-registered: |r(R-R2) − r(R-D)_027| ≤ 0.02
FP_SLACK_035 = 1e-9  # floating-point slack on the tolerance edge (0.8595 − 0.8395 = 0.02 + 2e-17)


def group_of_035(a: str, b: str) -> str:
    """Pair group of two pretraining seeds (base labels) with the second replicate, closed over
    `SEEDS_035`. The `group_of` groups keep their names and membership (MZ_006, U_006, MZ_rep,
    rep_other); `rung2` is `rung2_RD` here."""
    _check_labels(a, b, SEEDS_035)
    if a == b:
        return {REPLICATE: "MZ_rep", REPLICATE2: "MZ_R2"}.get(a, "MZ_006")
    s = {a, b}
    if s == {DEFAULT, REPLICATE}:
        return "rung2_RD"
    if s == {REPLICATE, REPLICATE2}:
        return "rung2_RR2"
    if s == {DEFAULT, REPLICATE2}:
        return "rung2_R2D"
    if REPLICATE in s:
        return "rep_other"
    if REPLICATE2 in s:
        return "R2_other"
    return "U_006"


def summary_035(res: dict, info: dict[str, reliability.ModelInfo]) -> dict[str, float]:
    """Means over pairs per group (raw and corrected), the three-pair mean `rung2_mean3`
    (mean of the three same-seed pair means; equal to the 48-pair mean when each pair has its
    16 twin pairs), the branch's row 2b `rung2_diffstack` (mean of R-D and R2-D: the different-
    stack pairs; row 2a is `rung2_RR2` itself), the pre-registered difference `RR2_minus_RD`,
    its replicate-side analogue `R2D_minus_RD`, the same-stack minus different-stack contrast
    `RR2_minus_diffstack` (= 2a − 2b), and the mean3 contrasts against U and MZ as `summary`
    gives them for rung 2. Descriptive, as rung2 − rep_other: `RD_minus_rep_other`,
    `RR2_minus_other`, `mean3_minus_other`, where `other` pools rep_other and R2_other (64
    pairs)."""
    out, by = _group_means(res["pairs"], info, GROUPS_035, group_of_035)
    other = by["rep_other"] + by["R2_other"]
    out["n_other"] = len(other)
    for sfx, key in (("", "r"), ("_corrected", "r_corrected")):
        out[f"r_other{sfx}"] = (float(np.nanmean([x[key] for x in other]))
                                if other else float("nan"))
        rd, rr2, r2d = (out[f"r_{g}{sfx}"] for g in RUNG2_PAIRS_035)
        mz, u = out[f"r_MZ_006{sfx}"], out[f"r_U_006{sfx}"]
        m3, diffstack = float(np.mean([rd, rr2, r2d])), (rd + r2d) / 2
        out[f"rung2_mean3{sfx}"] = m3
        out[f"rung2_diffstack{sfx}"] = diffstack
        out[f"RR2_minus_RD{sfx}"] = rr2 - rd
        out[f"R2D_minus_RD{sfx}"] = r2d - rd
        out[f"RR2_minus_diffstack{sfx}"] = rr2 - diffstack
        out[f"mean3_minus_U{sfx}"] = m3 - u
        out[f"MZ_minus_mean3{sfx}"] = mz - m3
        out[f"mean3_gap_fraction{sfx}"] = (m3 - u) / (mz - u) if mz != u else float("nan")
        out[f"RD_minus_rep_other{sfx}"] = rd - out[f"r_rep_other{sfx}"]
        out[f"RR2_minus_other{sfx}"] = rr2 - out[f"r_other{sfx}"]
        out[f"mean3_minus_other{sfx}"] = m3 - out[f"r_other{sfx}"]
    out["mean_reliability"] = float(np.nanmean(list(res["reliability"].values())))
    return out


def stat_for_035(info: dict[str, reliability.ModelInfo]) -> Callable[[dict], dict]:
    return lambda res: summary_035(res, info)


def prediction_035(r_rr2: float, r_rd_027: float, diff: float, diff_lo: float, diff_hi: float,
                   tol: float = TOL_035) -> dict:
    """The pre-registered prediction for R2 and its branch rule. Holds when r(R-R2) lies within
    ± `tol` of the earlier r(R-D) and the prompt-bootstrap CI of r(R-R2) − r(R-D) covers 0.
    Branch: `two_rows` when the difference is positive and its CI excludes 0 (rows 2a same stack
    and 2b different stack beside the three-pair mean); otherwise one row (the mean, three pair
    values shown). A negative difference with a CI excluding 0 is not a pre-registered branch:
    it is flagged (`ci_excludes_zero_negative`) and falls under the one-row rule.

    The tolerance test is `abs_difference_from_027 <= tolerance + tolerance_fp_slack` on exactly
    the numbers reported (slack `FP_SLACK_035`, for a difference landing on the edge in floating
    point), so the printed values and the flag always agree."""
    d = abs(r_rr2 - r_rd_027)
    within = d <= tol + FP_SLACK_035
    covers0 = diff_lo <= 0.0 <= diff_hi
    two_rows = bool(diff > 0.0 and diff_lo > 0.0)
    return {"r_RR2": r_rr2, "r_RD_027": r_rd_027, "tolerance": tol,
            "tolerance_fp_slack": FP_SLACK_035,
            "abs_difference_from_027": d, "within_tolerance": bool(within),
            "RR2_minus_RD": diff, "ci": [diff_lo, diff_hi], "ci_covers_zero": bool(covers0),
            "holds": bool(within and covers0),
            "two_rows": two_rows,
            "branch": "two_rows_2a_2b" if two_rows else "one_row_mean3",
            "ci_excludes_zero_negative": bool(diff < 0.0 and diff_hi < 0.0)}
