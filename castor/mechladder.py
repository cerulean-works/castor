"""The projection ladder and the twin-difference link.

Per model m and prompt i of a subset, the readout is x_i · r̂_m: the model's activation at the
common (layer, position) projected on its own unit refusal direction (castor.directions).
No alignment between models is needed: two readouts on different spaces are compared by their
values on shared prompts.

Rows of one model are packed into one array so that castor.reliability's prompt bootstrap,
base-cluster bootstrap and jackknife resample every quantity together:

    columns [0]            projection on the full direction (256 + 256 training prompts)
            [1 : 1+S]      projection on half-direction 1 of split s (128 + 128)
            [1+S : 1+2S]   projection on half-direction 2 of split s (the other halves)
            [1+2S : ]      the K sampled outcomes (1 = not refused, NaN = unusable)

**Reliability of the projection** (2609.14759, 2609.30275): for each of
the S random half-splits, the Pearson correlation across prompts of the two half-direction
projections, Spearman–Brown corrected, then averaged over splits.

**Ladder** = castor.reliability.twin_correlations with this measure: per-pair Pearson r of the
projections, divided by √(rel_a · rel_b); MZ (same base) and U (different pretraining seed)
means; primary = corrected r_MZ − r_U.

**Link** (U pairs, A, B): link = −corr(z(proj_A) − z(proj_B), P_A − P_B), z within model over
the subset; positive = the twin that projects more refuses more. Corrected by
√(rel(Δproj) · rel(ΔP)): rel(ΔP) = Spearman–Brown of the correlation between the even-sample
and odd-sample ΔP; rel(Δproj) = the mean over splits of the Spearman–Brown correlation between
the half-1 and half-2 Δproj (each half's projections z-scored within model). Same-base (MZ)
pairs are reported raw only.

Reading rule (pre-registered): corrected U link ≥ 0.5 "runs through the refusal direction";
0.3 – 0.5 "partly"; < 0.3 "elsewhere" (Cohen 1988 conventions).
"""

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from castor import reliability as R

LINK_READING = ((0.5, "runs through the refusal direction"), (0.3, "partly"),
                (-np.inf, "elsewhere"))


def pack(full: np.ndarray, h1: np.ndarray, h2: np.ndarray, y: np.ndarray | None = None
         ) -> np.ndarray:
    cols = [np.asarray(full, float)[:, None], np.asarray(h1, float), np.asarray(h2, float)]
    if y is not None:
        cols.append(np.asarray(y, float))
    return np.concatenate(cols, axis=1)


def unpack(a: np.ndarray, n_splits: int):
    s = n_splits
    return a[:, 0], a[:, 1:1 + s], a[:, 1 + s:1 + 2 * s], a[:, 1 + 2 * s:]


def colcorr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation of matching columns of two (n, S) arrays (finite rows only,
    shared across columns: rows with any NaN in either are dropped)."""
    ok = np.all(np.isfinite(x), 1) & np.all(np.isfinite(y), 1)
    x, y = x[ok] - x[ok].mean(0), y[ok] - y[ok].mean(0)
    den = np.sqrt((x * x).sum(0) * (y * y).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (x * y).sum(0) / den, np.nan)


def zscore(x: np.ndarray) -> np.ndarray:
    """Standardise columns over finite rows (population sd)."""
    x = np.asarray(x, float)
    mu = np.nanmean(x, 0)
    sd = np.nanstd(x, 0)
    return (x - mu) / np.where(sd > 0, sd, np.nan)


def split_half_mean(h1: np.ndarray, h2: np.ndarray) -> float:
    """Mean over splits of the Spearman–Brown corrected half correlation."""
    r = colcorr(h1, h2)
    return float(np.nanmean(R.spearman_brown(r)))


def projection_measure(n_splits: int):
    """Measure for R.twin_correlations: (projection values, split-half reliability)."""
    def measure(a: np.ndarray, method: str = "pearson") -> tuple[np.ndarray, float]:
        assert method == "pearson", "the projection reliability is Pearson-based"
        full, h1, h2, _ = unpack(a, n_splits)
        return full, split_half_mean(h1, h2)
    return measure


def ladder(n_splits: int, level_of=R.pair_level, levels: Sequence[str] = R.LEVELS):
    """analyze() for R.bootstrap_*: the ladder on packed arrays. `level_of` names a pair's
    level (MZ / U / R by default, or castor.factorial's levels) and `levels` lists the levels
    summarised."""
    measure = projection_measure(n_splits)

    def analyze(ys, info, method="pearson", pairs=None):
        return R.twin_correlations(ys, info, method, pairs=pairs, measure=measure,
                                   level_of=level_of, levels=list(levels))
    return analyze


def ladder_summary(res: dict) -> dict:
    """Every level's raw and corrected mean (r_<level>, r_<level>_corrected) and, when MZ and
    U are both summarised, MZ_minus_U and MZ_minus_U_corrected."""
    lv = res["levels"]
    out = {}
    for lvl, v in lv.items():
        out[f"r_{lvl}"] = v["r"]
        out[f"r_{lvl}_corrected"] = v["r_corrected"]
    if "MZ" in lv and "U" in lv:
        out |= {"MZ_minus_U": lv["MZ"]["r"] - lv["U"]["r"],
                "MZ_minus_U_corrected": lv["MZ"]["r_corrected"] - lv["U"]["r_corrected"]}
    out["mean_reliability"] = float(np.nanmean(list(res["reliability"].values())))
    return out


def pack_two(ref: tuple, harm: tuple, y: np.ndarray) -> np.ndarray:
    """Refusal readout (full, h1, h2), harm-direction readout (full, h1,
    h2) and the outcomes in one array, so a bootstrap resamples both ladders together."""
    return np.concatenate([pack(*ref), pack(*harm, y)], axis=1)


def unpack_two(a: np.ndarray, n_splits: int) -> tuple[np.ndarray, np.ndarray]:
    w = 1 + 2 * n_splits
    return a[:, :w], a[:, w:]


def ladder_contrast(n_splits: int):
    """analyze() for R.bootstrap_*: the ladder of the refusal readout and of the harm readout
    on the same (re)sampled prompts (arrays from pack_two)."""
    an = ladder(n_splits)

    def analyze(ys, info, method="pearson", pairs=None):
        parts = {m: unpack_two(a, n_splits) for m, a in ys.items()}
        return {"refusal": an({m: v[0] for m, v in parts.items()}, info, method, pairs),
                "harm": an({m: v[1] for m, v in parts.items()}, info, method, pairs)}
    return analyze


def ladder_contrast_summary(res: dict) -> dict:
    """Corrected (MZ − U)_refusal − (MZ − U)_harm and both ladders' summaries."""
    r, h = ladder_summary(res["refusal"]), ladder_summary(res["harm"])
    out = {f"refusal_{k}": v for k, v in r.items()} | {f"harm_{k}": v for k, v in h.items()}
    out["gap_refusal_minus_harm_corrected"] = (r["MZ_minus_U_corrected"]
                                               - h["MZ_minus_U_corrected"])
    out["gap_refusal_minus_harm"] = r["MZ_minus_U"] - h["MZ_minus_U"]
    return out


def site_point(ys: dict, info: dict, n_splits: int) -> dict:
    """The primary gap and the link at one site, point estimates (the gap is NaN for a set
    without MZ pairs)."""
    lad = ladder_summary(ladder(n_splits)(ys, info))
    lk = link_summary(link(n_splits)(ys, info))
    return {"MZ_minus_U_corrected": lad.get("MZ_minus_U_corrected", float("nan")),
            "MZ_minus_U": lad.get("MZ_minus_U", float("nan")),
            "mean_reliability": lad["mean_reliability"],
            "link_U_corrected": lk["link_U_corrected"], "link_U": lk["link_U"]}


def features(a: np.ndarray, n_splits: int) -> dict:
    """Per-model quantities the link needs, computed once per (re)sample."""
    full, h1, h2, y = unpack(a, n_splits)
    return {"z": zscore(full), "z1": zscore(h1), "z2": zscore(h2), "p": R.per_prompt_p(y),
            "p_even": R.per_prompt_p(y[:, 0::2]), "p_odd": R.per_prompt_p(y[:, 1::2])}


def pair_link_f(fa: dict, fb: dict, corrected: bool = True) -> dict:
    raw = -R.corr(fa["z"] - fb["z"], fa["p"] - fb["p"])
    out = {"link": raw}
    if corrected:
        rel_dp = R.spearman_brown(R.corr(fa["p_even"] - fb["p_even"], fa["p_odd"] - fb["p_odd"]))
        rel_dproj = split_half_mean(fa["z1"] - fb["z1"], fa["z2"] - fb["z2"])
        out |= {"rel_dP": rel_dp, "rel_dproj": rel_dproj,
                "link_corrected": R.disattenuate(raw, rel_dproj, rel_dp)}
    return out


def pair_link(a: np.ndarray, b: np.ndarray, n_splits: int, corrected: bool = True) -> dict:
    """The link for one pair of packed arrays (rows aligned on prompts)."""
    return pair_link_f(features(a, n_splits), features(b, n_splits), corrected)


def link(n_splits: int, level_of=R.pair_level, corrected_levels: Sequence[str] = ("U",),
         raw_levels: Sequence[str] = ("MZ",)):
    """analyze() for R.bootstrap_*: link per pair; `corrected_levels` (different-base pairs,
    U) get the disattenuated link, `raw_levels` (same-base pairs, MZ) the raw one; other levels
    are dropped."""
    corrected, raw = tuple(corrected_levels), tuple(raw_levels)
    assert not set(corrected) & set(raw)

    def analyze(ys, info, method="pearson", pairs=None):
        assert method == "pearson"
        names = sorted(ys)
        pairs = list(combinations(names, 2)) if pairs is None else list(pairs)
        levels = {(a, b): level_of(info[a], info[b]) for a, b in pairs}
        keep = set(corrected) | set(raw)
        used = {m for (a, b), lvl in levels.items() if lvl in keep for m in (a, b)}
        f = {m: features(ys[m], n_splits) for m in used}
        rows = [{"a": a, "b": b, "level": lvl, **pair_link_f(f[a], f[b], lvl in corrected)}
                for (a, b), lvl in levels.items() if lvl in keep]
        return {"pairs": rows, "corrected_levels": list(corrected), "raw_levels": list(raw)}
    return analyze


def link_summary(res: dict) -> dict:
    """Per level: link_<level> (raw mean), n_<level>; corrected levels add
    link_<level>_corrected, rel_dP_<level>, rel_dproj_<level>. The keys link_U,
    link_U_corrected, rel_dP_U, rel_dproj_U, link_MZ_raw, n_U, n_MZ are always present."""
    def m(rows, k):
        return float(np.nanmean([r[k] for r in rows])) if rows else float("nan")
    corrected = res.get("corrected_levels", ["U"])
    raw = res.get("raw_levels", ["MZ"])
    out = {}
    for lvl in list(corrected) + list(raw):
        rows = [r for r in res["pairs"] if r["level"] == lvl]
        out[f"link_{lvl}"] = m(rows, "link")
        out[f"n_{lvl}"] = len(rows)
        if lvl in corrected:
            out[f"link_{lvl}_corrected"] = m(rows, "link_corrected")
            out[f"rel_dP_{lvl}"] = m(rows, "rel_dP")
            out[f"rel_dproj_{lvl}"] = m(rows, "rel_dproj")
    u = [r for r in res["pairs"] if r["level"] == "U"]
    mz = [r for r in res["pairs"] if r["level"] == "MZ"]
    out |= {"link_U": m(u, "link"), "link_U_corrected": m(u, "link_corrected"),
            "rel_dP_U": m(u, "rel_dP"), "rel_dproj_U": m(u, "rel_dproj"),
            "link_MZ_raw": m(mz, "link"), "n_U": len(u), "n_MZ": len(mz)}
    return out


def link_reading(value: float) -> str:
    if not np.isfinite(value):
        return "undefined"
    return next(text for lo, text in LINK_READING if value >= lo)


def validity(a: np.ndarray, n_splits: int) -> dict:
    """Per-twin validity: projection vs P, Pearson and Spearman, raw and
    corrected by √(rel_proj · rel_P) (rel_P = sample split-half, Pearson or Spearman)."""
    full, h1, h2, y = unpack(a, n_splits)
    p = R.per_prompt_p(y)
    rel_proj = split_half_mean(h1, h2)
    out = {"rel_proj": rel_proj}
    for method in ("pearson", "spearman"):
        r = R.corr(full, p, method)
        rel_p = R.split_half(y, method)["reliability"]
        out[method] = {"r": r, "rel_P": rel_p, "r_corrected": R.disattenuate(r, rel_proj, rel_p)}
    return out


def direction_cosines(dirs: dict[str, np.ndarray], half1: dict[str, np.ndarray],
                      half2: dict[str, np.ndarray], info: dict[str, R.ModelInfo]) -> dict:
    """Cosines between models' unit directions, no alignment. Reliability of a
    direction = mean over splits of the Spearman–Brown half-direction cosine (2609.14759);
    same-base pairs are also reported disattenuated."""
    rel = {m: float(np.mean(R.spearman_brown(np.sum(half1[m] * half2[m], 1)))) for m in dirs}
    rows = []
    for a, b in combinations(sorted(dirs), 2):
        lvl = R.pair_level(info[a], info[b])
        c = float(np.dot(dirs[a], dirs[b]))
        rows.append({"a": a, "b": b, "level": lvl, "cos": c,
                     "cos_disattenuated": R.disattenuate(c, rel[a], rel[b])
                     if lvl == "MZ" else None})
    levels = {}
    for lvl in ("MZ", "U"):
        sel = [r for r in rows if r["level"] == lvl]
        if not sel:
            levels[lvl] = {"n_pairs": 0}
            continue
        levels[lvl] = {"n_pairs": len(sel), "cos": float(np.mean([r["cos"] for r in sel])),
                       "cos_abs_max": float(np.max(np.abs([r["cos"] for r in sel])))}
        if lvl == "MZ":
            levels[lvl]["cos_disattenuated"] = float(np.mean([r["cos_disattenuated"]
                                                              for r in sel]))
    return {"reliability": rel, "pairs": rows, "levels": levels}


def ablation_verdict(base: dict[str, float], own: dict[str, float],
                     random: dict[str, float]) -> dict:
    """Causal at 1B: in every twin, plain-harmful vulnerability under own-direction
    ablation >= 2 × the unablated value; and for each random-control twin, |random − base| <
    (own − base) / 4. `random` covers the control twins only."""
    per = {}
    for m in sorted(base):
        inc = own[m] - base[m]
        row = {"base": base[m], "own": own[m], "increase": inc,
               "doubles": bool(own[m] >= 2 * base[m])}
        if m in random:
            row["random"] = random[m]
            row["random_change"] = random[m] - base[m]
            row["random_ok"] = bool(abs(random[m] - base[m]) < inc / 4)
        per[m] = row
    causal = all(r["doubles"] for r in per.values()) and \
        all(r.get("random_ok", True) for r in per.values()) and len(random) > 0
    return {"per_model": per, "causal": bool(causal),
            "rule": "every twin: own-ablation plain-harmful vulnerability >= 2x base; random "
                    "control |change| < 1/4 of that twin's own-ablation increase"}


def levels_of(names: Sequence[str], info: dict[str, R.ModelInfo]) -> dict[str, int]:
    out: dict[str, int] = {}
    for a, b in combinations(sorted(names), 2):
        lvl = R.pair_level(info[a], info[b])
        if lvl:
            out[lvl] = out.get(lvl, 0) + 1
    return out


# ---- two sites on the same pairs and prompts, the paired difference ----------------------------


def two_readouts(n_splits: int, analyze_one, summary_one, names: Sequence[str] = ("site",
                                                                                   "compare")):
    """(analyze, summary) for R.bootstrap_* on arrays from `pack_two(first, second, y)`: the
    same analysis on each readout with the shared outcomes, and `diff_<key>` = first − second
    for every numeric summary key, so a bootstrap draw resamples both readouts together (e.g. the
    link at the fixed site minus the link at the common site, on the same pairs and prompts)."""
    a, b = names

    def analyze(ys, info, method="pearson", pairs=None):
        parts = {m: unpack_two(arr, n_splits) for m, arr in ys.items()}
        y = {m: unpack(v[1], n_splits)[3] for m, v in parts.items()}
        first = {m: np.concatenate([v[0], y[m]], axis=1) for m, v in parts.items()}
        return {a: analyze_one(first, info, method, pairs),
                b: analyze_one({m: v[1] for m, v in parts.items()}, info, method, pairs)}

    def summary(res):
        sa, sb = summary_one(res[a]), summary_one(res[b])
        out = {f"{a}_{k}": v for k, v in sa.items()} | {f"{b}_{k}": v for k, v in sb.items()}
        for k, v in sa.items():
            if isinstance(v, (int, float, np.floating, np.integer)) and k in sb \
                    and not k.startswith("n_"):
                out[f"diff_{k}"] = float(v) - float(sb[k])
        return out
    return analyze, summary


def jackknife_groups(ys: dict, info: dict, groups: dict[str, str], stat, analyze,
                     method: str = "pearson") -> dict:
    """Leave-one-group-out jackknife SE of every summary key (e.g. groups = recipes). Mirrors
    R.jackknife_bases with an arbitrary grouping."""
    point = stat(analyze(ys, info, method))
    names = sorted({groups[m] for m in ys})
    reps = []
    for g in names:
        keep = {m: y for m, y in ys.items() if groups[m] != g}
        reps.append(stat(analyze(keep, info, method)))
    out = {}
    for k, v in point.items():
        if not isinstance(v, (int, float, np.floating, np.integer)):
            continue
        d = np.array([r[k] for r in reps], dtype=float)
        d = d[np.isfinite(d)]
        se = float(np.sqrt((len(d) - 1) / len(d) * np.sum((d - d.mean()) ** 2))) \
            if len(d) > 1 else float("nan")
        out[k] = {"estimate": float(v), "se": se, "n_reps": len(d), "n_groups": len(names)}
    return out


def prediction_021(site_link: dict, diff_link: dict, floor: float = 0.5,
                   exclude: float = 0.3) -> dict:
    """The pre-registered rule: (1) the corrected U link at the fixed site (13, −1) is ≥ 0.5
    with its prompt-bootstrap CI excluding 0.3; (2) it exceeds the link at the common site
    (11, −2) on the same pairs, with the paired CI of the difference excluding
    0. Both parts are required; each is reported holds / fails. Inputs are `{"estimate", "lo",
    "hi"}` dicts of the prompt bootstrap."""
    p1 = bool(site_link["estimate"] >= floor and site_link["lo"] > exclude)
    p2 = bool(diff_link["estimate"] > 0 and diff_link["lo"] > 0)
    return {"part1": {"holds": p1, "link": site_link,
                      "rule": f"link >= {floor} and CI lower bound > {exclude}"},
            "part2": {"holds": p2, "difference": diff_link,
                      "rule": "link(site) - link(compare) > 0 with the paired CI excluding 0"},
            "holds": p1 and p2, "reading_site": link_reading(site_link["estimate"])}
