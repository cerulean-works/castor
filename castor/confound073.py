"""073: the refusal-direction readout at (13, -1) under within-last-token centring, and a
refusal-logit control at the same site.

Pure numpy; `scripts/confound_073.py` does the I/O and reuses 007's per-model instrument
(`scripts/ladder_007.py`) and 046's last-token groups (`scripts/harm_046.py`) unchanged.

Packed rows (one model, one prompt per row), so that one prompt resample moves every quantity:

    [0 : 1+2S]   007's projection pack (full, S half-1, S half-2 projections)
    [1+2S]       the refusal-logit readout: Arditi's refusal score (log-odds of the model's own
                 refusal-token set) of the logit lens at the site (final RMSNorm, unembedding)
    [2+2S]       the prompt's last instruction token id (046's grouping, stored position -8)
    [3+2S : ]    the K sampled outcomes (1 = not refused, NaN = unusable)

**Within-last-token centring** (046's `within_level`): every column a statistic uses
(projections, logit, per-prompt P and its even / odd halves) has its last-token-id group means
removed, recomputed on each (re)sample. The ladder and the link are then 007's statistics on the
centred columns. Centring both the projection and P gives the within-token association; centring
the projection only is reported as a sensitivity.

**Logit link**: the link of the logit readout, corrected for rel(dP) only (the logit is computed
from the stored activation without sampling, so its reliability is 1). **Partial link**: the link
of the projection controlling for the logit difference, -partial(dz, dP | dlogit); corrected
from the disattenuated zero-order correlations (rel(dproj), rel(dP), 1).
"""

from __future__ import annotations

import numpy as np

from castor import harmcontrol as HC
from castor import mechladder as ML
from castor import reliability as R


def pack(proj: tuple, logit: np.ndarray, group: np.ndarray, y: np.ndarray) -> np.ndarray:
    full, h1, h2 = proj
    return np.concatenate([ML.pack(full, h1, h2), np.asarray(logit, float)[:, None],
                           np.asarray(group, float)[:, None], np.asarray(y, float)], axis=1)


def unpack(a: np.ndarray, n_splits: int):
    w = 1 + 2 * n_splits
    return a[:, :w], a[:, w], a[:, w + 1], a[:, w + 2:]


def demean_within(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """castor.harmcontrol.demean_within (group nan-means removed per column) by one sort and
    np.add.reduceat instead of a loop over groups; equal to it up to float rounding."""
    x = np.asarray(x, float)
    g = np.asarray(groups)
    order = np.argsort(g, kind="stable")
    gs, xs = g[order], x[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    lengths = np.diff(np.r_[starts, len(gs)])
    fin = np.isfinite(xs)
    s = np.add.reduceat(np.where(fin, xs, 0.0), starts, axis=0)
    n = np.add.reduceat(fin.astype(float), starts, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    out = np.empty_like(xs)
    out[order] = xs - np.repeat(mean, lengths, axis=0)
    return out


def as_ml(a: np.ndarray, n_splits: int, centre: bool) -> np.ndarray:
    """007's pack (projections + outcomes), projections centred within last token if asked."""
    proj, _, g, y = unpack(a, n_splits)
    if centre:
        proj = demean_within(proj, g)
    return np.concatenate([proj, y], axis=1)


def features(a: np.ndarray, n_splits: int) -> dict:
    """Per-model quantities, raw and centred within last token."""
    proj, logit, g, y = unpack(a, n_splits)
    full, h1, h2 = proj[:, 0], proj[:, 1:1 + n_splits], proj[:, 1 + n_splits:]
    ps = np.stack([R.per_prompt_p(y), R.per_prompt_p(y[:, 0::2]), R.per_prompt_p(y[:, 1::2])],
                  axis=1)
    cols = np.concatenate([full[:, None], h1, h2, logit[:, None]], axis=1)
    out = {}
    for tag, c, p in (("raw", cols, ps), ("cen", demean_within(cols, g),
                                          demean_within(ps, g))):
        out[tag] = {"z": ML.zscore(c[:, 0]), "z1": ML.zscore(c[:, 1:1 + n_splits]),
                    "z2": ML.zscore(c[:, 1 + n_splits:1 + 2 * n_splits]),
                    "zl": ML.zscore(c[:, -1]), "p": p[:, 0], "p_even": p[:, 1],
                    "p_odd": p[:, 2]}
    return out


def partial(r_xy: float, r_xw: float, r_yw: float) -> float:
    den = (1 - r_xw ** 2) * (1 - r_yw ** 2)
    if not (np.isfinite(den) and den > 0):
        return float("nan")
    return float((r_xy - r_xw * r_yw) / np.sqrt(den))


def pair_stats(fa: dict, fb: dict, proj_f: str, p_f: str) -> dict:
    """The link family for one pair: projection from feature set `proj_f`, P (and its halves)
    and the logit from `p_f` ("raw" or "cen")."""
    a, b = fa[proj_f], fb[proj_f]
    pa, pb = fa[p_f], fb[p_f]
    dz, dp, dl = a["z"] - b["z"], pa["p"] - pb["p"], pa["zl"] - pb["zl"]
    ok = np.isfinite(dz) & np.isfinite(dp) & np.isfinite(dl)
    r_xy, r_yw, r_xw = (R.corr(u[ok], v[ok]) for u, v in ((dz, dp), (dp, dl), (dz, dl)))
    rel_dp = R.spearman_brown(R.corr(pa["p_even"] - pb["p_even"], pa["p_odd"] - pb["p_odd"]))
    rel_dproj = ML.split_half_mean(a["z1"] - b["z1"], a["z2"] - b["z2"])
    link = -r_xy
    out = {"link": link, "rel_dP": rel_dp, "rel_dproj": rel_dproj,
           "link_corrected": R.disattenuate(link, rel_dproj, rel_dp),
           "logit_link": -r_yw, "logit_link_corrected": R.disattenuate(-r_yw, 1.0, rel_dp),
           "r_dproj_dlogit": r_xw, "partial_link": -partial(r_xy, r_xw, r_yw)}
    if rel_dp > 0 and rel_dproj > 0:
        c_xy = r_xy / np.sqrt(rel_dproj * rel_dp)
        c_yw = r_yw / np.sqrt(rel_dp)
        c_xw = r_xw / np.sqrt(rel_dproj)
        out["partial_link_corrected"] = -partial(c_xy, c_xw, c_yw)
    else:
        out["partial_link_corrected"] = float("nan")
    return out


VARIANTS = {"raw": ("raw", "raw"), "centred": ("cen", "cen"), "proj_centred": ("cen", "raw")}
PAIR_KEYS = ("link", "link_corrected", "rel_dP", "rel_dproj", "logit_link",
             "logit_link_corrected", "r_dproj_dlogit", "partial_link", "partial_link_corrected")


def analyze(n_splits: int, gap: bool, groups_of: dict[str, str] | None = None):
    """analyze() for R.bootstrap_prompts: per U pair the link family under each variant; with
    `gap`, 007's ladder on raw and centred projections (MZ and U levels)."""
    ladder = ML.ladder(n_splits)

    def an(ys, info, method="pearson", pairs=None):
        names = sorted(ys)
        u = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
             if R.pair_level(info[a], info[b]) == "U"]
        f = {m: features(ys[m], n_splits) for m in {m for p in u for m in p}}
        rows = []
        for a, b in u:
            row = {"a": a, "b": b}
            if groups_of is not None:
                row["group"] = groups_of[a]
            for v, (pf, qf) in VARIANTS.items():
                row[v] = pair_stats(f[a], f[b], pf, qf)
            rows.append(row)
        out = {"pairs": rows}
        if gap:
            out["ladder"] = {c: ML.ladder_summary(ladder(
                {m: as_ml(y, n_splits, c == "centred") for m, y in ys.items()}, info))
                for c in ("raw", "centred")}
        return out
    return an


def summary(res: dict) -> dict:
    out: dict = {}
    rows = res["pairs"]
    out["n_U"] = len(rows)
    for v in VARIANTS:
        for k in PAIR_KEYS:
            out[f"{v}_{k}"] = float(np.nanmean([r[v][k] for r in rows]))
    groups = sorted({r["group"] for r in rows if "group" in r})
    for g in groups:
        sel = [r for r in rows if r.get("group") == g]
        for v in ("raw", "centred"):
            for k in ("link_corrected", "partial_link_corrected", "logit_link_corrected"):
                out[f"group:{g}:{v}_{k}"] = float(np.nanmean([r[v][k] for r in sel]))
    for c, lad in res.get("ladder", {}).items():
        for k in ("r_MZ_corrected", "r_U_corrected", "MZ_minus_U_corrected", "MZ_minus_U",
                  "mean_reliability"):
            out[f"ladder_{c}_{k}"] = lad[k]
    return out


def summary_pooled(res: dict) -> dict:
    """summary() without the per-group keys (a leave-one-group-out replicate lacks its group)."""
    return {k: v for k, v in summary(res).items() if not k.startswith("group:")}


def validity(a: np.ndarray, n_splits: int) -> dict:
    """Per model on the subset: Pearson correlation of P with the projection and with the logit,
    and of the projection with the logit, raw and within last token."""
    proj, logit, g, y = unpack(a, n_splits)
    p = R.per_prompt_p(y)
    cols = np.stack([proj[:, 0], logit, p], axis=1)
    out = {}
    for tag, c in (("raw", cols), ("centred", demean_within(cols, g))):
        out[tag] = {"r_proj_P": R.corr(c[:, 0], c[:, 2]), "r_logit_P": R.corr(c[:, 1], c[:, 2]),
                    "r_proj_logit": R.corr(c[:, 0], c[:, 1])}
    out["eta2_logit_last_token"] = HC.eta_squared(logit, g)
    out["eta2_proj_last_token"] = HC.eta_squared(proj[:, 0], g)
    return out


def ci_entry(ci: dict, key: str) -> dict:
    return {k: float(ci[key][k]) for k in ("estimate", "lo", "hi")}


def prediction(ci_a0: dict, ci_a: dict, valid_a: dict) -> dict:
    """073's pre-registration, frozen before any number (the prompt; operationalised in the note
    before the jobs ran). Inputs are prompt-bootstrap CI dicts and per-model validity rows."""
    gap = ci_entry(ci_a0, "ladder_centred_MZ_minus_U_corrected")
    l48 = ci_entry(ci_a0, "centred_link_corrected")
    l72 = ci_entry(ci_a, "centred_link_corrected")
    part = ci_entry(ci_a, "raw_partial_link_corrected")
    r_lp = [v["raw"]["r_logit_P"] for v in valid_a.values()]
    return {
        "gap_centred_A0": {"value": gap, "rule": "> 0, prompt CI excluding 0",
                           "holds": gap["estimate"] > 0 and gap["lo"] > 0},
        "link_centred_A0": {"value": l48, "rule": ">= 0.3, prompt CI excluding 0",
                            "holds": l48["estimate"] >= 0.3 and l48["lo"] > 0},
        "link_centred_A": {"value": l72, "rule": ">= 0.5 (primary metric)",
                           "holds": l72["estimate"] >= 0.5},
        "logit_validity_A": {"n_models": len(r_lp), "n_negative": int(sum(r < 0 for r in r_lp)),
                             "mean": float(np.mean(r_lp)),
                             "rule": "corr(logit, P) < 0 in every model",
                             "holds": all(r < 0 for r in r_lp)},
        "partial_link_A": {"value": part, "rule": ">= 0.3, prompt CI excluding 0",
                           "holds": part["estimate"] >= 0.3 and part["lo"] > 0},
    }
