"""The 2×2 init × data-order factorial: per-prompt twin correlations by what a pair of
post-trained models shares in pretraining.

Four 1B DCLM-Baseline bases, each post-trained with seeds k = 1..4:
    D = released seed-default      (init 6198, data 6198)
    A = released seed-large-aux-2  (init 4,    data 4)
    X = ours init-default-data-aux2 (init 6198, data 4)
    Y = ours init-aux2-data-default (init 4,    data 6198)
A pair of models from two bases shares the init only (D–X, A–Y), the order only (D–Y, A–X) or
neither (D–A, X–Y); two models from one base share both (the MZ level). Every pair type has
two replicates at the base level.

Correlations, reliabilities and ceiling correction are castor.reliability's (per-prompt P,
sample split-half reliability, Spearman disattenuation); only the level a pair is assigned to
differs. Every model pair across two bases is used (all k × k').

Confound: X and Y were pretrained on our hardware, D and A on Ai2's, so D–X, D–Y, A–X, A–Y also
differ in run-to-run numerics; X–Y and D–A do not cross hardware. Every init and order pair
crosses hardware, every neither pair does not, so `*_minus_neither` mixes the shared seed with
the hardware crossing and is biased down. Each level carries `crosses_hardware`;
`init_minus_order` is the one contrast balanced on it.

Uncertainty: prompts are resampled (`bootstrap_prompts`), but each level averages only two base
pairs, so between-base-pair variation is not in those CIs. `jackknife_bases` (leave one of the
four bases out; every level stays defined on the remaining three) gives that component.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from castor import reliability

LEVELS = ["both", "init", "order", "neither"]
REPLICATE = "R"  # fifth base: a same-seed replicate of D, init 6198 / data 6198
OURS = {"X", "Y", REPLICATE}  # pretrained on our hardware; D and A on Ai2's
# R's pairs get levels of their own, so the four LEVELS over 20 models are the 4-base table's
FIFTH_LEVELS = LEVELS + ["replicate", "R_both", "R_init", "R_order", "R_neither"]


@dataclass(frozen=True)
class Twin:
    base: str  # D, A, X, Y
    init_seed: int
    data_seed: int
    k: int  # post-training seed


def level(a: Twin, b: Twin) -> str | None:
    if a.base == b.base:
        return "both" if a.k != b.k else None
    si, sd = a.init_seed == b.init_seed, a.data_seed == b.data_seed
    assert not (si and sd), f"two bases share init and data: {a.base}, {b.base}"
    return "init" if si else "order" if sd else "neither"


def level_fifth(a: Twin, b: Twin) -> str | None:
    """`level` with the fifth base admitted: R and D share both seeds across two
    pretraining runs ("replicate"); every other pair involving R is `level`'s
    reading prefixed "R_", never pooled with the 4-base pairs (R's SFT configuration differs);
    pairs without R as `level`."""
    if REPLICATE not in (a.base, b.base):
        return level(a, b)
    if a.base != b.base and (a.init_seed, a.data_seed) == (b.init_seed, b.data_seed):
        return "replicate"
    lvl = level(a, b)
    return None if lvl is None else f"R_{lvl}"


def pairs(twins: dict[str, Twin]) -> list[tuple[str, str, str]]:
    """(model a, model b, level) for every pair with a level."""
    out = []
    for a, b in combinations(sorted(twins), 2):
        lvl = level(twins[a], twins[b])
        if lvl is not None:
            out.append((a, b, lvl))
    return out


def crosses_hardware(a: Twin, b: Twin) -> bool:
    return (a.base in OURS) != (b.base in OURS)


def analyze(ys: dict[str, np.ndarray], twins: dict[str, Twin], method: str = "pearson",
            pairs: Sequence[tuple[str, str]] | None = None, level_of=level,
            levels: Sequence[str] = LEVELS) -> dict:
    """Per-pair raw and corrected r (castor.reliability.twin_correlations with `level_of`,
    this module's `level` by default), averaged per level and per base pair. `pairs`
    restricts the model pairs used."""
    tw = reliability.twin_correlations(ys, twins, method, pairs=pairs, level_of=level_of,
                                       levels=list(levels))
    rows = [dict(x, base_pair="".join(sorted({twins[x["a"]].base, twins[x["b"]].base})),
                 crosses_hardware=crosses_hardware(twins[x["a"]], twins[x["b"]]))
            for x in tw["pairs"]]
    out_levels = {}
    for lvl in levels:
        sel = [x for x in rows if x["level"] == lvl]
        hw = {x["crosses_hardware"] for x in sel}
        out_levels[lvl] = dict(tw["levels"][lvl], n_base_pairs=len({x["base_pair"] for x in sel}),
                           crosses_hardware=hw.pop() if len(hw) == 1 else None)
    by_base_pair = {}
    for bp in sorted({x["base_pair"] for x in rows}):
        sel = [x for x in rows if x["base_pair"] == bp]
        by_base_pair[bp] = {"level": sel[0]["level"], "n_pairs": len(sel),
                            "n_pairs_defined": int(sum(np.isfinite(x["r_corrected"])
                                                       for x in sel)),
                            "crosses_hardware": sel[0]["crosses_hardware"],
                            "r": _mean(sel, "r"), "r_corrected": _mean(sel, "r_corrected")}
    return {"reliability": tw["reliability"], "pairs": rows, "levels": out_levels,
            "by_base_pair": by_base_pair}


def _mean(rows: Sequence[dict], key: str) -> float:
    with np.errstate(invalid="ignore"):
        return float(np.nanmean([x[key] for x in rows])) if rows else float("nan")


def summary(res: dict) -> dict[str, float]:
    """The bootstrapped quantities: each level raw / corrected, the contrasts that read the
    factorial (what sharing only the init, or only the order, adds over sharing neither; these
    are confounded with crossing hardware, see the module docstring), and init − order, which
    is balanced on hardware."""
    lv = res["levels"]
    out = {}
    for lvl in LEVELS:
        out[f"r_{lvl}"] = lv[lvl]["r"]
        out[f"r_{lvl}_corrected"] = lv[lvl]["r_corrected"]
    for lvl in ["init", "order", "both"]:
        for suf in ["", "_corrected"]:
            out[f"{lvl}_minus_neither{suf}"] = lv[lvl][f"r{suf}"] - lv["neither"][f"r{suf}"]
    for suf in ["", "_corrected"]:
        out[f"init_minus_order{suf}"] = lv["init"][f"r{suf}"] - lv["order"][f"r{suf}"]
    for bp, v in res["by_base_pair"].items():
        out[f"bp_{bp}_corrected"] = v["r_corrected"]
    return out


def bootstrap_prompts(ys: dict[str, np.ndarray], twins: dict[str, Twin], n_boot: int = 1000,
                      seed: int = 0, alpha: float = 0.05, method: str = "pearson") -> dict:
    """Percentile CIs from resampling prompts (castor.reliability.bootstrap_prompts). Between-
    base-pair variation is not in these; see `jackknife_bases`."""
    return reliability.bootstrap_prompts(ys, twins, n_boot, seed, alpha, method, stat=summary,
                                         analyze=analyze)


def jackknife_bases(ys: dict[str, np.ndarray], twins: dict[str, Twin],
                    method: str = "pearson") -> dict:
    """Leave-one-base-out jackknife SEs (castor.reliability.jackknife_bases): dropping any one
    of D, A, X, Y leaves one base pair per cross-base level, so the SEs carry the spread
    between the two replicate base pairs of each level. Per-base-pair values are left out: a
    replicate holds out one of the bases they are defined on."""
    def stat(res: dict) -> dict[str, float]:
        return {k: v for k, v in summary(res).items() if not k.startswith("bp_")}
    return reliability.jackknife_bases(ys, twins, method, stat=stat, analyze=analyze)


# --- Fifth base. R's twins ran SFT on ZeRO-1, D/A/X/Y's on ZeRO-3, so every R pair also crosses
# that configuration. The R rows are read by base pair or under their own R_* levels, never
# pooled into LEVELS.
# Within our hardware R–X shares the init only, R–Y the order only and X–Y neither, with no
# hardware crossing; R–X − R–Y is also balanced on configuration (both pairs cross it).

def analyze_fifth(ys: dict[str, np.ndarray], twins: dict[str, Twin], method: str = "pearson",
                  pairs: Sequence[tuple[str, str]] | None = None) -> dict:
    return analyze(ys, twins, method, pairs, level_of=level_fifth, levels=FIFTH_LEVELS)


def summary_fifth(res: dict) -> dict[str, float]:
    """Every base pair that involves R, X–Y beside them, and the within-hardware contrasts of
    the trio R / X / Y: init − order (RX − RY), init − neither (RX − XY), order − neither
    (RY − XY); the last two also cross configuration on one side only."""
    bp = res["by_base_pair"]
    out = {}
    for k, v in bp.items():
        if REPLICATE in k or k == "XY":
            out[f"bp_{k}"] = v["r"]
            out[f"bp_{k}_corrected"] = v["r_corrected"]
    for suf in ["", "_corrected"]:
        rx, ry, xy = (bp[k][f"r{suf}"] for k in ("RX", "RY", "XY"))
        out[f"hw_init_minus_order{suf}"] = rx - ry
        out[f"hw_init_minus_neither{suf}"] = rx - xy
        out[f"hw_order_minus_neither{suf}"] = ry - xy
    return out


def fifth_pairs(twins: dict[str, Twin]) -> list[tuple[str, str]]:
    """The model pairs `summary_fifth` reads: every pair involving R, and X–Y."""
    return [(a, b) for a, b in combinations(sorted(twins), 2)
            if REPLICATE in (twins[a].base, twins[b].base)
            or {twins[a].base, twins[b].base} == {"X", "Y"}]


def bootstrap_prompts_fifth(ys: dict[str, np.ndarray], twins: dict[str, Twin],
                            n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
                            method: str = "pearson") -> dict:
    """Prompt-bootstrap percentile CIs of `summary_fifth` (all models resampled jointly; only
    the `fifth_pairs` it reads are analysed)."""
    keep = fifth_pairs(twins)

    def analyze_kept(ys, twins, method, pairs=None):
        return analyze_fifth(ys, twins, method, pairs=keep)
    return reliability.bootstrap_prompts(ys, twins, n_boot, seed, alpha, method,
                                         stat=summary_fifth, analyze=analyze_kept)
