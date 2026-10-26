"""ZeRO-1 vs ZeRO-3 configuration effect on n = 12 same-cell pairs.

Each of twelve twins (three DCLM-Baseline 1B bases x post-training seeds 1-4, SFT on ZeRO-3) has
a rerun with the same base and post-training seed, SFT on ZeRO-1, DPO unchanged. The effect is
stated on three criteria: weights, per-prompt behaviour, levels.

Pair kinds, within one base (pairs across bases are dropped):
    same_cell_cross    same post-training seed, different configuration      12 pairs (the effect)
    MZ_within_zero3    different post-training seed, both ZeRO-3              18
    MZ_within_zero1    different post-training seed, both ZeRO-1              18
    MZ_cross           different post-training seed and configuration         36
Contrasts (corrected r):
    config_effect = MZ_cross - MZ_within: what changing the configuration adds to a post-training
        seed change (~0 = none);
    same_cell_minus_MZ_within: how much of a post-training seed's effect a configuration change
        alone reproduces (1 - this/(1 - MZ) is the share of a seed change);
    zero1_minus_zero3: whether ZeRO-1 twins are more or less alike than ZeRO-3 twins.
Correlations, reliabilities and the ceiling correction are castor.reliability's. The weights
criterion is the global fp32 L2 after DPO against an identical-seed floor times WEIGHT_FACTOR;
the levels criterion is the per-source pooled refusal rate against RATE_BAND.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from castor import reliability

ZERO1, ZERO3 = "zero1", "zero3"
SAME_CELL = "same_cell_cross"
KINDS = (SAME_CELL, "MZ_within_zero3", "MZ_within_zero1", "MZ_cross")
WITHIN = ("MZ_within_zero3", "MZ_within_zero1")
WEIGHT_FACTOR = 1.25
RATE_BAND = 0.03  # in refusal-rate units


@dataclass(frozen=True)
class CfgInfo:
    """reliability.ModelInfo plus the launcher configuration of the SFT stage."""
    recipe: str
    pretrain_seed: str
    posttrain_seed: str
    config: str

    def __post_init__(self):
        if self.config not in (ZERO1, ZERO3):
            raise ValueError(self.config)

    @property
    def base(self) -> tuple[str, str]:
        return (self.recipe, self.pretrain_seed)

    @property
    def cell(self) -> str:
        return f"{self.pretrain_seed}-s{self.posttrain_seed}"


def kind_of(a: CfgInfo, b: CfgInfo) -> str | None:
    """Pair kind; None across bases. Two models of one cell and configuration are a duplicate."""
    if a.base != b.base:
        return None
    same_k, same_cfg = a.posttrain_seed == b.posttrain_seed, a.config == b.config
    if same_k and same_cfg:
        raise ValueError(f"two models of one cell and configuration: {a} / {b}")
    if same_k:
        return SAME_CELL
    if same_cfg:
        return f"MZ_within_{a.config}"
    return "MZ_cross"


def analyze(ys: dict[str, np.ndarray], info: dict[str, CfgInfo], method: str = "pearson",
            pairs=None) -> dict:
    """reliability.twin_correlations with the configuration pair kinds as levels."""
    return reliability.twin_correlations(ys, info, method, pairs, level_of=kind_of, levels=KINDS)


def summary(res: dict, info: dict[str, CfgInfo]) -> dict[str, float]:
    """Per-kind means (raw and corrected), the three contrasts, and the twelve same-cell r."""
    lv, out = res["levels"], {}
    for k in KINDS:
        out[f"n_{k}"] = lv[k]["n_pairs"]
        out[f"r_{k}"] = lv[k]["r"]
        out[f"r_{k}_corrected"] = lv[k]["r_corrected"]
    for sfx in ("", "_corrected"):
        key = "r" + sfx
        within = [x[key] for x in res["pairs"] if x["level"] in WITHIN]
        out[f"r_MZ_within{sfx}"] = float(np.nanmean(within)) if within else float("nan")
        out[f"config_effect{sfx}"] = out[f"r_MZ_cross{sfx}"] - out[f"r_MZ_within{sfx}"]
        out[f"same_cell_minus_MZ_within{sfx}"] = out[f"r_{SAME_CELL}{sfx}"] - out[f"r_MZ_within{sfx}"]
        out[f"zero1_minus_zero3{sfx}"] = out[f"r_MZ_within_zero1{sfx}"] - out[f"r_MZ_within_zero3{sfx}"]
    for x in res["pairs"]:
        if x["level"] == SAME_CELL:
            out[f"r_cell_{info[x['a']].cell}_corrected"] = x["r_corrected"]
    out["mean_reliability"] = float(np.nanmean(list(res["reliability"].values())))
    return out


def stat_for(info: dict[str, CfgInfo]) -> Callable[[dict], dict]:
    return lambda res: summary(res, info)


def weight_table(rows: list[dict], floor: float, factor: float = WEIGHT_FACTOR) -> dict:
    """rows: {a, b, kind, l2} for every within-base pair of one stage. Per-kind mean / min / max,
    and each same-cell distance as a ratio to the identical-seed floor against its bound."""
    by: dict[str, list[float]] = {k: [] for k in KINDS}
    for r in rows:
        by[r["kind"]].append(r["l2"])
    out = {"floor": floor, "factor": factor, "bound": factor * floor,
           "by_kind": {k: {"n": len(v), "mean": float(np.mean(v)) if v else float("nan"),
                           "min": float(np.min(v)) if v else float("nan"),
                           "max": float(np.max(v)) if v else float("nan")} for k, v in by.items()}}
    same = [r | {"ratio": r["l2"] / floor, "within_bound": bool(r["l2"] <= factor * floor)}
            for r in rows if r["kind"] == SAME_CELL]
    ratios = [r["ratio"] for r in same]
    out["same_cell"] = same
    out["ratio_mean"] = float(np.mean(ratios)) if ratios else float("nan")
    out["ratio_min"] = float(np.min(ratios)) if ratios else float("nan")
    out["ratio_max"] = float(np.max(ratios)) if ratios else float("nan")
    out["n_within_bound"] = sum(r["within_bound"] for r in same)
    out["n_same_cell"] = len(same)
    # a configuration change as a fraction of a post-training seed change, in weight space
    mz = by["MZ_within_zero3"] + by["MZ_within_zero1"]
    out["same_cell_over_MZ_within"] = (out["by_kind"][SAME_CELL]["mean"] / float(np.mean(mz))
                                       if mz and same else float("nan"))
    return out


def level_table(rates: dict[str, dict[str, float]], cells: list[tuple[str, str, str]],
                band: float = RATE_BAND) -> dict:
    """rates: source -> model id -> pooled refusal rate; cells: (ZeRO-1 id, ZeRO-3 id, cell tag).
    Per cell and source the ZeRO-1 minus ZeRO-3 difference against the band; summary over all."""
    per_cell, deltas = {}, []
    for a0, m006, tag in cells:
        d = {s: rates[s][a0] - rates[s][m006] for s in rates}
        per_cell[tag] = {"a0": a0, "006": m006, "delta": d,
                         "max_abs_delta": max(abs(v) for v in d.values()),
                         "n_within": sum(abs(v) <= band for v in d.values())}
        deltas += list(d.values())
    deltas = np.array(deltas, dtype=float)
    by_source = {s: {"mean_delta": float(np.mean([per_cell[t]["delta"][s] for t in per_cell])),
                     "max_abs_delta": float(np.max([abs(per_cell[t]["delta"][s]) for t in per_cell]))}
                 for s in rates}
    return {"band": band, "per_cell": per_cell, "by_source": by_source,
            "n_within": int(np.sum(np.abs(deltas) <= band)), "n_total": int(deltas.size),
            "max_abs_delta": float(np.max(np.abs(deltas))) if deltas.size else float("nan"),
            "mean_delta": float(np.mean(deltas)) if deltas.size else float("nan"),
            "mean_abs_delta": float(np.mean(np.abs(deltas))) if deltas.size else float("nan")}


def flatten(state: dict) -> tuple[list[str], "torch.Tensor"]:  # noqa: F821
    """A checkpoint's tensors as one fp32 vector in sorted name order (the names are returned
    so two checkpoints can be checked for the same layout). Holding the vector in RAM makes the
    28 within-base distances cheap; reading mmap'd tensors per pair is ~100 s each."""
    import torch

    names = sorted(state)
    return names, torch.cat([state[n].detach().reshape(-1).float() for n in names])


def l2_flat(a: tuple, b: tuple, chunk: int = 1 << 26) -> float:
    """Global L2 of the parameter difference of two flatten() results. The sum of squares is
    accumulated in float64 over chunks: fp32 `torch.linalg.vector_norm` on a 1.28 B-element
    vector lost 22 % of the norm (26.62 vs the float64 reference 34.29, runs/027/check_l2.json)
    and is 3e-3 off already at 4 M elements."""
    import torch

    (na, xa), (nb, xb) = a, b
    assert na == nb, f"tensor names differ: {sorted(set(na) ^ set(nb))[:5]}"
    assert xa.shape == xb.shape, (xa.shape, xb.shape)
    d = xa - xb
    ss = 0.0
    for c in torch.split(d, chunk):
        c64 = c.double()
        ss += float(torch.sum(c64 * c64))
    return math.sqrt(ss)
