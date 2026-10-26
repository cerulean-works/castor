"""Per-prompt vulnerability, its split-half reliability, and ceiling-corrected twin
correlations.

Inputs are outcome matrices Y[m] of shape (n_prompts, K) per model m, entries 0/1 or NaN
(NaN = no usable outcome: incoherent or judge error). Rows are aligned on one prompt
order across models.

- P[m] = nanmean over the K samples: per-prompt P(non-refusal | coherent).
- Split-half reliability: correlate P from the even and the odd sample columns across
  prompts, then Spearman–Brown to full length, rel = 2r / (1 + r). This is the reliability of
  P[m] as a measurement of the model's true per-prompt rate (the test–retest ceiling).
- Twin correlation of two models a, b: r_ab = corr(P[a], P[b]) over prompts where both are
  defined, corrected for attenuation (Spearman 1904): r_ab / sqrt(rel_a * rel_b).
- Levels: MZ = same base (recipe and pretraining seed), different post-training seed;
  U = same recipe, different pretraining seed; R = different recipe. A level's value is the
  mean over its model pairs, raw and corrected (per-pair correction, then mean).
- Uncertainty: percentile bootstrap over prompts (rows resampled, every quantity including
  the reliabilities recomputed), and a cluster bootstrap / jackknife over bases (pretraining
  seeds), which resamples whole bases and keeps only pairs of distinct base identities.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

LEVELS = ("MZ", "U", "R")


@dataclass(frozen=True)
class ModelInfo:
    recipe: str
    pretrain_seed: str
    posttrain_seed: str

    @property
    def base(self) -> tuple[str, str]:
        return (self.recipe, self.pretrain_seed)


def per_prompt_p(y: np.ndarray) -> np.ndarray:
    """(n_prompts, K) -> (n_prompts,), NaN where a prompt has no usable sample."""
    y = np.asarray(y, dtype=float)
    n = np.sum(~np.isnan(y), axis=1)
    s = np.nansum(y, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    """Correlation over positions where both are finite; NaN if < 3 or constant."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")
    if method == "spearman":
        x, y = _rank(x), _rank(y)
    elif method != "pearson":
        raise ValueError(method)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank(v: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    r[order] = np.arange(len(v), dtype=float)
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / cnt)[inv]


def spearman_brown(r: float, factor: float = 2.0) -> float:
    return factor * r / (1 + (factor - 1) * r)


def split_half(y: np.ndarray, method: str = "pearson") -> dict:
    """Odd/even sample halves -> half correlation and its Spearman–Brown reliability."""
    y = np.asarray(y, dtype=float)
    r = corr(per_prompt_p(y[:, 0::2]), per_prompt_p(y[:, 1::2]), method)
    return {"r_half": r, "reliability": spearman_brown(r)}


def disattenuate(r: float, rel_a: float, rel_b: float) -> float:
    """Spearman (1904): r / sqrt(rel_a rel_b); NaN if a reliability is not positive."""
    if not (rel_a > 0 and rel_b > 0):
        return float("nan")
    return r / np.sqrt(rel_a * rel_b)


def pair_level(a: ModelInfo, b: ModelInfo) -> str | None:
    if a.recipe != b.recipe:
        return "R"
    if a.pretrain_seed != b.pretrain_seed:
        return "U"
    if a.posttrain_seed != b.posttrain_seed:
        return "MZ"
    return None  # same model


def outcome_measure(y: np.ndarray, method: str = "pearson") -> tuple[np.ndarray, float]:
    """The default per-model measure: per-prompt P and its sample split-half reliability."""
    return per_prompt_p(y), split_half(y, method)["reliability"]


def twin_correlations(ys: dict[str, np.ndarray], info: dict[str, ModelInfo],
                      method: str = "pearson",
                      pairs: Sequence[tuple[str, str]] | None = None,
                      measure: Callable[[np.ndarray, str], tuple[np.ndarray, float]]
                      = outcome_measure,
                      level_of: Callable[[Any, Any], str | None] = pair_level,
                      levels: Sequence[str] = LEVELS) -> dict:
    """Reliabilities per model, raw and corrected correlations per pair and per level.
    `measure` maps one model's rows to (per-prompt values, reliability); the default is the
    outcome matrix's P and sample split-half.
    `level_of(info[a], info[b])` names a pair's level (None drops it) and `levels` lists the
    levels summarised; the defaults are MZ / U / R."""
    names = sorted(ys)
    vals = {m: measure(ys[m], method) for m in names}
    p = {m: v[0] for m, v in vals.items()}
    rel = {m: v[1] for m, v in vals.items()}
    pairs = list(combinations(names, 2)) if pairs is None else list(pairs)
    rows = []
    for a, b in pairs:
        lvl = level_of(info[a], info[b])
        if lvl is None:
            continue
        r = corr(p[a], p[b], method)
        rows.append({"a": a, "b": b, "level": lvl, "r": r,
                     "r_corrected": disattenuate(r, rel[a], rel[b])})
    out_levels = {}
    for lvl in levels:
        sel = [x for x in rows if x["level"] == lvl]
        out_levels[lvl] = {
            "n_pairs": len(sel),
            # pairs whose correction is finite: the nanmean below averages over these only, so
            # a reader can see when r_corrected is a subset mean (a twin's reliability <= 0)
            "n_pairs_defined": int(sum(np.isfinite(x["r_corrected"]) for x in sel)),
            "r": float(np.nanmean([x["r"] for x in sel])) if sel else float("nan"),
            "r_corrected": float(np.nanmean([x["r_corrected"] for x in sel]))
            if sel else float("nan"),
        }
    return {"reliability": rel, "pairs": rows, "levels": out_levels}


def _summary(res: dict) -> dict[str, float]:
    """Flatten the quantities bootstrapped: per level raw/corrected, and contrasts."""
    lv = res["levels"]
    out = {}
    for lvl in LEVELS:
        out[f"r_{lvl}"] = lv[lvl]["r"]
        out[f"r_{lvl}_corrected"] = lv[lvl]["r_corrected"]
    out["MZ_minus_U"] = lv["MZ"]["r"] - lv["U"]["r"]
    out["MZ_minus_U_corrected"] = lv["MZ"]["r_corrected"] - lv["U"]["r_corrected"]
    out["U_minus_R"] = lv["U"]["r"] - lv["R"]["r"]
    out["U_minus_R_corrected"] = lv["U"]["r_corrected"] - lv["R"]["r_corrected"]
    out["mean_reliability"] = float(np.nanmean(list(res["reliability"].values())))
    return out


def _percentile_ci(draws: list[dict], point: dict, alpha: float) -> dict:
    out = {}
    for k, v in point.items():
        d = np.array([x[k] for x in draws], dtype=float)
        d = d[np.isfinite(d)]
        lo, hi = (np.quantile(d, [alpha / 2, 1 - alpha / 2]) if len(d)
                  else (float("nan"), float("nan")))
        out[k] = {"estimate": v, "lo": float(lo), "hi": float(hi), "n_draws": len(d)}
    return out


def bootstrap_prompts(ys: dict[str, np.ndarray], info: dict[str, ModelInfo], n_boot: int = 1000,
                      seed: int = 0, alpha: float = 0.05, method: str = "pearson",
                      stat: Callable[[dict], dict] = _summary,
                      analyze: Callable[..., dict] = twin_correlations) -> dict:
    """Percentile CIs from resampling prompts (rows, the same rows for every model).
    `analyze(ys, info, method, pairs=None)` computes what `stat` summarises (default
    twin_correlations)."""
    rng = np.random.default_rng(seed)
    n = next(iter(ys.values())).shape[0]
    point = stat(analyze(ys, info, method))
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        draws.append(stat(analyze({m: y[idx] for m, y in ys.items()}, info, method)))
    return _percentile_ci(draws, point, alpha)


def bootstrap_prompt_clusters(ys: dict[str, np.ndarray], info: dict[str, ModelInfo],
                              clusters: Sequence[Any], n_boot: int = 1000, seed: int = 0,
                              alpha: float = 0.05, method: str = "pearson",
                              strata: Sequence[Any] | None = None,
                              stat: Callable[[dict], dict] = _summary,
                              analyze: Callable[..., dict] = twin_correlations) -> dict:
    """Cluster bootstrap over prompts: resample whole clusters (e.g. a jailbreak template) with
    replacement, every row of a drawn cluster coming along, the same rows for every model.
    With `strata` (e.g. the source benchmark), clusters are drawn within each stratum so its
    share of clusters is fixed. Singleton clusters reduce this to `bootstrap_prompts`."""
    rng = np.random.default_rng(seed)
    n = next(iter(ys.values())).shape[0]
    assert len(clusters) == n, f"{len(clusters)} cluster labels for {n} prompt rows"
    point = stat(analyze(ys, info, method))
    strata = list(strata) if strata is not None else [0] * len(clusters)
    members: dict[Any, dict[Any, list[int]]] = {}
    for i, (st, c) in enumerate(zip(strata, clusters, strict=True)):
        members.setdefault(st, {}).setdefault((st, c), []).append(i)
    groups = [list(g.values()) for g in members.values()]
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))])
                              for g in groups])
        draws.append(stat(analyze({m: y[idx] for m, y in ys.items()}, info, method)))
    return _percentile_ci(draws, point, alpha)


def bootstrap_bases(ys: dict[str, np.ndarray], info: dict[str, ModelInfo], n_boot: int = 1000,
                    seed: int = 0, alpha: float = 0.05, method: str = "pearson",
                    stat: Callable[[dict], dict] = _summary,
                    analyze: Callable[..., dict] = twin_correlations) -> dict:
    """Cluster bootstrap over bases: within each recipe, resample its bases with replacement;
    every model of a drawn base comes along, renamed per draw. Pairs between two copies of
    the same base are dropped for U (they would be MZ-like), so only distinct bases form U.
    With 3 bases per recipe this is coarse; report it next to `jackknife_bases`."""
    rng = np.random.default_rng(seed)
    point = stat(analyze(ys, info, method))
    by_base: dict[tuple, list[str]] = {}
    for m, i in info.items():
        if m in ys:
            by_base.setdefault(i.base, []).append(m)
    recipes: dict[str, list[tuple]] = {}
    for b in by_base:
        recipes.setdefault(b[0], []).append(b)
    draws = []
    for _ in range(n_boot):
        ys_b, info_b = {}, {}
        for recipe, bases in recipes.items():
            for j, bi in enumerate(rng.integers(0, len(bases), len(bases))):
                b = bases[bi]
                for m in by_base[b]:
                    new = f"{m}#{j}"
                    ys_b[new] = ys[m]
                    info_b[new] = ModelInfo(recipe, f"{b[1]}#{j}", info[m].posttrain_seed)
        origin = {n: n.rsplit("#", 1)[0] for n in ys_b}
        base_of = {n: info[origin[n]].base for n in ys_b}
        keep = [(a, b) for a, b in combinations(sorted(ys_b), 2)
                if not (base_of[a] == base_of[b] and info_b[a].base != info_b[b].base)]
        draws.append(stat(analyze(ys_b, info_b, method, pairs=keep)))
    return _percentile_ci(draws, point, alpha)


def jackknife_bases(ys: dict[str, np.ndarray], info: dict[str, ModelInfo],
                    method: str = "pearson", stat: Callable[[dict], dict] = _summary,
                    analyze: Callable[..., dict] = twin_correlations) -> dict:
    """Leave-one-base-out jackknife standard errors."""
    point = stat(analyze(ys, info, method))
    bases = sorted({info[m].base for m in ys})
    reps = []
    for b in bases:
        keep = {m: y for m, y in ys.items() if info[m].base != b}
        reps.append(stat(analyze(keep, info, method)))
    g = len(reps)
    out = {}
    for k, v in point.items():
        d = np.array([r[k] for r in reps], dtype=float)
        d = d[np.isfinite(d)]
        se = float(np.sqrt((len(d) - 1) / len(d) * np.sum((d - d.mean()) ** 2))) \
            if len(d) > 1 else float("nan")
        out[k] = {"estimate": v, "se": se, "n_reps": len(d), "n_bases": g}
    return out


def by_group(ys: dict[str, np.ndarray], info: dict[str, ModelInfo],
             groups: dict[str, Sequence[int]], n_boot: int = 1000, seed: int = 0,
             alpha: float = 0.05, method: str = "pearson") -> dict[str, dict]:
    """The twin statistics on each group of prompt rows separately (e.g. by prompt source).
    Each group is its own panel: reliabilities, pair correlations and the prompt bootstrap
    (`bootstrap_prompts`, same seed for every group) are recomputed on that group's rows only."""
    out = {}
    for name, rows in groups.items():
        idx = np.asarray(rows, dtype=int)
        sub = {m: y[idx] for m, y in ys.items()}
        tw = twin_correlations(sub, info, method)
        out[name] = {"n_prompts": len(idx), "levels": tw["levels"],
                     "reliability": tw["reliability"],
                     "ci_prompts": bootstrap_prompts(sub, info, n_boot, seed, alpha, method)}
    return out
