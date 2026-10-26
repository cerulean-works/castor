"""Does a readout of the base model, before any post-training, predict the run's per-prompt
jailbreak residual?

Definitions (pre-registered):

- Runs: the 75 grid bases (25 recipes × 3 pretraining runs). A run's **profile** is the
  per-prompt P(non-refusal | coherent) averaged over its post-training seeds; its **residual**
  is the profile minus the mean profile of the recipe's three runs (per prompt). A base
  readout (one number per prompt) is residualised the same way, against the recipe's three
  bases. Rows are runs, columns prompts, one prompt order everywhere.
- Primary: mean over runs of corr(base residual of run i, post-trained residual of run i),
  Pearson over prompts (castor.gridstats.rowwise_corr), each corrected for the target's
  reliability only (the readout is deterministic): r / sqrt(rel_i), where rel_i is the
  Spearman–Brown split-half reliability of the residual built from the odd and the even
  sample columns separately.
- Mismatched baseline: for run i the mean of corr(base residual of run j, residual of run i)
  over the other runs j of i's recipe, corrected by the same rel_i, then the mean over runs.
  The signal is matched − mismatched.
- Profile level (secondary): the same with raw readout and raw profile (no residualisation),
  reliability of the raw profile.
- Intervals: percentile bootstrap over recipes (resampling recipes with replacement, per-run
  values held fixed) and over prompts (columns resampled, every correlation and
  reliability recomputed), 1,000 draws each.
- Reverse fingerprint: each post-trained model's residual (its P minus its recipe's mean
  profile) is assigned to the base whose residual readout correlates best with it, among all
  75 bases and among its recipe's three (castor.fingerprint's nearest-reference rule); exact
  binomial intervals; prompt-count curve over random prompt subsets.
"""

from collections.abc import Sequence

import numpy as np

from castor import fingerprint, reliability
from castor.gridstats import rowwise_corr


def recipe_residual(M: np.ndarray, groups: Sequence) -> np.ndarray:
    """(n_rows, n_cols) minus, per column, the mean of the rows in the same group."""
    M = np.asarray(M, dtype=float)
    g = np.asarray(groups)
    out = np.empty_like(M)
    for q in np.unique(g):
        sel = g == q
        out[sel] = M[sel] - M[sel].mean(axis=0, keepdims=True)
    return out


def split_half_rel(Ro: np.ndarray, Re: np.ndarray) -> np.ndarray:
    """Spearman–Brown reliability per row from the odd and even halves (castor.reliability)."""
    return np.array([reliability.spearman_brown(r) for r in rowwise_corr(Ro, Re)])


def _correct(r: np.ndarray, rel: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.where(rel > 0, r / np.sqrt(np.where(rel > 0, rel, np.nan)), np.nan)


def per_run(X: np.ndarray, Y: np.ndarray, Yo: np.ndarray, Ye: np.ndarray, groups: Sequence,
            residual: bool = True) -> dict:
    """Per-run matched and mismatched correlations, raw and corrected.

    X: readouts (n_runs, n_p); Y, Yo, Ye: target and its odd / even halves (n_runs, n_p);
    groups: recipe per run. residual=False reads the profile level (no residualisation)."""
    g = np.asarray(groups)
    if residual:
        X, Y, Yo, Ye = (recipe_residual(M, g) for M in (X, Y, Yo, Ye))
    rel = split_half_rel(Yo, Ye)
    matched = rowwise_corr(X, Y)
    mism = np.full(len(g), np.nan)
    for i in range(len(g)):
        others = [j for j in np.flatnonzero(g == g[i]) if j != i]
        if others:
            mism[i] = np.mean(rowwise_corr(X[others], np.repeat(Y[i:i + 1], len(others), 0)))
    mc, mmc = _correct(matched, rel), _correct(mism, rel)
    return {"rel": rel, "matched": matched, "mismatched": mism, "matched_c": mc,
            "mismatched_c": mmc, "diff_c": mc - mmc, "diff": matched - mism}


STATS = ("matched", "mismatched", "diff", "matched_c", "mismatched_c", "diff_c", "rel")


def summarise(pr: dict, rows: np.ndarray | None = None) -> dict:
    """Means over runs (all, or the rows given, repeats allowed) of every per-run quantity."""
    rows = np.arange(len(pr["rel"])) if rows is None else rows
    return {k: float(np.nanmean(pr[k][rows])) for k in STATS}


def _pct(draws: list[dict], point: dict) -> dict:
    out = {}
    for k, v in point.items():
        d = np.array([x[k] for x in draws], dtype=float)
        d = d[np.isfinite(d)]
        lo, hi = np.quantile(d, [0.025, 0.975]) if len(d) else (np.nan, np.nan)
        out[k] = {"estimate": v, "lo": float(lo), "hi": float(hi), "n_draws": len(d)}
    return out


def recipe_bootstrap(pr: dict, groups: Sequence, n_boot: int = 1000, seed: int = 0) -> dict:
    """Recipes resampled with replacement; per-run values fixed; mean over the drawn runs."""
    g = np.asarray(groups)
    rec = list(dict.fromkeys(g.tolist()))
    idx = {q: np.flatnonzero(g == q) for q in rec}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(rec), len(rec))
        draws.append(summarise(pr, np.concatenate([idx[rec[i]] for i in pick])))
    return _pct(draws, summarise(pr))


def per_run_avg(arms: Sequence[tuple], groups: Sequence, residual: bool = True) -> dict:
    """per_run of each arm (X, Y, Yo, Ye) on the same runs, averaged per run (e.g. the two
    orderings of the leave-one-descendant-out guard)."""
    prs = [per_run(*arm, groups, residual) for arm in arms]
    return {k: np.mean([p[k] for p in prs], axis=0) for k in prs[0]}


def prompt_bootstrap(arms: Sequence[tuple], groups: Sequence, n_boot: int = 1000,
                     seed: int = 0, residual: bool = True) -> tuple[dict, dict]:
    """Prompts resampled (the same columns for every run and arm); everything recomputed.
    `arms` = [(X, Y, Yo, Ye), ...], averaged per run (one arm: plain per_run). Returns the
    overall intervals and per-recipe intervals of the mean corrected matched correlation."""
    g = np.asarray(groups)
    rec = list(dict.fromkeys(g.tolist()))
    rng = np.random.default_rng(seed)
    n = arms[0][0].shape[1]
    pr0 = per_run_avg(arms, g, residual)
    draws, per_rec = [], {q: [] for q in rec}
    for _ in range(n_boot):
        c = rng.integers(0, n, n)
        pr = per_run_avg([tuple(M[:, c] for M in arm) for arm in arms], g, residual)
        draws.append(summarise(pr))
        for q in rec:
            per_rec[q].append(float(np.nanmean(pr["matched_c"][g == q])))
    rec_out = {}
    for q in rec:
        d = np.asarray(per_rec[q])
        d = d[np.isfinite(d)]
        rec_out[q] = {"estimate": float(np.nanmean(pr0["matched_c"][g == q])),
                      "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975))}
    return _pct(draws, summarise(pr0)), rec_out


# ---- reverse fingerprint ----------------------------------------------------------------------


def assign_within(C: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Nearest reference per query among the allowed ones (fingerprint.assign's rule)."""
    return fingerprint.assign(np.where(allowed, C, np.nan))


def reverse_fingerprint(Q: np.ndarray, R: np.ndarray, truth: Sequence[int],
                        q_groups: Sequence, r_groups: Sequence) -> dict:
    """Queries (post-trained residual profiles) against references (base residual readouts):
    accuracy among all references and among the query's own group, with exact intervals."""
    truth = np.asarray(truth)
    C = fingerprint.corr_cross(Q, R)
    same = np.asarray(q_groups)[:, None] == np.asarray(r_groups)[None, :]
    out = {}
    for name, allowed in (("all", np.ones_like(same)), ("within_recipe", same)):
        got = assign_within(C, allowed)
        ok = got == truth
        k, n = int(ok.sum()), len(ok)
        out[name] = {"n_correct": k, "n_queries": n, "accuracy": k / n,
                     "chance": float(np.mean(1 / allowed.sum(1))),
                     "ci_exact": fingerprint.clopper_pearson(k, n)}
    return out


def reverse_curve(Q: np.ndarray, R: np.ndarray, truth: Sequence[int], q_groups: Sequence,
                  r_groups: Sequence, counts: Sequence[int], n_sub: int = 200,
                  seed: int = 0) -> list[dict]:
    """Reverse-fingerprint accuracy on random prompt subsets (fingerprint.prompt_curve's
    draws: one generator per count, without replacement; the full set evaluated once)."""
    n_p = Q.shape[1]
    out = []
    for n in counts:
        if n >= n_p:
            subs = [np.arange(n_p)]
        else:
            rng = np.random.default_rng([seed, n])
            subs = [rng.choice(n_p, n, replace=False) for _ in range(n_sub)]
        acc = {"all": [], "within_recipe": []}
        for s in subs:
            r = reverse_fingerprint(Q[:, s], R[:, s], truth, q_groups, r_groups)
            for k, v in acc.items():
                v.append(r[k]["accuracy"])
        row = {"n_prompts": int(min(n, n_p)), "n_subsets": len(subs)}
        for k, v in acc.items():
            v = np.asarray(v)
            row[k] = {"mean": float(v.mean()), "lo": float(np.quantile(v, 0.025)),
                      "hi": float(np.quantile(v, 0.975))}
        out.append(row)
    return out
