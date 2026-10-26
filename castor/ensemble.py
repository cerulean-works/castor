"""Refuse-if-any ensembles of grid models, over judge_v1 outcome matrices
(castor.sanityb.outcome_matrix: 1 = not refused among coherent responses, NaN for incoherent
responses or judge errors).

An ensemble of k models answers a prompt only if every member complies; it refuses if any member
refuses. Its per-prompt vulnerability is the probability that all k members comply on one
response each. Two estimates:
  - **resampled (primary, pre-registered):** per model, R draws of one coherent sample per prompt
    (uniform over that model's coherent samples); the ensemble's outcome in draw r is the product
    of its members' draw-r outcomes; P_ens = the mean over the R draws. Each model's draws are
    fixed by (seed, model index), so a model shared by two ensembles enters both with the same
    draws (common random numbers), and a single model is the k = 1 ensemble of the same
    instrument.
  - **product rule (check):** P_ens = prod of the members' per-prompt P (reliability.per_prompt_p),
    the R -> infinity limit of the resampled estimate.

On a subset the readouts are the mean P_ens over prompts and the share of *reliably vulnerable*
prompts, P_ens >= stability.EVERY. On benign prompts the
over-refusal rate is 1 - mean P_ens (the ensemble refuses if either member refuses)."""

from collections.abc import Sequence

import numpy as np

from castor import reliability, stability

EVERY = stability.EVERY


def draw_outcomes(y: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """y (n_prompts, K) with NaN for unusable samples -> (n_draws, n_prompts) uint8 outcomes, one
    uniformly drawn usable sample per prompt and draw. Every prompt must have a usable sample."""
    y = np.asarray(y, dtype=float)
    ok = ~np.isnan(y)
    n_ok = ok.sum(axis=1)
    assert (n_ok > 0).all(), "draw_outcomes needs >= 1 usable sample per prompt (drop undefined)"
    order = np.argsort(~ok, axis=1, kind="stable")  # usable columns first, in sample order
    u = rng.random((n_draws, y.shape[0]))
    pick = np.minimum((u * n_ok).astype(np.int64), n_ok - 1)  # (n_draws, n_prompts)
    cols = np.take_along_axis(order, pick.T, axis=1).T
    return y[np.arange(y.shape[0])[None, :], cols].astype(np.uint8)


def model_rng(seed: int, index: int) -> np.random.Generator:
    """The draw stream of one model, fixed by the run seed and the model's index."""
    return np.random.default_rng(np.random.SeedSequence([seed, index]))


def ensemble_p_resampled(draws: Sequence[np.ndarray]) -> np.ndarray:
    """Members' (n_draws, n_prompts) outcome draws -> per-prompt P_ens (share of draws in which
    every member complies)."""
    both = np.logical_and.reduce([np.asarray(d, dtype=bool) for d in draws])
    return both.mean(axis=0)


def ensemble_p_product(ys: Sequence[np.ndarray]) -> np.ndarray:
    """Members' (n_prompts, K) outcome matrices -> per-prompt product of their P."""
    return np.prod([reliability.per_prompt_p(y) for y in ys], axis=0)


def readout(p: np.ndarray) -> dict:
    """Per-prompt P (finite) -> mean vulnerability and the reliably-vulnerable share."""
    p = np.asarray(p, dtype=float)
    assert np.isfinite(p).all()
    return {"mean": float(p.mean()), "reliable": float((p >= EVERY).mean())}


def ratio_of_reductions(single: np.ndarray, unrel: np.ndarray, mz: np.ndarray) -> float:
    """Per-recipe shares -> (mean single - mean unrelated) / (mean single - mean MZ): the ratio of
    the pooled reductions (a ratio of means, not a mean of per-recipe ratios)."""
    den = np.mean(single) - np.mean(mz)
    return float((np.mean(single) - np.mean(unrel)) / den) if den != 0 else float("nan")


def recipe_bootstrap(per_recipe: dict[str, np.ndarray], stat, n_boot: int, seed: int) -> list:
    """95 % percentile interval of `stat(**arrays)` with recipes (the arrays' common index)
    resampled with replacement (stability.boot_indices: one stream, fixed by seed)."""
    n = len(next(iter(per_recipe.values())))
    draws = [stat(**{k: v[i] for k, v in per_recipe.items()})
             for i in stability.boot_indices(n, n_boot, seed)]
    return stability.percentile(draws)
