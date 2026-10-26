"""Per-prompt vulnerability profiles as fingerprints of the pretraining run.

Definitions (pre-registered):

- A profile is a model's per-prompt P(non-refusal | coherent) over the adversarial-harmful
  prompts (judge_v1, `wg_vulnerable`, K = 20).
- Identification: a query profile is assigned to the reference profile with the highest
  Pearson correlation over prompts (nearest neighbour by correlation). Accuracy = share of
  queries assigned to their own reference.
- Run identification: references = the 75 bases' post-training-seed-1 models, queries = the
  seed-2 models (and the reverse direction).
- Recipe identification: references = each recipe's mean profile over its first two bases
  (both post-training seeds, 4 models), queries = the third base's models (50); chance 1/25.
- Prompt-count curve: accuracy on random prompt subsets of size n (200 subsets per n),
  mean and 2.5–97.5 % range over subsets.
- Interval: percentile bootstrap over queries (references fixed), 1,000 draws.
"""

from collections.abc import Sequence

import numpy as np


def zscore_rows(X: np.ndarray) -> np.ndarray:
    """Rows centred and scaled to unit norm, so that A @ B.T is the Pearson correlation.
    Requires finite X; a constant row gives NaN."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return Xc / np.linalg.norm(Xc, axis=1, keepdims=True)


def corr_cross(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """(n_q, n_p), (n_r, n_p) -> (n_q, n_r) Pearson correlations over prompts."""
    assert Q.shape[1] == R.shape[1]
    return zscore_rows(Q) @ zscore_rows(R).T


def assign(C: np.ndarray) -> np.ndarray:
    """Index of the highest-correlation reference per query (first on ties); NaN correlations
    (a constant profile on a small prompt subset) never win, and a query whose correlations
    are all NaN is assigned -1 (a failure)."""
    D = np.where(np.isnan(C), -np.inf, C)
    return np.where(np.isfinite(D).any(axis=1), np.argmax(D, axis=1), -1)


def identify(Q: np.ndarray, R: np.ndarray, truth: Sequence[int]) -> dict:
    """Nearest-reference assignment of each query, with per-query detail: the true reference's
    correlation and rank (1 = assigned), the best impostor and the margin true − impostor."""
    truth = np.asarray(truth)
    C = corr_cross(Q, R)
    got = assign(C)
    rows = np.arange(len(truth))
    c_true = C[rows, truth]
    imp = C.copy()
    imp[rows, truth] = -np.inf
    best_imp = np.argmax(imp, axis=1)
    rank = 1 + (C > c_true[:, None]).sum(axis=1)
    return {"correct": got == truth, "assigned": got, "c_true": c_true,
            "impostor": best_imp, "c_impostor": imp[rows, best_imp],
            "margin": c_true - imp[rows, best_imp], "rank": rank, "C": C}


def accuracy(Q: np.ndarray, R: np.ndarray, truth: Sequence[int]) -> float:
    return float(np.mean(assign(corr_cross(Q, R)) == np.asarray(truth)))


def boot_ci(correct: np.ndarray, n_boot: int = 1000, seed: int = 0,
            alpha: float = 0.05) -> list[float]:
    """Percentile bootstrap over queries of the accuracy (references fixed)."""
    correct = np.asarray(correct, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    d = correct[idx].mean(axis=1)
    return [float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))]


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k), X ~ Binomial(n, p)."""
    from math import comb
    return float(sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1)))


def _bisect(f, lo: float = 0.0, hi: float = 1.0, it: int = 100) -> float:
    """Root of a decreasing f on [lo, hi]."""
    for _ in range(it):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float]:
    """Exact binomial interval (Clopper & Pearson 1934), secondary to the pre-registered
    bootstrap, which is degenerate at 0 or n correct. Bisection on the binomial CDF."""
    lo = 0.0 if k == 0 else _bisect(lambda p: alpha / 2 - (1 - _binom_cdf(k - 1, n, p)))
    hi = 1.0 if k == n else _bisect(lambda p: _binom_cdf(k, n, p) - alpha / 2)
    return [lo, hi]


def prompt_curve(Q: np.ndarray, R: np.ndarray, truth: Sequence[int], counts: Sequence[int],
                 n_sub: int = 200, seed: int = 0) -> list[dict]:
    """Accuracy on random prompt subsets (without replacement) of each size; the full set is
    evaluated once. Subsets of one size are drawn from one generator seeded per size, so a
    count's draws do not depend on which other counts are requested."""
    n_p = Q.shape[1]
    out = []
    for n in counts:
        if n >= n_p:
            acc = np.array([accuracy(Q, R, truth)])
        else:
            rng = np.random.default_rng([seed, n])
            acc = np.array([accuracy(Q[:, s], R[:, s], truth)
                            for s in (rng.choice(n_p, n, replace=False) for _ in range(n_sub))])
        out.append({"n_prompts": int(min(n, n_p)), "n_subsets": len(acc),
                    "mean": float(acc.mean()), "lo": float(np.quantile(acc, 0.025)),
                    "hi": float(np.quantile(acc, 0.975)), "min": float(acc.min())})
    return out
