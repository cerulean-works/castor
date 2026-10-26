"""Helpers behind the appendix figures of scripts/figures_041.py: which pair a scatter shows,
the per-prompt range across many twins, and label confusion counts."""

from collections.abc import Iterable, Sequence

import numpy as np


def lower_median_pair(pairs: list[dict], level: str, key: str = "r_corrected") -> dict:
    """The pair of `level` at the lower median of `key`: pairs sorted ascending by (key, a, b),
    index (n - 1) // 2. With an odd count it is the median pair; with an even count
    the lower of the two middle pairs. Never the best pair."""
    sel = sorted((p for p in pairs if p["level"] == level), key=lambda p: (p[key], p["a"], p["b"]))
    if not sel:
        raise ValueError(f"no pair at level {level}")
    return sel[(len(sel) - 1) // 2]


def prompt_range(ps: dict[str, np.ndarray]) -> dict:
    """Per prompt, the mean, min and max of the models' per-prompt values over prompts where every
    model is defined (a prompt with any NaN is dropped and counted)."""
    names = sorted(ps)
    m = np.vstack([np.asarray(ps[n], dtype=float) for n in names])
    ok = np.all(np.isfinite(m), axis=0)
    m = m[:, ok]
    return {"models": names, "keep": ok, "n_dropped": int((~ok).sum()),
            "mean": m.mean(axis=0), "min": m.min(axis=0), "max": m.max(axis=0)}


def stable_shares(lo: np.ndarray, hi: np.ndarray, every: float = 0.9, none: float = 0.1) -> dict:
    """Share of prompts on which every model's value is >= `every` (min >= every: every twin
    breaks) and on which no model's value exceeds `none` (max <= none: no twin breaks)."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    n = len(lo)
    a, b = int((lo >= every).sum()), int((hi <= none).sum())
    return {"n": n, "every": every, "none": none, "n_every": a, "n_none": b,
            "share_every": a / n if n else float("nan"), "share_none": b / n if n else float("nan")}


def confusion(pairs: Iterable[tuple[str, str]], rows: Sequence[str], cols: Sequence[str]) -> dict:
    """Counts of (row label, column label) pairs as {row: {col: count}} in the given orders; a label
    outside the orders raises (nothing is silently dropped)."""
    out = {r: dict.fromkeys(cols, 0) for r in rows}
    for r, c in pairs:
        if r not in out or c not in out[r]:
            raise ValueError(f"label pair {(r, c)} outside {list(rows)} x {list(cols)}")
        out[r][c] += 1
    return out
