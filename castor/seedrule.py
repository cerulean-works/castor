"""Is a new pretraining run inside the released seed spread? (z-rule)

    z = |x - m̄_recipe| / (σ_seed · √(4/3))  ≤ 2

m̄_recipe is the mean of the 3 released seeds of the run's recipe; σ_seed is the within-recipe
seed SD pooled over all recipes at that size (25 recipes × (3 − 1) = 50 df). √(4/3) is the SD of
x − m̄ in units of σ when x is a 4th draw from the same seed distribution:
Var(x − m̄) = σ² + σ²/3. OLMES-10 macro and mean held-out CE both use this rule; both must pass.
"""

import math

import pandas as pd


def pooled_sd(df: pd.DataFrame, group: str = "recipe", value: str = "value") -> tuple[float, int]:
    """Within-group SD pooled over groups: sqrt(Σ_g Σ_i (x_gi − x̄_g)² / Σ_g (n_g − 1)).
    Returns (sd, degrees of freedom). Groups with a single value contribute nothing."""
    g = df.groupby(group)[value]
    ss = g.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum()
    dof = int((g.count() - 1).clip(lower=0).sum())
    if dof == 0:
        raise ValueError("no group has more than one value")
    return math.sqrt(ss / dof), dof


def z_score(x: float, released: list[float], sigma: float) -> float:
    """z of a new run's value x against the released seeds of its recipe."""
    if len(released) != 3:
        raise ValueError(f"expected the 3 released seeds, got {len(released)}")
    m = sum(released) / len(released)
    return abs(x - m) / (sigma * math.sqrt(1 + 1 / len(released)))


def passes(z: float, threshold: float = 2.0) -> bool:
    return z <= threshold
