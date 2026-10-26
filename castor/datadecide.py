"""DataDecide released-eval helpers."""

import json

import pandas as pd

SEEDS = ["default", "large aux 2", "large aux 3"]
FINAL_STEP = 69369


def seed_spread(ev: pd.DataFrame, params: str = "1B", step: int = FINAL_STEP,
                seeds: list[str] = SEEDS) -> pd.DataFrame:
    """One row per (recipe, task): the primary metric of each seed, their mean, max-min
    range and sample SD (ddof=1). `ev` has DataDecide-eval-results columns
    (params, step, seed, data, task, metrics as a JSON string)."""
    ev = ev[(ev.params == params) & (ev.step == step) & ev.seed.isin(seeds)]
    ev = ev.assign(score=ev.metrics.map(lambda s: json.loads(s)["primary_metric"]))
    dup = ev.duplicated(["data", "task", "seed"]).sum()
    assert dup == 0, f"{dup} duplicate rows"
    rows = []
    for (data, task), g in ev.groupby(["data", "task"]):
        s = g.set_index("seed").score.reindex(seeds)
        rows.append(dict(data=data, task=task, **{k: s[k] for k in seeds},
                         mean=s.mean(), range=s.max() - s.min(), sd=s.std(ddof=1),
                         n=s.notna().sum()))
    return pd.DataFrame(rows)
