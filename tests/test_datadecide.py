import json

import pandas as pd
import pytest

from castor.datadecide import seed_spread


def _ev(rows):
    return pd.DataFrame([{"params": p, "step": s, "seed": sd, "data": d, "task": t,
                          "metrics": json.dumps({"primary_metric": v})}
                         for p, s, sd, d, t, v in rows])


def test_seed_spread():
    ev = _ev([("1B", 69369, "default", "r", "t", 0.5),
              ("1B", 69369, "large aux 2", "r", "t", 0.6),
              ("1B", 69369, "large aux 3", "r", "t", 0.7),
              ("1B", 100, "default", "r", "t", 0.0),        # other step: ignored
              ("150M", 69369, "default", "r", "t", 0.0),    # other size: ignored
              ("1B", 69369, "small aux 2", "r", "t", 0.0)])  # other seed: ignored
    row = seed_spread(ev).iloc[0]
    assert row["mean"] == pytest.approx(0.6)
    assert row["range"] == pytest.approx(0.2)
    assert row["sd"] == pytest.approx(0.1)  # sample SD, ddof=1
    assert row["n"] == 3


def test_seed_spread_rejects_duplicates():
    ev = _ev([("1B", 69369, "default", "r", "t", 0.5)] * 2)
    with pytest.raises(AssertionError):
        seed_spread(ev)
