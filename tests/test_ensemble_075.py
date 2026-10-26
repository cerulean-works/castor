"""castor.ensemble (075) on synthetic labels with known answers: the resampled estimate converges
to the product rule under independence, draws respect incoherent samples, readouts and the ratio
of reductions; and (when the grid's labels are on disk) 044's shares are reproduced by the run
of record."""

import json
from pathlib import Path

import numpy as np
import pytest

from castor import ensemble

ROOT = Path(__file__).resolve().parents[1]


def _labels(p: np.ndarray, k: int, rng, nan_frac: float = 0.0) -> np.ndarray:
    y = (rng.random((len(p), k)) < p[:, None]).astype(float)
    if nan_frac:
        m = rng.random(y.shape) < nan_frac
        m[:, 0] = False  # keep >= 1 usable sample per prompt
        y[m] = np.nan
    return y


def test_resampled_equals_product_rule_in_the_limit():
    rng = np.random.default_rng(1)
    pa, pb = rng.random(400), rng.random(400)
    ya, yb = _labels(pa, 20, rng, 0.2), _labels(pb, 20, rng, 0.2)
    da = ensemble.draw_outcomes(ya, 20000, ensemble.model_rng(0, 0))
    db = ensemble.draw_outcomes(yb, 20000, ensemble.model_rng(0, 1))
    pr = ensemble.ensemble_p_resampled([da, db])
    pp = ensemble.ensemble_p_product([ya, yb])
    assert np.abs(pr - pp).max() < 0.02  # MC error at 20,000 draws: SD <= 0.0035
    assert abs(pr.mean() - pp.mean()) < 2e-3


def test_draws_use_only_usable_samples():
    y = np.array([[1, np.nan, np.nan], [0, 0, np.nan], [np.nan, np.nan, 1]], dtype=float)
    d = ensemble.draw_outcomes(y, 500, np.random.default_rng(0))
    assert d.shape == (500, 3)
    assert (d[:, 0] == 1).all() and (d[:, 1] == 0).all() and (d[:, 2] == 1).all()
    with pytest.raises(AssertionError):
        ensemble.draw_outcomes(np.array([[np.nan, np.nan]]), 5, np.random.default_rng(0))


def test_draws_are_uniform_over_usable_samples():
    y = np.array([[1, 0, 0, 0, np.nan]], dtype=float)  # P = 0.25 among 4 usable
    d = ensemble.draw_outcomes(y, 40000, np.random.default_rng(3))
    assert d.mean() == pytest.approx(0.25, abs=0.01)


def test_common_random_numbers_and_single_is_k1():
    rng = np.random.default_rng(2)
    y = _labels(rng.random(50), 20, rng)
    a = ensemble.draw_outcomes(y, 100, ensemble.model_rng(7, 3))
    b = ensemble.draw_outcomes(y, 100, ensemble.model_rng(7, 3))
    assert (a == b).all()
    assert np.allclose(ensemble.ensemble_p_resampled([a]), a.mean(axis=0))
    # refuse-if-any can only lower vulnerability
    c = ensemble.draw_outcomes(y, 100, ensemble.model_rng(7, 4))
    assert (ensemble.ensemble_p_resampled([a, c]) <= a.mean(axis=0) + 1e-12).all()


def test_readout_ratio_and_bootstrap():
    r = ensemble.readout(np.array([0.95, 0.9, 0.5, 0.0]))
    assert r == {"mean": pytest.approx(0.5875), "reliable": 0.5}
    s, u, m = np.array([0.2, 0.3]), np.array([0.1, 0.1]), np.array([0.15, 0.25])
    assert ensemble.ratio_of_reductions(s, u, m) == pytest.approx(0.15 / 0.05)
    assert np.isnan(ensemble.ratio_of_reductions(s, u, s))
    lo, hi = ensemble.recipe_bootstrap({"single": s, "unrel": u, "mz": m},
                                       ensemble.ratio_of_reductions, 200, 0)
    assert lo <= ensemble.ratio_of_reductions(s, u, m) <= hi


@pytest.mark.skipif(not (ROOT / "runs/075/summary.json").exists(), reason="no run of record")
def test_run_of_record_reproduces_044():
    s = json.loads((ROOT / "runs/075/summary.json").read_text())
    ref = json.loads((ROOT / "runs/044/summary.json").read_text())["grid"]["prompts"]["levels"]
    for k, v in s["check_044"].items():
        assert abs(v["recomputed"] - ref[k]["share_every"]) < 1e-9
    assert round(ref["seed_stable (within base)"]["share_every"], 3) == 0.162
    assert round(ref["ancestry_independent (within recipe)"]["share_every"], 3) == 0.068
