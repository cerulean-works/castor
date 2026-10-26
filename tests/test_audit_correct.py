import importlib.util
import pathlib

import numpy as np
import pytest

from castor import reliability

spec = importlib.util.spec_from_file_location(
    "audit_correct_006",
    pathlib.Path(__file__).parent.parent / "scripts" / "audit_correct_006.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def _toy(seed=0, n=60, k=20):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    info = {f"s{b}k{j}": reliability.ModelInfo("r", str(b), str(j))
            for b in range(2) for j in range(2)}
    ys = {name: (rng.uniform(size=(n, k)) < p[:, None]).astype(float) for name in info}
    ys["s0k0"][0, 0] = np.nan  # an incoherent sample stays NaN after scaling
    return ys, info


def test_per_model_constant_leaves_twin_correlations_unchanged():
    ys, info = _toy()
    f = {name: np.full(60, c) for name, c in zip(ys, (0.9, 0.7, 0.5, 0.95))}
    a = reliability.twin_correlations(ys, info)
    b = reliability.twin_correlations(m.scaled(ys, f), info)
    for lvl in ("MZ", "U"):
        assert b["levels"][lvl]["r"] == pytest.approx(a["levels"][lvl]["r"], abs=1e-12)
        assert b["levels"][lvl]["r_corrected"] == pytest.approx(
            a["levels"][lvl]["r_corrected"], abs=1e-12)


def test_per_prompt_factor_scales_rates_and_keeps_nan():
    ys, _ = _toy()
    f = {name: np.r_[np.full(30, 0.5), np.ones(30)] for name in ys}
    s = m.scaled(ys, f)
    assert np.isnan(s["s0k0"][0, 0])
    np.testing.assert_allclose(reliability.per_prompt_p(s["s1k1"])[:30],
                               0.5 * reliability.per_prompt_p(ys["s1k1"])[:30])


def test_check_factors_rejects_nan_missing_and_out_of_range():
    m.check_factors({"a": 0.9, "b": 1.0}, "x", ["a", "b"])
    for bad in (float("nan"), 0.0, 1.2, float("inf")):
        with pytest.raises(SystemExit):
            m.check_factors({"a": bad, "b": 1.0}, "x", ["a", "b"])
    with pytest.raises(SystemExit, match="no factor for"):
        m.check_factors({"a": 0.9}, "x", ["a", "b"])
