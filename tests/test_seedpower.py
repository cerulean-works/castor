"""castor.seedpower on synthetic nested data with known components."""

import numpy as np
import pytest

from castor import gridstats, reliability, seedpower
from castor.reliability import ModelInfo


def nested(a, b, n, s_r, s_b, s_e, rng, mu=0.5):
    return (mu + rng.normal(0, s_r, (a, 1, 1)) + rng.normal(0, s_b, (a, b, 1))
            + rng.normal(0, s_e, (a, b, n)))


def test_anova_recovers_known_components():
    rng = np.random.default_rng(0)
    est = np.array([list(seedpower.anova(nested(25, 3, 2, 0.035, 0.027, 0.021, rng)).values())
                    for _ in range(2000)])
    # method of moments is unbiased: the mean over replicates recovers the truth
    assert est.mean(axis=0) == pytest.approx([0.035 ** 2, 0.027 ** 2, 0.021 ** 2], rel=0.05)


def test_anova_equals_gridstats_nested_variance():
    rng = np.random.default_rng(1)
    y = nested(5, 3, 2, 0.04, 0.03, 0.02, rng)
    means, infos = {}, {}
    for i in range(5):
        for j in range(3):
            for k in range(2):
                m = f"r{i}-s{j}-k{k}"
                means[m], infos[m] = y[i, j, k], ModelInfo(f"r{i}", f"s{j}", str(k + 1))
    ref = gridstats.nested_variance(means, infos)["components"]
    assert seedpower.anova(y) == pytest.approx(ref, rel=1e-12)


def test_sampling_variance_matches_repeat_sampling():
    rng = np.random.default_rng(2)
    p_true = rng.beta(0.7, 0.7, 3000)
    levels, sv, bv = [], [], []
    for _ in range(300):
        y = (rng.random((3000, 20)) < p_true[:, None]).astype(float)
        P = reliability.per_prompt_p(y)
        rel = reliability.split_half(y)["reliability"]
        levels.append(P.mean())
        sv.append(seedpower.sampling_variance(P, rel))
        bv.append(seedpower.binomial_sampling_variance(y))
    truth = np.var(levels, ddof=1)
    assert np.mean(sv) == pytest.approx(truth, rel=0.2)
    assert np.mean(bv) == pytest.approx(truth, rel=0.2)


def test_components_split_sampling_out_of_posttrain():
    rng = np.random.default_rng(3)
    c = seedpower.components(nested(25, 3, 2, 0.03, 0.02, 0.02, rng), sampling=1e-4)
    assert c["posttrain_true"] == pytest.approx(c["posttrain_seed"] - 1e-4)
    assert sum(c["shares"].values()) == pytest.approx(1.0)


def test_se_level_and_smallest_design():
    comp = {"pretrain_seed": 4e-4, "posttrain_seed": 2e-4}
    assert seedpower.se_level(comp, 1, 1) == pytest.approx(np.sqrt(6e-4))
    assert seedpower.se_level(comp, 2, 4) == pytest.approx(np.sqrt(2e-4 + 2e-4 / 8))
    # SE ≤ 0.02 needs var ≤ 4e-4: 2×1 gives 3e-4 (2 models) and 1×q never does (≥ 4e-4 + …)
    assert seedpower.smallest_design(comp, 0.02) == [2, 1]
    assert seedpower.smallest_design(comp, 0.001) is None
    # negative components count as 0
    assert seedpower.se_level({"pretrain_seed": -1.0, "posttrain_seed": 1e-4}, 1, 1) == \
        pytest.approx(0.01)


def test_flip_rate_half_at_zero_true_difference():
    assert seedpower.model_flip(0.0, 1e-3) == 0.5
    rng = np.random.default_rng(4)
    # two recipes whose true levels differ by 1e-9 against seed noise of SD 0.03
    rates = [seedpower.pair_flip(rng.normal(0, 0.03, 6), rng.normal(1e-9, 0.03, 6), -1e-9)
             for _ in range(4000)]
    assert np.mean(rates) == pytest.approx(0.5, abs=0.01)


def test_flip_rate_known_difference_and_ties():
    assert seedpower.pair_flip(np.array([1.0, 2.0]), np.array([1.5]), 1.0) == 0.5
    assert seedpower.pair_flip(np.array([1.0]), np.array([1.0]), 1.0) == 0.5
    assert np.isnan(seedpower.pair_flip(np.array([1.0]), np.array([2.0]), 0.0))
    rng = np.random.default_rng(5)
    s, d = 0.02, 0.02
    rates = [seedpower.pair_flip(rng.normal(d, s, 50), rng.normal(0, s, 50), d)
             for _ in range(400)]
    assert np.mean(rates) == pytest.approx(seedpower.model_flip(d, s ** 2), abs=0.01)


def test_flip_rates_pairs_and_bootstrap_skip_self_pairs():
    y = np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2)
    pairs = seedpower.flip_rates(y)
    assert len(pairs) == 6 and all(p["flip"] == 0.0 for p in pairs)
    assert len(seedpower.flip_rates(y, recipes=[0, 0, 1])) == 2
    s = seedpower.summarise_flips(pairs, min_delta=10)
    assert s["n_pairs"] == 6 and s["n_pairs_ge_2pt"] == 3


def test_g_coefficient():
    assert seedpower.g_coefficient(0.75, 0.9, 1, 1) == pytest.approx(0.75)
    for m in (2, 3, 6):
        assert seedpower.g_coefficient(0.75, 0.9, m, 1) == \
            pytest.approx(reliability.spearman_brown(0.75, m))
    # averaging post-training seeds of one base cannot remove base-level noise
    assert seedpower.g_coefficient(0.75, 0.9, 1, 10 ** 9) == pytest.approx(0.75 / 0.9)


def test_ci_percentile():
    out = seedpower.ci(list(np.linspace(0, 1, 1001)), 0.5)
    assert out["lo"] == pytest.approx(0.025) and out["hi"] == pytest.approx(0.975)
