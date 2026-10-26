"""Synthetic data with known ceilings.

With p_i ~ Beta(a, b) and K Bernoulli(p_i) samples per prompt, the reliability of the
per-prompt mean is var(p) / (var(p) + E[p(1-p)]/K) = K / (K + a + b).
"""

import numpy as np
import pytest

from castor import reliability as rl
from castor.reliability import ModelInfo


def sample(p: np.ndarray, k: int, rng) -> np.ndarray:
    return (rng.random((len(p), k)) < p[:, None]).astype(float)


def test_per_prompt_p_ignores_nan():
    y = np.array([[1, 0, np.nan, 1], [np.nan] * 4, [0, 0, 0, 1]], dtype=float)
    p = rl.per_prompt_p(y)
    assert p[0] == pytest.approx(2 / 3) and np.isnan(p[1]) and p[2] == pytest.approx(0.25)


def test_spearman_brown():
    assert rl.spearman_brown(0.5) == pytest.approx(2 / 3)
    assert rl.spearman_brown(1.0) == pytest.approx(1.0)


def test_rank_ties_and_spearman():
    assert list(rl._rank(np.array([3.0, 1.0, 3.0, 2.0]))) == [2.5, 0.0, 2.5, 1.0]
    x = np.arange(10.0)
    assert rl.corr(x, x ** 3, "spearman") == pytest.approx(1.0)
    assert np.isnan(rl.corr(np.ones(5), x[:5]))
    assert np.isnan(rl.corr(np.array([1.0, np.nan]), np.array([1.0, 2.0])))


@pytest.mark.parametrize("a,b,k", [(1, 1, 20), (0.5, 0.5, 20), (2, 5, 10)])
def test_split_half_recovers_known_reliability(a, b, k):
    rng = np.random.default_rng(1)
    p = rng.beta(a, b, 20000)
    rel = rl.split_half(sample(p, k, rng))["reliability"]
    assert rel == pytest.approx(k / (k + a + b), abs=0.015)


def test_split_half_with_missing_samples():
    """NaNs at random (incoherent responses) lower K, and reliability with it."""
    rng = np.random.default_rng(2)
    p = rng.beta(1, 1, 20000)
    y = sample(p, 20, rng)
    y[rng.random(y.shape) < 0.5] = np.nan  # effective K ~ 10
    rel = rl.split_half(y)["reliability"]
    assert 10 / 12 - 0.03 < rel < 20 / 22


def test_disattenuate():
    assert rl.disattenuate(0.4, 0.8, 0.5) == pytest.approx(0.4 / np.sqrt(0.4))
    assert np.isnan(rl.disattenuate(0.4, 0.0, 0.5))
    assert np.isnan(rl.disattenuate(0.4, float("nan"), 0.5))


def test_pair_level():
    a = ModelInfo("dclm", "s1", "p1")
    assert rl.pair_level(a, ModelInfo("dclm", "s1", "p2")) == "MZ"
    assert rl.pair_level(a, ModelInfo("dclm", "s2", "p1")) == "U"
    assert rl.pair_level(a, ModelInfo("dolma", "s1", "p1")) == "R"
    assert rl.pair_level(a, a) is None


def synthetic_grid(n=4000, k=20, seed=0):
    """Two recipes x 3 bases x 2 post-training seeds.

    True per-prompt rates: p = (r + s + u) / 3 with r shared within a recipe, s within a base,
    u per model, all Uniform(0,1) and independent. Then the true correlations are
    MZ: 2/3 (r and s shared), U: 1/3 (r shared), R: 0.
    """
    rng = np.random.default_rng(seed)
    ys, info = {}, {}
    for recipe in ("A", "B"):
        r = rng.random(n)
        for base in ("s1", "s2", "s3"):
            s = rng.random(n)
            for post in ("p1", "p2"):
                p = (r + s + rng.random(n)) / 3
                name = f"{recipe}-{base}-{post}"
                ys[name] = sample(p, k, rng)
                info[name] = ModelInfo(recipe, base, post)
    return ys, info


def test_twin_correlations_recover_truth():
    ys, info = synthetic_grid()
    res = rl.twin_correlations(ys, info)
    lv = res["levels"]
    assert lv["MZ"]["n_pairs"] == 6 and lv["U"]["n_pairs"] == 24 and lv["R"]["n_pairs"] == 36
    assert all(lv[k]["n_pairs_defined"] == lv[k]["n_pairs"] for k in lv)
    assert lv["MZ"]["r_corrected"] == pytest.approx(2 / 3, abs=0.03)
    assert lv["U"]["r_corrected"] == pytest.approx(1 / 3, abs=0.03)
    assert lv["R"]["r_corrected"] == pytest.approx(0.0, abs=0.03)
    # raw correlations are attenuated by the ceiling
    assert lv["MZ"]["r"] < lv["MZ"]["r_corrected"]


def test_identical_twins_correct_to_one():
    rng = np.random.default_rng(3)
    p = rng.beta(1, 1, 20000)
    ys = {"a": sample(p, 20, rng), "b": sample(p, 20, rng)}
    info = {"a": ModelInfo("A", "s", "1"), "b": ModelInfo("A", "s", "2")}
    res = rl.twin_correlations(ys, info)
    assert res["levels"]["MZ"]["r_corrected"] == pytest.approx(1.0, abs=0.02)
    assert res["levels"]["MZ"]["r"] == pytest.approx(20 / 22, abs=0.02)


def test_bootstraps_cover_truth():
    ys, info = synthetic_grid(n=1500, seed=4)
    bp = rl.bootstrap_prompts(ys, info, n_boot=60, seed=0)
    assert bp["r_MZ_corrected"]["lo"] < 2 / 3 < bp["r_MZ_corrected"]["hi"]
    assert bp["MZ_minus_U_corrected"]["lo"] > 0
    bb = rl.bootstrap_bases(ys, info, n_boot=40, seed=0)
    assert bb["r_U_corrected"]["n_draws"] > 0
    assert bb["r_MZ"]["lo"] <= bb["r_MZ"]["estimate"] <= bb["r_MZ"]["hi"]
    jk = rl.jackknife_bases(ys, info)
    assert jk["r_MZ_corrected"]["n_bases"] == 6 and jk["r_MZ_corrected"]["se"] > 0


def test_bootstrap_bases_never_pairs_a_base_with_its_copy_as_U():
    ys, info = synthetic_grid(n=300, seed=5)
    seen = []

    def stat(res):
        seen.extend(p for p in res["pairs"] if p["level"] == "U")
        return {"x": 0.0}

    rl.bootstrap_bases(ys, info, n_boot=20, seed=1, stat=stat)
    for p in seen:
        assert p["a"].rsplit("#", 1)[0].rsplit("-", 1)[0] != \
            p["b"].rsplit("#", 1)[0].rsplit("-", 1)[0]


def test_cluster_bootstrap_singletons_match_and_clusters_widen():
    ys, info = synthetic_grid(n=600, seed=6)
    n = next(iter(ys.values())).shape[0]
    single = rl.bootstrap_prompt_clusters(ys, info, list(range(n)), n_boot=30, seed=2)
    plain = rl.bootstrap_prompts(ys, info, n_boot=30, seed=2)
    assert single["MZ_minus_U_corrected"] == plain["MZ_minus_U_corrected"]
    # duplicate every row 5 times: clustering by the original row must give a wider interval
    dup = {m: np.repeat(y, 5, axis=0) for m, y in ys.items()}
    rows = np.repeat(np.arange(n), 5)
    naive = rl.bootstrap_prompts(dup, info, n_boot=60, seed=3)["r_MZ"]
    clus = rl.bootstrap_prompt_clusters(dup, info, rows, n_boot=60, seed=3)["r_MZ"]
    assert clus["hi"] - clus["lo"] > 1.5 * (naive["hi"] - naive["lo"])
    strat = rl.bootstrap_prompt_clusters(dup, info, rows, n_boot=10, seed=3, strata=rows % 2)
    assert strat["r_MZ"]["n_draws"] == 10


def test_cluster_bootstrap_refuses_a_cluster_list_of_another_length():
    """A clusters list shorter than the matrix would leave the trailing prompts out of every
    draw (the CI for another statistic than the estimate), so it is refused."""
    ys, info = synthetic_grid(n=60, seed=1)
    with pytest.raises(AssertionError, match="cluster labels for 60 prompt rows"):
        rl.bootstrap_prompt_clusters(ys, info, list(range(59)), n_boot=2)
    with pytest.raises(AssertionError, match="cluster labels"):
        rl.bootstrap_prompt_clusters(ys, info, list(range(61)), n_boot=2)
