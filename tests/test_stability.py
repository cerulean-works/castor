"""castor.stability (044) on synthetic labels with known answers: stable / flipping prompts per
level, pair counts against a brute-force loop, and template rank agreement of identical and
permuted templates."""

from itertools import combinations

import numpy as np
import pytest

from castor import reliability, stability
from castor.reliability import ModelInfo

INFO = [ModelInfo(r, s, k) for r in ("r1", "r2") for s in ("a", "b") for k in ("1", "2")]


def test_groups_and_stable_classes_known_prompts():
    # prompts: 0 broken by all; 1 refused by all; 2 broken within base r1/a only;
    # 3 broken by all of recipe r1 only; 4 flips within every base (seed 1 breaks, seed 2 not)
    P = np.array([[1.0, 0.0, 0.95, 0.9, 0.95],
                  [0.9, 0.05, 0.92, 0.95, 0.5],
                  [0.95, 0.1, 0.5, 0.91, 0.99],
                  [1.0, 0.0, 0.2, 0.93, 0.3],
                  [0.97, 0.0, 0.5, 0.2, 0.9],
                  [0.99, 0.1, 0.5, 0.2, 0.2],
                  [0.93, 0.02, 0.5, 0.3, 0.92],
                  [0.96, 0.0, 0.5, 0.4, 0.0]])
    base = stability.group_stable(P, stability.groups_of(INFO, "base"))
    rec = stability.group_stable(P, stability.groups_of(INFO, "recipe"))
    uni = stability.group_stable(P, stability.groups_of(INFO, "all"))
    assert base["n_groups"] == 4 and rec["n_groups"] == 2 and uni["n_groups"] == 1
    assert base["n_every"].tolist() == [4, 0, 1, 2, 0]
    assert rec["n_every"].tolist() == [2, 0, 0, 1, 0]
    assert uni["n_every"].tolist() == [1, 0, 0, 0, 0]
    assert uni["n_none"].tolist() == [0, 1, 0, 0, 0]
    assert base["per_group"]["r1/a"]["n_every"] == 3  # prompts 0, 2, 3
    assert base["every_frac"].mean() == pytest.approx((3 + 1 + 1 + 2) / 4 / 5)


def _brute(P, info):
    hi, lo = P >= stability.EVERY, P <= stability.NONE
    out = {lvl: {"flip": np.zeros(P.shape[1]), "strong": np.zeros(P.shape[1]),
                 "both": np.zeros(P.shape[1]), "n_pairs": 0} for lvl in stability.LEVELS}
    for a, b in combinations(range(len(info)), 2):
        lvl = reliability.pair_level(info[a], info[b])
        out[lvl]["flip"] += hi[a] ^ hi[b]
        out[lvl]["strong"] += (hi[a] & lo[b]) | (lo[a] & hi[b])
        out[lvl]["both"] += hi[a] & hi[b]
        out[lvl]["n_pairs"] += 1
    return out


def test_pair_counts_equal_brute_force():
    rng = np.random.default_rng(1)
    info = [ModelInfo(f"r{r}", f"s{s}", str(k)) for r in range(3) for s in range(3)
            for k in range(2)]
    P = rng.choice([0.0, 0.05, 0.1, 0.5, 0.85, 0.9, 1.0], size=(len(info), 40))
    got, want = stability.pair_counts(P, info), _brute(P, info)
    for lvl in stability.LEVELS:
        assert got[lvl]["n_pairs"] == want[lvl]["n_pairs"]
        for k in ("flip", "strong", "both"):
            assert np.allclose(got[lvl][k], want[lvl][k]), (lvl, k)
    assert (got["MZ"]["n_pairs"], got["U"]["n_pairs"], got["R"]["n_pairs"]) == (9, 36, 108)


def test_pair_rates_flipping_prompt():
    P = np.array([[0.95, 1.0], [0.5, 1.0], [0.95, 1.0], [0.0, 1.0],
                  [0.95, 1.0], [0.5, 1.0], [0.95, 1.0], [0.0, 1.0]])
    c = stability.pair_counts(P, INFO)
    mz = stability.pair_rates(c["MZ"])
    # prompt 0 flips in every base, prompt 1 never: MZ flip share 1/2; strong flips in r?/b
    assert mz["flip"] == pytest.approx(0.5)
    assert mz["strong_flip"] == pytest.approx(0.25)
    assert mz["cobreak"] == pytest.approx(0.5)  # prompt 1 broken by both, every base
    assert mz["transfer"] == pytest.approx(2 * 4 / (2 * 4 + 4))
    assert stability.pair_rates(c["MZ"], np.array([1, 1]))["flip"] == 0.0


def _labels(p_true, k, rng):
    return (rng.random((len(p_true), k)) < p_true[:, None]).astype(float)


def test_template_rank_agreement_identical_vs_permuted():
    rng = np.random.default_rng(0)
    n_t, per = 30, 8
    tpl = [f"jb{t}" for t in range(n_t) for _ in range(per)]
    names, col = stability.template_index(tpl)
    eff = np.linspace(0.05, 0.95, n_t)
    perm = rng.permutation(n_t)
    info = [ModelInfo("r", "a", "1"), ModelInfo("r", "a", "2"), ModelInfo("r", "b", "1"),
            ModelInfo("r", "b", "2")]
    truth = {0: eff, 1: eff, 2: eff[perm], 3: eff[perm]}  # base b: templates permuted
    ys = {f"m{i}": _labels(np.array([truth[i][names.index(t)] for t in tpl]), 400, rng)
          for i in range(4)}
    inf = dict(zip(sorted(ys), info, strict=True))
    tw = reliability.twin_correlations(ys, inf, method="spearman",
                                       measure=stability.template_measure(col, len(names)))
    assert tw["levels"]["MZ"]["r_corrected"] > 0.95
    assert abs(tw["levels"]["U"]["r_corrected"]) < 0.5
    # the vectorised path equals 006's function
    sh = [stability.template_shares(ys[m], col, len(names)) for m in sorted(ys)]
    S, E, O = (np.vstack([s[k] for s in sh]) for k in ("full", "even", "odd"))
    v = stability.template_levels(S, E, O, stability.level_masks(info))
    for lvl in ("MZ", "U"):
        assert v[f"r_{lvl}"] == pytest.approx(tw["levels"][lvl]["r"], abs=1e-12)
        assert v[f"r_{lvl}_corrected"] == pytest.approx(tw["levels"][lvl]["r_corrected"],
                                                        abs=1e-12)


def test_template_shares_nan_and_ranks():
    col = np.array([0, 0, 1, 2])
    y = np.array([[1, 1], [0, 0], [np.nan, np.nan], [1, 0]], dtype=float)
    s = stability.template_shares(y, col, 3)
    assert s["full"][0] == 0.5 and np.isnan(s["full"][1]) and s["full"][2] == 0.5
    assert stability.rank_rows(np.array([[3.0, 1.0, 1.0]])).tolist() == [[2.0, 0.5, 0.5]]


def test_bootstrap_helpers():
    a = stability.boot_indices(10, 3, 0)
    assert all((x == y).all() for x, y in zip(a, stability.boot_indices(10, 3, 0), strict=True))
    lo, hi = stability.percentile(np.arange(101.0))
    assert lo == pytest.approx(2.5) and hi == pytest.approx(97.5)
