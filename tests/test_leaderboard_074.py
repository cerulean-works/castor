"""castor.leaderboard on synthetic shares with known flip rates, pooled / bootstrap weights
against brute-force loops, top-k change with ties, and the runs/044 reproduction (synthetic:
per-recipe means of template levels equal the grid level; artifact: runs/074 equals runs/044 to
1e-9)."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from castor import leaderboard as lb
from castor import reliability, stability
from castor.reliability import ModelInfo

ROOT = Path(__file__).resolve().parents[1]


def test_flip_rule_and_ties():
    sa = np.array([1, -1, 0, 1, 0], dtype=np.int8)
    sb = np.array([-1, -1, 1, 0, 0], dtype=np.int8)
    assert lb.flip(sa, sb).tolist() == [1.0, 0.0, 0.5, 0.5, 0.0]


def test_gap_bins_edges():
    m = np.array([0.0, 0.0, 0.019, 0.02, 0.049, 0.05, 0.1, 0.2, 0.9])
    iu = (np.zeros(8, dtype=int), np.arange(1, 9))
    assert lb.gap_bins(m, iu).tolist() == [0, 0, 1, 1, 2, 3, 4, 4]


def test_known_flip_rates():
    # model a orders three templates 0 > 1 > 2, b reverses all, c equals a: a-b flips 3 / 3,
    # a-c 0 / 3, b-c 3 / 3
    S = np.array([[0.5, 0.4, 0.3], [0.3, 0.4, 0.5], [0.5, 0.4, 0.3]])
    iu = lb.template_pairs(3)
    sg = lb.signs(S, iu)
    assert lb.flip(sg[0], sg[1]).mean() == 1.0
    assert lb.flip(sg[0], sg[2]).mean() == 0.0
    bins = lb.gap_bins(S.mean(axis=0), iu)  # gaps 0.0333, 0.0667, 0.0333 points 3.3 / 6.7 / 3.3
    assert bins.tolist() == [1, 2, 1]
    F = np.vstack([lb.flip(sg[p], sg[q]) for p, q in ((0, 1), (0, 2), (1, 2))])
    f, c = lb.bin_sums(F, bins)
    assert f.tolist() == [0, 4, 2, 0, 0] and c.tolist() == [0, 6, 3, 0, 0]


def _rand_shares(rng, n_m, n_t):
    return np.round(rng.random((n_m, n_t)), 1)  # coarse: ties occur


def test_observed_gap_brute_force():
    rng = np.random.default_rng(1)
    S = _rand_shares(rng, 4, 7)
    iu = lb.template_pairs(7)
    sg = lb.signs(S, iu)
    pairs = [(0, 1), (2, 3), (0, 3)]
    f, c = lb.observed_gap_sums(S, sg, pairs, iu)
    bf, bc = np.zeros(5), np.zeros(5)
    for x, y in pairs:
        for a, b in ((x, y), (y, x)):
            for i, j in zip(*iu, strict=True):
                da, db = S[a, i] - S[a, j], S[b, i] - S[b, j]
                if da == 0:
                    continue
                k = sum(abs(da) >= e for e in lb.EDGES) - 1
                bc[k] += 1
                bf[k] += 1.0 if np.sign(db) == -np.sign(da) else (0.5 if db == 0 else 0.0)
    assert np.allclose(f, bf) and np.allclose(c, bc)


def test_pooled_weights_brute_force():
    rng = np.random.default_rng(2)
    f, c = rng.random((4, 5)), rng.random((4, 5)) + 1
    w = np.array([2, 0, 1, 1])
    want = sum(w[r] * f[r] for r in range(4)) / sum(w[r] * c[r] for r in range(4))
    assert np.allclose(lb.pooled_within(f, c, w), want)
    fx, cx = rng.random((4, 4, 5)), rng.random((4, 4, 5)) + 1
    fx, cx = fx + fx.transpose(1, 0, 2), cx + cx.transpose(1, 0, 2)
    for A in (fx, cx):
        for r in range(4):
            A[r, r] = 0
    draw = [0, 0, 2, 3]  # recipe 0 twice: the (0, 0) position pair is skipped
    num, den = np.zeros(5), np.zeros(5)
    for i, j in combinations(range(4), 2):
        if draw[i] != draw[j]:
            num += fx[draw[i], draw[j]]
            den += cx[draw[i], draw[j]]
    w = np.bincount(draw, minlength=4)
    assert np.allclose(lb.pooled_cross(fx, cx, w), num / den)


def test_template_pair_weights():
    iu = lb.template_pairs(4)
    W = lb.template_pair_weights(4, iu, 3, 0)
    rng = np.random.default_rng(0)
    for d in range(3):
        w = np.bincount(rng.integers(0, 4, 4), minlength=4)
        assert W[d].tolist() == [w[i] * w[j] for i, j in zip(*iu, strict=True)]


def test_top_changes_with_ties():
    assert lb.top1_change(np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])) == 0.5
    assert lb.top1_change(np.array([0.2, 0.9]), np.array([0.9, 0.2])) == 1.0
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    tb = np.zeros(7)
    assert lb.top5_change(s, s, tb) == (0.0, 1.0, 0)
    t = s.copy()
    t[[4, 5]] = t[[5, 4]]  # 5th and 6th swap
    assert lb.top5_change(s, t, tb) == (1.0, 0.8, 0)
    u = s.copy()
    u[5] = u[4]  # tie at the 5th place, broken by tiebreak, then index
    assert lb.top5_change(s, u, tb)[2] == 1
    assert lb.top_set(u, 5, np.array([0, 0, 0, 0, 0, 1.0, 0]))[0] == frozenset({0, 1, 2, 3, 5})


def test_per_recipe_means_equal_grid_levels():
    # equal pair counts per recipe: the grid's MZ / U levels are the means of per-recipe levels
    rng = np.random.default_rng(3)
    info = [ModelInfo(r, s, k) for r in ("r1", "r2", "r3") for s in ("a", "b", "c")
            for k in ("1", "2")]
    S = rng.random((18, 12))
    E, O = (S + 0.1 * rng.random((18, 12)) for _ in range(2))  # reliable halves
    full = stability.template_levels(S, E, O, stability.level_masks(info))
    for k in ("r_MZ", "r_MZ_corrected", "r_U", "r_U_corrected"):
        per = [stability.template_levels(S[v], E[v], O[v],
                                         stability.level_masks([info[j] for j in v]))[k]
               for v in (range(6), range(6, 12), range(12, 18))]
        assert np.mean(per) == pytest.approx(full[k], abs=1e-12)
    assert reliability.LEVELS == ("MZ", "U", "R")


@pytest.mark.skipif(not (ROOT / "runs/074/summary.json").exists(), reason="no 074 run yet")
def test_074_reproduces_044():
    s74 = json.loads((ROOT / "runs/074/summary.json").read_text())
    s44 = json.loads((ROOT / "runs/044/summary.json").read_text())["grid"]["templates"]
    assert s74["n_templates"] == s44["n_templates"] == 91
    for k, v in s74["check_044"].items():
        assert abs(v["074"] - s44["levels"][k.split(" ")[0]]["value"]) < 1e-9, k
