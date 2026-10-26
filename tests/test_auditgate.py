import math

import numpy as np
import pytest

from castor import auditgate as G


def test_tier_edges():
    assert G.tier(0.85, 0.75) == "GO"
    assert G.tier(0.8499, 0.75) == "KILL of the 0.85 gate, tier 1"
    assert G.tier(0.80, 0.80) == "KILL of the 0.85 gate, tier 1"
    assert G.tier(0.7999, 0.9) == "KILL, tier 2"
    assert G.tier(0.95, 0.7499) == "KILL, tier 2"  # one bad model is tier 2 even if overall GO
    assert G.tier(0.82, 0.74) == "KILL, tier 2"
    assert G.tier(float("nan"), 0.9) == G.tier(0.9, float("nan")) == "KILL, tier 2"


def test_weighted_accuracy():
    assert G.weighted_accuracy([True, False], [3, 1]) == pytest.approx(0.75)
    assert math.isnan(G.weighted_accuracy([], []))


def test_flips_down_takes_largest_weights_first():
    # acc = 9/10 = 0.9; the largest correct item (weight 2) is flipped first
    correct = [True] * 5 + [False]
    w = [1, 2, 2, 2, 2, 1]
    f = G.flips_to_cross(correct, w, 0.85)
    assert f["direction"] == "down" and f["accuracy"] == pytest.approx(0.9)
    assert f["n_flips"] == 1  # the weight-2 item: 7/10 < 0.85
    f = G.flips_to_cross(correct, w, 0.35)  # 9 -> 7 -> 5 -> 3 (< 3.5): three flips
    assert f["n_flips"] == 3


def test_flips_up_and_exact_threshold():
    correct = [True, False, False, False]
    w = [1, 1, 1, 1]
    f = G.flips_to_cross(correct, w, 0.75)
    assert f["direction"] == "up" and f["n_flips"] == 2  # 0.25 -> 0.5 -> 0.75 (>= crosses)
    g = G.flips_to_cross([True, True, True, False], w, 0.75)  # exactly at: down direction
    assert g["direction"] == "down" and g["n_flips"] == 1
    assert G.flips_to_cross([False] * 3, [1] * 3, 0.0)["n_flips"] is None  # 0 >= 0: can't go down


def test_threshold_is_decided_at_equality_despite_float_sums():
    # 0.85 of the mass, reached through non-integer weights: the float sum is 1 ulp low
    w = [7723.2, 767.5, 1498.52352941]
    tot = sum(w)
    w.append(tot * 0.15 / 0.85)
    acc = G.weighted_accuracy([True, True, True, False], w)
    assert abs(acc - 0.85) < 1e-12
    assert G.tier(0.85 - 1e-15, 0.75 - 1e-15) == "GO"
    assert G.tier(0.8499, 0.75) == "KILL of the 0.85 gate, tier 1"
    f = G.flips_to_cross([True, True, True, False], w, acc + 1e-15)
    assert f["direction"] == "down"


def _wmean(xs):
    return sum(x["c"] * x["weight"] for x in xs) / sum(x["weight"] for x in xs)


def test_stratified_bootstrap_keeps_strata_and_point():
    items = ([{"stratum": "a", "c": 1.0, "weight": 4.0}] * 5
             + [{"stratum": "b", "c": 0.0, "weight": 4.0}] * 5)

    def stat(xs):
        # every item appears once per draw; stratum weight totals are fixed by Rao-Wu
        wa = sum(x["weight"] for x in xs if x["stratum"] == "a")
        return {"share_a": wa / sum(x["weight"] for x in xs), "mean": _wmean(xs)}

    b = G.stratified_bootstrap(items, stat, n_boot=50, seed=1)
    assert b["share_a"]["point"] == 0.5
    assert b["share_a"]["lo"] == pytest.approx(0.5) and b["share_a"]["hi"] == pytest.approx(0.5)
    assert b["mean"]["lo"] == pytest.approx(0.5) == b["mean"]["hi"]  # constant within strata
    assert b["mean"]["frac_undefined"] == 0.0


def test_stratified_bootstrap_spread():
    items = [{"stratum": "a", "c": float(i % 2), "weight": 10.0} for i in range(40)]
    b = G.stratified_bootstrap(items, lambda xs: {"m": _wmean(xs)}, n_boot=500, seed=0)
    assert b["m"]["point"] == 0.5 and b["m"]["lo"] < 0.5 < b["m"]["hi"]


def test_rao_wu_matches_stratified_variance_in_small_strata():
    # 200 strata of 2 items, negligible sampling fraction: the naive bootstrap would give half
    # the variance; Rao-Wu should give the stratified-estimator variance sum W_h^2 s_h^2 / n_h
    rng = np.random.default_rng(0)
    H = 200
    items = [{"stratum": f"h{h}", "c": float(rng.normal(h % 3, 1)), "weight": 1e9}
             for h in range(H) for _ in range(2)]
    var = 0.0
    for h in range(H):
        y = [x["c"] for x in items if x["stratum"] == f"h{h}"]
        var += (1 / H) ** 2 * np.var(y, ddof=1) / 2
    b = G.stratified_bootstrap(items, lambda xs: {"m": _wmean(xs)}, n_boot=4000, seed=0)
    sd = (b["m"]["hi"] - b["m"]["lo"]) / (2 * 1.959964)
    assert sd == pytest.approx(math.sqrt(var), rel=0.1)


def test_fully_sampled_stratum_has_no_variance():
    items = [{"stratum": "a", "c": float(i % 2), "weight": 1.0} for i in range(10)]
    b = G.stratified_bootstrap(items, lambda xs: {"m": _wmean(xs)}, n_boot=50, seed=0)
    assert b["m"]["lo"] == pytest.approx(0.5) == b["m"]["hi"]


def test_singleton_strata_are_collapsed():
    items = ([{"stratum": "x|1", "cell": "x", "c": 0.0, "weight": 5.0},
              {"stratum": "x|2", "cell": "x", "c": 1.0, "weight": 5.0},  # pool of two
              {"stratum": "y|1", "cell": "y", "c": 1.0, "weight": 5.0}]  # alone in its cell
             + [{"stratum": "y|big", "cell": "y", "c": 0.0, "weight": 5.0}] * 3
             + [{"stratum": "z|big", "cell": "z", "c": 0.0, "weight": 5.0}] * 2)
    groups = G.bootstrap_groups(items, collapse=lambda x: x["cell"])
    sizes = sorted(sorted({x["stratum"] for x in g}) for g in groups)
    assert sizes == [["x|1", "x|2"], ["y|1", "y|big"], ["z|big"]]
    assert sum(map(len, groups)) == len(items)
    # no collapse key: the lone singletons pool together
    assert sorted(map(len, G.bootstrap_groups(items))) == [2, 3, 3]
    b = G.stratified_bootstrap(items, lambda xs: {"m": _wmean(xs)}, n_boot=200, seed=0,
                               collapse=lambda x: x["cell"])
    assert b["m"]["lo"] < b["m"]["hi"]  # the singletons now contribute variance


def test_undefined_draws_are_not_dropped():
    items = [{"stratum": "a", "c": float(i == 0), "weight": 1e9} for i in range(3)]

    def stat(xs):
        # undefined when the only c=1 item is not drawn ((2/3)^2 of draws; its weight then
        # keeps only the tiny finite-population term)
        pos = sum(x["weight"] for x in xs if x["c"])
        v = 1.0 if pos > 1 else float("nan")
        return {"floor_acc": v, "precision": v}

    b = G.stratified_bootstrap(items, stat, n_boot=400, seed=0, nan_low={"floor_acc"})
    assert 0.35 < b["precision"]["frac_undefined"] < 0.55
    assert math.isnan(b["precision"]["lo"]) and math.isnan(b["precision"]["hi"])
    assert b["floor_acc"]["lo"] == -math.inf and b["floor_acc"]["hi"] == 1.0


def test_singletons_in_separate_cells_form_one_group():
    items = [{"stratum": "x|1", "cell": "x", "c": 0.0, "weight": 5.0},
             {"stratum": "y|1", "cell": "y", "c": 1.0, "weight": 5.0}]
    groups = G.bootstrap_groups(items, collapse=lambda x: x["cell"])
    assert [sorted(x["stratum"] for x in g) for g in groups] == [["x|1", "y|1"]]
    # a lone singleton met before any group exists joins one created later
    more = [*items, {"stratum": "z|1", "cell": "z", "c": 0.0, "weight": 5.0},
            {"stratum": "z|2", "cell": "z", "c": 1.0, "weight": 5.0}]
    groups = G.bootstrap_groups(more[:1] + more[2:], collapse=lambda x: x["cell"])
    assert [len(g) for g in groups] == [3]
    with pytest.raises(ValueError):
        G.bootstrap_groups(items[:1])
