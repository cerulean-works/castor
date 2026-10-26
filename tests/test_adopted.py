import numpy as np
import pytest

from castor import adopted, reliability

R = "dclm-baseline"


def _info():
    out = {}
    for b in ("b1", "b2", "b3"):
        for k in (1, 2, 3, 4):
            out[f"t-{b}-{k}"] = adopted.info(R, b, adopted.TULU, k)
        for k in (1, 2):
            out[f"s-{b}-{k}"] = adopted.info(R, b, adopted.SWAP, k)
    return out


def test_levels_count_as_preregistered():
    info = _info()
    tw = reliability.twin_correlations(
        {m: np.random.default_rng(0).random((50, 20)) for m in info}, info,
        level_of=adopted.level_of, levels=adopted.LEVELS)
    n = {k: v["n_pairs"] for k, v in tw["levels"].items()}
    assert n == {"adopted": 24, "swapMZ": 3, "swapU": 12, "cross": 48, "MZ": 18, "U": 48}


def test_level_of_edges():
    a = adopted.info(R, "b1", adopted.TULU, 1)
    assert adopted.level_of(a, a) is None
    assert adopted.level_of(a, adopted.info("c4", "b1", adopted.SWAP, 1)) is None
    # a Tulu seed 1 and a swap seed 1 on one base are two models, not the same one
    assert adopted.level_of(a, adopted.info(R, "b1", adopted.SWAP, 1)) == "adopted"
    with pytest.raises(AssertionError):
        adopted.info(R, "b1", "other", 1)


def test_base_signal_recovered():
    """Profiles = base effect + arm effect + noise: same-base pairs beat other-base pairs, and
    the adopted level sits between MZ and U when the arm shifts the profile."""
    rng = np.random.default_rng(1)
    n = 400
    base = {b: rng.normal(size=n) for b in ("b1", "b2", "b3")}
    arm = {adopted.TULU: rng.normal(size=n), adopted.SWAP: rng.normal(size=n)}
    info = _info()
    ys = {}
    for m, i in info.items():
        z = base[i.pretrain_seed] + 0.6 * arm[adopted.arm(i)]
        p = 1 / (1 + np.exp(-z))
        ys[m] = (rng.random((n, 20)) < p[:, None]).astype(float)
    tw = reliability.twin_correlations(ys, info, level_of=adopted.level_of,
                                       levels=adopted.LEVELS)
    s = adopted.summary(tw)
    assert s["r_MZ_corrected"] > s["r_adopted_corrected"] > s["r_U_corrected"]
    assert s["r_swapMZ_corrected"] > s["r_swapU_corrected"]
    assert s["r_adopted_corrected"] > s["r_cross_corrected"]
    assert s["adopted_minus_cross_corrected"] == pytest.approx(
        s["r_adopted_corrected"] - s["r_cross_corrected"])
    assert s["MZ_minus_adopted_corrected"] == pytest.approx(
        s["r_MZ_corrected"] - s["r_adopted_corrected"])


def test_reading_rule():
    assert adopted.reading(0.85, 0.82, 0.811, 0.7) == {
        "band": "above r_U", "prediction_holds": True, "point_above_r_U": True,
        "interval_excludes_r_U": True}
    r = adopted.reading(0.82, 0.80, 0.811, 0.7)
    assert r["band"] == "above r_U" and not r["prediction_holds"]
    assert adopted.reading(0.75, 0.70, 0.811, 0.7)["band"] == "between swap U and r_U"
    assert adopted.reading(0.70, 0.65, 0.811, 0.7)["band"] == "at or below swap U"


def test_level_row_ceiling():
    p = np.array([0.95, 0.92, np.nan])
    r = adopted.level_row(p, 0.9)
    assert r["ceiling_flag"] and r["n_prompts_defined"] == 2
    assert not adopted.level_row(np.array([0.5, 0.6]), 0.9)["ceiling_flag"]


Q25, Q50 = "dclm-baseline-25p-dolma1.7-75p", "dclm-baseline-50p-dolma1.7-50p"


def _info_three_recipes():
    """080's 36 models: Tulu main seeds 1-4, Tulu query recipes seeds 1-2, swap main seeds 1-2,
    swap query recipes seed 1."""
    out = _info()
    for q in (Q25, Q50):
        for b in ("b1", "b2", "b3"):
            for k in (1, 2):
                out[f"t-{q}-{b}-{k}"] = adopted.info(q, b, adopted.TULU, k)
            out[f"s-{q}-{b}-1"] = adopted.info(q, b, adopted.SWAP, 1)
    return out


def test_ladder_arms_and_pairs():
    """(B, 093): the arms hold the right cells and 008's level definitions give the pair counts."""
    info = _info_three_recipes()
    arms = adopted.ladder_arms(info)
    assert list(arms) == ["tulu_matched", "swap", "tulu_grid008"]
    assert [len(v) for v in arms.values()] == [12, 12, 18]
    cell = lambda i: (i.recipe, i.pretrain_seed, i.posttrain_seed.split(":")[1])
    assert {cell(i) for i in arms["tulu_matched"].values()} == {
        cell(i) for i in arms["swap"].values()}
    assert all(i.posttrain_seed in ("tulu:1", "tulu:2") for i in arms["tulu_grid008"].values())
    rng = np.random.default_rng(2)
    ys = {m: (rng.random((40, 20)) < 0.5).astype(float) for m in info}
    lad = adopted.ladder(ys, arms, n_boot=3, seed=0)
    sw, tg = lad["pairs"]["swap"], lad["pairs"]["tulu_grid008"]
    assert sw[R] == {"MZ": 3, "U": 12, "R": 6 * 6} and sw[Q25] == {"MZ": 0, "U": 3, "R": 3 * 9}
    assert lad["pairs"]["tulu_matched"] == sw
    assert tg[R] == {"MZ": 3, "U": 12, "R": 6 * 12} and tg[Q50] == {"MZ": 3, "U": 12, "R": 6 * 12}
    ci = lad["ci_prompts"]
    d = ci["swap_minus_tulu_matched.U_minus_R_corrected"]["estimate"]
    assert d == pytest.approx(ci["swap.U_minus_R_corrected"]["estimate"]
                              - ci["tulu_matched.U_minus_R_corrected"]["estimate"])
    # the swap arm's main-recipe U is the mean of the same 12 pairs as the 2x2's swapU
    tw = reliability.twin_correlations({m: ys[m] for m in info if info[m].recipe == R}, info,
                                       level_of=adopted.level_of, levels=adopted.LEVELS)
    assert ci[f"swap.{R}.U_r_corrected"]["estimate"] == pytest.approx(
        tw["levels"]["swapU"]["r_corrected"], abs=1e-12)
    assert ci[f"swap.{R}.MZ_r_corrected"]["estimate"] == pytest.approx(
        tw["levels"]["swapMZ"]["r_corrected"], abs=1e-12)


def test_source_split_points():
    """(A, 093): each source's point is the 2x2 on that source's rows alone, and the difference
    key is the second source minus the first."""
    info = _info()
    rng = np.random.default_rng(3)
    ys = {m: (rng.random((30, 20)) < 0.4).astype(float) for m in info}
    groups = {"wj": np.arange(0, 18), "hj": np.arange(18, 30)}
    res = adopted.source_split(ys, info, groups, n_boot=5, seed=0)
    assert res["n_prompts"] == {"wj": 18, "hj": 12}
    ci = res["ci_prompts"]
    for g, rows in groups.items():
        s = adopted.summary(reliability.twin_correlations(
            {m: y[rows] for m, y in ys.items()}, info, level_of=adopted.level_of,
            levels=adopted.LEVELS))
        assert ci[f"{g}.r_adopted_corrected"]["estimate"] == pytest.approx(
            s["r_adopted_corrected"], abs=1e-12)
    assert ci["hj_minus_wj.r_cross_corrected"]["estimate"] == pytest.approx(
        ci["hj.r_cross_corrected"]["estimate"] - ci["wj.r_cross_corrected"]["estimate"])
    assert ci["wj.r_adopted_corrected"]["n_draws"] + ci["wj.r_adopted_corrected"][
        "n_draws_undefined"] == 5
