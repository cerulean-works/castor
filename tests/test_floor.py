import numpy as np
import pytest

from castor import floor, reliability

SEEDS = ("default", "large-aux-2", "large-aux-3", "replicate")


@pytest.fixture
def models(synthetic_twins):
    """16 synthetic models on 011's 4 bases; the replicate shares half of default's latent."""
    return lambda seed=0, n=200, k=20: synthetic_twins(SEEDS, {"replicate": 0.5}, seed, n, k)


@pytest.mark.parametrize("a, b, g", [
    ("default", "default", "MZ_006"), ("large-aux-2", "large-aux-2", "MZ_006"),
    ("default", "large-aux-3", "U_006"), ("replicate", "replicate", "MZ_rep"),
    ("default", "replicate", "rung2"), ("replicate", "default", "rung2"),
    ("large-aux-2", "replicate", "rep_other"),
])
def test_group_of(a, b, g):
    assert floor.group_of(a, b) == g


@pytest.mark.parametrize("a, b", [("replicate2", "default"), ("default", "replicate-i6198-d6198")])
def test_group_of_raises_on_an_unknown_label(a, b):
    """011's groups are closed too: a fifth base (035's replicate2) or a revision string must
    not fall silently into U_006 / MZ_006."""
    with pytest.raises(ValueError, match="unknown pretraining-seed label"):
        floor.group_of(a, b)


def test_pair_counts_and_006_levels_reproduced(models):
    ys, info = models()
    tw = reliability.twin_correlations(ys, info)
    s = floor.summary(tw, info)
    assert [s[f"n_{g}"] for g in floor.GROUPS] == [18, 48, 16, 6, 32]
    # 006's MZ / U levels, computed on 006's 12 models alone, equal the MZ_006 / U_006 groups
    six = {m: y for m, y in ys.items() if info[m].pretrain_seed != "replicate"}
    lv = reliability.twin_correlations(six, info)["levels"]
    assert s["r_MZ_006_corrected"] == pytest.approx(lv["MZ"]["r_corrected"], abs=1e-12)
    assert s["r_U_006_corrected"] == pytest.approx(lv["U"]["r_corrected"], abs=1e-12)
    assert s["r_U_006"] == pytest.approx(lv["U"]["r"], abs=1e-12)
    # the synthetic replicate shares part of default's base latent: between U and MZ
    assert s["r_U_006_corrected"] < s["r_rung2_corrected"] < s["r_MZ_006_corrected"]
    assert s["rung2_minus_rep_other_corrected"] == pytest.approx(
        s["r_rung2_corrected"] - s["r_rep_other_corrected"])
    frac = s["rung2_gap_fraction_corrected"]
    assert frac == pytest.approx((s["r_rung2_corrected"] - s["r_U_006_corrected"])
                                 / (s["r_MZ_006_corrected"] - s["r_U_006_corrected"]))


def test_jackknife_over_the_8_rung2_models(models):
    ys, info = models(n=120)
    drop = sorted(m for m in ys if info[m].pretrain_seed in ("default", "replicate"))
    jk = floor.jackknife_models(ys, info, drop)
    r = jk["r_rung2_corrected"]
    assert r["n_reps"] == 8 and r["se"] > 0 and r["lo"] < r["estimate"] < r["hi"]


@pytest.mark.parametrize("r2, lo, hi, holds", [
    (0.85, 0.83, 0.87, True),    # between, nearer U, CI clear of both
    (0.95, 0.94, 0.96, False),   # nearer MZ
    (0.82, 0.80, 0.84, False),   # CI covers r_U
    (0.80, 0.79, 0.81, False),   # below r_U
    (0.98, 0.975, 0.99, False),  # CI covers r_MZ
])
def test_prediction_rule(r2, lo, hi, holds):
    assert floor.prediction(r2, lo, hi, 0.811, 0.984)["holds"] is holds


def test_by_base_pair_is_the_pairs_regrouped(models):
    ys, info = models(n=150)
    tw = reliability.twin_correlations(ys, info)
    bp = floor.by_base_pair(tw["pairs"], info)
    assert len(bp) == 10  # 4 within-base + 6 across-base keys
    assert {k: v["n_pairs"] for k, v in bp.items() if v["group"] in ("MZ_006", "MZ_rep")} == {
        f"{s}|{s}": 6 for s in SEEDS}
    assert all(v["n_pairs"] == 16 for v in bp.values() if "|".join(v) and v["group"] not in ("MZ_006", "MZ_rep"))
    assert bp["default|replicate"]["group"] == "rung2"
    assert bp["large-aux-2|replicate"]["group"] == "rep_other"
    rows = [x for x in tw["pairs"] if {info[x["a"]].pretrain_seed, info[x["b"]].pretrain_seed}
            == {"default", "replicate"}]
    want = np.mean([x["r_corrected"] for x in rows])
    got = bp["default|replicate"]
    assert got["r_corrected"] == pytest.approx(want, abs=1e-12)
    assert got["min_corrected"] <= got["r_corrected"] <= got["max_corrected"]
    assert got["min_corrected"] == pytest.approx(min(x["r_corrected"] for x in rows))
    # the group means of floor.summary are the pair-weighted means of the base-pair rows
    s = floor.summary(tw, info)
    u_rows = [v for v in bp.values() if v["group"] == "U_006"]
    assert s["r_U_006_corrected"] == pytest.approx(
        sum(v["r_corrected"] * v["n_pairs"] for v in u_rows) / sum(v["n_pairs"] for v in u_rows))


def test_base_pair_key_is_unordered():
    assert floor.base_pair_key("replicate", "default") == "default|replicate"
    assert floor.base_pair_key("default", "default") == "default|default"
