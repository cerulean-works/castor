import numpy as np
import pytest

from castor import gridstats, reliability
from castor.reliability import ModelInfo


def _sim(n_rec=4, n_seed=3, n_k=2, n=400, K=20, seed=0, recipe_sd=0.8, base_sd=0.6,
         post_sd=0.2, nan_frac=0.01):
    """Nested latent logits: prompt + recipe×prompt + base×prompt + post×prompt."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 1, n)
    ys, infos = {}, {}
    for r in range(n_rec):
        rec = common + rng.normal(0, recipe_sd, n)
        for s in range(n_seed):
            base = rec + rng.normal(0, base_sd, n)
            for k in range(n_k):
                p = 1 / (1 + np.exp(-(base + rng.normal(0, post_sd, n))))
                y = (rng.random((n, K)) < p[:, None]).astype(float)
                y[rng.random((n, K)) < nan_frac] = np.nan
                m = f"r{r}-s{s}-k{k}"
                ys[m], infos[m] = y, ModelInfo(f"r{r}", f"s{s}", str(k))
    return ys, infos


def test_corr_matrix_equals_pairwise_corr():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 50))
    X[rng.random(X.shape) < 0.1] = np.nan
    X[4] = 1.0  # constant row -> NaN (exactly representable, so both agree)
    C = gridstats.corr_matrix(X)
    for i in range(5):
        for j in range(5):
            ref = reliability.corr(X[i], X[j])
            assert (np.isnan(ref) and np.isnan(C[i, j])) or C[i, j] == pytest.approx(ref)
    r = gridstats.rowwise_corr(X[:3], X[1:4])
    assert r == pytest.approx([reliability.corr(X[i], X[i + 1]) for i in range(3)])


def test_levels_match_twin_correlations():
    ys, info = _sim()
    names = sorted(ys)
    infos = [info[m] for m in names]
    P, Po, Pe = gridstats.halves(ys, names)
    rel = gridstats.reliabilities(Po, Pe)
    ref = reliability.twin_correlations(ys, info)
    assert rel == pytest.approx([ref["reliability"][m] for m in names])
    lv = gridstats.levels(P, rel, infos)
    assert lv["R_overall"]["r_corrected"] == pytest.approx(ref["levels"]["R"]["r_corrected"])
    assert lv["R_overall"]["n_pairs"] == ref["levels"]["R"]["n_pairs"]
    for q in ("r0", "r2"):  # one recipe alone through twin_correlations
        sub = {m: y for m, y in ys.items() if info[m].recipe == q}
        t = reliability.twin_correlations(sub, info)["levels"]
        d = lv["per_recipe"][q]
        assert d["MZ"]["n_pairs"] == 3 and d["U"]["n_pairs"] == 12
        assert d["MZ"]["r_corrected"] == pytest.approx(t["MZ"]["r_corrected"])
        assert d["U"]["r"] == pytest.approx(t["U"]["r"])
        assert d["R"]["n_pairs"] == 6 * 18
    # the simulation's nesting shows up in the right order
    assert lv["MZ_r_corrected"] > lv["U_r_corrected"] > lv["R_r_corrected"]
    assert lv["MZ_minus_U_corrected"] == pytest.approx(
        np.mean([lv["per_recipe"][q]["MZ_minus_U_corrected"] for q in lv["per_recipe"]]))


def test_bootstraps_bracket_the_estimate():
    ys, info = _sim(n=200)
    names = sorted(ys)
    infos = [info[m] for m in names]
    P, Po, Pe = gridstats.halves(ys, names)
    lv = gridstats.levels(P, gridstats.reliabilities(Po, Pe), infos)
    cr = gridstats.recipe_bootstrap(lv["per_recipe"], 200, 0)["MZ_minus_U_corrected"]
    cp = gridstats.prompt_bootstrap(P, Po, Pe, infos, 50, 0)
    for c in (cr, cp["MZ_minus_U_corrected"]):
        assert c["lo"] <= c["estimate"] <= c["hi"]
    assert set(cp["per_recipe"]) == {"r0", "r1", "r2", "r3"}


def test_nested_variance_recovers_components():
    rng = np.random.default_rng(0)
    means, infos = {}, {}
    for r in range(200):
        a = rng.normal(0, 2.0)
        for s in range(3):
            b = a + rng.normal(0, 1.0)
            for k in range(2):
                m = f"{r}-{s}-{k}"
                means[m] = b + rng.normal(0, 0.5)
                infos[m] = ModelInfo(str(r), str(s), str(k))
    v = gridstats.nested_variance(means, infos)
    c = v["components"]
    assert c["recipe"] == pytest.approx(4.0, rel=0.25)
    assert c["pretrain_seed"] == pytest.approx(1.0, rel=0.25)
    assert c["posttrain_seed"] == pytest.approx(0.25, rel=0.25)
    assert sum(v["shares"].values()) == pytest.approx(1.0)
    del means["0-0-0"]
    with pytest.raises(AssertionError, match="balanced"):
        gridstats.nested_variance(means, infos)


def test_additive_residual_removes_main_effects():
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 30, 300)
    cols = rng.integers(0, 8, 300)
    re, ce = rng.normal(size=30), rng.normal(size=8)
    inter = rng.normal(0, 0.1, 300)
    p = 0.5 + re[rows] + ce[cols] + inter
    p[5] = np.nan
    res = gridstats.additive_residual(p, rows, cols)
    assert np.isnan(res[5])
    ok = np.isfinite(res)
    # purely additive data leaves no residual; the interaction survives (shrunk)
    pure = gridstats.additive_residual(0.5 + re[rows] + ce[cols], rows, cols)
    assert np.abs(pure).max() < 1e-9
    assert np.corrcoef(res[ok], inter[ok])[0, 1] > 0.8
    assert abs(np.nanmean(res)) < 1e-9


def test_top_decile_transfer():
    ys, info = _sim(n=500)
    names = sorted(ys)
    infos = [info[m] for m in names]
    P, Po, Pe = gridstats.halves(ys, names)
    t = gridstats.top_decile_transfer(P, Po, Pe, infos)
    assert t["k_prompts"] == 50
    m = t["mean_over_recipes"]
    assert 0 < m["R"]["transfer"] < m["U"]["transfer"] < m["MZ"]["transfer"] <= 1.1
    # selection on the odd half, own ceiling on the even half: an identical twin transfers ≈ 1
    same = {"a": ys[names[0]], "b": ys[names[0]].copy()}
    inf2 = [ModelInfo("r", "s", "1"), ModelInfo("r", "s", "2")]
    Q, Qo, Qe = gridstats.halves(same, ["a", "b"])
    tt = gridstats.top_decile_transfer(Q, Qo, Qe, inf2)["mean_over_recipes"]["MZ"]
    assert tt["mean_P_A_even"] == pytest.approx(
        gridstats.top_decile_transfer(Q, Qo, Qe, inf2)["per_recipe"]["r"]["MZ"]["mean_P_A_even"])
    assert tt["mean_P_B"] >= tt["mean_P_A_even"]  # P_B includes the selected odd half


def test_olmes_names_cover_25_recipes():
    assert len(set(gridstats.OLMES_NAMES.values())) == 25


# ---- recipes with one post-training seed (stage C not run) -------------------------------------


def _mixed(n_two=2, n_one=3, **kw):
    """n_two recipes with two post-training seeds, n_one with seed 1 only."""
    ys, infos = _sim(n_rec=n_two + n_one, **kw)
    drop = [m for m, i in infos.items() if int(i.recipe[1:]) >= n_two and i.posttrain_seed == "1"]
    for m in drop:
        del ys[m], infos[m]
    return ys, infos


def test_levels_with_one_seed_recipes():
    ys, info = _mixed()
    names = sorted(ys)
    infos = [info[m] for m in names]
    P, Po, Pe = gridstats.halves(ys, names)
    lv = gridstats.levels(P, gridstats.reliabilities(Po, Pe), infos)
    per = lv["per_recipe"]
    assert per["r0"]["MZ"]["n_pairs"] == 3 and per["r0"]["U"]["n_pairs"] == 12
    assert per["r3"]["MZ"]["n_pairs"] == 0 and np.isnan(per["r3"]["MZ_minus_U_corrected"])
    assert per["r3"]["U"]["n_pairs"] == 3 and np.isfinite(per["r3"]["U"]["r_corrected"])
    assert lv["n_recipes"] == 5 and lv["n_recipes_MZ"] == 2
    # MZ − U over the two-seed recipes, U − R over all
    assert lv["MZ_minus_U_corrected"] == pytest.approx(
        np.mean([per[q]["MZ_minus_U_corrected"] for q in ("r0", "r1")]))
    assert lv["U_minus_R_corrected"] == pytest.approx(
        np.mean([per[q]["U_minus_R_corrected"] for q in per]))
    # the stratified recipe bootstrap keeps both strata's sizes in every draw
    rb = gridstats.recipe_bootstrap(per, 100, 0)
    assert rb["MZ_minus_U_corrected"]["n_draws"] == 100
    assert rb["MZ_minus_U_corrected"]["lo"] <= lv["MZ_minus_U_corrected"] \
        <= rb["MZ_minus_U_corrected"]["hi"]
    pred = gridstats.prediction(lv, {"components": {"recipe": 1.0, "pretrain_seed": 0.5}})
    assert pred["n_recipes"] == 2 and set(pred["per_recipe"]) == {"r0", "r1"}
    cp = gridstats.prompt_bootstrap(P, Po, Pe, infos, 20, 0)
    assert np.isfinite(cp["MZ_minus_U_corrected"]["lo"])


def test_recipe_bootstrap_strata_sizes(monkeypatch):
    # r1's MZ is undefined (NaN) but it has MZ pairs: it stays in the MZ stratum
    per = {f"r{i}": {"MZ": {"n_pairs": 3 if i < 2 else 0,
                            "r_corrected": 0.9 if i == 0 else np.nan}} for i in range(5)}
    seen = []
    monkeypatch.setattr(gridstats, "summarise",
                        lambda p, recipes=None: seen.append(recipes) or {"x": 0.0})
    gridstats.recipe_bootstrap(per, 30, 0)
    for r in seen[1:]:  # seen[0] is the point estimate
        assert len(r) == 5 and sum(q in ("r0", "r1") for q in r) == 2


def test_nested_variance_one_seed_is_base_level():
    rng = np.random.default_rng(0)
    means, infos = {}, {}
    for r in range(300):
        a = rng.normal(0, 2.0)
        for s in range(3):
            m = f"{r}-{s}"
            means[m] = a + rng.normal(0, 1.0) + rng.normal(0, 0.5)
            infos[m] = ModelInfo(str(r), str(s), "1")
    v = gridstats.nested_variance(means, infos)
    assert v["n_posttrain"] == 1 and np.isnan(v["components"]["posttrain_seed"])
    assert v["components"]["pretrain_seed"] == pytest.approx(1.25, rel=0.2)  # 1 + 0.25
    assert v["components"]["recipe"] == pytest.approx(4.0, rel=0.25)
    assert sum(v["shares"].values()) == pytest.approx(1.0) and "posttrain_seed" not in v["shares"]


def test_analyze_variance_design(analyze_008):
    an = analyze_008
    rng = np.random.default_rng(0)
    means, infos = {}, {}
    for r in range(6):
        for s in range(3):
            for k in ((1, 2) if r < 2 else (1,)):
                m = f"{r}-{s}-{k}"
                means[m], infos[m] = rng.normal(), ModelInfo(str(r), str(s), str(k))
    v = an.variance(means, infos)
    assert v["a_recipes"] == 6 and v["n_posttrain"] == 1 and "seed 1 only" in v["design"]
    assert v["two_seed_recipes"]["a_recipes"] == 2 and v["two_seed_recipes"]["n_posttrain"] == 2
    full = {m: x for m, x in means.items() if infos[m].recipe in ("0", "1")}
    assert an.variance(full, infos)["design"] == "all models"


def test_undefined_mz_keeps_its_recipe_in_the_design():
    """A two-seed recipe whose corrected MZ is NaN (a reliability ≤ 0) stays a two-seed recipe:
    counted in n_recipes_MZ and the prediction, flagged in n_recipes_MZ_undefined."""
    ys, info = _mixed()
    names = sorted(ys)
    infos = [info[m] for m in names]
    P, Po, Pe = gridstats.halves(ys, names)
    rel = gridstats.reliabilities(Po, Pe)
    rel[[i for i, m in enumerate(names) if info[m].recipe == "r1"]] = -0.1
    lv = gridstats.levels(P, rel, infos)
    assert np.isnan(lv["per_recipe"]["r1"]["MZ"]["r_corrected"])
    assert lv["n_recipes_MZ"] == 2 and lv["n_recipes_MZ_undefined"] == 1
    pred = gridstats.prediction(lv, {"components": {"recipe": 1.0, "pretrain_seed": 0.5}})
    assert pred["n_recipes"] == 2 and pred["per_recipe"]["r1"] is False
    assert pred["n_recipes_MZ_undefined"] == 1


def test_analyze_variance_with_one_two_seed_recipe(analyze_008):
    """After stage A alone only the main recipe has two post-training seeds: no crash."""
    an = analyze_008
    rng = np.random.default_rng(0)
    means, infos = {}, {}
    for r in range(4):
        for s in range(3):
            for k in ((1, 2) if r == 0 else (1,)):
                m = f"{r}-{s}-{k}"
                means[m], infos[m] = rng.normal(), ModelInfo(str(r), str(s), str(k))
    v = an.variance(means, infos)
    assert v["a_recipes"] == 4 and v["two_seed_recipes"] is None
    assert "1 here" in v["two_seed_recipes_skipped"]


def test_seeds_per_recipe():
    infos = [ModelInfo("a", "s", "1"), ModelInfo("a", "t", "2"), ModelInfo("b", "s", "1")]
    assert gridstats.seeds_per_recipe(infos) == {"a": {"1", "2"}, "b": {"1"}}


def test_prompt_bootstrap_with_no_draws_keeps_its_shape():
    """signal_016 --n-boot 0 (per-size blocks only) reaches prompt_bootstrap through the
    template residual; it must return the point estimates with
    empty intervals (n_draws 0), not index an empty draw list."""
    ys, info = _sim(n=120)
    names = sorted(ys)
    infos = [info[m] for m in names]
    P, Po, Pe = gridstats.halves(ys, names)
    zero = gridstats.prompt_bootstrap(P, Po, Pe, infos, 0, 0)
    some = gridstats.prompt_bootstrap(P, Po, Pe, infos, 5, 0)
    assert set(zero) == set(some) and set(zero["per_recipe"]) == set(some["per_recipe"])
    c = zero["MZ_minus_U_corrected"]
    assert c["estimate"] == some["MZ_minus_U_corrected"]["estimate"]
    assert c["n_draws"] == 0 and np.isnan(c["lo"]) and np.isnan(c["hi"])
    q = next(iter(zero["per_recipe"].values()))
    assert set(q) == set(next(iter(some["per_recipe"].values())))
    assert all(np.isnan(v["lo"]) for v in q.values())
