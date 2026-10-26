"""castor.forecast on synthetic matrices with a known planted run residual."""

import numpy as np

from castor import forecast


def synth(n_rec=40, n_p=1500, plant=1.0, noise_x=1.0, noise_y=1.0, seed=0):
    """Runs = n_rec recipes × 3. True run profile = prompt difficulty + recipe effect + u (the
    run's own part, var 1). Readout X = template effect + recipe effect + plant·u + noise.
    Target halves = profile + independent sample noise; Y = their mean."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_rec), 3)
    n = len(groups)
    d = rng.normal(size=n_p)
    rec = rng.normal(size=(n_rec, n_p))[groups]
    u = rng.normal(size=(n, n_p))
    profile = 2 * d + rec + u
    X = 3 * rng.normal(size=n_p) + 0.5 * rec + plant * u + noise_x * rng.normal(size=(n, n_p))
    Yo = profile + noise_y * rng.normal(size=(n, n_p))
    Ye = profile + noise_y * rng.normal(size=(n, n_p))
    return X, (Yo + Ye) / 2, Yo, Ye, groups


def test_residual_sums_to_zero_within_recipe():
    X, *_, g = synth(n_rec=3, n_p=50)
    R = forecast.recipe_residual(X, g)
    for q in np.unique(g):
        assert np.allclose(R[g == q].sum(0), 0)


def test_recovers_planted_correlation():
    X, Y, Yo, Ye, g = synth()
    s = forecast.summarise(forecast.per_run(X, Y, Yo, Ye, g))
    # residual space: X_res = u_res + e_res, both with the same 2/3 shrink -> corr(X, u) = 1/sqrt 2;
    # the corrected target is u_res itself
    assert abs(s["matched_c"] - 1 / np.sqrt(2)) < 0.03
    assert s["matched"] < s["matched_c"]  # the target is noisy, so correction raises r
    assert s["mismatched_c"] <= 0
    assert abs(s["mismatched_c"] + s["matched_c"] / 2) < 0.03  # three residuals sum to zero


def test_no_plant_gives_zero_and_chance():
    X, Y, Yo, Ye, g = synth(plant=0.0, n_rec=100, n_p=400)
    s = forecast.summarise(forecast.per_run(X, Y, Yo, Ye, g))
    assert abs(s["matched_c"]) < 0.03 and abs(s["diff_c"]) < 0.05
    Rx = forecast.recipe_residual(X, g)
    Ry = forecast.recipe_residual(Y, g)
    fp = forecast.reverse_fingerprint(Ry, Rx, np.arange(len(g)), g, g)
    w = fp["within_recipe"]
    assert w["chance"] == 1 / 3
    assert w["ci_exact"][0] < 1 / 3 < w["ci_exact"][1]


def test_plant_identifies_run():
    X, Y, _, _, g = synth(n_rec=30, n_p=800)
    Rx, Ry = forecast.recipe_residual(X, g), forecast.recipe_residual(Y, g)
    fp = forecast.reverse_fingerprint(Ry, Rx, np.arange(len(g)), g, g)
    assert fp["within_recipe"]["accuracy"] == 1.0 and fp["all"]["accuracy"] > 0.9
    curve = forecast.reverse_curve(Ry, Rx, np.arange(len(g)), g, g, [10, 800], n_sub=20)
    assert curve[1]["n_subsets"] == 1 and curve[1]["within_recipe"]["mean"] == 1.0
    assert curve[0]["within_recipe"]["mean"] < 1.0


def test_bootstraps_bracket_the_estimate():
    X, Y, Yo, Ye, g = synth(n_rec=12, n_p=300)
    pr = forecast.per_run(X, Y, Yo, Ye, g)
    rb = forecast.recipe_bootstrap(pr, g, n_boot=200)
    pb, per_rec = forecast.prompt_bootstrap([(X, Y, Yo, Ye)], g, n_boot=50)
    for b in (rb, pb):
        assert b["matched_c"]["lo"] <= b["matched_c"]["estimate"] <= b["matched_c"]["hi"]
    assert set(per_rec) == set(range(12))


def test_profile_level_mode():
    X, Y, Yo, Ye, g = synth()
    s = forecast.summarise(forecast.per_run(X, Y, Yo, Ye, g, residual=False))
    # raw level: shared difficulty dominates Y but the readout's template part is unrelated
    assert np.isfinite(s["matched_c"]) and abs(s["matched_c"] - s["mismatched_c"]) < 0.2


def test_two_arms_average_per_run():
    a1 = synth(n_rec=5, n_p=200, seed=1)
    a2 = synth(n_rec=5, n_p=200, seed=2)
    g = a1[-1]
    avg = forecast.per_run_avg([a1[:4], a2[:4]], g)
    p1, p2 = forecast.per_run(*a1[:4], g), forecast.per_run(*a2[:4], g)
    assert np.allclose(avg["matched_c"], (p1["matched_c"] + p2["matched_c"]) / 2)
