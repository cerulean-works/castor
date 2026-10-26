import numpy as np
import pytest

from castor import fingerprint, provenance

fp076 = provenance.load_script("fingerprint_076")


def synth(recipe_sd: float, base_sd: float, seed_sd: float, n_p: int = 400, seed: int = 0):
    """Profiles keyed (recipe, base, post-training seed) with known variance structure."""
    rng = np.random.default_rng(seed)
    recipes = [f"r{i:02d}" for i in range(25)]
    bases = ["b1", "b2", "b3"]
    common = rng.normal(size=n_p)
    P = {}
    for r in recipes:
        rr = rng.normal(scale=recipe_sd, size=n_p)
        for b in bases:
            bb = rng.normal(scale=base_sd, size=n_p)
            for s in (1, 2):
                P[(r, b, s)] = common + rr + bb + rng.normal(scale=seed_sd, size=n_p)
    return P, recipes, bases


def test_corr_cross_is_pearson():
    rng = np.random.default_rng(1)
    Q, R = rng.normal(size=(4, 50)), rng.normal(size=(6, 50))
    C = fingerprint.corr_cross(Q, R)
    for i in range(4):
        for j in range(6):
            assert C[i, j] == pytest.approx(np.corrcoef(Q[i], R[j])[0, 1], abs=1e-12)


def test_identity_and_permutation():
    rng = np.random.default_rng(2)
    R = rng.normal(size=(30, 200))
    assert fingerprint.accuracy(R, R, range(30)) == 1.0
    # a wrong truth assignment (shifted by one) is never correct on self-queries
    assert fingerprint.accuracy(R, R, [(i + 1) % 30 for i in range(30)]) == 0.0


def test_run_structure_identifies_run_not_recipe():
    """Strong base signal, no recipe signal beyond the base: runs are identified, recipes from
    an unseen base are at chance level."""
    P, recipes, bases = synth(recipe_sd=0.0, base_sd=1.0, seed_sd=0.2)
    runs = [(r, b) for r in recipes for b in bases]
    Q = np.vstack([P[(r, b, 2)] for r, b in runs])
    R = np.vstack([P[(r, b, 1)] for r, b in runs])
    assert fingerprint.accuracy(Q, R, range(len(runs))) == 1.0
    rec, _ = fp076.recipe_test(P, recipes, bases, "b3", n_boot=200, seed=0)
    assert rec["accuracy"] < 0.2  # chance 0.04; 50 queries


def test_recipe_structure_identifies_recipe():
    P, recipes, bases = synth(recipe_sd=1.0, base_sd=0.1, seed_sd=0.1)
    rec, _ = fp076.recipe_test(P, recipes, bases, "b3", n_boot=200, seed=0)
    assert rec["accuracy"] == 1.0 and rec["ci_exact"][0] > 0.9


def test_recipe_test_leaves_held_base_out():
    """Leave-one-out: make the held base's profiles identical to a decoy written into nothing
    else; if the held base entered the references, accuracy would be 1."""
    P, recipes, bases = synth(recipe_sd=0.0, base_sd=1.0, seed_sd=0.0)
    rec, _ = fp076.recipe_test(P, recipes, bases, "b3", n_boot=200, seed=0)
    assert rec["reference_bases"] == ["b1", "b2"]
    assert rec["accuracy"] < 0.2
    # control: holding nothing out (references include the query's own base) is perfect
    R = np.vstack([np.mean([P[(r, b, s)] for b in bases for s in (1, 2)], axis=0)
                   for r in recipes])
    Q = np.vstack([P[(r, "b3", s)] for r in recipes for s in (1, 2)])
    assert fingerprint.accuracy(Q, R, [i // 2 for i in range(50)]) == 1.0


def test_identify_detail_and_failures():
    R = np.array([[0, 1, 2, 3.0], [3, 2, 1, 0.0], [0, 1, 3, 2.0]])
    Q = np.array([[0, 1, 2.9, 3.1], [0, 1, 3.1, 2.0]])
    res = fingerprint.identify(Q, R, [0, 0])
    assert res["correct"].tolist() == [True, False]
    assert res["assigned"].tolist() == [0, 2]
    assert res["rank"].tolist() == [1, 2]
    assert (res["margin"][0] > 0) and (res["margin"][1] < 0)


def test_assign_nan_never_wins():
    C = np.array([[np.nan, 0.2, 0.1], [np.nan, np.nan, np.nan]])
    assert fingerprint.assign(C).tolist() == [1, -1]


def test_prompt_curve_full_and_subsets():
    P, recipes, bases = synth(recipe_sd=0.0, base_sd=0.3, seed_sd=1.0, n_p=300)
    runs = [(r, b) for r in recipes for b in bases]
    Q = np.vstack([P[(r, b, 2)] for r, b in runs])
    R = np.vstack([P[(r, b, 1)] for r, b in runs])
    cur = fingerprint.prompt_curve(Q, R, range(75), [300, 100, 10], n_sub=30, seed=0)
    assert cur[0]["n_subsets"] == 1 and cur[1]["n_subsets"] == 30
    assert cur[0]["mean"] >= cur[1]["mean"] >= cur[2]["mean"]
    assert all(c["lo"] <= c["mean"] <= c["hi"] for c in cur)
    # a count's draws do not depend on the other counts requested
    again = fingerprint.prompt_curve(Q, R, range(75), [10], n_sub=30, seed=0)
    assert again[0] == cur[2]


def test_clopper_pearson_known_values():
    assert fingerprint.clopper_pearson(75, 75)[0] == pytest.approx(0.025 ** (1 / 75), abs=1e-9)
    lo, hi = fingerprint.clopper_pearson(5, 10)
    assert lo == pytest.approx(0.18709, abs=1e-4) and hi == pytest.approx(0.81291, abs=1e-4)
    assert fingerprint.clopper_pearson(0, 10)[0] == 0.0


def test_boot_ci_degenerate_at_all_correct():
    assert fingerprint.boot_ci(np.ones(75, bool), 200, 0) == [1.0, 1.0]
