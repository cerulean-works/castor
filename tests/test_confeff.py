import numpy as np
import pytest

from castor import confeff, reliability

SEEDS = ("default", "large-aux-2", "large-aux-3")


def _models(seed=0, n=300, k=20, scramble_a0=False):
    """24 synthetic models: 12 cells x {zero3, zero1}. The zero1 copy of a cell shares the cell's
    latent (a configuration change is noise); with scramble_a0 the zero1 copies lose it."""
    rng = np.random.default_rng(seed)
    recipe = rng.normal(size=n)
    base = {s: rng.normal(size=n) for s in SEEDS}
    ys, info = {}, {}
    for s in SEEDS:
        for kk in range(1, 5):
            cell = 0.3 * rng.normal(size=n)
            for cfg in (confeff.ZERO3, confeff.ZERO1):
                logit = recipe + 0.7 * base[s] + cell + 0.15 * rng.normal(size=n)
                if scramble_a0 and cfg == confeff.ZERO1:
                    logit = recipe + 0.7 * base[s] + 0.3 * rng.normal(size=n) + 0.15 * rng.normal(size=n)
                p = 1 / (1 + np.exp(-logit))
                m = f"{cfg}-{s}-s{kk}"
                ys[m] = (rng.random((n, k)) < p[:, None]).astype(float)
                info[m] = confeff.CfgInfo("dclm-baseline", s, str(kk), cfg)
    return ys, info


def test_kind_of():
    i = lambda s, k, c: confeff.CfgInfo("r", s, str(k), c)
    assert confeff.kind_of(i("d", 1, "zero3"), i("d", 1, "zero1")) == "same_cell_cross"
    assert confeff.kind_of(i("d", 1, "zero3"), i("d", 2, "zero3")) == "MZ_within_zero3"
    assert confeff.kind_of(i("d", 1, "zero1"), i("d", 2, "zero1")) == "MZ_within_zero1"
    assert confeff.kind_of(i("d", 1, "zero1"), i("d", 2, "zero3")) == "MZ_cross"
    assert confeff.kind_of(i("d", 1, "zero1"), i("a", 1, "zero1")) is None
    with pytest.raises(ValueError):
        confeff.kind_of(i("d", 1, "zero1"), i("d", 1, "zero1"))
    with pytest.raises(ValueError):
        confeff.CfgInfo("r", "d", "1", "zero2")


def test_pair_counts_and_contrasts_no_effect():
    ys, info = _models()
    res = confeff.analyze(ys, info)
    s = confeff.summary(res, info)
    assert [s[f"n_{k}"] for k in confeff.KINDS] == [12, 18, 18, 36]
    assert sum(1 for k in s if k.startswith("r_cell_")) == 12
    # same-cell pairs are far more alike than different-seed pairs; configuration adds ~nothing
    assert s["r_same_cell_cross_corrected"] > s["r_MZ_within_corrected"] + 0.05
    assert abs(s["config_effect_corrected"]) < 0.02
    assert s["r_MZ_within_corrected"] == pytest.approx(np.mean(
        [x["r_corrected"] for x in res["pairs"] if x["level"] in confeff.WITHIN]))
    # the MZ level on the zero3 models alone equals MZ_within_zero3
    z3 = {m: y for m, y in ys.items() if info[m].config == "zero3"}
    info3 = {m: reliability.ModelInfo(i.recipe, i.pretrain_seed, i.posttrain_seed)
             for m, i in info.items() if i.config == "zero3"}
    lv = reliability.twin_correlations(z3, info3)["levels"]["MZ"]
    assert s["r_MZ_within_zero3_corrected"] == pytest.approx(lv["r_corrected"], abs=1e-12)


def test_planted_configuration_effect_is_detected():
    ys, info = _models(scramble_a0=True)
    s = confeff.summary(confeff.analyze(ys, info), info)
    # the zero1 copies lost the cell latent: same-cell pairs fall to the different-seed level
    assert s["same_cell_minus_MZ_within_corrected"] < 0.02


def test_bootstrap_runs_with_the_kind_levels():
    ys, info = _models(n=120)
    ci = reliability.bootstrap_prompts(ys, info, n_boot=20, stat=confeff.stat_for(info),
                                       analyze=confeff.analyze)
    c = ci["config_effect_corrected"]
    assert c["lo"] <= c["estimate"] <= c["hi"] and c["n_draws"] == 20


def test_weight_table():
    rows = [{"a": "z1", "b": "z3", "kind": "same_cell_cross", "l2": 7.0},
            {"a": "z1b", "b": "z3b", "kind": "same_cell_cross", "l2": 8.5},
            {"a": "z3", "b": "z3b", "kind": "MZ_within_zero3", "l2": 34.0},
            {"a": "z1", "b": "z1b", "kind": "MZ_within_zero1", "l2": 36.0},
            {"a": "z1", "b": "z3b", "kind": "MZ_cross", "l2": 35.0}]
    w = confeff.weight_table(rows, floor=6.4)
    assert w["bound"] == pytest.approx(8.0)
    assert [r["within_bound"] for r in w["same_cell"]] == [True, False]
    assert w["n_within_bound"] == 1 and w["n_same_cell"] == 2
    assert w["ratio_min"] == pytest.approx(7.0 / 6.4) and w["ratio_max"] == pytest.approx(8.5 / 6.4)
    assert w["by_kind"]["MZ_cross"] == {"n": 1, "mean": 35.0, "min": 35.0, "max": 35.0}
    assert w["same_cell_over_MZ_within"] == pytest.approx(7.75 / 35.0)


def test_level_table():
    rates = {"s1": {"a": 0.50, "b": 0.52, "c": 0.40, "d": 0.45},
             "s2": {"a": 0.30, "b": 0.30, "c": 0.20, "d": 0.21}}
    lv = confeff.level_table(rates, [("a", "b", "c1"), ("c", "d", "c2")])
    assert lv["n_total"] == 4 and lv["n_within"] == 3
    assert lv["per_cell"]["c2"]["delta"]["s1"] == pytest.approx(-0.05)
    assert lv["max_abs_delta"] == pytest.approx(0.05)
    assert lv["by_source"]["s2"]["mean_delta"] == pytest.approx(-0.005)


def test_l2_flat_equals_024_l2():
    import importlib.util
    from pathlib import Path

    import torch

    spec = importlib.util.spec_from_file_location(
        "equiv_024", Path(__file__).parents[1] / "scripts/equiv_024.py")
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    g = torch.Generator().manual_seed(0)
    x = {"b": torch.randn(20, generator=g), "a.w": torch.randn(50, 20, generator=g).bfloat16()}
    y = {"a.w": torch.randn(50, 20, generator=g).bfloat16(), "b": torch.randn(20, generator=g)}
    fx, fy = confeff.flatten(x), confeff.flatten(y)
    assert fx[0] == ["a.w", "b"] and fx[1].dtype == torch.float32 and fx[1].numel() == 1020
    assert confeff.l2_flat(fx, fy) == pytest.approx(E.l2(x, y), rel=1e-5)
    assert confeff.l2_flat(fx, fx) == 0.0
    with pytest.raises(AssertionError):
        confeff.l2_flat(fx, confeff.flatten({"a.w": x["a.w"], "c": x["b"]}))


def test_l2_flat_is_exact_on_a_long_vector():
    """fp32 torch.linalg.vector_norm loses 1e-2 at 2^24 constant elements of 1e-3; the float64
    chunked sum must match the exact value to 1e-6, including across chunk boundaries."""
    import math

    import torch

    n = 1 << 24
    x = {"w": torch.full((n,), 1e-3, dtype=torch.float32)}
    z = {"w": torch.zeros(n, dtype=torch.float32)}
    exact = 1e-3 * math.sqrt(n)
    fx, fz = confeff.flatten(x), confeff.flatten(z)
    assert confeff.l2_flat(fx, fz) == pytest.approx(exact, rel=1e-6)
    assert confeff.l2_flat(fx, fz, chunk=1 << 20) == pytest.approx(exact, rel=1e-6)
    assert abs(float(torch.linalg.vector_norm(fx[1] - fz[1])) - exact) / exact > 1e-3  # the bug


def test_weight_rows_holds_two_vectors_at_a_time():
    """config_effect_027.weight_rows: one anchor + one partner, the partner reloaded per pair
    (n + n(n-1)/2 reads per base), every within-base pair once, L2 equal to equiv_024's l2."""
    import importlib.util
    from pathlib import Path

    import torch

    def load(name):
        spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / f"scripts/{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    C, E = load("config_effect_027"), load("equiv_024")
    g = torch.Generator().manual_seed(1)
    states = {f"m{i}": {"w": torch.randn(7, 3, generator=g).bfloat16(), "b": torch.randn(5, generator=g)}
              for i in range(4)}
    ms = {}
    for i, m in enumerate(states):
        cfg = confeff.ZERO1 if i % 2 else confeff.ZERO3
        ms[m] = {"info": confeff.CfgInfo("r", "default", str(1 + i // 2), cfg),
                 "dpo": Path("/nonexistent") / m}
    ms["x0"] = {"info": confeff.CfgInfo("r", "large-aux-2", "1", confeff.ZERO3), "dpo": Path("/nonexistent/x0")}
    states["x0"] = states["m0"]

    reads = []

    class FakeE:
        @staticmethod
        def load_state(path):
            reads.append(path.parent.name)
            return states[path.parent.name]

    rows = C.weight_rows(ms, "dpo", FakeE, log=lambda s: None)
    # 4 models of base default -> 6 pairs + 4 anchors = 10 reads; the lone aux-2 model -> 1 read
    assert len(rows) == 6 and len(reads) == 11
    assert {(r["a"], r["b"]) for r in rows} == {("m0", "m1"), ("m0", "m2"), ("m0", "m3"),
                                                ("m1", "m2"), ("m1", "m3"), ("m2", "m3")}
    kinds = {(r["a"], r["b"]): r["kind"] for r in rows}
    assert kinds[("m0", "m1")] == "same_cell_cross" and kinds[("m0", "m2")] == "MZ_within_zero3"
    assert kinds[("m1", "m3")] == "MZ_within_zero1" and kinds[("m0", "m3")] == "MZ_cross"
    for r in rows:
        assert r["l2"] == pytest.approx(E.l2(states[r["a"]], states[r["b"]]), rel=1e-5)
