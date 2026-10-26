import importlib.util
import pathlib
import random

import numpy as np
import pytest

from castor import audit008 as A8
from castor import auditgate, gridstats
from castor.judge import labels as L

ROOT = pathlib.Path(__file__).parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RECIPES = sorted(gridstats.OLMES_NAMES.values())  # the grid's 25 recipes


def synth(prec=None, per_cell=16, seed=0, judge_pos=0.7, greedy=0.05):
    """015-format key rows + gold on the 008 design: recipe x source x pool x flag strata with
    design weights N_h / n_h; `prec(recipe, source)` = the true precision of the judge's
    vulnerable call. Weights differ by stratum, as in the real draw."""
    rng = random.Random(seed)
    prec = prec or (lambda r, s: {"wildjailbreak": 0.6, "harmbench_humanjb": 0.5}[s])
    items, n = [], 0
    for r in RECIPES:
        for s in A8.SOURCES:
            for pool, frac, N in (("disagree", 0.4, 5000), ("random", 0.6, 60000)):
                for flag in (False, True):
                    k = max(1, round(per_cell * frac / 2))
                    Nh = N * (3 if not flag else 1)
                    for _ in range(k):
                        jv = rng.random() < judge_pos
                        wg = rng.choice([L.HARMFUL, L.NONHARMFUL]) if jv else L.REFUSAL
                        if jv:
                            gold = L.HARMFUL if rng.random() < prec(r, s) else \
                                rng.choice([L.REFUSAL, L.REFUSAL, L.INCOHERENT])
                        else:
                            gold = L.REFUSAL if rng.random() < 0.95 else L.NONHARMFUL
                        items.append({
                            "item_id": f"{n:06x}", "model": r, "recipe": r, "source": s,
                            "pool": pool, "status": "disagree" if pool == "disagree" else "agree",
                            "flag": flag, "stratum": f"{r}|{s}|{pool}|{flag}",
                            "weight": Nh / k, "sample": -1 if rng.random() < greedy else 1,
                            "wg_label": wg, "gold": gold, "prompt_id": f"p{n}"})
                        n += 1
    return items


def test_groups_cover_the_grid():
    assert len(RECIPES) == 25
    g = {r: A8.group_of(r) for r in RECIPES}
    assert set(g.values()) == set(A8.GROUPS)
    assert g["dclm-baseline-25p-dolma1.7-75p"] == "DCLM"  # mixes count as DCLM
    assert sum(v == "Dolma" for v in g.values()) == 6
    assert sum(v == "DCLM" for v in g.values()) == 10
    assert sum(v == "Falcon" for v in g.values()) == 6
    with pytest.raises(ValueError):
        A8.group_of("pile")


def test_rates_equal_015s_definition():
    r015 = _load("audit_report_015")
    items = synth(seed=3)
    for s in A8.SOURCES:
        sel = [x for x in items if x["source"] == s]
        mine = A8.rates(sel)
        theirs = r015.rates([x for x in sel if x["sample"] >= 0])
        for k in ("judge_rate", "precision", "human_rate", "recall"):
            assert mine[k] == pytest.approx(theirs[k], abs=1e-15)


def test_greedy_items_are_left_out():
    items = synth(seed=1, greedy=0.3)
    pop = A8.population(items)
    assert all(x["sample"] >= 0 for x in pop) and len(pop) < len(items)


def test_known_per_source_precision_is_recovered():
    items = synth(per_cell=160, seed=2)
    st = A8.make_stat(recipes=RECIPES)(A8.population(items))
    assert st["source:wildjailbreak"] == pytest.approx(0.6, abs=0.03)
    assert st["source:harmbench_humanjb"] == pytest.approx(0.5, abs=0.03)
    for g in A8.GROUPS:
        assert st[f"group_source:{g}|wildjailbreak"] == pytest.approx(0.6, abs=0.12)


def test_components_and_correction():
    rng = np.random.default_rng(0)
    P = rng.uniform(size=50)
    P[[3, 17]] = np.nan
    src = np.array(["wildjailbreak"] * 20 + ["harmbench_humanjb"] * 30)
    c = A8.components(P, src)
    assert sum(c.values()) == pytest.approx(np.nanmean(P), abs=1e-15)
    f = {"wildjailbreak": 0.6, "harmbench_humanjb": 0.5}
    want = np.nanmean(P * np.array([f[s] for s in src]))
    assert A8.corrected(c, f) == pytest.approx(want, abs=1e-15)


def test_level_keys_in_stat():
    items = synth(seed=4)
    comp = {f"m{i}": {"wildjailbreak": 0.3, "harmbench_humanjb": 0.2} for i in range(4)}
    rof = {"m0": "c4", "m1": "c4", "m2": "falcon", "m3": "dolma1_7"}
    st = A8.make_stat(comp, rof, RECIPES)(A8.population(items))
    f = (st["source:wildjailbreak"], st["source:harmbench_humanjb"])
    assert st["level:per_source:model:m0"] == pytest.approx(0.3 * f[0] + 0.2 * f[1])
    g = (st["group_source:Falcon|wildjailbreak"], st["group_source:Falcon|harmbench_humanjb"])
    assert st["level:per_group_source:model:m2"] == pytest.approx(0.3 * g[0] + 0.2 * g[1])
    assert st["level:per_source:recipe:c4"] == pytest.approx(st["level:per_source:model:m0"])


def test_bootstrap_runs_on_the_design():
    items = A8.population(synth(seed=5))
    stat = A8.make_stat(recipes=RECIPES)
    b = auditgate.stratified_bootstrap(items, stat, 50, 0,
                                       collapse=lambda x: f"{x['model']}|{x['source']}")
    s = b["source:wildjailbreak"]
    assert s["lo"] <= s["point"] <= s["hi"]


def test_heterogeneity_detected():
    hi = set(RECIPES[::2])
    items = synth(prec=lambda r, s: 0.95 if r in hi else 0.15, seed=6)
    t = A8.Heterogeneity(A8.population(items), lambda x: x["model"], 2000, 0).test()
    assert t["p"] < A8.ALPHA_RULE


def test_null_p_is_roughly_uniform():
    ps = []
    for seed in range(60):
        items = synth(seed=100 + seed)
        ps.append(A8.Heterogeneity(A8.population(items), lambda x: x["model"], 199, seed)
                  .test()["p"])
    ps = np.array(ps)
    assert 0.35 < ps.mean() < 0.65
    assert (ps < 0.05).mean() <= 0.15


def test_permutation_keeps_block_counts():
    items = A8.population(synth(seed=7))
    het = A8.Heterogeneity(items, lambda x: x["model"], 20, 0)
    pos = [x for x in items if A8.judge_vulnerable(x)]
    blocks = np.array([f"{x['source']}|{x['pool']}" for x in pos])
    for b in set(blocks):
        sel = blocks == b
        for row in het.perm:
            assert sorted(row[sel]) == sorted(het.lab[sel])


def test_flips_to_cross():
    hi = set(RECIPES[:3])
    items = A8.population(synth(prec=lambda r, s: 0.98 if r in hi else 0.5, seed=8))
    het = A8.Heterogeneity(items, lambda x: x["model"], 500, 0)
    base = het.test()
    alt = {x["item_id"]: not A8.human_vulnerable(x["gold"]) for x in items
           if A8.judge_vulnerable(x)}
    r = A8.flips_to_cross(het, alt, alpha=base["p"] * 1.5 if base["p"] < 0.5 else 0.01,
                          max_flips=40)
    assert r["n_flips"] is not None and r["n_flips"] == len(r["path"])
    assert (r["path"][-1]["p"] < r["alpha"]) != (base["p"] < r["alpha"])
    none = A8.flips_to_cross(het, {})
    assert none["n_flips"] is None and none["path"] == []


def test_rejected_alternatives():
    R, H, N, I = L.REFUSAL, L.HARMFUL, L.NONHARMFUL, L.INCOHERENT
    a = {"x": {"label": R}, "y": {"label": H}, "z": {"label": H}, "w": {"label": R}}
    b = {"x": {"label": H}, "y": {"label": N}, "z": {"label": H}, "w": {"label": I}}
    adj = {"x": {"label": N}, "y": {"label": H}}
    assert A8.rejected_alternatives([a, b], adj) == {"x": False}


def test_check_complete_refuses():
    with pytest.raises(SystemExit):
        auditgate.check_complete({"a": {}, "b": {}}, [{"a": 1, "b": 1}, {"a": 1}], {"a": 1},
                          {"unresolved": 0})
    with pytest.raises(SystemExit):
        auditgate.check_complete({"a": {}}, [{"a": 1}, {"a": 1}], {}, {"unresolved": 1})
    auditgate.check_complete({"a": {}}, [{"a": 1}, {"a": 1}], {"a": 1}, {"unresolved": 0})


def test_model_factor():
    pt = {"source:wildjailbreak": 0.6, "source:harmbench_humanjb": 0.5,
          "group_source:C4|wildjailbreak": 0.7, "group_source:C4|harmbench_humanjb": 0.4}
    assert A8.model_factor("none", "c4", pt) == {"wildjailbreak": 1.0, "harmbench_humanjb": 1.0}
    assert A8.model_factor("per_source", "c4", pt)["wildjailbreak"] == 0.6
    assert A8.model_factor("per_group_source", "c4", pt)["harmbench_humanjb"] == 0.4


def test_contrast_none_is_the_grid_block_and_scaling_matches_components():
    m = _load("audit_correct_008")
    from castor import reliability
    rng = np.random.default_rng(0)
    n, k = 40, 20
    p = rng.uniform(0.05, 0.95, n)
    infos = [reliability.ModelInfo(r, b, str(j)) for r in ("c4", "falcon")
             for b in ("default", "large-aux-2") for j in (1, 2)]
    names = [f"{i.recipe}-{i.pretrain_seed}-{i.posttrain_seed}" for i in infos]
    ys = {nm: (rng.uniform(size=(n, k)) < p[:, None]).astype(float) for nm in names}
    ys[names[0]][0, :] = np.nan
    src = np.array(["wildjailbreak"] * 15 + ["harmbench_humanjb"] * 25)
    ones = {nm: np.ones(n) for nm in names}
    c = m.contrast(ys, names, infos, ones, 20, 0)
    P, Po, Pe = gridstats.halves(ys, names)
    lv = gridstats.levels(P, gridstats.reliabilities(Po, Pe), infos)
    assert c["levels"]["MZ_minus_U_corrected"] == lv["MZ_minus_U_corrected"]
    f = {"wildjailbreak": 0.6, "harmbench_humanjb": 0.5}
    fac = {nm: np.array([f[s] for s in src]) for nm in names}
    cs = m.contrast(ys, names, infos, fac, 5, 0)
    for nm in names:
        comp = A8.components(reliability.per_prompt_p(ys[nm]), src)
        assert cs["mean_vulnerability"][nm] == pytest.approx(A8.corrected(comp, f), abs=1e-12)


def test_disagree_reports_kappa():
    d = _load("audit_disagree")
    R, H = L.REFUSAL, L.HARMFUL
    a = {str(i): {"label": R if i < 6 else H, "partial": False} for i in range(10)}
    b = {str(i): {"label": R if i < 5 else H, "partial": False} for i in range(10)}
    res = d.compare(a, b)
    from castor import audit
    la = [a[i]["label"] for i in sorted(a)]
    lb = [b[i]["label"] for i in sorted(b)]
    assert res["kappa_4class"] == audit.cohen_kappa(la, lb)
    assert res["kappa_refusal"] == pytest.approx(0.8)


def test_flips_tie_break_moves_towards_crossing():
    """At the permutation floor every flip ties on p; the chosen flip must lower T (p must
    rise), not raise it."""
    hi = set(RECIPES[::2])
    items = A8.population(synth(prec=lambda r, s: 0.95 if r in hi else 0.15, seed=6))
    het = A8.Heterogeneity(items, lambda x: x["model"], 50, 0)
    base = het.test()
    assert base["p"] == 1 / 51
    alt = {x["item_id"]: not A8.human_vulnerable(x["gold"]) for x in items
           if A8.judge_vulnerable(x)}
    pos = {i: k for k, i in enumerate(het.ids)}
    ts = []
    for i, v in alt.items():
        h = het.h.copy()
        h[pos[i]] = float(v)
        r = het.test(h)
        ts.append((r["p"], r["T"]))
    floor = [t for p, t in ts if p == base["p"]]
    assert len(floor) > 1 and max(p for p, _ in ts) == base["p"]
    step = A8.flips_to_cross(het, alt, alpha=0.05, max_flips=1)["path"][0]
    assert step["T"] == min(floor) < base["T"]


def test_precomputed_T_equals_direct():
    items = A8.population(synth(seed=9))
    het = A8.Heterogeneity(items, lambda x: x["model"], 30, 0)
    h = het.h.copy()
    h[::3] = 1 - h[::3]
    w, lab = het.w, het.lab
    pbar = (w * h).sum() / w.sum()
    T = sum((lab == r).sum() * ((w * h)[lab == r].sum() / w[lab == r].sum() - pbar) ** 2
            for r in range(het.k) if (lab == r).any())
    assert het.test(h)["T"] == pytest.approx(T, rel=1e-12)


def test_one_flip_change_covers_group_source_factors():
    items = A8.population(synth(seed=10))
    alt = {x["item_id"]: not A8.human_vulnerable(x["gold"]) for x in items}
    out = A8.max_factor_change(items, alt)
    assert set(out) == set(A8.factor_keys()["per_source"] + A8.factor_keys()["per_group_source"])
    st = A8.make_stat(recipes=RECIPES)(items)
    for k, v in out.items():
        assert v["precision"] == pytest.approx(st[k], abs=1e-15)
    # a group x source factor rests on fewer items, so one flip moves it at least as much
    worst_src = max(out[k]["max_abs_change_one_flip"] for k in A8.factor_keys()["per_source"])
    worst_gs = max(out[k]["max_abs_change_one_flip"]
                   for k in A8.factor_keys()["per_group_source"])
    assert worst_gs >= worst_src > 0


def test_contrast_levels_from_halves_equal_contrast():
    m = _load("audit_correct_008")
    from castor import reliability
    rng = np.random.default_rng(1)
    n, k = 40, 20
    p = rng.uniform(0.05, 0.95, n)
    infos = [reliability.ModelInfo(r, b, str(j)) for r in ("c4", "falcon")
             for b in ("default", "large-aux-2") for j in (1, 2)]
    names = [f"{i.recipe}-{i.pretrain_seed}-{i.posttrain_seed}" for i in infos]
    ys = {nm: (rng.uniform(size=(n, k)) < p[:, None]).astype(float) for nm in names}
    ys[names[1]][2, :5] = np.nan
    src = np.array(["wildjailbreak"] * 15 + ["harmbench_humanjb"] * 25)
    point = {"group_source:C4|wildjailbreak": 0.7, "group_source:C4|harmbench_humanjb": 0.4,
             "group_source:Falcon|wildjailbreak": 0.5, "group_source:Falcon|harmbench_humanjb": 0.9}
    rof = {nm: i.recipe for nm, i in zip(names, infos)}
    F = m.factor_matrix("per_group_source", names, rof, src, point)
    fast = m.contrast_levels(gridstats.halves(ys, names), infos, gridstats.pair_masks(infos), F)
    slow = m.contrast(ys, names, infos, dict(zip(names, F)), 0, 0)["levels"]
    for key in m.CONTRAST_KEYS:
        assert fast[key] == pytest.approx(slow[key], abs=1e-12)


def test_same_is_nan_aware_and_exact():
    m = _load("audit_correct_008")
    nan = float("nan")
    assert m.same({"a": [1.0, nan]}, {"a": [1.0, nan]})
    assert not m.same({"a": 1.0}, {"a": 1.0 + 1e-15})
    assert not m.same({"a": 1.0}, {"b": 1.0})
    with pytest.raises(SystemExit):
        m.require(False, "x")


def test_bootstrap_over_all_items_keeps_strata():
    """The audit bootstrap resamples every gold item (domain estimation, as in
    audit_report_015): filtering first would shrink strata into singletons and change the
    groups."""
    items = synth(seed=11, greedy=0.3)
    st = A8.make_stat(recipes=RECIPES)
    assert st(items) == st(A8.population(items))
    g_all = auditgate.bootstrap_groups(items, auditgate.source_cell)
    g_pop = auditgate.bootstrap_groups(A8.population(items), auditgate.source_cell)
    assert sorted(map(len, g_all)) != sorted(map(len, g_pop))


def test_check_sheet_reports_missing_text(tmp_path):
    c = _load("audit_check_sheets")
    from castor import audit
    keyrows = {"a1": {"model_id": "m", "prompt_id": "p", "sample": 0},
               "a2": {"model_id": "m", "prompt_id": "q", "sample": 0}}
    audit.write_sheet(tmp_path / "s.csv", [("a1", "P", "R"), ("a2", "Q", "S")])
    text = {("m", "p", 0): ("P", "R")}
    r = c.check_sheet(str(tmp_path / "s.csv"), keyrows, text,
                      lambda x: tuple(x[k] for k in c.TEXT_KEY_008))
    assert r["text_missing"] == ["a2"] and r["text_changed"] == [] and not r["ok"]


def test_read_generations_joins_on_model_id(tmp_path):
    from castor import audit
    from castor.judge.common import write_jsonl
    write_jsonl(tmp_path / "a.jsonl", [{"prompt_id": "p", "sample": 0, "prompt": "x",
                                        "response": "1"}])
    write_jsonl(tmp_path / "b.jsonl", [{"prompt_id": "p", "sample": 0, "prompt": "x",
                                        "response": "2"}])
    got = audit.read_generations({"ma": [str(tmp_path / "a.jsonl")],
                                  "mb": [str(tmp_path / "b.jsonl")]}, {("mb", "p", 0)})
    assert list(got) == [("mb", "p", 0)] and got[("mb", "p", 0)]["response"] == "2"
