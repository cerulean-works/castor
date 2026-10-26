"""Two-site paired link, level rules, the pre-registered prediction rule, merge."""

import importlib.util
import json

import numpy as np
import pytest

from castor import factorial
from castor import mechladder as ML
from castor import reliability as R


def _twin_packs(rng, n, s, loadings, noise=0.3, bases="ABC", posts=(1, 2)):
    """Per model a readout = loading on a shared factor + base-specific + noise, returned as
    (full, h1, h2) tuples with ModelInfo."""
    info = {f"{b}{p}": R.ModelInfo("r", b, str(p)) for b in bases for p in posts}
    shared = rng.standard_normal(n)
    base = {b: rng.standard_normal(n) for b in bases}
    out = {}
    for m, i in info.items():
        t = loadings[0] * shared + loadings[1] * base[i.pretrain_seed]
        h1 = t[:, None] + rng.standard_normal((n, s)) * noise
        h2 = t[:, None] + rng.standard_normal((n, s)) * noise
        out[m] = (t + rng.standard_normal(n) * noise / np.sqrt(2), h1, h2)
    return out, info


def test_two_readouts_matches_separate_analyses_and_differences():
    rng = np.random.default_rng(21)
    n, s, k = 1500, 12, 6
    a, info = _twin_packs(rng, n, s, (1.0, 0.6))
    b, _ = _twin_packs(rng, n, s, (1.0, 0.2))
    y = {m: (rng.random((n, k)) < 0.5).astype(float) for m in info}
    ys2 = {m: ML.pack_two(a[m], b[m], y[m]) for m in info}
    an, st = ML.two_readouts(s, ML.link(s), ML.link_summary)
    out = st(an(ys2, info))
    sa = ML.link_summary(ML.link(s)({m: ML.pack(*a[m], y[m]) for m in info}, info))
    sb = ML.link_summary(ML.link(s)({m: ML.pack(*b[m], y[m]) for m in info}, info))
    assert out["site_link_U_corrected"] == sa["link_U_corrected"]
    assert out["compare_link_U_corrected"] == sb["link_U_corrected"]
    assert out["diff_link_U_corrected"] == pytest.approx(sa["link_U_corrected"]
                                                         - sb["link_U_corrected"])
    assert "diff_n_U" not in out and out["site_n_U"] == 12
    # identical readouts: every difference is exactly zero
    same = {m: ML.pack_two(a[m], a[m], y[m]) for m in info}
    z = st(an(same, info))
    assert z["diff_link_U_corrected"] == 0.0 and z["diff_link_MZ_raw"] == 0.0
    # the ladder through the same machinery
    an_g, st_g = ML.two_readouts(s, ML.ladder(s), ML.ladder_summary)
    g = st_g(an_g(ys2, info))
    lad = ML.ladder_summary(ML.ladder(s)({m: ML.pack(*a[m], y[m]) for m in info}, info))
    assert g["site_MZ_minus_U_corrected"] == lad["MZ_minus_U_corrected"]
    ci = R.bootstrap_prompts(ys2, info, n_boot=4, stat=st, analyze=an)
    assert ci["diff_link_U_corrected"]["n_draws"] == 4


def test_link_and_ladder_with_factorial_levels():
    rng = np.random.default_rng(5)
    n, s, k = 1200, 10, 4
    twins = {"D1": factorial.Twin("D", 6198, 6198, 1), "D2": factorial.Twin("D", 6198, 6198, 2),
             "X1": factorial.Twin("X", 6198, 4, 1), "Y1": factorial.Twin("Y", 4, 6198, 1),
             "R1": factorial.Twin("R", 6198, 6198, 1)}
    ys = {}
    for m in twins:
        t = rng.standard_normal(n)
        ys[m] = ML.pack(t, t[:, None] + rng.standard_normal((n, s)) * 0.3,
                        t[:, None] + rng.standard_normal((n, s)) * 0.3,
                        (rng.random((n, k)) < 0.5).astype(float))
    levels = factorial.FIFTH_LEVELS
    corrected = [x for x in levels if x != "both"]
    res = ML.link(s, factorial.level_fifth, corrected, ("both",))(ys, twins)
    got = {(r["a"], r["b"]): r["level"] for r in res["pairs"]}
    assert got[("D1", "D2")] == "both" and got[("D1", "R1")] == "replicate"
    assert got[("D1", "X1")] == "init" and got[("D1", "Y1")] == "order"
    assert got[("X1", "Y1")] == "neither"
    assert "link_corrected" not in next(r for r in res["pairs"] if r["level"] == "both")
    st = ML.link_summary(res)
    assert st["n_replicate"] == 2 and st["n_both"] == 1 and np.isfinite(st["link_init_corrected"])
    assert st["n_U"] == 0 and np.isnan(st["link_U_corrected"])  # the 007 keys stay present
    lad = ML.ladder_summary(ML.ladder(s, factorial.level_fifth, levels)(ys, twins))
    assert set(lad) >= {f"r_{lvl}_corrected" for lvl in levels} and "MZ_minus_U" not in lad


def test_ladder_summary_without_mz_pairs_and_site_point():
    rng = np.random.default_rng(3)
    n, s, k = 600, 8, 4
    a, info = _twin_packs(rng, n, s, (1.0, 0.5), posts=(1,))  # seed 1 only: no MZ pairs
    ys = {m: ML.pack(*a[m], (rng.random((n, k)) < 0.4).astype(float)) for m in info}
    lad = ML.ladder_summary(ML.ladder(s)(ys, info))
    assert np.isnan(lad["MZ_minus_U_corrected"]) and np.isfinite(lad["r_U_corrected"])
    pt = ML.site_point(ys, info, s)
    assert np.isnan(pt["MZ_minus_U_corrected"]) and np.isfinite(pt["link_U_corrected"])


def test_jackknife_groups_matches_jackknife_bases_when_groups_are_bases():
    rng = np.random.default_rng(9)
    n, s, k = 800, 8, 4
    a, info = _twin_packs(rng, n, s, (1.0, 0.5))
    ys = {m: ML.pack(*a[m], (rng.random((n, k)) < 0.4).astype(float)) for m in info}
    an, st = ML.link(s), ML.link_summary
    jb = R.jackknife_bases(ys, info, stat=st, analyze=an)
    jg = ML.jackknife_groups(ys, info, {m: info[m].pretrain_seed for m in ys}, st, an)
    assert jg["link_U_corrected"]["se"] == pytest.approx(jb["link_U_corrected"]["se"])


def test_prediction_021_rule():
    ok = {"estimate": 0.58, "lo": 0.52, "hi": 0.63}
    assert ML.prediction_021(ok, {"estimate": 0.2, "lo": 0.1, "hi": 0.3})["holds"]
    r = ML.prediction_021({"estimate": 0.52, "lo": 0.28, "hi": 0.7},
                          {"estimate": 0.2, "lo": 0.1, "hi": 0.3})
    assert r["part1"]["holds"] is False and r["part2"]["holds"] is True and not r["holds"]
    r = ML.prediction_021({"estimate": 0.49, "lo": 0.45, "hi": 0.53},
                          {"estimate": 0.2, "lo": 0.1, "hi": 0.3})
    assert r["part1"]["holds"] is False and r["reading_site"] == "partly"
    r = ML.prediction_021(ok, {"estimate": 0.05, "lo": -0.02, "hi": 0.1})
    assert r["part1"]["holds"] and r["part2"]["holds"] is False and not r["holds"]


def _load_021():
    spec = importlib.util.spec_from_file_location("ladder_021", "scripts/ladder_021.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_site_and_merge(tmp_path):
    LD = _load_021()
    s = LD.parse_site("13,-1")
    assert (s["layer"], s["pos_index"], s["stored"]) == (13, 6, 7)
    assert LD.parse_site("11,-2")["stored"] == 6
    with pytest.raises(AssertionError):
        LD.parse_site("11,-8")  # the instruction token is not a post-instruction site
    shared = {k: 1 for k in LD.SHARED}
    parts = []
    for name in ("A", "B"):
        for b in LD.BLOCKS:
            d = shared | {"set": name, "blocks": [b], "seconds": 1.0, "command": b}
            d |= {k: f"{name}-{b}" for k in LD.BLOCK_KEYS[b]}
            p = tmp_path / f"{name}-{b}.json"
            p.write_text(json.dumps(d))
            parts.append(p)
    LD.merge(parts, tmp_path / "m.json")
    m = json.loads((tmp_path / "m.json").read_text())
    assert sorted(m["sets"]) == ["A", "B"] and m["sets"]["A"]["link"] == "A-main"
    assert m["sets"]["B"]["site_map"] == "B-sites" and m["sets"]["A"]["blocks"] == ["main", "sites"]
    bad = json.loads(parts[1].read_text()) | {"git_commit": 2}
    parts[1].write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        LD.merge(parts, tmp_path / "m2.json")


def test_per_recipe_gaps_from_pair_rows():
    LD = _load_021()
    rows = {"refusal": {"pairs": [{"a": "r1-A1", "b": "r1-A2", "level": "MZ", "r_corrected": 0.9},
                                  {"a": "r1-A1", "b": "r1-B1", "level": "U", "r_corrected": 0.5},
                                  {"a": "r2-A1", "b": "r2-B1", "level": "U", "r_corrected": 0.4}]},
            "harm": {"pairs": [{"a": "r1-A1", "b": "r1-A2", "level": "MZ", "r_corrected": 1.0},
                               {"a": "r1-A1", "b": "r1-B1", "level": "U", "r_corrected": 0.7},
                               {"a": "r2-A1", "b": "r2-B1", "level": "U", "r_corrected": 0.6}]}}
    recipe = {m: m.split("-")[0] for m in ("r1-A1", "r1-A2", "r1-B1", "r2-A1", "r2-B1")}
    g = LD.per_recipe_gaps(rows, recipe)
    assert g["r1"]["refusal_MZ_minus_U_corrected"] == pytest.approx(0.4)
    assert g["r1"]["harm_MZ_minus_U_corrected"] == pytest.approx(0.3)
    assert g["r1"]["gap_refusal_minus_harm_corrected"] == pytest.approx(0.1)
    assert np.isnan(g["r2"]["refusal_MZ_minus_U_corrected"]) and g["r2"]["n_MZ"] == 0
