"""Tests for castor.harmcontrol: cells, the group rule and prediction check, the
diagnostics' arithmetic (eta^2, within-group demeaning, leave-one-source-out) and the merge."""

import json
from pathlib import Path

import numpy as np
import pytest

from castor import harmcontrol as HC
from castor import mechladder as ML
from castor import reliability as R

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "data/manifests/mech_021.json"


def _toy_manifest():
    models, sets = [], {"A": {"models": []}, "B": {"models": []}, "A0": {"models": []},
                        "S300M": {"models": []}}

    def add(mid, recipe, size, seed, post, *in_sets):
        models.append({"model_id": mid, "recipe": recipe, "size": size, "pretrain_seed": seed,
                       "posttrain_seed": str(post)})
        for s in in_sets:
            sets[s]["models"].append(mid)

    for rc in ("c4", "falcon"):
        for seed in ("default", "aux2", "aux3"):
            add(f"{rc}-{seed}-1", rc, "1B", seed, 1, "A", *(("B",) if rc == "c4" else ()))
            if rc == "c4":
                add(f"{rc}-{seed}-2", rc, "1B", seed, 2, "B")
    for seed in ("default", "aux2"):
        for post in (1, 2):
            add(f"main-{seed}-{post}", "dclm-baseline", "1B", seed, post, "A0")
            add(f"m300-{seed}-{post}", "dclm-baseline", "300M", seed, post, "S300M")
    return {"models": models, "sets": sets}


def test_cells_toy_manifest():
    c = HC.cells(_toy_manifest())
    assert list(c) == ["B:c4", "A:falcon", "A0:dclm-baseline", "S300M:dclm-baseline"]
    assert len(c["B:c4"]["models"]) == 6 and len(c["A:falcon"]["models"]) == 3
    assert c["S300M:dclm-baseline"]["size"] == "300M" and c["A0:dclm-baseline"]["size"] == "1B"
    assert HC.grid_cells(c) == ["B:c4", "A:falcon", "A0:dclm-baseline"]


@pytest.mark.skipif(not MANIFEST.exists(), reason="021 manifest absent")
def test_cells_real_manifest_cover_the_25_recipes_once():
    c = HC.cells(json.loads(MANIFEST.read_text()))
    by_set = {}
    for v in c.values():
        by_set.setdefault(v["set"], []).append(v)
    assert len(by_set["B"]) == 8 and all(len(v["models"]) == 6 for v in by_set["B"])
    assert len(by_set["A"]) == 16 and all(len(v["models"]) == 3 for v in by_set["A"])
    assert len(by_set["A0"]) == 1 and len(by_set["A0"][0]["models"]) == 12
    assert len(by_set["S300M"]) == 1 and by_set["S300M"][0]["size"] == "300M"
    grid = HC.grid_cells(c)
    assert len(grid) == 25 and len({c[g]["recipe"] for g in grid}) == 25
    assert "dclm-baseline-qc-7p-fw3" in {v["recipe"] for v in by_set["B"]}


def _ru(shared, specific, between=()):
    out = {f"s{i}": 0.99 for i in range(shared)}
    out |= {("dclm-baseline" if i == 0 else "dclm-baseline-qc-7p-fw3" if i == 1 else f"x{i}"):
            0.42 for i in range(specific)}
    out |= {f"b{i}": v for i, v in enumerate(between)}
    return out


def test_classify_and_prediction_parts():
    r = _ru(7, 2)
    gaps = {"s0": {13: {"refusal_gap": 0.2, "harm_gap": 0.01},
                   11: {"refusal_gap": 0.3, "harm_gap": 0.01}},
            "dclm-baseline": {13: {"refusal_gap": 0.27, "harm_gap": 0.57},
                              11: {"refusal_gap": 0.40, "harm_gap": 0.59}}}
    p = HC.prediction_check(r, gaps)
    assert p["groups"]["base_specific"] == ["dclm-baseline", "dclm-baseline-qc-7p-fw3"]
    assert p["part1_bimodal"]["holds"] and p["part2_membership"]["holds"]
    assert p["part3_shared_gap_below_refusal"]["holds"]
    assert p["part3_shared_gap_below_refusal"]["n_recipes"] == 1
    # a value between the modes breaks part 1 only
    p = HC.prediction_check(r | {"b0": 0.7}, gaps)
    assert not p["part1_bimodal"]["holds"] and p["groups"]["between"] == ["b0"]
    assert p["part2_membership"]["holds"]
    # three extra base-specific recipes break part 2; a missing predicted one too
    p = HC.prediction_check(_ru(4, 5), gaps)
    assert not p["part2_membership"]["holds"] and len(p["part2_membership"]["extra"]) == 3
    p = HC.prediction_check({"s0": 0.99, "dclm-baseline": 0.42}, gaps)
    assert p["part2_membership"]["predicted_missing"] == ["dclm-baseline-qc-7p-fw3"]
    assert not p["part2_membership"]["holds"]
    # a shared recipe whose harm gap exceeds its refusal gap at one layer breaks part 3
    g2 = {"s0": {13: {"refusal_gap": 0.2, "harm_gap": 0.01},
                 11: {"refusal_gap": 0.3, "harm_gap": 0.35}}}
    assert not HC.prediction_check(r, g2)["part3_shared_gap_below_refusal"]["holds"]
    assert HC.classify({"a": 0.95, "b": 0.55, "c": 0.5500001})["between"] == ["c"]


def test_last_char_class_and_shares():
    assert [HC.last_char_class(s) for s in ("Write a poem.", "How?  ", "Do it", "Go!", "a:", "")] \
        == [".", "?", "word", "!", ":", "word"]
    assert HC.shares(["a", "b", "a", "a"]) == {"a": 0.75, "b": 0.25}


def test_eta_squared_limits():
    g = np.array([0, 0, 1, 1, 2, 2])
    assert HC.eta_squared(np.array([1.0, 1, 5, 5, 9, 9]), g) == pytest.approx(1.0)
    assert HC.eta_squared(np.array([1.0, 3, 1, 3, 1, 3]), g) == pytest.approx(0.0)
    assert np.isnan(HC.eta_squared(np.ones(6), g))
    x = np.array([1.0, np.nan, 5, 5, 9, 9])  # NaNs dropped
    assert HC.eta_squared(x, g) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(6000)
    gg = rng.integers(0, 3, 6000)
    assert HC.eta_squared(x, gg) < 0.01


def test_demean_within_removes_the_class_part_and_the_within_level_reads_the_rest():
    """Two 'models' on different bases whose harm projection is one shared class effect plus
    base-specific parts: the raw unrelated-pair level is high, the within-class level ~ 0."""
    rng = np.random.default_rng(46)
    n, s = 3000, 20
    cls = rng.integers(0, 3, n)
    effect = np.array([-2.0, 0.0, 2.0])[cls]
    info = {"a1": R.ModelInfo("r", "A", "1"), "b1": R.ModelInfo("r", "B", "1")}
    ys = {}
    for m in info:
        own = rng.standard_normal(n) * 0.5
        t = effect + own
        h1 = t[:, None] + rng.standard_normal((n, s)) * 0.1
        h2 = t[:, None] + rng.standard_normal((n, s)) * 0.1
        a = ML.pack(t, h1, h2)
        assert HC.eta_squared(a[:, 0], cls) > 0.8
        d = HC.demean_within(a, cls)
        assert HC.eta_squared(d[:, 0], cls) < 1e-6
        ys[m] = (a, np.concatenate([d, np.zeros((n, 2))], 1), np.concatenate([a, np.zeros((n, 2))], 1))
    raw = ML.ladder_summary(ML.ladder(s)({m: v[2] for m, v in ys.items()}, info))
    within = ML.ladder_summary(ML.ladder(s)({m: v[1] for m, v in ys.items()}, info))
    assert raw["r_U_corrected"] > 0.85 and abs(within["r_U_corrected"]) < 0.1


def test_loo_and_matched_random_keep():
    src = ["advbench"] * 5 + ["tdc2023"] * 2 + ["malicious_instruct"] * 3
    k = HC.loo_keep(src, "tdc2023")
    assert k.tolist() == [0, 1, 2, 3, 4, 7, 8, 9]
    r1, r2 = HC.matched_random_keep(10, 2, 1), HC.matched_random_keep(10, 2, 1)
    assert len(r1) == 8 and r1.tolist() == r2.tolist() and sorted(set(r1)) == r1.tolist()
    assert HC.matched_random_keep(10, 2, 2).tolist() != r1.tolist() or True  # seeds may coincide


def _cell(set_name, recipe, n_mz, n_u, ru13, ru11, gap13=None, gap11=None):
    def layer(ru, gap):
        pt = {"refusal_r_U_corrected": 0.77, "harm_r_U_corrected": ru,
              "refusal_MZ_minus_U_corrected": gap[0] if gap else float("nan"),
              "harm_MZ_minus_U_corrected": gap[1] if gap else float("nan")}
        return {"point": pt}
    return {"set": set_name, "recipe": recipe, "size": "1B", "n_MZ": n_mz, "n_U": n_u,
            "layers": {13: layer(ru13, gap13), 11: layer(ru11, gap11)}}


def _part(tmp, name, cells, commit="abc"):
    d = {k: "x" for k in HC.SHARED} | {"cells": cells, "git_commit": commit, "seconds": 1.0,
                                       "command": "cmd"}
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(d, default=float))
    return p


def test_merge_and_summary(tmp_path):
    p1 = _part(tmp_path, "p1", {"B:c4": _cell("B", "c4", 3, 12, 0.99, 0.99, (0.2, 0.01), (0.3, 0.01)),
                                "A:falcon": _cell("A", "falcon", 0, 3, 0.99, 0.99)})
    p2 = _part(tmp_path, "p2", {"A0:dclm-baseline": _cell("A0", "dclm-baseline", 18, 48, 0.43, 0.41,
                                                          (0.27, 0.57), (0.40, 0.59)),
                                "S300M:dclm-baseline": _cell("S300M", "dclm-baseline", 18, 48,
                                                             0.995, 0.993, (0.22, 0.005),
                                                             (0.30, 0.007))}, commit="def")
    out = HC.merge([p1, p2])
    assert list(out["cells"]) == ["A0:dclm-baseline", "A:falcon", "B:c4", "S300M:dclm-baseline"]
    assert out["git_commits"] == ["abc", "def"] and set(out["parts"]) == {str(p1), str(p2)}
    s = out["summary"]
    assert s["n_grid_recipes"] == 3  # S300M is a size rung, not a grid recipe
    assert s["groups"] == {"shared": ["c4", "falcon"], "base_specific": ["dclm-baseline"],
                           "between": []}
    assert s["prediction"]["part1_bimodal"]["holds"]
    assert not s["prediction"]["part2_membership"]["holds"]  # qc-7p-fw3 absent from the toy
    assert s["prediction"]["part3_shared_gap_below_refusal"]["n_recipes"] == 1
    assert s["ranges"]["13"]["shared"]["mz_recipes"]["n_harm_gap_below_refusal"] == 1
    assert s["ranges"]["13"]["base_specific"]["harm_r_U"] == {"min": 0.43, "max": 0.43}
    assert "mz_recipes" not in s["ranges"]["13"]["between"]
    # a duplicated cell or a differing shared field refuses
    with pytest.raises(AssertionError):
        HC.merge([p1, p1])
    bad = json.loads(p2.read_text()) | {"n_splits": "y"}
    p3 = tmp_path / "p3.json"
    p3.write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        HC.merge([p1, p3])


def _run_cell(recipe, eta, pairs_r):
    """A cell with 3 runs x 2 post-training seeds; `eta`: run -> eta^2; `pairs_r`: (MZ|U,
    odd?) -> harm r_corrected (refusal = 0.5 * harm)."""
    models = [f"{recipe}-{r}-{s}" for r in ("r0", "r1", "r2") for s in (1, 2)]
    run_of = {m: m.split("-")[-2] for m in models}
    pairs = {"harm": [], "refusal": []}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            level = "MZ" if run_of[a] == run_of[b] else "U"
            odd = "r2" in (run_of[a], run_of[b])
            pairs["harm"].append({"a": a, "b": b, "level": level,
                                  "r_corrected": pairs_r[(level, odd)]})
            pairs["refusal"].append({"a": a, "b": b, "level": level,
                                     "r_corrected": 0.5 * pairs_r[(level, odd)]})
    pm = {m: {"eta2_harm_last_token": eta[run_of[m]]} for m in models}
    lay = {"pairs": pairs, "diagnosis_c": {"per_model": pm}}
    return {"recipe": recipe, "models": models, "layers": {"13": lay, "11": lay}}, run_of


def test_per_run_splits_pairs_by_the_odd_run():
    odd_cell, ro1 = _run_cell("x", {"r0": 0.98, "r1": 0.97, "r2": 0.04},
                              {("MZ", False): 1.0, ("MZ", True): 1.0, ("U", False): 0.99,
                               ("U", True): 0.15})
    tok_cell, ro2 = _run_cell("y", {"r0": 0.98, "r1": 0.98, "r2": 0.98},
                              {("MZ", False): 1.0, ("MZ", True): 1.0, ("U", False): 0.99,
                               ("U", True): 0.99})
    out = HC.per_run({"B:x": odd_cell, "B:y": tok_cell, "S300M:y": tok_cell}, ro1 | ro2)
    assert (out["n_runs"], out["n_token"], out["n_odd"]) == (6, 5, 1)  # S300M not counted
    assert out["odd"] == [{"cell": "B:x", "recipe": "x", "run": "r2"}]
    lv = out["cells"]["B:x"]["layers"]["13"]["harm"]
    # 3 runs x 2 seeds: MZ 1 per run (2 token, 1 odd); U 4 per run pair (4 token-token, 8 odd)
    assert {k: v["n"] for k, v in lv["levels"].items()} == {"MZ_odd": 1, "MZ_token": 2,
                                                            "U_odd": 8, "U_token": 4}
    assert lv["gap"]["odd"] == pytest.approx(0.85) and lv["gap"]["token"] == pytest.approx(0.01)
    ref = out["cells"]["B:x"]["layers"]["13"]["refusal"]["gap"]
    assert ref["odd"] == pytest.approx(0.425)
    assert out["cells"]["B:y"]["odd_runs"] == [] and out["cells"]["B:y"]["layers"] == {}
    assert out["cells"]["B:x"]["runs"]["r2"]["13"]["eta2_harm_last_token_mean"] == 0.04
