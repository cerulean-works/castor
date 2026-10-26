import importlib.util
import json
import pathlib

import numpy as np
import pytest

from castor import factorial
from castor.factorial import Twin

spec = importlib.util.spec_from_file_location(
    "factorial_009", pathlib.Path(__file__).parent.parent / "scripts" / "factorial_009.py")
F9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F9)

SEEDS = {"D": (6198, 6198), "A": (4, 4), "X": (6198, 4), "Y": (4, 6198)}


def _twins(ks=(1, 2, 3, 4)):
    return {f"{b}{k}": Twin(b, i, d, k) for b, (i, d) in SEEDS.items() for k in ks}


def test_levels_and_counts():
    tw = _twins()
    assert factorial.level(tw["D1"], tw["D2"]) == "both"
    assert factorial.level(tw["D1"], tw["D1"]) is None
    assert factorial.level(tw["D1"], tw["X3"]) == "init"
    assert factorial.level(tw["A1"], tw["Y1"]) == "init"
    assert factorial.level(tw["D1"], tw["Y2"]) == "order"
    assert factorial.level(tw["A2"], tw["X2"]) == "order"
    assert factorial.level(tw["D1"], tw["A1"]) == "neither"
    assert factorial.level(tw["X1"], tw["Y4"]) == "neither"
    n = {lvl: 0 for lvl in factorial.LEVELS}
    for *_, lvl in factorial.pairs(tw):
        n[lvl] += 1
    assert n == {"both": 24, "init": 32, "order": 32, "neither": 32}


def test_same_init_and_data_across_bases_refused():
    with pytest.raises(AssertionError):
        factorial.level(Twin("D", 1, 2, 1), Twin("Z", 1, 2, 1))


def _sim(w_init, w_order, n=3000, K=20, seed=0):
    """Per-prompt logits = init effect + order effect + base-specific + model-specific parts."""
    rng = np.random.default_rng(seed)
    init = {s: rng.normal(size=n) for s in (6198, 4)}
    order = {s: rng.normal(size=n) for s in (6198, 4)}
    base = {b: rng.normal(size=n) for b in SEEDS}
    ys = {}
    for m, t in _twins().items():
        z = (w_init * init[t.init_seed] + w_order * order[t.data_seed] + 0.5 * base[t.base]
             + 0.3 * rng.normal(size=n))
        p = 1 / (1 + np.exp(-z))
        ys[m] = (rng.random((n, K)) < p[:, None]).astype(float)
    return ys


def test_init_dominant_reads_off_the_contrasts():
    tw = _twins()
    res = factorial.analyze(_sim(1.0, 0.2), tw)
    lv = res["levels"]
    assert lv["both"]["r_corrected"] > lv["init"]["r_corrected"] > lv["order"]["r_corrected"] \
        > lv["neither"]["r_corrected"]
    assert abs(lv["neither"]["r_corrected"]) < 0.1  # independent seeds share nothing here
    assert set(res["by_base_pair"]) == {"A", "D", "X", "Y", "AD", "AX", "AY", "DX", "DY", "XY"}
    assert res["by_base_pair"]["DX"]["level"] == "init"
    s = factorial.summary(res)
    assert s["init_minus_neither_corrected"] > s["order_minus_neither_corrected"] > 0


def test_order_dominant_flips():
    lv = factorial.analyze(_sim(0.2, 1.0, seed=1), _twins())["levels"]
    assert lv["order"]["r_corrected"] > lv["init"]["r_corrected"]


def test_bootstrap_ci_shape():
    tw = _twins(ks=(1, 2))
    ys = _sim(1.0, 0.5, n=400, seed=2)
    ys = {m: y for m, y in ys.items() if m in tw}
    ci = factorial.bootstrap_prompts(ys, tw, n_boot=30, seed=0)
    point = factorial.summary(factorial.analyze(ys, tw))
    assert set(ci) == set(point)
    for k in ["r_init_corrected", "init_minus_neither_corrected", "init_minus_order"]:
        assert ci[k]["estimate"] == point[k]
        assert ci[k]["lo"] <= ci[k]["hi"] and ci[k]["n_draws"] == 30


def test_pairs_argument_restricts():
    tw = _twins(ks=(1, 2))
    ys = {m: y for m, y in _sim(1.0, 0.5, n=200).items() if m in tw}
    res = factorial.analyze(ys, tw, pairs=[("D1", "X1"), ("D1", "D2")])
    assert [(x["a"], x["b"], x["level"]) for x in res["pairs"]] == [
        ("D1", "X1", "init"), ("D1", "D2", "both")]
    assert res["levels"]["order"]["n_pairs"] == 0


def test_hardware_aligned_with_levels():
    lv = factorial.analyze(_sim(1.0, 0.2, n=300), _twins())["levels"]
    assert {k: v["crosses_hardware"] for k, v in lv.items()} == {
        "both": False, "init": True, "order": True, "neither": False}


def test_n_pairs_defined_counts_undefined_corrections():
    tw = _twins(ks=(1, 2))
    ys = {m: y for m, y in _sim(1.0, 0.5, n=300).items() if m in tw}
    ys["X1"] = np.zeros_like(ys["X1"])  # constant -> NaN reliability, NaN corrected r
    lv = factorial.analyze(ys, tw)["levels"]
    assert lv["init"]["n_pairs"] == 8 and lv["init"]["n_pairs_defined"] == 6


def test_jackknife_bases_keeps_every_level():
    tw = _twins(ks=(1, 2))
    jk = factorial.jackknife_bases({m: y for m, y in _sim(1.0, 0.2, n=500).items() if m in tw}, tw)
    for lvl in factorial.LEVELS:
        assert jk[f"r_{lvl}_corrected"]["n_reps"] == 4
    assert jk["init_minus_order_corrected"]["se"] > 0
    assert not any(k.startswith("bp_") for k in jk)


def _registry(tmp_path, swap=False):
    entries = []
    for rev, (i, d) in {"init-default-data-aux2": (6198, 4),
                        "init-aux2-data-default": (4, 6198)}.items():
        ck = tmp_path / rev
        ck.mkdir()
        ci, cd = (d, i) if swap else (i, d)
        (ck / "config.yaml").write_text(f"seed: {ci}\ndata:\n  seed: {cd}\n")
        entries.append({"repo": "castor/DataDecide-dclm-baseline-1B", "revision": rev,
                        "init_seed": i, "data_seed": d, "source_checkpoint": str(ck)})
    p = tmp_path / "twins.json"
    p.write_text(json.dumps({"entries": entries}))
    return p


def test_models_maps_bases_to_letters_and_seeds(tmp_path):
    tw, runs, refs, sft = F9.models(F9.TWINS_MANIFEST.parent / "grid_009.json",
                                    _registry(tmp_path))
    assert len(tw) == 16 and set(runs) == set(refs) == set(sft) == set(tw)
    by = {(t.base, t.init_seed, t.data_seed) for t in tw.values()}
    assert by == {("D", 6198, 6198), ("A", 4, 4), ("X", 6198, 4), ("Y", 4, 6198)}
    for m, t in tw.items():
        rev = refs[m].split("@")[1]
        assert {"D": "seed-default", "A": "seed-large-aux-2", "X": "init-default-data-aux2",
                "Y": "init-aux2-data-default"}[t.base] in rev
        assert runs[m][1] == ("006" if t.base in "DA" else "009")


def test_models_refuses_config_seeds_that_disagree(tmp_path):
    with pytest.raises(AssertionError, match="OLMo config seeds"):
        F9.models(F9.TWINS_MANIFEST.parent / "grid_009.json", _registry(tmp_path, swap=True))


# --- fifth base ---
SEEDS5 = SEEDS | {"R": (6198, 6198)}


def _twins5(ks=(1, 2)):
    return {f"{b}{k}": Twin(b, i, d, k) for b, (i, d) in SEEDS5.items() for k in ks}


def _sim5(w_init, w_order, n=600, K=20, seed=0, ks=(1, 2)):
    rng = np.random.default_rng(seed)
    init = {s: rng.normal(size=n) for s in (6198, 4)}
    order = {s: rng.normal(size=n) for s in (6198, 4)}
    base = {b: rng.normal(size=n) for b in SEEDS5}
    ys = {}
    for m, t in _twins5(ks).items():
        z = (w_init * init[t.init_seed] + w_order * order[t.data_seed] + 0.5 * base[t.base]
             + 0.3 * rng.normal(size=n))
        ys[m] = (rng.random((n, K)) < 1 / (1 + np.exp(-z))[:, None]).astype(float)
    return ys


def test_level_fifth_and_hardware():
    tw = _twins5()
    lv = {(a, b): factorial.level_fifth(tw[a], tw[b]) for a, b in
          [("R1", "D1"), ("R1", "X2"), ("R1", "Y1"), ("R1", "A1"), ("R1", "R2"), ("X1", "Y1"),
           ("D1", "X1")]}
    assert list(lv.values()) == ["replicate", "R_init", "R_order", "R_neither", "R_both",
                                 "neither", "init"]
    assert factorial.level_fifth(tw["R1"], tw["R1"]) is None
    assert factorial.crosses_hardware(tw["R1"], tw["D1"])
    assert not factorial.crosses_hardware(tw["R1"], tw["X1"])
    assert not factorial.crosses_hardware(tw["R1"], tw["Y1"])
    with pytest.raises(AssertionError):  # the 4-base level still refuses R–D
        factorial.level(tw["R1"], tw["D1"])


def test_fifth_does_not_change_the_four_base_rows():
    ys, tw = _sim5(1.0, 0.3), _twins5()
    four = {m: t for m, t in tw.items() if t.base != "R"}
    a4 = factorial.analyze({m: ys[m] for m in four}, four)
    a5 = factorial.analyze_fifth(ys, tw)
    for bp, v in a4["by_base_pair"].items():
        assert a5["by_base_pair"][bp] == v
    assert set(a5["by_base_pair"]) - set(a4["by_base_pair"]) == {"R", "AR", "DR", "RX", "RY"}
    assert a5["by_base_pair"]["DR"]["level"] == "replicate"
    for lvl in factorial.LEVELS:  # no R pair is pooled into the 4-base levels
        assert a5["levels"][lvl] == a4["levels"][lvl]
    assert {k: a5["levels"][k]["crosses_hardware"] for k in factorial.FIFTH_LEVELS[4:]} == {
        "replicate": True, "R_both": False, "R_init": False, "R_order": False,
        "R_neither": True}


def test_summary_fifth_contrasts():
    ys, tw = _sim5(1.0, 0.2, seed=3), _twins5()
    res = factorial.analyze_fifth(ys, tw)
    s, bp = factorial.summary_fifth(res), res["by_base_pair"]
    assert s["hw_init_minus_order_corrected"] == bp["RX"]["r_corrected"] - bp["RY"]["r_corrected"]
    assert s["hw_init_minus_neither"] == bp["RX"]["r"] - bp["XY"]["r"]
    assert s["hw_init_minus_order_corrected"] > 0 and s["hw_init_minus_neither_corrected"] > 0
    assert not any(k in s for k in ("bp_DX", "bp_AD"))
    ci = factorial.bootstrap_prompts_fifth(ys, tw, n_boot=20)
    assert set(ci) == set(s)
    assert all(ci[k]["estimate"] == v or (np.isnan(v) and np.isnan(ci[k]["estimate"]))
               for k, v in s.items())  # restricting to fifth_pairs leaves the stat unchanged
    assert {frozenset((tw[a].base, tw[b].base)) for a, b in factorial.fifth_pairs(tw)} == {
        frozenset(x) for x in ("R", "RD", "RA", "RX", "RY", "XY")}


def _replicate_registry(tmp_path, seeds=(6198, 6198), data_seed_in_config: str | None = "null"):
    p = _registry(tmp_path)
    d = json.loads(p.read_text())
    ck = tmp_path / "rep"
    ck.mkdir()
    data = "{}" if data_seed_in_config is None else f"\n  seed: {data_seed_in_config}"
    ck.joinpath("config.yaml").write_text(f"seed: {seeds[0]}\ndata: {data}\n")
    d["entries"].append({"repo": "castor/DataDecide-dclm-baseline-1B",
                         "revision": "replicate-i6198-d6198", "init_seed": seeds[0],
                         "data_seed": seeds[1], "source_checkpoint": str(ck)})
    p.write_text(json.dumps(d))
    return p


def test_extra_models_reads_the_replicate(tmp_path):
    tw, runs, _, _ = F9.extra_models(F9.TWINS_MANIFEST.parent / "grid_011.json",
                                    _replicate_registry(tmp_path))
    assert {(t.base, t.init_seed, t.data_seed) for t in tw.values()} == {("R", 6198, 6198)}
    assert sorted(t.k for t in tw.values()) == [1, 2, 3, 4]
    assert {r[1] for r in runs.values()} == {"011"}


def test_extra_models_refuses_seeds_that_disagree(tmp_path):
    with pytest.raises(AssertionError, match="revision name"):
        F9.extra_models(F9.TWINS_MANIFEST.parent / "grid_011.json",
                        _replicate_registry(tmp_path, seeds=(4, 6198)))


def test_check_seeds_null_data_seed_falls_back_to_seed(tmp_path):
    F9.extra_models(F9.TWINS_MANIFEST.parent / "grid_011.json",
                    _replicate_registry(tmp_path, data_seed_in_config="6198"))
    (b := tmp_path / "b").mkdir()
    with pytest.raises(AssertionError, match="OLMo config seeds"):  # data.seed set and wrong
        F9.extra_models(F9.TWINS_MANIFEST.parent / "grid_011.json",
                        _replicate_registry(b, data_seed_in_config="4"))

    (c := tmp_path / "c").mkdir()  # data.seed absent altogether: also falls back to seed
    F9.extra_models(F9.TWINS_MANIFEST.parent / "grid_011.json",
                    _replicate_registry(c, data_seed_in_config=None))


def _prepare(monkeypatch, tmp_path, ds5="zero1", instrument="v1", missing=(), allow=True,
             collide=False):
    """prepare_fifth with extra_models, SFT provenance, the instrument and label rows stubbed."""
    tw5 = {f"r{k}": Twin("R", 6198, 6198, k) for k in (1, 2, 3, 4)}
    twins = {"r1" if collide else "d1": Twin("D", 6198, 6198, 1)}
    monkeypatch.setattr(F9, "extra_models", lambda g: (
        tw5, {m: (m, "011") for m in tw5}, {m: "ref" for m in tw5}, {m: m for m in tw5}))
    monkeypatch.setattr(F9, "sft_ds_config", lambda d: ds5)
    monkeypatch.setattr(F9.pipeval, "check_instrument", lambda runs, sha: instrument)
    monkeypatch.setattr(F9.pipeval, "load_rows", lambda run: run[0])
    monkeypatch.setattr(F9.sanityb, "outcome_matrix", lambda r, ids, f: (
        None, {"missing": ["p1"] if r in missing else []}))
    a = type("A", (), {"extra_grid": tmp_path / "grid_011.json", "allow_mixed_sft": allow})
    return F9.prepare_fifth(a, [{"id": "p1"}], "sha", "v1", twins, {}, {"d1": "zero3"})


def test_prepare_fifth_gates(monkeypatch, tmp_path):
    f = _prepare(monkeypatch, tmp_path)
    assert f["mixed"] and set(f["rows"]) == set(f["twins"])
    assert not _prepare(monkeypatch, tmp_path, ds5="zero3", allow=False)["mixed"]
    with pytest.raises(AssertionError, match="--allow-mixed-sft"):
        _prepare(monkeypatch, tmp_path, allow=False)
    with pytest.raises(AssertionError, match="different instrument"):
        _prepare(monkeypatch, tmp_path, instrument="v2")
    with pytest.raises(AssertionError, match="missing label cells"):
        _prepare(monkeypatch, tmp_path, missing=("r3",))
    with pytest.raises(AssertionError, match="collide"):
        _prepare(monkeypatch, tmp_path, collide=True)


def test_ds_describe_reads_the_zero_stage(monkeypatch, tmp_path):
    (oi := tmp_path / "oi" / "configs/ds_configs").mkdir(parents=True)
    (oi / "stage1.conf").write_text(json.dumps({"zero_optimization": {"stage": 1}}))
    monkeypatch.setattr(F9.posttrain, "OI_DIR", tmp_path / "oi")
    monkeypatch.setattr(F9.posttrain, "REPO_ROOT", tmp_path)
    sha = F9.posttrain.sha256_file(oi / "stage1.conf")
    assert F9.ds_describe(sha) == "ZeRO-1 (stage1.conf)"
    assert F9.ds_describe("configs/ds_configs/stage1.conf") == "ZeRO-1 (stage1.conf)"
    assert F9.ds_describe("0" * 64) == "0" * 64
