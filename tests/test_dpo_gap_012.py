import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "dpo_gap_012", Path(__file__).parents[1] / "scripts/dpo_gap_012.py")
dg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dg)


def _r(s1, a, b, c, d, ref):
    return {"s1": s1, "A": a, "B": b, "C": c, "D": d, "ref": ref}


def test_gap_rule_per_source_band():
    # x: seed spread 0.10 (s1 vs C) lets A sit 0.08 below ref; y: spread 0.01 -> band 3 pt
    rates = {"x": _r(0.50, 0.52, 0.50, 0.40, 0.45, 0.60),
             "y": _r(0.70, 0.66, 0.70, 0.70, 0.71, 0.70)}
    out = dg.gap_rule(rates)
    x, y = out["sources"]["x"], out["sources"]["y"]
    assert x["W"] == pytest.approx(0.10) and x["within"]
    assert y["band"] == pytest.approx(0.03) and y["A_minus_ref"] == pytest.approx(-0.04)
    assert not y["within"] and out["reading"] == "b" and out["n_within"] == 1
    assert x["F"] == pytest.approx(0.0)


def test_gap_rule_reading_a_inside_band():
    out = dg.gap_rule({"x": _r(0.50, 0.475, 0.51, 0.50, 0.50, 0.50)})
    assert out["reading"] == "a" and out["sources"]["x"]["F"] == pytest.approx(0.01)


def test_band_is_not_global():
    # a noisy source must not widen the band of a quiet one
    rates = {"noisy": _r(0.3, 0.3, 0.3, 0.1, 0.3, 0.3), "quiet": _r(0.5, 0.45, 0.5, 0.5, 0.5, 0.5)}
    assert dg.gap_rule(rates)["reading"] == "b"


def _runs(tmp_path, seeds=None, a_parent=None):
    models = tmp_path / "models"
    sft = models / "sft"
    (sft / "model").mkdir(parents=True)
    runs = tmp_path / "runs"
    for k, (model, seed, parent) in dg.ARMS.items():
        mdir = models / k
        (mdir / "model").mkdir(parents=True)
        if isinstance(parent, tuple):
            base = a_parent or parent
        else:
            base = (str(sft / "model"), None)
        (mdir / "provenance.json").write_text(json.dumps(
            {"stage": "dpo", "oi_seed": (seeds or {}).get(k, seed), "base_model": base[0],
             "base_revision": base[1], "git_commit": "c"}))
        g = runs / model[0] / model[1]
        g.mkdir(parents=True)
        g.joinpath("provenance.json").write_text(json.dumps(
            {"model": str(mdir / "model"), "git_commit": "g"}))
    g = runs / dg.REF[0] / dg.REF[1]
    g.mkdir(parents=True)
    g.joinpath("provenance.json").write_text(json.dumps({"model": dg.REF_MODEL}))
    return runs, str(sft / "model")


def test_check_lineage(tmp_path, monkeypatch):
    runs, sft = _runs(tmp_path)
    monkeypatch.setattr(dg, "ARMS", {k: (m, s, sft if isinstance(p, str) else p)
                                     for k, (m, s, p) in dg.ARMS.items()})
    out = dg.check_lineage(runs)
    assert out["A"]["parent"] == dg.SFT_AI2 and out["C"]["oi_seed"] == 113


def test_check_lineage_refuses_wrong_seed_or_parent(tmp_path, monkeypatch):
    for i, kw in enumerate(({"seeds": {"C": 111}},
                            {"a_parent": ("allenai/OLMo-2-0425-1B-SFT", "main")})):
        d = tmp_path / str(i)
        runs, sft = _runs(d, **kw)
        monkeypatch.setattr(dg, "ARMS", {k: (m, s, sft if isinstance(p, str) else p)
                                         for k, (m, s, p) in dg.ARMS.items()})
        with pytest.raises(AssertionError):
            dg.check_lineage(runs)
        monkeypatch.undo()
