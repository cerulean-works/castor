"""summary_008: derived readings of grid.json."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "summary_008", Path(__file__).resolve().parent.parent / "scripts" / "summary_008.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def test_non_main_excludes_main_recipe():
    pr = {"dclm-baseline": {"MZ_minus_U_corrected": 0.9}, "a": {"MZ_minus_U_corrected": 0.1},
          "b": {"MZ_minus_U_corrected": 0.3}}
    res = m.non_main(pr)
    assert res["mean_non_main"] == pytest.approx(0.2) and res["n_non_main"] == 2
    assert res["main"] == 0.9


def test_recipe_vulnerability_skips_reference():
    models = {"x1": {"recipe": "a"}, "x2": {"recipe": "a"}, "y1": {"recipe": "b"}}
    pm = {"x1": {"mean_vulnerability": 0.2}, "x2": {"mean_vulnerability": 0.4},
          "y1": {"mean_vulnerability": 0.5}, "olmo2-1b-dpo-ai2": {"mean_vulnerability": 0.9}}
    assert m.recipe_vulnerability(pm, models) == {"a": pytest.approx(0.3), "b": 0.5}


def test_reference_main_block():
    models = {"x": {"recipe": "dclm-baseline"}, "y": {"recipe": "c4"}}
    pm = {"x": {"r": 0.6, "r_corrected": 0.7}, "y": {"r": 0.4, "r_corrected": 0.5}}
    res = m.reference(pm, models)
    assert res["main_recipe"] == {"n": 1, "r": 0.6, "r_corrected": 0.7}
    assert res["all"]["r_corrected"] == pytest.approx(0.6)
