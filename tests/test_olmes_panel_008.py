"""olmes_panel_008: the OLMES-10 join and the per-recipe / per-base means."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "olmes_panel_008", Path(__file__).resolve().parent.parent / "scripts" / "olmes_panel_008.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def test_olmes_table_reads_macro_row(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("data,task,default,large aux 2,large aux 3,mean,range,sd,n\n"
                 "C4,arc_easy,0.1,0.2,0.3,0.2,0.2,0.1,3\n"
                 "C4,olmes_10_macro_avg,0.5,0.6,0.7,0.6,0.2,0.1,3\n")
    assert m.olmes_table(p) == {("c4", "default"): 0.5, ("c4", "large-aux-2"): 0.6,
                                ("c4", "large-aux-3"): 0.7}


def test_panel_means_and_correlation():
    models, per_model, olmes = {}, {}, {}
    for i, r in enumerate(("a", "b", "c")):
        for j, s in enumerate(("default", "large-aux-2", "large-aux-3")):
            olmes[(r, s)] = 0.5 + 0.1 * i + 0.01 * j
            for k in (1, 2):
                mid = f"{r}-{s}-{k}"
                models[mid] = {"recipe": r, "pretrain_seed": s}
                per_model[mid] = {"mean_vulnerability": 0.1 * i + 0.01 * j + 0.001 * k}
    per_model["olmo2-1b-dpo-ai2"] = {"mean_vulnerability": 0.9}  # not a grid model: skipped
    res = m.panel(per_model, models, olmes)
    assert res["per_recipe"]["b"]["mean_vulnerability"] == pytest.approx(0.1 + 0.01 + 0.0015)
    assert res["per_base"]["c|large-aux-3"]["olmes10"] == pytest.approx(0.72)
    assert res["recipes"]["n"] == 3 and res["bases"]["n"] == 9
    assert res["recipes"]["pearson"] == pytest.approx(1.0)
    assert res["bases"]["spearman"] == pytest.approx(1.0)
