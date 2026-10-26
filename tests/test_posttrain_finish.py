import argparse
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "posttrain_run", Path(__file__).parents[1] / "scripts/posttrain_run.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)


def _setup(tmp_path, dirs):
    (tmp_path / "provenance.json").write_text(json.dumps(
        {"overrides": {"exp_name": "e"}, "oi_seed": 111}))
    for d, saved in dirs.items():
        (tmp_path / "oi" / d).mkdir(parents=True)
        if saved:
            (tmp_path / "oi" / d / "config.json").write_text("{}")
    return argparse.Namespace(out=tmp_path)


def test_finish_skips_empty_rank_dirs(tmp_path):
    # rank dirs one second apart; the main process saved into the earlier one
    a = _setup(tmp_path, {"e__111__149": True, "e__111__150": False})
    pr.finish(a)
    assert (tmp_path / "model").resolve() == (tmp_path / "oi" / "e__111__149").resolve()
    assert json.loads((tmp_path / "provenance.json").read_text())["checkpoint"].endswith("149")


def test_finish_refuses_two_saved_models(tmp_path):
    a = _setup(tmp_path, {"e__111__149": True, "e__111__150": True})
    with pytest.raises(AssertionError):
        pr.finish(a)


def test_parent_lineage_local_reads_run_provenance(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "provenance.json").write_text(json.dumps(
        {"stage": "sft", "oi_seed": 1, "git_commit": "c", "checkpoint": "x",
         "base_model": "allenai/OLMo-2-0425-1B", "base_revision": "f83c"}))
    p = pr.parent_lineage(str(tmp_path / "model"), None, None)
    assert (p["base_model"], p["base_revision"], p["stage"]) == ("allenai/OLMo-2-0425-1B", "f83c", "sft")
    with pytest.raises(AssertionError):
        pr.parent_lineage(str(tmp_path / "model"), None, "a@b")


def test_parent_lineage_hf_uses_flag():
    p = pr.parent_lineage("allenai/OLMo-2-0425-1B-SFT", "0d85", "allenai/OLMo-2-0425-1B@f83c")
    assert (p["revision"], p["base_model"], p["base_revision"]) == ("0d85", "allenai/OLMo-2-0425-1B", "f83c")
    assert pr.parent_lineage("allenai/x", "r", None)["base_revision"] is None
