import importlib.util
import os
from pathlib import Path

import pytest
import yaml

spec = importlib.util.spec_from_file_location(
    "weight_distance_004", Path(__file__).parents[1] / "scripts/weight_distance_004.py")
wd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wd)

BASE = ("allenai/OLMo-2-0425-1B", "f83cd641")


def _root(tmp_path, sft_rev="f83cd641", dpo2_parent="sft/seed2"):
    for stage, k in [(s, k) for s in ("sft", "dpo") for k in (1, 2)]:
        d = tmp_path / stage / f"seed{k}"
        (d / "model").mkdir(parents=True)
        cfg = ({"model_name_or_path": BASE[0], "model_revision": sft_rev} if stage == "sft" else
               {"model_name_or_path": str(tmp_path / (dpo2_parent if k == 2 else "sft/seed1")
                                          / "model")})
        (d / "oi_config.yaml").write_text(yaml.safe_dump(cfg))
    return str(tmp_path)


def test_check_parents_ok(tmp_path):
    got = wd.check_parents(_root(tmp_path), BASE)
    assert got["sft1"] == BASE
    assert got["dpo2"][0] == os.path.realpath(tmp_path / "sft/seed2/model")


def test_check_parents_wrong_base_revision(tmp_path):
    with pytest.raises(AssertionError, match="sft1"):
        wd.check_parents(_root(tmp_path, sft_rev="a1847dff"), BASE)


def test_check_parents_dpo_from_other_seed(tmp_path):
    with pytest.raises(AssertionError, match="dpo2"):
        wd.check_parents(_root(tmp_path, dpo2_parent="sft/seed1"), BASE)


def test_check_extra_local_and_hf_parent(tmp_path):
    root = _root(tmp_path)
    hf = wd.HF | {"base": BASE}
    run = tmp_path / "012" / "B"
    (run / "model").mkdir(parents=True)
    (run / "oi_config.yaml").write_text(yaml.safe_dump(
        {"model_name_or_path": str(tmp_path / "sft/seed1/model")}))
    key, path, parent, _ = wd.check_extra(f"dpoB={run}:sft1", root, hf)
    assert (key, parent, path) == ("dpoB", "sft1", os.path.realpath(run / "model"))
    with pytest.raises(AssertionError):
        wd.check_extra(f"dpoB={run}:sft2", root, hf)
    (run / "oi_config.yaml").write_text(yaml.safe_dump(
        {"model_name_or_path": hf["sft_ai2"][0], "model_revision": hf["sft_ai2"][1]}))
    assert wd.check_extra(f"dpoA={run}:sft_ai2", root, hf)[2] == "sft_ai2"
    with pytest.raises(AssertionError):
        wd.check_extra(f"dpo1={run}:sft_ai2", root, hf)
