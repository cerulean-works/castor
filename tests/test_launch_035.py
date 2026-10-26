import json
import sys

import pytest

from castor import sanityb
from castor.provenance import load_script

L35 = load_script("launch_035")
L6 = L35.L

REV = "replicate2-i6198-d6198"


def _manifest(tmp_path, **kw):
    m = {"prompt_id": "035", "sft_config": sanityb.SFT_CONFIG, "dpo_config": sanityb.DPO_CONFIG,
         "posttrain_seeds": list(sanityb.POSTTRAIN_SEEDS),
         "bases": [{"repo": "castor/DataDecide-dclm-baseline-1B", "revision": REV}]} | kw
    p = tmp_path / "grid_035.json"
    p.write_text(json.dumps(m))
    return p


@pytest.mark.parametrize("kw, msg", [
    # the variant lets grid.load pass (its guard, b7b7fc2), so the launcher's own check fires
    ({"sft_config": "configs/posttrain/other.yaml", "variant": "other"}, "006's post-training configs"),
    ({"posttrain_seeds": [1, 2]}, r"posttrain_seeds \(1, 2\) != 006's"),
    ({"prompt_id": "009"}, "manifest prompt_id 009 != 035"),
    ({"bases": [{"repo": "castor/DataDecide-dclm-baseline-1B", "revision": REV,
                 "posttrain_seeds": [1, 2]}]}, "base-level posttrain_seeds"),
    ({"bases": [{"repo": "castor/DataDecide-dclm-baseline-1B", "revision": REV,
                 "base_dir": "/elsewhere"}]}, "base_dir override"),
])
def test_cells_refuse_a_different_instrument(tmp_path, kw, msg):
    with pytest.raises(ValueError, match=msg):
        L35.cells(_manifest(tmp_path, **kw))


def test_four_cells_of_the_replicate(tmp_path):
    cs = L35.cells(_manifest(tmp_path))
    assert [c.k for c in cs] == [1, 2, 3, 4]
    assert {c.base_ref for c in cs} == {f"castor/DataDecide-dclm-baseline-1B@{REV}"}


def test_ratified_ds_config_sha():
    L35.check_ds()  # the committed file is the ratified one


def test_train_cmd_is_006_plus_zero1_on_sft_only(tmp_path, monkeypatch):
    monkeypatch.setattr(sanityb, "check_base", lambda c: {})
    c = L35.cells(_manifest(tmp_path))[0]
    cmd = L35.train_cmd(c, tmp_path)
    plain = L6.train_cmd(c, tmp_path, None)
    sft, dpo = cmd.split(" && ")
    psft, pdpo = plain.split(" && ")
    assert sft == psft + " --ds-config " + L35.SFT_DS
    assert dpo == pdpo  # DPO unchanged


def test_gj_cmd_is_guarded_and_names_prompt_035(tmp_path):
    c = L35.cells(_manifest(tmp_path))[0]
    guard, gj = L35.gj_cmd(c, tmp_path).split(" && ")
    assert "launch_035.py trained --only" in guard and c.tag in guard
    assert gj.endswith(f"{c.model_id} 035")


def test_trained_guard_exits_unless_done(tmp_path, monkeypatch):
    cs = L35.cells(_manifest(tmp_path))[:1]
    monkeypatch.setattr(L35, "cells", lambda: cs)
    monkeypatch.setattr(sys, "argv", ["launch_035.py", "trained"])
    monkeypatch.setattr(L6, "status", lambda c, root: "started")
    with pytest.raises(SystemExit, match="training not finished"):
        L35.main()
    monkeypatch.setattr(L6, "status", lambda c, root: "done")
    L35.main()


def test_submit_chains_gj_after_train(tmp_path, monkeypatch):
    cs = L35.cells(_manifest(tmp_path))[:1]
    monkeypatch.setattr(L35, "cells", lambda: cs)
    monkeypatch.setattr(sys, "argv", ["launch_035.py", "submit"])
    monkeypatch.setattr(L6, "queued", lambda: set())
    monkeypatch.setattr(L6, "status", lambda c, root: "todo")
    monkeypatch.setattr(L35, "train_cmd", lambda c, root: "TRAIN")
    calls = []
    monkeypatch.setattr(L35, "submit", lambda opts, cmd, dry: calls.append((opts, cmd)) or "123")
    L35.main()
    (topts, tcmd), (gopts, gcmd) = calls
    assert "launch_035.py trained" in gcmd
    assert tcmd == "TRAIN" and "035-train-" + cs[0].tag in topts
    assert gopts[-2:] == ["--after", "123"] and "035-gj-" + cs[0].tag in gopts
