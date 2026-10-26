import json
import sys

import pytest

from castor import sanityb
from castor.provenance import load_script

L11 = load_script("launch_011")
L6 = L11.L

REV = "replicate-i6198-d6198"


def _manifest(tmp_path, **kw):
    m = {"prompt_id": "011", "sft_config": sanityb.SFT_CONFIG, "dpo_config": sanityb.DPO_CONFIG,
         "posttrain_seeds": list(sanityb.POSTTRAIN_SEEDS),
         "bases": [{"repo": "castor/DataDecide-dclm-baseline-1B", "revision": REV}]} | kw
    p = tmp_path / "grid_011.json"
    p.write_text(json.dumps(m))
    return p


@pytest.mark.parametrize("kw, msg", [
    # the variant lets grid.load pass (its guard, b7b7fc2), so the launcher's own check fires
    ({"sft_config": "configs/posttrain/other.yaml", "variant": "other"}, "006's post-training configs"),
    ({"posttrain_seeds": [1, 2]}, r"posttrain_seeds \(1, 2\) != 006's"),
    ({"prompt_id": "009"}, "manifest prompt_id 009 != 011"),
    ({"bases": [{"repo": "castor/DataDecide-dclm-baseline-1B", "revision": REV,
                 "posttrain_seeds": [1, 2]}]}, "base-level posttrain_seeds"),
    ({"bases": [{"repo": "castor/DataDecide-dclm-baseline-1B", "revision": REV,
                 "base_dir": "/elsewhere"}]}, "base_dir override"),
])
def test_cells_refuse_a_different_instrument(tmp_path, kw, msg):
    with pytest.raises(ValueError, match=msg):
        L11.cells(_manifest(tmp_path, **kw))


def test_four_cells_of_the_replicate(tmp_path):
    cs = L11.cells(_manifest(tmp_path))
    assert [c.k for c in cs] == [1, 2, 3, 4]
    assert {c.base_ref for c in cs} == {f"castor/DataDecide-dclm-baseline-1B@{REV}"}


def test_ratified_ds_config_sha():
    L11.check_ds()  # the committed file is the ratified one


def test_train_cmd_is_006_plus_zero1_on_sft_only(tmp_path, monkeypatch):
    monkeypatch.setattr(sanityb, "check_base", lambda c: {})
    c = L11.cells(_manifest(tmp_path))[0]
    cmd = L11.train_cmd(c, tmp_path)
    plain = L6.train_cmd(c, tmp_path, None)
    sft, dpo = cmd.split(" && ")
    psft, pdpo = plain.split(" && ")
    assert sft == psft + " --ds-config " + L11.SFT_DS
    assert dpo == pdpo  # DPO unchanged


def test_gj_cmd_is_guarded_and_names_prompt_011(tmp_path):
    c = L11.cells(_manifest(tmp_path))[0]
    guard, gj = L11.gj_cmd(c, tmp_path).split(" && ")
    assert "launch_011.py trained --only" in guard and c.tag in guard
    assert gj.endswith(f"{c.model_id} 011")


def test_trained_guard_exits_unless_done(tmp_path, monkeypatch):
    cs = L11.cells(_manifest(tmp_path))[:1]
    monkeypatch.setattr(L11, "cells", lambda: cs)
    monkeypatch.setattr(sys, "argv", ["launch_011.py", "trained"])
    monkeypatch.setattr(L6, "status", lambda c, root: "started")
    with pytest.raises(SystemExit, match="training not finished"):
        L11.main()
    monkeypatch.setattr(L6, "status", lambda c, root: "done")
    L11.main()


def test_submit_chains_gj_after_train(tmp_path, monkeypatch):
    cs = L11.cells(_manifest(tmp_path))[:1]
    monkeypatch.setattr(L11, "cells", lambda: cs)
    monkeypatch.setattr(sys, "argv", ["launch_011.py", "submit"])
    monkeypatch.setattr(L6, "queued", lambda: set())
    monkeypatch.setattr(L6, "status", lambda c, root: "todo")
    monkeypatch.setattr(L11, "train_cmd", lambda c, root: "TRAIN")
    calls = []
    monkeypatch.setattr(L11, "submit", lambda opts, cmd, dry: calls.append((opts, cmd)) or "123")
    L11.main()
    (topts, tcmd), (gopts, gcmd) = calls
    assert "launch_011.py trained" in gcmd
    assert tcmd == "TRAIN" and "011-train-" + cs[0].tag in topts
    assert gopts[-2:] == ["--after", "123"] and "011-gj-" + cs[0].tag in gopts
