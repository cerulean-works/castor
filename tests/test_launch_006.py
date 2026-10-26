import json

import pytest

from castor import convert, sanityb
from castor.provenance import load_script

L = load_script("launch_006")

CELL = sanityb.Cell("default", 2)


def _write(path, d):
    path.mkdir(parents=True, exist_ok=True)
    (path / "provenance.json").write_text(json.dumps(d))


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(convert, "converted_dir", lambda repo, rev: tmp_path / "base" / rev)
    monkeypatch.setattr(sanityb, "RUNS", tmp_path / "runs")
    return tmp_path / "models"


def _sft(root, **kw):
    _write(CELL.out("sft", root), {"post_training_seed_k": 2, "base_model": str(CELL.base_dir),
                                   "base_ref": CELL.base_ref, "finished": "x"} | kw)


def _dpo(root, **kw):
    ckpt = CELL.out("dpo", root) / "oi" / "ck"
    ckpt.mkdir(parents=True)
    _write(CELL.out("dpo", root), {"post_training_seed_k": 2, "finished": "x",
                                   "base_model": str(CELL.out("sft", root) / "model"),
                                   "checkpoint": str(ckpt)} | kw)
    return ckpt


def test_status_reads_lineage_from_provenance(root):
    assert L.status(CELL, root) == "todo"
    _sft(root)
    assert L.status(CELL, root) == "started"
    _dpo(root)
    assert L.status(CELL, root) == "done"
    _sft(root, base_ref=None)  # an SFT with no recorded REPO@revision is not this cell's
    with pytest.raises(AssertionError, match="base_ref"):
        L.status(CELL, root)
    _sft(root, base_model="/elsewhere")
    with pytest.raises(AssertionError, match="base"):
        L.status(CELL, root)
    _sft(root, post_training_seed_k=3)
    with pytest.raises(AssertionError, match="seed"):
        L.status(CELL, root)


def test_genjudge_status_keys_on_checkpoint(root):
    _sft(root)
    ckpt = _dpo(root)
    out = sanityb.RUNS / CELL.model_id / sanityb.PROMPT_ID
    assert L.genjudge_status(CELL, root) == "todo"
    _write(out, {"limit": 5, "model_resolved": str(ckpt.resolve())})
    assert L.genjudge_status(CELL, root) == "todo"  # partial panel
    _write(out, {"limit": None, "model_resolved": str(ckpt.resolve())})
    assert L.genjudge_status(CELL, root) == "todo"  # not judged yet
    (out / "judge_v1").mkdir()
    (out / "judge_v1" / "labels.jsonl").write_text("")
    assert L.genjudge_status(CELL, root) == "done"
    _write(out, {"limit": None, "model_resolved": "/old/checkpoint"})
    assert L.genjudge_status(CELL, root) == "stale"
    _write(out, {"limit": None})  # generated before model_resolved was recorded
    assert L.genjudge_status(CELL, root) == "stale"


def test_train_cmd_records_base_and_smoke_zero_is_not_falsy(root, monkeypatch):
    monkeypatch.setattr(sanityb, "check_base", lambda c: {})
    cmd = L.train_cmd(CELL, root, None)
    assert f"--base-ref {CELL.base_ref}" in cmd and "max_train_steps" not in cmd
    assert "max_train_steps=50" in L.train_cmd(CELL, root, 50)


@pytest.mark.parametrize("argv, msg", [
    (["train", "--smoke", "0"], "positive"),
    (["train", "--dev"], "smoke only"),
    (["genjudge", "--smoke", "5"], "training only"),
])
def test_main_rejects_unsafe_flags(monkeypatch, argv, msg):
    monkeypatch.setattr("sys.argv", ["launch_006.py", *argv])
    monkeypatch.setattr(L, "queued", lambda: pytest.fail("checks must run before squeue"))
    with pytest.raises(AssertionError, match=msg):
        L.main()


def test_main_skips_queued_cells(root, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["launch_006.py", "train", "--only", CELL.tag])
    monkeypatch.setattr(L, "queued", lambda: {f"006-train-{CELL.tag}"})
    monkeypatch.setattr(L, "submit", lambda *a: pytest.fail("submitted a queued cell"))
    monkeypatch.setattr(sanityb, "MODELS_ROOT", root)
    L.main()
    assert "queued or running" in capsys.readouterr().out
