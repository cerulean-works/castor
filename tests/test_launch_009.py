import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

from castor import grid, provenance, sanityb

spec = importlib.util.spec_from_file_location(
    "launch_009", pathlib.Path(__file__).parent.parent / "scripts" / "launch_009.py")
L9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L9)
L6 = L9.L


def _manifest(tmp_path, **kw):
    m = {"prompt_id": "009", "sft_config": sanityb.SFT_CONFIG, "dpo_config": sanityb.DPO_CONFIG,
         "posttrain_seeds": list(sanityb.POSTTRAIN_SEEDS),
         "bases": [{"repo": "castor/DataDecide-dclm-baseline-1B",
                    "revision": "init-default-data-aux2"},
                   {"repo": "castor/DataDecide-dclm-baseline-1B",
                    "revision": "init-aux2-data-default"}]} | kw
    p = tmp_path / "grid_009.json"
    p.write_text(json.dumps(m))
    return p


def test_real_manifest_is_006_instrument():
    cs = L9.cells()
    assert len(cs) == 8 and {c.k for c in cs} == {1, 2, 3, 4}
    assert {c.base_ref for c in cs} == {"castor/DataDecide-dclm-baseline-1B@init-default-data-aux2",
                                        "castor/DataDecide-dclm-baseline-1B@init-aux2-data-default"}


@pytest.mark.parametrize("kw, msg", [
    # the variant lets grid.load pass (its guard, b7b7fc2), so the launcher's own check fires
    ({"sft_config": "configs/posttrain/other.yaml", "variant": "other"}, "006's post-training configs"),
    ({"posttrain_seeds": [1, 2]}, None),
    ({"prompt_id": "008"}, None),
])
def test_cells_refuse_a_different_instrument(tmp_path, kw, msg):
    with pytest.raises(ValueError, match=msg):
        L9.cells(_manifest(tmp_path, **kw))


def test_adapter_matches_grid_cell(tmp_path):
    c = L9.cells(_manifest(tmp_path))[0]
    g = grid.load(_manifest(tmp_path)).cells()[0]
    assert (c.k, c.tag, c.model_id, c.base_ref) == (g.k, g.tag, g.model_id, g.base.ref)
    assert c.out("dpo", tmp_path) == g.out("dpo", tmp_path)


N = 3  # prompts in the fake panel


def _prov(path, **kw):
    path.mkdir(parents=True, exist_ok=True)
    (path / "provenance.json").write_text(json.dumps(kw))


def _trained(c, root, sft_sha=None):
    """Finished SFT and DPO provenance for cell `c` as train.sh writes it."""
    g = c.g
    _prov(c.out("sft", root), stage="sft", post_training_seed_k=c.k,
          base_model=str(g.base.base_dir), base_ref=g.base.ref, config=sanityb.SFT_CONFIG,
          config_sha256=sft_sha or grid._config_sha(sanityb.SFT_CONFIG), finished="t")
    ckpt = c.out("dpo", root) / "oi" / "ckpt"
    ckpt.mkdir(parents=True)
    _prov(c.out("dpo", root), stage="dpo", post_training_seed_k=c.k,
          base_model=str(c.out("sft", root) / "model"), config=sanityb.DPO_CONFIG,
          config_sha256=grid._config_sha(sanityb.DPO_CONFIG), checkpoint=str(ckpt), finished="t")
    return ckpt


def _generated(c, runs, ckpt, rows=N * (grid.K + 1)):
    out = c.g.gen_dir(runs)
    _prov(out, limit=None, model_resolved=str(pathlib.Path(ckpt).resolve()))
    (out / "judge_v1").mkdir()
    (out / "judge_v1" / "labels.jsonl").write_text("{}\n" * rows)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """One real cell, models and runs under tmp_path, a 3-prompt panel, submit recorded."""
    root, runs = tmp_path / "models", tmp_path / "runs"
    monkeypatch.setattr(sanityb, "MODELS_ROOT", root)
    monkeypatch.setattr(sanityb, "RUNS", runs)
    monkeypatch.setattr(L9.evalset, "load_eval", lambda: [{}] * N)
    cs = L9.cells(_manifest(tmp_path))[:1]
    monkeypatch.setattr(L9, "cells", lambda: cs)
    monkeypatch.setattr(L6, "queued", lambda: set())
    monkeypatch.setattr(L6, "train_cmd", lambda c, root, smoke: "TRAIN")
    heads = []
    monkeypatch.setattr(L9, "head", lambda: heads.append(1) or "c0ffee")
    calls = []
    monkeypatch.setattr(L6, "submit", lambda opts, cmd, dry, commit=None:
                        calls.append((opts, cmd, commit)) or str(100 + len(calls)))

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["launch_009.py", *argv])
        L9.main()
    return {"c": cs[0], "root": root, "runs": runs, "calls": calls, "heads": heads, "run": run}


def test_gj_cmd_is_guarded_and_names_prompt_009(tmp_path):
    c = L9.cells(_manifest(tmp_path))[0]
    cmd = L9.gj_cmd(c, tmp_path, "abc")
    guard, gj = cmd.split(" && ")
    assert "launch_009.py trained --only" in guard and c.tag in guard
    assert guard.endswith("--manifest-sha256 abc")
    assert gj.endswith(f"{c.model_id} 009")


def test_submit_chains_gj_after_train_on_one_commit(env):
    """One head() per invocation; both jobs of a cell carry the same pin."""
    env["run"]("submit")
    (t_opts, t_cmd, t_commit), (g_opts, g_cmd, g_commit) = env["calls"]
    assert t_cmd == "TRAIN" and t_opts[t_opts.index("--gpus") + 1] == "8"
    assert g_opts[g_opts.index("--after") + 1] == "101" and "gen_judge.sh" in g_cmd
    assert t_commit == g_commit == "c0ffee" and len(env["heads"]) == 1


def test_todo_training_with_old_generations_is_refused(env, capsys):
    """A cell retrained after its runs were moved aside still has generations from
    the old checkpoint; nothing is submitted over them."""
    _generated(env["c"], env["runs"], env["root"] / "old-ckpt")
    env["run"]("submit")
    assert env["calls"] == [] and "generations stale" in capsys.readouterr().out


def test_guard_refuses_stale_or_partial_generations(env):
    """In the job: training done, but the generation directory is from another
    checkpoint (stale) or holds a crashed judge (failed): gen_judge.sh would reuse it."""
    c = env["c"]
    _trained(c, env["root"])
    sha = L9.manifest_sha()
    env["run"]("trained", "--manifest-sha256", sha)  # todo: generates
    _generated(c, env["runs"], env["root"] / "old-ckpt")
    with pytest.raises(SystemExit, match="genjudge=stale"):
        env["run"]("trained", "--manifest-sha256", sha)


def test_guard_exits_unless_trained(env):
    with pytest.raises(SystemExit, match="train=todo"):
        env["run"]("trained", "--manifest-sha256", L9.manifest_sha())


def test_truncated_labels_are_not_done(env, capsys):
    """A judge that crashed mid-write leaves fewer label rows than N x (K + 1)."""
    c = env["c"]
    _generated(c, env["runs"], _trained(c, env["root"]), rows=5)
    assert L9.gj_st(c, env["root"], N) == "failed"
    env["run"]("submit")
    assert env["calls"] == [] and "generations failed" in capsys.readouterr().out
    (c.g.gen_dir(env["runs"]) / "judge_v1" / "labels.jsonl").write_text(
        "{}\n" * (N * (grid.K + 1)))
    assert L9.gj_st(c, env["root"], N) == "done"


def test_run_on_another_config_is_refused(env):
    """grid.stage_state's config sha256 check, which launch_006.status lacks."""
    _trained(env["c"], env["root"], sft_sha="0" * 64)
    with pytest.raises(AssertionError, match="SFT config changed"):
        env["run"]("submit")
    assert env["calls"] == []


def test_head_refuses_any_dirty_code_path(monkeypatch):
    """The pin is provenance.git_commit, so a dirty scripts/generate.py refuses the submission."""
    monkeypatch.setattr(provenance, "dirty_code", lambda root: ["scripts/generate.py"])
    with pytest.raises(RuntimeError, match="generate.py"):
        L9.head()


def test_manifest_is_cwd_independent_and_guarded(env, tmp_path, monkeypatch):
    """The manifest path is anchored at the repo; the in-job guard refuses a
    manifest that changed since submission (data/ is the live tree, not the snapshot)."""
    monkeypatch.chdir(tmp_path)
    assert L9.MANIFEST.is_absolute() and L9.MANIFEST.exists()
    _trained(env["c"], env["root"])
    with pytest.raises(SystemExit, match="changed since submission"):
        env["run"]("trained", "--manifest-sha256", "0" * 64)


def test_submit_failure_shows_submit_sh_reason(tmp_path, monkeypatch, capsys):
    """submit.sh's refusal reaches the user, and the error is not a bare
    CalledProcessError."""
    fake = tmp_path / "submit.sh"
    fake.write_text("#!/bin/bash\necho '19 of 20 jobs in Slurm' >&2\nexit 3\n")
    fake.chmod(0o755)
    monkeypatch.setattr(L6, "SUBMIT", fake)
    with pytest.raises(RuntimeError, match="exited 3: 19 of 20 jobs"):
        L6.submit(["--gpus", "1"], "echo hi", False, commit="abc")
    assert "19 of 20 jobs in Slurm" in capsys.readouterr().err


def test_gj_submit_failure_names_the_orphaned_training_job(env, monkeypatch, capsys):
    def fail_gj(opts, cmd, dry, commit=None):
        if "gen_judge.sh" in cmd:
            raise RuntimeError("submit.sh exited 3")
        return "777"
    monkeypatch.setattr(L6, "submit", fail_gj)
    with pytest.raises(RuntimeError):
        env["run"]("submit")
    assert "training job 777 is queued WITHOUT" in capsys.readouterr().err


def test_instrument_check_survives_python_O(tmp_path):
    """The 006-instrument checks are not asserts, so -O does not strip them."""
    m = _manifest(tmp_path, posttrain_seeds=[1, 2])
    code = (f"import importlib.util as u; s = u.spec_from_file_location('l9', {str(spec.origin)!r});"
            f" m = u.module_from_spec(s); s.loader.exec_module(m); m.cells(__import__('pathlib')"
            f".Path({str(m)!r}))")
    r = subprocess.run([sys.executable, "-O", "-c", code], check=False, capture_output=True, text=True,
                       cwd=pathlib.Path(__file__).parent.parent)
    assert r.returncode != 0 and "ValueError" in r.stderr and "posttrain_seeds" in r.stderr


def test_submit_skips_queued_and_unfinished(env, monkeypatch, capsys):
    c = env["c"]
    monkeypatch.setattr(L6, "queued", lambda: {f"009-gj-{c.tag}"})
    env["run"]("submit")
    assert "queued or running" in capsys.readouterr().out
    monkeypatch.setattr(L6, "queued", lambda: set())
    _prov(c.out("sft", env["root"]), stage="sft", post_training_seed_k=c.k,
          base_model=str(c.g.base.base_dir), base_ref=c.g.base.ref, config=sanityb.SFT_CONFIG,
          config_sha256=grid._config_sha(sanityb.SFT_CONFIG))  # started, never finished
    env["run"]("submit")
    assert env["calls"] == [] and "training failed-sft" in capsys.readouterr().out
