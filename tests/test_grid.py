import json

import pytest

from castor import convert, grid
from castor.provenance import load_script

MAN = {"prompt_id": "008", "sft_config": "configs/posttrain/sft_datadecide_eosbos.yaml",
       "dpo_config": "configs/posttrain/dpo_datadecide_eosbos.yaml", "posttrain_seeds": [1, 2],
       "bases": [{"repo": "allenai/DataDecide-dclm-baseline-qc-10p-1B",
                  "revision": "step69369-seed-large-aux-2"},
                 {"repo": "allenai/DataDecide-c4-1B", "revision": "step69369-seed-default"}]}


@pytest.fixture
def g(tmp_path, monkeypatch):
    p = tmp_path / "grid.json"
    p.write_text(json.dumps(MAN))
    monkeypatch.setattr(convert, "converted_dir", lambda repo, rev: tmp_path / "base" / rev)
    return grid.load(p)


def test_names(g):
    cells = g.cells()
    assert len(cells) == 4
    c = cells[1]
    assert c.base.recipe == "dclm-baseline-qc-10p" and c.base.size == "1B"
    assert c.base.pretrain_seed == "large-aux-2" and c.k == 2
    assert c.model_id == "dd-dclm-baseline-qc-10p-1b-large-aux-2-dpo-seed2"
    assert c.tag == "dclm-baseline-qc-10p-1B-large-aux-2-s2"
    assert c.job_name("train") == "008-train-dclm-baseline-qc-10p-1B-large-aux-2-s2"
    assert c.out("sft").parts[-3:] == ("dd-dclm-baseline-qc-10p-1B-seed-large-aux-2", "sft",
                                       "seed2")


def test_a_base_with_no_posttrain_seeds_has_no_cells(tmp_path, monkeypatch):
    """A base's own posttrain_seeds replace the grid's, an empty list included."""
    monkeypatch.setattr(convert, "converted_dir", lambda repo, rev: tmp_path / rev)
    p = tmp_path / "g.json"
    p.write_text(json.dumps(MAN | {"bases": [MAN["bases"][0] | {"posttrain_seeds": []},
                                             MAN["bases"][1] | {"posttrain_seeds": [3]}]}))
    assert [(c.base.recipe, c.k) for c in grid.load(p).cells()] == [("c4", 3)]


def test_duplicate_base_refused(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps(MAN | {"bases": MAN["bases"] + MAN["bases"][:1]}))
    with pytest.raises(AssertionError, match="duplicate"):
        grid.load(p)


def _write(path, d):
    path.mkdir(parents=True, exist_ok=True)
    (path / "provenance.json").write_text(json.dumps(d))


def _sft(c, root, **kw):
    _write(c.out("sft", root), {
        "stage": "sft", "post_training_seed_k": c.k, "base_model": str(c.base.base_dir),
        "base_ref": c.base.ref, "config": c.grid.sft_config, "slurm_job": "11",
        "config_sha256": grid._config_sha(c.grid.sft_config), "finished": "x"} | kw)


def _dpo(c, root, **kw):
    ckpt = c.out("dpo", root) / "oi" / "ck"
    ckpt.mkdir(parents=True, exist_ok=True)
    _write(c.out("dpo", root), {
        "stage": "dpo", "post_training_seed_k": c.k,
        "base_model": str(c.out("sft", root) / "model"), "config": c.grid.dpo_config,
        "config_sha256": grid._config_sha(c.grid.dpo_config), "checkpoint": str(ckpt),
        "slurm_job": "12", "finished": "x"} | kw)
    return ckpt


def test_train_states(g, tmp_path):
    c, root = g.cells()[0], tmp_path / "models"
    st = grid.stage_state(c, root)
    assert grid.train_state(st, False) == "todo"
    assert grid.train_state(st, True) == "in-queue"
    _sft(c, root, finished=None)
    prov = json.loads((c.out("sft", root) / "provenance.json").read_text())
    del prov["finished"]
    (c.out("sft", root) / "provenance.json").write_text(json.dumps(prov))
    assert grid.train_state(grid.stage_state(c, root), False) == "failed-sft"
    assert grid.train_state(grid.stage_state(c, root), True) == "in-queue"
    _sft(c, root)
    assert grid.train_state(grid.stage_state(c, root), False) == "failed-dpo"  # SFT only
    _dpo(c, root)
    assert grid.train_state(grid.stage_state(c, root), False) == "done"
    assert grid.train_state(grid.stage_state(c, root), True) == "done"


@pytest.mark.parametrize("field,value,match", [
    ("post_training_seed_k", 2, "seed"), ("base_model", "/elsewhere", "base"),
    ("base_ref", None, "base_ref"), ("config_sha256", "0" * 64, "config changed"),
    ("config", "configs/posttrain/sft_olmo2_1b.yaml", "config")])
def test_foreign_sft_provenance_is_an_error(g, tmp_path, field, value, match):
    c, root = g.cells()[0], tmp_path / "models"
    _sft(c, root, **{field: value})
    with pytest.raises(AssertionError, match=match):
        grid.stage_state(c, root)


def test_foreign_dpo_provenance_is_an_error(g, tmp_path):
    c, root = g.cells()[0], tmp_path / "models"
    _sft(c, root)
    _dpo(c, root, base_model="/other/sft/model")
    with pytest.raises(AssertionError, match="base differs"):
        grid.stage_state(c, root)


def test_genjudge_states(g, tmp_path):
    c, root, runs = g.cells()[0], tmp_path / "models", tmp_path / "runs"
    _sft(c, root)
    ckpt = _dpo(c, root)
    assert grid.genjudge_state(c, 2, False, root, runs) == "todo"
    assert grid.genjudge_state(c, 2, True, root, runs) == "in-queue"
    out = c.gen_dir(runs)
    out.mkdir(parents=True)
    assert grid.genjudge_state(c, 2, False, root, runs) == "failed"  # started, no provenance
    (out / "provenance.json").write_text(json.dumps({"limit": None,
                                                     "model_resolved": str(ckpt.resolve())}))
    assert grid.genjudge_state(c, 2, False, root, runs) == "failed"  # no labels
    (out / "judge_v1").mkdir()
    (out / "judge_v1" / "labels.jsonl").write_text("{}\n" * 41)
    assert grid.genjudge_state(c, 2, False, root, runs) == "failed"  # incomplete labels
    (out / "judge_v1" / "labels.jsonl").write_text("{}\n" * 42)  # 2 prompts × (20 + greedy)
    assert grid.genjudge_state(c, 2, False, root, runs) == "done"
    (out / "provenance.json").write_text(json.dumps({"limit": None, "model_resolved": "/x"}))
    assert grid.genjudge_state(c, 2, False, root, runs) == "stale"


def _log(steps, t0=0, dt=2.0, start=1):
    lines = []
    for i, n in enumerate(range(start, start + steps)):
        s = int(t0 + i * dt)
        lines.append(f"  x/y [..]09/29/2026 {s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"
                     f" - INFO - __main__ -   Step: {n}, LR: 1e-5, Loss: 1.0")
    return "\n".join(lines)


def test_step_rate():
    assert grid.step_rate("nothing") is None
    r = grid.step_rate(_log(150))
    assert r["s_per_step"] is None and r["step"] == 150
    r = grid.step_rate(_log(400, dt=2.0))
    assert r["s_per_step"] == pytest.approx(2.0) and not r["slow"]
    r = grid.step_rate(_log(400, dt=5.0))
    assert r["slow"] and r["stage_index"] == 1
    # a second stage (DPO) restarts at step 1: only the current stage counts
    r = grid.step_rate(_log(400, dt=9.0) + "\n" + _log(300, t0=4000, dt=1.5))
    assert r["stage_index"] == 2 and r["s_per_step"] == pytest.approx(1.5, abs=0.01)


# ---- re-entrant driver ----------------------------------------------------------------------


def test_genjudge_limit_is_its_own_run(g, tmp_path):
    c, root, runs = g.cells()[0], tmp_path / "models", tmp_path / "runs"
    _sft(c, root)
    ckpt = _dpo(c, root)
    out = c.gen_dir(runs, smoke=True)
    assert out.name == "008-smoke"
    out.mkdir(parents=True)
    (out / "provenance.json").write_text(json.dumps({"limit": 2,
                                                     "model_resolved": str(ckpt.resolve())}))
    (out / "judge_v1").mkdir()
    (out / "judge_v1" / "labels.jsonl").write_text("{}\n" * 42)
    cache = {}
    assert grid.genjudge_state(c, 2, False, root, runs, limit=2, cache=cache) == "done"
    assert len(cache) == 1
    # a limited generation never counts as the full panel's
    assert grid.genjudge_state(c, 2, False, root, runs) == "todo"


def test_move_aside_and_attempts(tmp_path):
    d = tmp_path / "seed1"
    assert grid.move_aside(d, "5") is None
    d.mkdir()
    assert grid.move_aside(d, "5").name == "seed1.failed-5"
    d.mkdir()
    assert grid.move_aside(d, "5").name == "seed1.failed-5-2"  # never overwrites
    assert grid.attempts(d) == ["5", "5-2"]
    assert not d.exists()


def test_max_jobs_schedule():
    from datetime import datetime
    ctl = {"max_jobs": [[None, 12], ["2026-10-02T16:00:00", 16]]}
    assert grid.max_jobs_now(ctl, datetime(2026, 10, 2, 15, 59))  # noqa: DTZ001 == 12
    assert grid.max_jobs_now(ctl, datetime(2026, 10, 2, 16, 0))  # noqa: DTZ001 == 16


def test_lock_excludes_and_takes_over_stale(tmp_path):
    import os
    import time
    with grid.lock(tmp_path), pytest.raises(TimeoutError), grid.lock(tmp_path, wait_s=0):
        pass
    assert not (tmp_path / "lock").exists()
    (tmp_path / "lock").mkdir()  # a dead holder's lock
    old = time.time() - 10_000
    os.utime(tmp_path / "lock", (old, old))
    with grid.lock(tmp_path, stale_s=60):
        pass
    assert list(tmp_path.glob("lock.stale-*"))


def test_plan_order_and_retry_cap(g):
    gs = _script()
    cells = g.cells()
    rows = {cells[0].tag: {"train": "todo", "genjudge": "waiting"},
            cells[1].tag: {"train": "failed-dpo", "genjudge": "waiting", "retries_used": 0},
            cells[2].tag: {"train": "failed-sft", "genjudge": "waiting", "retries_used": 1},
            cells[3].tag: {"train": "done", "genjudge": "todo"}}
    p = gs.plan(cells, rows, max_retries=1)
    assert [(k, c.tag, s) for k, c, s in p] == [
        ("gj", cells[3].tag, None), ("retry", cells[1].tag, "dpo"), ("train", cells[0].tag, "sft")]


def test_own_pretraining_runs_as_bases():
    b = grid.Base("castor/DataDecide-dclm-baseline-1B", "init-default-data-aux2")
    assert (b.recipe, b.size, b.pretrain_seed) == ("dclm-baseline", "1B", "init-default-data-aux2")
    assert b.group == "dd-dclm-baseline-1B-seed-init-default-data-aux2"
    g = grid.Grid("009", "s.yaml", "d.yaml", (1,), (b,))
    assert g.cells()[0].model_id == "dd-dclm-baseline-1b-init-default-data-aux2-dpo-seed1"
    r = grid.Base("allenai/DataDecide-dclm-baseline-1B", "step69369-seed-default")
    assert r.pretrain_seed == "default"
    with pytest.raises(TypeError):
        grid.Base("someone/DataDecide-x-1B", "step1-seed-a").recipe  # noqa: B018


# ---- driver hardening -------------------------------------------------------------------------


def _script():
    return load_script("grid")


def test_lock_taken_over_holder_leaves_successor_alone(tmp_path):
    import os
    import time
    with grid.lock(tmp_path, stale_s=60):
        old = time.time() - 10_000  # the holder looks dead: a waiter takes the lock over
        os.utime(tmp_path / "lock", (old, old))
        with grid.lock(tmp_path, stale_s=60):
            pass  # the successor released its own lock; now a third takes one
        (tmp_path / "lock").mkdir()
        (tmp_path / "lock" / "holder").write_text("othertoken node pid 1 job -\n")
    assert (tmp_path / "lock" / "holder").read_text().startswith("othertoken")  # not removed


def test_lock_beat_refreshes_age(tmp_path):
    import os
    import time
    with grid.lock(tmp_path) as beat:
        old = time.time() - 10_000
        os.utime(tmp_path / "lock", (old, old))
        beat()
        assert time.time() - (tmp_path / "lock").stat().st_mtime < 60


def test_by_name_counts_duplicates_in_list():
    jobs = [{"id": "5", "name": "bash", "state": "PENDING", "where": "(Priority)"},
            {"id": "6", "name": "bash", "state": "RUNNING", "where": "n1"},
            {"id": "7", "name": "bash", "state": "PENDING", "where": "(Priority)"}]
    assert grid.by_name(jobs) == {"bash": {"id": "6", "state": "RUNNING", "where": "n1"}}


def test_retry_ledger_and_atomic_write(tmp_path):
    assert grid.read_ledger(tmp_path) == {}
    a = grid.ledger_pending(tmp_path, "t|sft")
    b = grid.ledger_pending(tmp_path, "t|sft")
    assert grid.read_ledger(tmp_path) == {"t|sft": [a, b]} and a.startswith("pending-")
    grid.ledger_resolve(tmp_path, "t|sft", a, "101")
    grid.ledger_resolve(tmp_path, "t|sft", b, None)
    led = grid.read_ledger(tmp_path)
    assert led == {"t|sft": ["101"]}
    assert grid.retries_used(led, "t", "sft") == 0  # the first submission is not a retry
    assert grid.retries_used({"t|dpo": ["5"]}, "t", "dpo") == 1  # a DPO entry is always a retry
    assert grid.retries_used({"t|gj": ["5", "6"]}, "t", "gj") == 1
    assert grid.last_job({"t|sft": ["7", "pending-x"]}, "t|dpo", "t|sft") == "7"
    assert not list(tmp_path.glob(".*.tmp"))


def test_generation_complete(g, tmp_path):
    c, root, runs = g.cells()[0], tmp_path / "models", tmp_path / "runs"
    _sft(c, root)
    ckpt = _dpo(c, root)
    assert not grid.generation_complete(c, root, runs)
    _write(c.gen_dir(runs), {"limit": None, "model_resolved": str(ckpt.resolve())})
    assert grid.generation_complete(c, root, runs)
    assert not grid.generation_complete(c, root, runs, limit=40)
    _write(c.gen_dir(runs), {"limit": None, "model_resolved": "/x"})
    assert not grid.generation_complete(c, root, runs)


def test_wrap_traps_term_with_a_later_refill(monkeypatch):
    gs = _script()
    monkeypatch.setattr(grid, "main_repo", lambda: grid.ROOT)
    w = gs.wrap("a && b", grid.ROOT / "ctl.json", {"exclude": "n[1]", "commit": "abc"}, "008")
    assert "( a && b ) & wait $!; rc=$?; " in w and "trap - TERM" not in w  # trap kept to the end
    assert w.count(") & wait $!") == 3  # both refills in the background too: TERM acts at once
    assert w.startswith("trap 'trap \"\" TERM; ")  # a second TERM cannot submit twice
    assert "--name 008-refill" in w and "--commit abc" in w and "--after" in w
    assert "--use-reserve" in w and gs.TRAP_TAG in w


def _refill_setup(g, tmp_path, monkeypatch, gs, submit):
    c = g.cells()[0]
    root = grid.MODELS_ROOT
    d = tmp_path / "driver"
    d.mkdir()
    man = tmp_path / "grid.json"
    ctl = {"enabled": True, "manifest": str(man),
           "manifest_sha256": gs.posttrain.sha256_file(man), "commit": "abc",
           "max_jobs": [[None, 4]], "max_retries": 1, "exclude": "n[1]", "smoke": None,
           "gj_limit": None, "dev": False, "only": [c.tag]}
    sft, dpo = tmp_path / "sft", tmp_path / "dpo"
    dpo.mkdir()
    monkeypatch.setattr(grid.Cell, "out", lambda self, st, r=root: sft if st == "sft" else dpo)
    monkeypatch.setattr(grid, "squeue_list", list)
    monkeypatch.setattr(gs, "_train_cmd", lambda *a: "train")
    monkeypatch.setattr(gs, "survey", lambda *a, **k: [
        {"tag": c.tag, "train": "failed-dpo", "genjudge": "waiting",
         "failed_job": {"job": "77", "node": "None"},
         "retries_used": grid.retries_used(a[7], c.tag, "dpo")}])
    monkeypatch.setattr(gs, "submit", submit)
    return ctl, d, dpo, c


def test_refused_retry_restores_the_directory(g, tmp_path, monkeypatch):
    gs = _script()

    def refuse(*a):
        raise RuntimeError("submit failed (1): cap")
    ctl, d, dpo, _ = _refill_setup(g, tmp_path, monkeypatch, gs, refuse)
    gs.refill(ctl, d / "control.json")
    assert dpo.exists() and not list(tmp_path.glob("dpo.failed-*"))
    assert grid.read_ledger(d) == {}  # refused: the pending entry is removed


def test_submitted_retry_is_counted_even_without_a_directory(g, tmp_path, monkeypatch):
    gs = _script()
    ctl, d, dpo, c = _refill_setup(g, tmp_path, monkeypatch, gs,
                                   lambda *a: "Submitted batch job 900")
    dpo.rmdir()  # DPO died before writing anything: nothing to move aside
    gs.refill(ctl, d / "control.json")
    assert grid.read_ledger(d) == {f"{c.tag}|dpo": ["900"]}
    txt = gs.refill(ctl, d / "control.json")  # attempts used up: no second retry
    assert "planned 0" in txt


def test_beat_raises_once_taken_over(tmp_path):
    import os
    import time
    with grid.lock(tmp_path, stale_s=60) as beat:
        beat()
        old = time.time() - 10_000  # this holder stalled: a waiter takes the lock over
        os.utime(tmp_path / "lock", (old, old))
        with grid.lock(tmp_path, stale_s=60), pytest.raises(grid.LockLost):
            beat()  # the stalled holder stops instead of refilling beside its successor


def test_trap_refill_job_frees_its_own_slot(tmp_path, monkeypatch):
    """Run the wrapped command in bash, SIGTERM it, and follow the later refill job: the trap
    passes the dying job to --after, and the refill job passes its own id to --ended-job."""
    import os
    import signal
    import subprocess
    import time
    gs = _script()
    repo = tmp_path / "repo"
    (repo / "shell" / "nchc").mkdir(parents=True)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    submit = repo / "shell" / "nchc" / "submit.sh"  # records its options, runs the command as job 555
    submit.write_text('#!/bin/bash\nwhile [ "$1" != -- ]; do echo "$1" >> "$LOG.submit"; shift; '
                      'done; shift\nSLURM_JOB_ID=555 bash -c "$(printf \'%q \' "$@")"\n')
    (bin_ / "uv").write_text('#!/bin/bash\necho "$SLURM_JOB_ID: $*" >> "$LOG.uv"\n')
    for f in (submit, bin_ / "uv"):
        f.chmod(0o755)
    monkeypatch.setattr(grid, "main_repo", lambda: repo)
    w = gs.wrap("sleep 30", tmp_path / "ctl.json", {"exclude": "n[1]", "commit": "abc"}, "008")
    log = tmp_path / "log"
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "SLURM_JOB_ID": "444",
           "LOG": str(log)}
    p = subprocess.Popen(["bash", "-c", w], env=env, start_new_session=True)
    for _ in range(100):
        if (tmp_path / "log.uv").exists():
            break
        time.sleep(0.05)
    p.send_signal(signal.SIGTERM)
    time.sleep(0.2)
    p.send_signal(signal.SIGTERM)  # a second TERM while the trap runs: still one refill job
    assert p.wait(timeout=10) == 143
    os.killpg(p.pid, signal.SIGKILL)
    uv = (tmp_path / "log.uv").read_text().splitlines()
    assert uv[0].startswith("444: run --frozen python scripts/grid.py refill --control ")
    assert uv[-1].startswith("555: ") and uv[-1].endswith("--ended-job 555")
    opts = (tmp_path / "log.submit").read_text().split()
    assert opts[opts.index("--after") + 1] == "444" and "008-refill" in opts
    assert opts.count("--after") == 1


def test_budget_keeps_a_slot_for_the_trap(g, tmp_path, monkeypatch):
    gs = _script()
    ctl, d, _, _ = _refill_setup(g, tmp_path, monkeypatch, gs, lambda *a: "Submitted batch job 1")
    others = [{"id": str(i), "name": f"x{i}", "state": "RUNNING", "where": "n"}
              for i in range(gs.ACCOUNT_CAP - 1)]
    monkeypatch.setattr(grid, "squeue_list", lambda: others)
    assert "budget 0" in gs.refill(ctl, d / "control.json")


def test_retry_submitted_then_ledger_fails_keeps_directory_aside(g, tmp_path, monkeypatch):
    gs = _script()
    ctl, d, dpo, c = _refill_setup(g, tmp_path, monkeypatch, gs,
                                   lambda *a: "Submitted batch job 900")

    def broken(*a):
        raise OSError("disk full")
    monkeypatch.setattr(grid, "ledger_resolve", broken)
    txt = gs.refill(ctl, d / "control.json")
    assert not dpo.exists() and list(tmp_path.glob("dpo.failed-77"))  # the queued job owns dpo/
    assert "job submitted; its ledger entry stays pending" in txt
    led = grid.read_ledger(d)
    assert len(led[f"{c.tag}|dpo"]) == 1  # the pending entry still counts the attempt


# ---- staged launch and the grid budget ----------------------------------------------------------


def test_008_stages_partition_the_manifest():
    g = grid.load(grid.ROOT / "data/manifests/grid_008.json")
    stages = grid.load_stages("data/manifests/grid_008_stages.json", g)
    assert [s.name for s in stages] == ["A0", "A", "B", "C"]
    by = {}
    for c in g.cells():
        by.setdefault(grid.stage_of(c, stages), []).append(c)
    assert {k: len(v) for k, v in by.items()} == {"A0": 12, "A": 72, "B": 24, "C": 48}
    # A0: the 12 main-recipe rerun cells, 3 bases × seeds 1-4; no other stage has one
    assert {c.base.recipe for c in by["A0"]} == {"dclm-baseline"}
    assert sorted(c.k for c in by["A0"]) == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    assert "dclm-baseline" not in {c.base.recipe for c in by["A"] + by["B"] + by["C"]}
    assert {c.k for c in by["A"]} == {1} and {c.k for c in by["B"] + by["C"]} == {2}
    d035 = json.loads((grid.ROOT / "data/manifests/d035_recipes.json").read_text())["recipes"]
    assert {c.base.repo for c in by["B"]} == set(d035) and len(d035) == 8
    assert not {c.base.repo for c in by["C"]} & set(d035)
    assert len({c.base.recipe for c in by["C"]}) == 16


def test_stages_file_must_cover_every_cell(g, tmp_path, monkeypatch):
    monkeypatch.setattr(grid, "ROOT", tmp_path)
    (tmp_path / "s.json").write_text(json.dumps(
        {"prompt_id": "008", "stages": [{"name": "A", "seeds": [1], "repos": None}]}))
    with pytest.raises(AssertionError, match="in no stage"):
        grid.load_stages("s.json", g)
    (tmp_path / "s.json").write_text(json.dumps(
        {"prompt_id": "008", "stages": [{"name": "A", "seeds": [1], "repos": ["allenai/x-1B"]},
                                        {"name": "B", "seeds": [1, 2], "repos": None}]}))
    with pytest.raises(AssertionError, match="repos not in the grid"):
        grid.load_stages("s.json", g)
    assert grid.load_stages(None, g) == [grid.ALL_STAGE]
    assert grid.stage_of(g.cells()[0], [grid.ALL_STAGE]) == "all"


def test_order_cells_by_stage_and_enabled_only():
    gs = _script()
    g = grid.load(grid.ROOT / "data/manifests/grid_008.json")
    stages = grid.load_stages("data/manifests/grid_008_stages.json", g)
    cells = g.cells()  # manifest order interleaves seeds 1 and 2
    ab = gs.order_cells(cells, stages, ["A", "B"])
    assert [grid.stage_of(c, stages) for c in ab] == ["A"] * 72 + ["B"] * 24
    a_only = [c.tag for c in cells if c.k == 1 and c.base.recipe != "dclm-baseline"]
    assert [c.tag for c in ab[:72]] == a_only  # stable within
    assert gs.order_cells(cells, stages, ["C"])[0].k == 2
    assert len(gs.order_cells(cells, [grid.ALL_STAGE], None)) == 156
    # the main-recipe rerun cells are the first jobs, then stage A
    go = gs.order_cells(cells, stages, ["A0", "A", "B", "C"])
    assert [grid.stage_of(c, stages) for c in go[:13]] == ["A0"] * 12 + ["A"]
    # plan follows the order: stage A's new chains before stage B's
    rows = {c.tag: {"train": "todo", "genjudge": "waiting"} for c in cells}
    p = gs.plan(ab, rows, 1)
    assert [grid.stage_of(c, stages) for _, c, _ in p][:73] == ["A"] * 72 + ["B"]


def test_is_driver_job():
    for n in ("008-train-c4-1B-default-s1", "008-gj-c4-1B-default-s1", "008-refill",
              "008-smoke-train-c4-1B-default-s1", "008-smoke-gj-x-s2"):
        assert grid.is_driver_job(n, "008")
    for n in ("008-mock-analyze", "008-analyze", "0080-train-x", "009-train-x", "bash"):
        assert not grid.is_driver_job(n, "008")


def test_budget_counts_driver_jobs_only(g, tmp_path, monkeypatch):
    gs = _script()
    ctl, d, _, _ = _refill_setup(g, tmp_path, monkeypatch, gs, lambda *a: "Submitted batch job 1")
    jobs = [{"id": "1", "name": "008-mock-analyze", "state": "RUNNING", "where": "n"},
            {"id": "2", "name": "008-train-x-s1", "state": "PENDING", "where": "(Priority)"},
            {"id": "3", "name": "008-refill", "state": "PENDING", "where": "(Dependency)"}]
    monkeypatch.setattr(grid, "squeue_list", lambda: jobs)
    txt = gs.refill(ctl, d / "control.json")
    assert "3 jobs of this user, 2 of this grid" in txt and "budget 2" in txt  # max_jobs 4 − 2


def test_refill_submits_enabled_stages_only(g, tmp_path, monkeypatch):
    gs = _script()
    ctl, d, _, _ = _refill_setup(g, tmp_path, monkeypatch, gs, lambda *a: "Submitted batch job 1")
    sfile = tmp_path / "stages.json"
    sfile.write_text(json.dumps({"prompt_id": "008", "stages": [
        {"name": "A", "seeds": [1], "repos": None}, {"name": "B", "seeds": [2], "repos": None}]}))
    real_repo = grid.main_repo()
    monkeypatch.setattr(grid, "ROOT", tmp_path)  # stages file resolves under the fake root
    monkeypatch.setattr(grid, "main_repo", lambda: real_repo)
    ctl |= {"stages_file": "stages.json", "stages_sha256": gs.posttrain.sha256_file(sfile),
            "stages_enabled": ["B"], "only": None}
    rows = [{"tag": x.tag, "train": "todo", "genjudge": "waiting", "stage": "A" if x.k == 1
             else "B"} for x in g.cells()]
    monkeypatch.setattr(gs, "survey", lambda *a, **k: rows)
    txt = gs.refill(ctl, d / "control.json")
    subs = [x for x in txt.splitlines() if x.startswith("submitted train")]
    assert len(subs) == 2 and all(x.split()[2].endswith("-s2:") for x in subs)
    assert "stage A (disabled, 2 cells)" in txt and "stage B (enabled, 2 cells)" in txt
    assert gs.STOP_NOTE in txt
    sfile.write_text(sfile.read_text() + " ")  # the stages file changed after launch: refuse
    with pytest.raises(AssertionError, match="changed"):
        gs.refill(ctl, d / "control.json")


# ---- failed chains, the TERM trap and status --------------------------------------------------


def _survey_one(gs, g, monkeypatch, st, ledger, gj="todo"):
    c = g.cells()[0]
    monkeypatch.setattr(gs.evalset, "load_eval", lambda: [0] * 10)
    monkeypatch.setattr(grid, "stage_state", lambda *a: st)
    monkeypatch.setattr(grid, "genjudge_state", lambda *a, **k: gj)
    monkeypatch.setattr(grid, "generation_complete", lambda *a, **k: False)
    monkeypatch.setattr(grid, "sacct", lambda job_id=None, name=None: {"job": job_id or name,
                                                                        "node": "n7"})
    return c, gs.survey(g, [c], grid.MODELS_ROOT, False, {}, ledger=ledger)[0]


def test_job_dead_before_provenance_is_failed_not_todo(g, monkeypatch):
    """A submitted chain that left no provenance is failed (capped, its node excluded), never a
    fresh todo resubmitted without limit."""
    gs = _script()
    none = {"sft": None, "dpo": None, "jobs": {}}
    c, r = _survey_one(gs, g, monkeypatch, none, {})
    assert r["train"] == "todo"
    c, r = _survey_one(gs, g, monkeypatch, none, {f"{c.tag}|sft": ["5"]})
    assert r["train"] == "failed-sft" and r["retries_used"] == 0
    assert r["failed_job"] == {"job": "5", "node": "n7"}
    c, r = _survey_one(gs, g, monkeypatch, none, {f"{c.tag}|sft": ["5", "pending-x"]})
    assert r["retries_used"] == 1  # a refill killed after writing ahead still used the retry
    assert gs.plan([c], {c.tag: r}, max_retries=1) == []


def test_genjudge_dead_before_its_directory_is_failed(g, monkeypatch):
    gs = _script()
    done = {"sft": "finished", "dpo": "finished", "jobs": {}}
    c, r = _survey_one(gs, g, monkeypatch, done, {})
    assert r["genjudge"] == "todo"
    c, r = _survey_one(gs, g, monkeypatch, done, {f"{c.tag}|gj": ["8"]})
    assert r["genjudge"] == "failed" and r["retries_used"] == 0
    assert gs.plan([c], {c.tag: r}, max_retries=1) == [("gj-retry", c, None)]
    c, r = _survey_one(gs, g, monkeypatch, done, {f"{c.tag}|gj": ["8", "9"]})
    assert gs.plan([c], {c.tag: r}, max_retries=1) == []


def test_sft_retry_does_not_use_up_the_dpo_retry(g, monkeypatch):
    gs = _script()
    st = {"sft": "finished", "dpo": "started", "jobs": {"sft": "1", "dpo": "1"}}
    c, r = _survey_one(gs, g, monkeypatch, st, None)
    c, r = _survey_one(gs, g, monkeypatch, st, {f"{c.tag}|sft": ["1", "2"]})
    assert r["train"] == "failed-dpo" and r["retries_used"] == 0
    assert r["failed_job"]["job"] == "2"  # the newest chain, not the provenance's older job
    assert gs.plan([c], {c.tag: r}, max_retries=1) == [("retry", c, "dpo")]


def test_refill_killed_mid_retry_counts_the_attempt(g, tmp_path, monkeypatch):
    """SIGKILL between moving the directory aside and sbatch: the write-ahead entry stays, so
    the cell is not rerun uncounted."""
    gs = _script()

    def killed(*a):
        raise SystemExit(137)
    ctl, d, _, c = _refill_setup(g, tmp_path, monkeypatch, gs, killed)
    with pytest.raises(SystemExit):
        gs.refill(ctl, d / "control.json")
    led = grid.read_ledger(d)
    assert led[f"{c.tag}|dpo"][0].startswith("pending-")
    assert grid.retries_used(led, c.tag, "dpo") == 1


def test_pre_slurm_failure_does_not_use_the_budget(g, tmp_path, monkeypatch):
    gs = _script()
    sent = []
    ctl, d, _, _ = _refill_setup(g, tmp_path, monkeypatch, gs,
                                 lambda o, cmd, dry: sent.append(o) or "Submitted batch job 3")
    ctl |= {"only": None, "max_jobs": [[None, 1]]}
    cells = g.cells()
    monkeypatch.setattr(gs, "survey", lambda *a, **k: [
        {"tag": x.tag, "train": "todo", "genjudge": "waiting"} for x in cells])

    def train_cmd(c, *a):
        assert c.tag != cells[0].tag, "base converted by different code"
        return "train"
    monkeypatch.setattr(gs, "_train_cmd", train_cmd)
    txt = gs.refill(ctl, d / "control.json")
    assert f"SUBMIT-ERROR train {cells[0].tag}" in txt
    assert len(sent) == 1 and cells[1].job_name("train") in sent[0]
    assert grid.read_ledger(d) == {f"{cells[1].tag}|sft": ["3"]}  # the failed one left nothing


def test_refill_taken_over_writes_no_status(g, tmp_path, monkeypatch):
    gs = _script()
    ctl, d, _, _ = _refill_setup(g, tmp_path, monkeypatch, gs, lambda *a: "x")
    ctl["max_retries"] = 0  # nothing to submit: the only beat is before the status writes

    def lost():
        raise grid.LockLost("taken over")
    with pytest.raises(grid.LockLost):
        gs.refill(ctl, d / "control.json", beat=lost)
    assert not (d / "status.txt").exists()


def _run_wrap(tmp_path, monkeypatch, work, uv_body, submit_rc=0):
    """bash -c wrap(work) with fake uv and submit.sh; returns (popen, log prefix)."""
    import os
    import subprocess
    gs = _script()
    repo = tmp_path / "repo"
    (repo / "shell" / "nchc").mkdir(parents=True)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    submit = repo / "shell" / "nchc" / "submit.sh"
    submit.write_text(f'#!/bin/bash\necho "$*" >> "$LOG.submit"\nexit {submit_rc}\n')
    (bin_ / "uv").write_text(f'#!/bin/bash\necho "$SLURM_JOB_ID: $*" >> "$LOG.uv"\n{uv_body}\n')
    for f in (submit, bin_ / "uv"):
        f.chmod(0o755)
    monkeypatch.setattr(grid, "main_repo", lambda: repo)
    w = gs.wrap(work, tmp_path / "ctl.json", {"exclude": "n[1]", "commit": "abc"}, "008")
    log = tmp_path / "log"
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "SLURM_JOB_ID": "444",
           "LOG": str(log)}
    return subprocess.Popen(["bash", "-c", w], env=env, start_new_session=True), log


def _wait_for(path, n=1):
    import time
    for _ in range(200):
        if path.exists() and len(path.read_text().splitlines()) >= n:
            return
        time.sleep(0.05)
    raise AssertionError(f"{path} never had {n} lines")


def test_term_during_a_blocked_start_refill_acts_at_once(tmp_path, monkeypatch):
    """A start refill waiting on the lock (here: sleeping) does not defer the trap past
    KillWait: TERM submits the later refill job within seconds."""
    import os
    import signal
    import time
    p, _ = _run_wrap(tmp_path, monkeypatch, "true", "sleep 30")
    _wait_for(tmp_path / "log.uv")
    t0 = time.time()
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=10) == 143 and time.time() - t0 < 5
    os.killpg(p.pid, signal.SIGKILL)
    assert "--after 444" in (tmp_path / "log.submit").read_text()


def test_term_during_the_end_refill_still_submits_a_refill(tmp_path, monkeypatch):
    """The trap stays set through the end refill; the job exits with
    the work's rc, and a refused refill job is logged for `status`."""
    import os
    import signal
    p, _ = _run_wrap(tmp_path, monkeypatch, "exit 3",
                       '[[ "$*" == *--ended-job* ]] && sleep 30', submit_rc=1)
    _wait_for(tmp_path / "log.uv", 2)  # the end refill is running
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=10) == 3
    os.killpg(p.pid, signal.SIGKILL)
    assert "--use-reserve" in (tmp_path / "log.submit").read_text()
    gs = _script()
    traps = gs.trap_failures(tmp_path)
    assert len(traps) == 1 and "job 444" in traps[0]
    txt = gs.format_status([], ["h"], traps=traps)
    assert "TRAP REFILL REFUSED" in txt


def test_status_before_launch_and_after_a_stages_edit(g, tmp_path, monkeypatch):
    gs = _script()
    rows = [{"tag": "x", "train": "todo", "genjudge": "waiting", "stage": "A"}]
    assert "stage A (not launched, 1 cells)" in gs.format_status(rows, ["h"], None,
                                                                 launched=False)
    sfile = tmp_path / "stages.json"
    sfile.write_text(json.dumps({"prompt_id": "008", "stages": [
        {"name": "A", "seeds": [1, 2], "repos": None}]}))
    monkeypatch.setattr(grid, "ROOT", tmp_path)
    ctl = {"stages_file": "stages.json", "stages_sha256": "old", "stages_enabled": ["A"]}
    with pytest.raises(AssertionError, match="changed"):
        gs._stages(ctl, g)
    stages, enabled = gs._stages(ctl, g, check=False)  # the read-only status still reads it
    assert enabled == ["A"] and [s.name for s in stages] == ["A"]


def test_submit_sh_keeps_the_last_slot(tmp_path):
    import os
    import subprocess
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "squeue").write_text("#!/bin/bash\nfor i in $(seq 19); do echo $i; done\n")
    (bin_ / "squeue").chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}"}
    r = subprocess.run(["bash", str(grid.ROOT / "shell/nchc/submit.sh"), "--", "true"],
                       env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 3 and "--use-reserve" in r.stderr


# ---- cron on the login node ---------------------------------------------------------------------


@pytest.fixture
def fake_crontab(tmp_path, monkeypatch):
    """A `crontab` on PATH that keeps the table in a file (-l reads it, - writes stdin)."""
    import os
    bin_ = tmp_path / "cronbin"
    bin_.mkdir()
    tab = tmp_path / "crontab.txt"
    (bin_ / "crontab").write_text(
        f'#!/bin/bash\nT={tab}\nif [ "$1" = -l ]; then [ -f $T ] || '
        '{ echo "no crontab for x" >&2; exit 1; }; cat $T; else cat > $T; fi\n')
    (bin_ / "crontab").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    return tab


def test_cron_install_is_idempotent_and_keeps_other_lines(fake_crontab, tmp_path):
    fake_crontab.write_text("0 1 * * * echo other\n")
    tag = grid.cron_tag("008", False)
    line = grid.cron_line("008", False, tmp_path / "control.json", tmp_path, "/x/uv",
                          "/x:/usr/bin:/bin")
    assert line.startswith("*/30 * * * * /bin/bash -c 'export PATH=/x:/usr/bin:/bin && cd ")
    assert line.endswith(tag)
    assert f"/x/uv run --frozen python scripts/grid.py refill --control {tmp_path}" in line
    grid.install_cron(line, tag)
    grid.install_cron(line, tag)  # a second launch replaces, never duplicates
    assert fake_crontab.read_text().splitlines() == ["0 1 * * * echo other", line]
    smoke = grid.cron_tag("008", True)
    grid.install_cron("* * * * * s " + smoke, smoke)  # the smoke grid's line is separate
    assert grid.remove_cron(tag) and not grid.remove_cron(tag)
    assert fake_crontab.read_text().splitlines() == ["0 1 * * * echo other", "* * * * * s " + smoke]


def test_cron_on_records_host_and_stop_removes(g, fake_crontab, tmp_path, monkeypatch):
    gs = _script()
    ctl, d, _, _ = _refill_setup(g, tmp_path, monkeypatch, gs, lambda *a: "x")
    monkeypatch.setattr(grid, "main_repo", lambda: tmp_path)
    gs._cron_on(ctl, d / "control.json")
    assert ctl["cron"]["host"] == gs._host() and ctl["cron"]["tag"] == "# castor-grid 008"
    assert fake_crontab.read_text().count("castor-grid 008") == 1
    monkeypatch.setattr(gs, "_host", lambda: "elsewhere")  # stop on another login node: kept
    gs._cron_off(ctl, d, "stop")
    assert "castor-grid 008" in fake_crontab.read_text()
    monkeypatch.setattr(gs, "_host", lambda: ctl["cron"]["host"])
    gs._cron_off(ctl, d, "stop")
    assert fake_crontab.read_text() == ""
    assert "cron removed" in (d / "driver.log").read_text()
    monkeypatch.setenv("SLURM_JOB_ID", "5")  # never from inside a job
    ctl["cron"] = None
    gs._cron_on(ctl, d / "control.json")
    assert ctl["cron"] is None and fake_crontab.read_text() == ""


def test_refill_removes_cron_when_labelled_or_disabled(g, fake_crontab, tmp_path, monkeypatch):
    gs = _script()
    ctl, d, _, c = _refill_setup(g, tmp_path, monkeypatch, gs, lambda *a: "x")
    monkeypatch.setattr(grid, "main_repo", lambda: tmp_path)
    gs._cron_on(ctl, d / "control.json")
    gs.refill(ctl, d / "control.json")  # the cell still has work: the line stays
    assert "castor-grid 008" in fake_crontab.read_text()
    monkeypatch.setattr(gs, "survey", lambda *a, **k: [
        {"tag": c.tag, "train": "done", "genjudge": "done"}])
    monkeypatch.setenv("SLURM_JOB_ID", "9")  # a job's refill on another node cannot remove it
    monkeypatch.setattr(gs, "_host", lambda: "hgpn001")
    gs.refill(ctl, d / "control.json")
    assert "castor-grid 008" in fake_crontab.read_text()
    monkeypatch.setattr(gs, "_host", lambda: ctl["cron"]["host"])
    gs.refill(ctl, d / "control.json")  # the cron's own refill on the login node
    assert fake_crontab.read_text() == ""
    gs._cron_on(ctl, d / "control.json")
    monkeypatch.setattr(gs, "survey", lambda *a, **k: [
        {"tag": c.tag, "train": "todo", "genjudge": "waiting"}])
    gs.refill(ctl | {"enabled": False}, d / "control.json")  # stopped elsewhere: removed here
    assert fake_crontab.read_text() == ""


# ---- relaunch, slot counting, exclude list and trap bookkeeping -------------------------------


def test_relaunch_keeps_track_of_an_earlier_cron_line(g, fake_crontab, tmp_path, monkeypatch):
    """A relaunch with --no-cron removes the earlier line; on another host it refuses
    instead of writing a control without the line's record."""
    gs = _script()
    ctl, d, _, _ = _refill_setup(g, tmp_path, monkeypatch, gs, lambda *a: "x")
    monkeypatch.setattr(grid, "main_repo", lambda: tmp_path)
    gs._cron_on(ctl, d / "control.json")
    old = dict(ctl)
    monkeypatch.setattr(gs, "_host", lambda: "lgn-other")
    with pytest.raises(AssertionError, match="cron line is on"):
        gs._relaunch_cron(dict(old), dict(ctl, cron=None), d / "control.json", no_cron=True)
    assert "castor-grid 008" in fake_crontab.read_text()
    monkeypatch.setattr(gs, "_host", lambda: old["cron"]["host"])
    new = dict(ctl, cron=None)
    gs._relaunch_cron(dict(old), new, d / "control.json", no_cron=False)  # replaced, not doubled
    assert fake_crontab.read_text().count("castor-grid 008") == 1 and new["cron"]
    new2 = dict(ctl, cron=None)
    gs._relaunch_cron(dict(new), new2, d / "control.json", no_cron=True)
    assert fake_crontab.read_text() == "" and new2["cron"] is None


def test_budget_counts_a_submission_whose_ledger_update_failed(g, tmp_path, monkeypatch):
    """A job that reached Slurm counts against the budget even if ledger_resolve
    raised afterwards."""
    gs = _script()
    sent = []
    ctl, d, _, _ = _refill_setup(g, tmp_path, monkeypatch, gs,
                                 lambda o, cmd, dry: sent.append(o) or "Submitted batch job 3")
    ctl |= {"only": None, "max_jobs": [[None, 1]]}
    monkeypatch.setattr(gs, "survey", lambda *a, **k: [
        {"tag": x.tag, "train": "todo", "genjudge": "waiting"} for x in g.cells()])

    def broken(*a):
        raise OSError("disk full")
    monkeypatch.setattr(grid, "ledger_resolve", broken)
    txt = gs.refill(ctl, d / "control.json")
    assert len(sent) == 1 and "budget 1; planned 4; submitted 1" in txt


def test_refill_passes_the_ending_job_to_submit(g, tmp_path, monkeypatch):
    gs = _script()
    sent = []
    ctl, d, _, _ = _refill_setup(g, tmp_path, monkeypatch, gs,
                                 lambda o, cmd, dry: sent.append(o) or "Submitted batch job 3")
    gs.refill(ctl, d / "control.json", ended_job="444")
    assert sent[0][sent[0].index("--ending-job") + 1] == "444"
    gs.refill(ctl | {"max_retries": 2}, d / "control.json")
    assert "--ending-job" not in sent[1]


def _submit_sh(tmp_path, squeue_ids, *opts):
    import os
    import subprocess
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    (bin_ / "squeue").write_text("#!/bin/bash\n" + "".join(f"echo {i}\n" for i in squeue_ids))
    (bin_ / "sbatch").write_text("#!/bin/bash\ncat > /dev/null; echo Submitted batch job 9\n")
    for f in ("squeue", "sbatch"):
        (bin_ / f).chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}"}
    return subprocess.run(["bash", str(grid.ROOT / "shell/nchc/submit.sh"), "--live", *opts,
                           "--", "true"], env=env, capture_output=True, text=True, check=False)


def test_submit_sh_does_not_count_the_ending_job(tmp_path):
    """refill --ended-job counts its own job's slot as free; submit.sh must too."""
    ids = list(range(100, 118)) + [555]  # 19 jobs, one of them the job that is ending
    assert _submit_sh(tmp_path, ids).returncode == 3
    r = _submit_sh(tmp_path, ids, "--ending-job", "555")
    assert r.returncode == 0 and "Submitted batch job 9" in r.stdout
    assert _submit_sh(tmp_path, [*ids, 556], "--ending-job", "555").returncode == 3  # 19 others
    assert _submit_sh(tmp_path, [], "--ending-job", "555").returncode == 0


def test_default_exclude_lives_in_one_file():
    """One node list, read by both submit.sh and castor.grid."""
    import subprocess
    r = subprocess.run(["bash", str(grid.ROOT / "shell/nchc/submit.sh"), "--dry-run", "--",
                        "true"], capture_output=True, text=True, check=True)
    line = [x for x in r.stdout.splitlines() if x.startswith("#SBATCH --exclude=")]
    assert line == [f"#SBATCH --exclude={grid.read_exclude()}"]
    assert grid.read_exclude() == "25a-hgpn[001,003,007,036-039,062,071,083,111,125-173]"
    for f in ("castor/grid.py", "shell/nchc/submit.sh", "scripts/grid.py"):
        assert "hgpn[001" not in (grid.ROOT / f).read_text(), f  # no second copy
    assert not hasattr(grid, "DEFAULT_EXCLUDE")  # read at launch, not at import


def _submit_sh_copy(tmp_path, exclude_text):
    """submit.sh in a bare tree with its own exclude_nodes (the check runs before any git call)."""
    import shutil
    import subprocess
    tree = tmp_path / "tree" / "shell" / "nchc"
    tree.mkdir(parents=True, exist_ok=True)
    shutil.copy(grid.ROOT / "shell/nchc/submit.sh", tree / "submit.sh")
    (tree / "exclude_nodes").write_text(exclude_text)
    return subprocess.run(["bash", str(tree / "submit.sh"), "--dry-run", "--", "true"],
                          capture_output=True, text=True, check=False)


def test_submit_sh_refuses_a_malformed_exclude_file(tmp_path):
    """A second node list line or a comment-only file is an error (exit 2), never a
    multi-line #SBATCH --exclude (which ends the directive block) or a silent empty default."""
    r = _submit_sh_copy(tmp_path, "# why\nn[1-2]\nn[3]\n")
    assert r.returncode == 2 and "exactly one node list line" in r.stderr and "n[3]" in r.stderr
    r = _submit_sh_copy(tmp_path, "# nothing but comments\n")
    assert r.returncode == 2 and "exactly one node list line" in r.stderr
    with pytest.raises(AssertionError, match="expected one node list"):
        grid.read_exclude(tmp_path / "tree/shell/nchc/exclude_nodes")


def test_submit_sh_refuses_when_squeue_fails(tmp_path):
    """A failed squeue (controller timeout) is not a count of 0 jobs; the 19-slot check
    cannot pass on it."""
    import os
    import subprocess
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "squeue").write_text("#!/bin/bash\necho 'squeue: error: Socket timed out' >&2\nexit 1\n")
    (bin_ / "sbatch").write_text("#!/bin/bash\ncat > /dev/null; echo Submitted batch job 9\n")
    for f in ("squeue", "sbatch"):
        (bin_ / f).chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}"}
    r = subprocess.run(["bash", str(grid.ROOT / "shell/nchc/submit.sh"), "--live", "--", "true"],
                       env=env, capture_output=True, text=True, check=False)
    assert r.returncode != 0 and "Submitted" not in r.stdout and "Socket timed out" in r.stderr


def test_cron_removal_is_persisted_and_stages_reinstalls(g, fake_crontab, tmp_path, monkeypatch):
    """The refill that removes the line writes the cleared record to the control, so
    `status` stops reporting it; `stages --enable` still knows where to reinstall it."""
    gs = _script()
    ctl, d, _, c = _refill_setup(g, tmp_path, monkeypatch, gs, lambda *a: "x")
    monkeypatch.setattr(grid, "main_repo", lambda: tmp_path)
    gs._cron_on(ctl, d / "control.json")
    (d / "control.json").write_text(json.dumps(ctl))
    monkeypatch.setattr(gs, "survey", lambda *a, **k: [
        {"tag": c.tag, "train": "done", "genjudge": "done"}])
    gs.refill(json.loads((d / "control.json").read_text()), d / "control.json")
    on_disk = json.loads((d / "control.json").read_text())
    assert fake_crontab.read_text() == "" and on_disk["cron"] is None
    assert on_disk["cron_removed"]["host"] == gs._host()
    ctl2 = dict(on_disk)
    gs._cron_off(ctl2, d, "again")  # nothing recorded: no change
    assert ctl2 == on_disk
    sfile = tmp_path / "stages.json"
    sfile.write_text(json.dumps({"prompt_id": "008", "stages": [
        {"name": "A", "seeds": [1], "repos": None}, {"name": "B", "seeds": [2], "repos": None}]}))
    man = tmp_path / "grid.json"
    monkeypatch.setattr(grid, "ROOT", tmp_path)
    on_disk |= {"stages_file": "stages.json", "stages_sha256": gs.posttrain.sha256_file(sfile),
                "stages_enabled": ["A"], "manifest": "grid.json"}
    (d / "control.json").write_text(json.dumps(on_disk))
    monkeypatch.setattr(gs, "_control_path", lambda m, s: d / "control.json")
    monkeypatch.setattr(gs, "refill", lambda *a, **k: "refilled")
    import argparse
    gs.cmd_stages(argparse.Namespace(manifest=man, smoke=None, enable=["B"], disable=[]))
    after = json.loads((d / "control.json").read_text())
    assert after["cron"] and "cron_removed" not in after
    assert fake_crontab.read_text().count("castor-grid 008") == 1


def test_stop_persists_the_cleared_cron(g, fake_crontab, tmp_path, monkeypatch):
    gs = _script()
    ctl, d, _, _ = _refill_setup(g, tmp_path, monkeypatch, gs, lambda *a: "x")
    monkeypatch.setattr(grid, "main_repo", lambda: tmp_path)
    gs._cron_on(ctl, d / "control.json")
    (d / "control.json").write_text(json.dumps(ctl))
    monkeypatch.setattr(gs, "_control_path", lambda m, s: d / "control.json")
    import argparse
    gs.cmd_stop(argparse.Namespace(manifest=None, smoke=None))
    on_disk = json.loads((d / "control.json").read_text())
    assert not on_disk["enabled"] and on_disk["cron"] is None and fake_crontab.read_text() == ""


def test_no_cells_is_not_labelled(g, fake_crontab, tmp_path, monkeypatch):
    """Every stage disabled is not a finished grid; the cron line stays."""
    assert not grid.labelled([])
    assert grid.labelled([{"train": "done", "genjudge": "done"}])
    gs = _script()
    ctl, d, _, _ = _refill_setup(g, tmp_path, monkeypatch, gs, lambda *a: "x")
    monkeypatch.setattr(grid, "main_repo", lambda: tmp_path)
    gs._cron_on(ctl, d / "control.json")
    sfile = tmp_path / "stages.json"
    sfile.write_text(json.dumps({"prompt_id": "008", "stages": [
        {"name": "A", "seeds": [1, 2], "repos": None}]}))
    real_repo = grid.main_repo()
    monkeypatch.setattr(grid, "ROOT", tmp_path)
    monkeypatch.setattr(grid, "main_repo", lambda: real_repo)
    ctl |= {"stages_file": "stages.json", "stages_sha256": gs.posttrain.sha256_file(sfile),
            "stages_enabled": []}
    txt = gs.refill(ctl, d / "control.json")
    assert "castor-grid 008" in fake_crontab.read_text() and ctl["cron"]
    assert "stages=[]" in txt


def test_trap_refusal_clears_after_a_refill_pass(tmp_path):
    """A refused trap refill is an attention item until a refill pass surveyed with the trapped
    job gone from Slurm. A pass whose squeue snapshot still held the job (it was COMPLETING, or
    the pass began before the kill) logs after the trap without resolving it."""
    gs = _script()
    log = tmp_path / "driver.log"
    log.write_text(f"[t1 job 1] {gs.PASS_TAG}3 jobs of this user\n"
                   f"[t1 job 1] {gs.SNAPSHOT_TAG} 1 2 444\n"
                   "[t2 job 444] TRAP: refill job refused; run scripts/grid.py refill by hand\n")
    assert len(gs.trap_failures(tmp_path)) == 1
    with open(log, "a") as f:  # a pass that began before the kill: 444 still in its snapshot
        f.write(f"[t3 job 7] {gs.PASS_TAG}2 jobs of this user, 1 of this grid\n"
                f"[t3 job 7] {gs.SNAPSHOT_TAG} 7 444\n")
    assert len(gs.trap_failures(tmp_path)) == 1
    with open(log, "a") as f:  # a pass that saw 444 gone: its survey found the cell failed
        f.write(f"[t5 job 9] {gs.PASS_TAG}1 jobs of this user, 1 of this grid\n"
                f"[t5 job 9] {gs.SNAPSHOT_TAG} 9\n")
    assert gs.trap_failures(tmp_path) == []
    with open(log, "a") as f:
        f.write("[t4 job 8] TRAP: refill job refused; run scripts/grid.py refill by hand\n"
                f"[t6 lgn1 pid 5] {gs.PASS_TAG}0 jobs\n[t6 lgn1 pid 5] {gs.SNAPSHOT_TAG} -\n")
    assert gs.trap_failures(tmp_path) == []  # an empty snapshot resolves everything
    with open(log, "a") as f:
        f.write("[t7 job 8] TRAP: refill job refused; run scripts/grid.py refill by hand\n")
    assert [x[:9] for x in gs.trap_failures(tmp_path)] == ["[t7 job 8"]


def test_refill_status_and_log_agree_on_traps(g, tmp_path, monkeypatch):
    """The status a pass writes shows exactly the traps `status` reads from driver.log
    after that pass: open while the trapped job is still in Slurm, gone once it is not."""
    gs = _script()
    ctl, d, _, _ = _refill_setup(g, tmp_path, monkeypatch, gs, lambda *a: "Submitted batch job 1")
    ctl["max_retries"] = 0
    (d / "driver.log").write_text(
        "[t2 job 444] TRAP: refill job refused; run scripts/grid.py refill by hand\n")
    monkeypatch.setattr(grid, "squeue_list", lambda: [
        {"id": "444", "name": "008-train-x", "state": "COMPLETING", "where": "n1"}])
    txt = gs.refill(ctl, d / "control.json")
    assert "TRAP REFILL REFUSED" in txt and txt == (d / "status.txt").read_text()
    assert len(gs.trap_failures(d)) == 1
    assert f"{gs.SNAPSHOT_TAG} 444" in (d / "driver.log").read_text()
    monkeypatch.setattr(grid, "squeue_list", list)
    txt = gs.refill(ctl, d / "control.json")
    assert "TRAP REFILL REFUSED" not in txt and gs.trap_failures(d) == []
    assert gs.refill(ctl, d / "control.json", ended_job="444").count("TRAP") == 0


def test_pending_only_failure_does_not_look_up_sacct_by_name(g, monkeypatch):
    """No job id for the failed attempt (pending-only ledger entry): no sacct by
    name, which can return a job of an earlier launch or of the smoke grid."""
    gs = _script()
    none = {"sft": None, "dpo": None, "jobs": {}}
    calls = []
    c, r = _survey_one(gs, g, monkeypatch, none, {})
    monkeypatch.setattr(grid, "sacct", lambda job_id=None, name=None: calls.append(
        (job_id, name)) or {"job": job_id or name, "node": "n7"})
    r = gs.survey(g, [c], grid.MODELS_ROOT, False, {}, ledger={f"{c.tag}|sft": ["pending-x"]})[0]
    assert r["train"] == "failed-sft" and r["failed_job"] == {} and calls == []
    done = {"sft": "finished", "dpo": "finished", "jobs": {}}
    monkeypatch.setattr(grid, "stage_state", lambda *a: done)
    r = gs.survey(g, [c], grid.MODELS_ROOT, False, {}, ledger={f"{c.tag}|gj": ["pending-y"]})[0]
    assert r["genjudge"] == "failed" and r["failed_job"] == {} and calls == []


def test_cron_path_resolves_every_tool(tmp_path):
    """The cron line's PATH is built from absolute paths found at install time."""
    paths = {t: f"/opt/{t}/bin/{t}" for t in grid.CRON_TOOLS}
    p = grid.cron_path(paths.get)
    assert p.split(":")[:2] == ["/opt/uv/bin", "/opt/squeue/bin"] and p.endswith("/usr/bin:/bin")
    with pytest.raises(AssertionError, match="sbatch"):
        grid.cron_path(lambda t: None if t == "sbatch" else "/usr/bin/" + t)
    assert grid.cron_path(lambda t: "/usr/bin/" + t) == "/usr/bin:/bin"  # each directory once


# ---- launcher configuration (--ds-config / --launch-env) and the main-recipe rerun cells
# (per-base seeds, stage A0) -------------------------------------------------------------------

ZERO1 = "configs/posttrain/ds/sft_zero1.json"


def test_base_seeds_override_the_grid_seeds(tmp_path, monkeypatch):
    bases = MAN["bases"] + [{"repo": "allenai/DataDecide-dclm-baseline-1B",
                             "revision": "step69369-seed-default", "posttrain_seeds": [1, 2, 3, 4]}]
    p = tmp_path / "g.json"
    p.write_text(json.dumps(MAN | {"bases": bases}))
    g = grid.load(p)
    assert len(g.cells()) == 4 + 4
    main = [c for c in g.cells() if c.base.recipe == "dclm-baseline"]
    assert [c.k for c in main] == [1, 2, 3, 4]
    # never 006's paths or ids (dd-dclm-1B-seed-…, dd-dclm-1b-…)
    assert main[0].model_id == "dd-dclm-baseline-1b-default-dpo-seed1"
    assert main[0].out("sft").parts[-3] == "dd-dclm-baseline-1B-seed-default"
    s = tmp_path / "g_stages.json"
    s.write_text(json.dumps({"prompt_id": "008", "stages": [
        {"name": "A0", "seeds": [1, 2, 3, 4], "repos": ["allenai/DataDecide-dclm-baseline-1B"]},
        {"name": "A", "seeds": [1], "repos": None}, {"name": "C", "seeds": [2], "repos": None}]}))
    monkeypatch.setattr(grid, "ROOT", tmp_path)
    st = grid.load_stages("g_stages.json", g)
    assert [grid.stage_of(c, st) for c in main] == ["A0"] * 4
    assert {grid.stage_of(c, st) for c in g.cells() if c not in main} == {"A", "C"}


def test_train_cmd_passes_the_launcher_configuration(g, monkeypatch):
    gs = _script()
    monkeypatch.setattr(grid, "check_base", lambda b: {})
    c, root = g.cells()[0], grid.MODELS_ROOT
    plain = gs._train_cmd(c, root, "sft", None)
    assert plain == gs._train_cmd(c, root, "sft", None, {"ds_config": None, "launch_env": None})
    assert "--ds-config" not in plain and "--launch-env" not in plain
    launch = {"ds_config": {"sft": ZERO1}, "launch_env": ["NCCL_CUMEM_ENABLE=0"]}
    sft, dpo = gs._train_cmd(c, root, "sft", None, launch).split(" && ")
    assert sft == plain.split(" && ")[0] + f" --ds-config {ZERO1} --launch-env NCCL_CUMEM_ENABLE=0"
    assert "--ds-config" not in dpo and dpo.endswith("--launch-env NCCL_CUMEM_ENABLE=0")
    assert gs._train_cmd(c, root, "dpo", None, launch) == dpo  # a DPO-only retry: same flags


def _prov_launch(stage, ds=None, env=None, cpus=12):
    from castor import posttrain
    p = posttrain.ds_config_path(stage, ds)
    return {"open_instruct": {"deepspeed_config": str(p),
                              "deepspeed_config_sha256": posttrain.sha256_file(p)},
            "launcher": {"env": posttrain.launch_env(cpus, env)}}


def test_stage_state_checks_the_launcher_configuration(g, tmp_path):
    c, root = g.cells()[0], tmp_path / "models"
    zero1 = {"ds_config": {"sft": ZERO1}, "launch_env": None}
    current = {"ds_config": None, "launch_env": None}
    _sft(c, root, **_prov_launch("sft", ZERO1, cpus=96))
    _dpo(c, root, **_prov_launch("dpo"))
    res = grid.resolve_launcher
    assert grid.train_state(grid.stage_state(c, root, res(zero1)), False) == "done"
    assert grid.stage_state(c, root, res(None))["sft"] == "finished"  # no control: no check
    with pytest.raises(AssertionError, match="launcher configuration"):
        grid.stage_state(c, root, res(current))  # a ZeRO-1 SFT is not done under the current one
    with pytest.raises(AssertionError, match="launcher configuration"):
        grid.stage_state(c, root, res(zero1 | {"launch_env": ["NCCL_CUMEM_ENABLE=0"]}))


def test_pre_024_provenance_reads_as_the_current_configuration(g, tmp_path):
    """Provenance before 335ecd2 records neither sha nor environment (006's twins, old smokes)."""
    from castor import posttrain
    c, root = g.cells()[0], tmp_path / "models"
    _sft(c, root, open_instruct={"deepspeed_config": posttrain.OI_DS_CONFIG["sft"]})
    current = grid.resolve_launcher({"ds_config": None, "launch_env": None})
    assert grid.stage_state(c, root, current)["sft"] == "finished"
    with pytest.raises(AssertionError, match="launcher configuration"):
        grid.stage_state(c, root, grid.resolve_launcher({"ds_config": {"sft": ZERO1},
                                                         "launch_env": None}))
    legacy = {"open_instruct": {"deepspeed_config": "configs/ds_configs/other.conf"}}
    with pytest.raises(AssertionError, match="non-default"):
        grid.launcher_of(legacy, "sft")


def test_refill_refuses_a_changed_ds_config_and_sends_its_flags(g, tmp_path, monkeypatch):
    gs = _script()
    sent = []
    ctl, d, _, _ = _refill_setup(g, tmp_path, monkeypatch, gs,
                                 lambda opts, cmd, dry: sent.append(cmd) or "Submitted batch job 5")
    seen = []
    monkeypatch.setattr(gs, "_train_cmd", lambda c, root, st, smoke, launch: seen.append(launch)
                        or "train")
    cfg = tmp_path / "ds.json"
    cfg.write_text('{"zero_optimization": {"stage": 1}}')
    ctl |= {"ds_config": {"sft": str(cfg)}, "ds_config_sha256": gs.ds_files({"sft": str(cfg)}),
            "launch_env": None}
    gs.refill(ctl, d / "control.json")
    assert seen == [{"ds_config": {"sft": str(cfg)}, "launch_env": None}] and len(sent) == 1
    assert "launcher=sft:" in (d / "status.txt").read_text()
    cfg.write_text('{"zero_optimization": {"stage": 2}}')
    with pytest.raises(AssertionError, match="changed since launch"):
        gs.refill(ctl, d / "control.json")
    old = {k: v for k, v in ctl.items() if k not in ("ds_config", "ds_config_sha256", "launch_env")}
    assert gs.launcher(old) == {"ds_config": None, "launch_env": None}  # pre-ds control
    gs.check_ds(old)


def test_status_warns_of_control_files_changed_since_launch(tmp_path):
    """refill refuses a changed DeepSpeed config (check_ds), so status must say so, as
    it does for the stages file."""
    gs = _script()
    cfg, sf = tmp_path / "ds.json", tmp_path / "stages.json"
    cfg.write_text("{}")
    sf.write_text("{}")
    ctl = {"ds_config": {"sft": str(cfg)}, "ds_config_sha256": gs.ds_files({"sft": str(cfg)}),
           "stages_file": str(sf), "stages_sha256": gs.posttrain.sha256_file(sf)}
    assert gs.changed_since_launch(ctl) == []
    cfg.write_text("{ }")
    sf.write_text("{ }")
    w = gs.changed_since_launch(ctl)
    assert len(w) == 2 and all(x.startswith("WARNING") and "refuse" in x for x in w)
    assert str(sf) in w[0] and f"DeepSpeed config {ctl['ds_config']}" in w[1]
    assert gs.changed_since_launch({"ds_config": None, "ds_config_sha256": None}) == []


def test_launch_refuses_a_ds_config_that_is_not_committed_as_is(tmp_path, monkeypatch):
    """The jobs render from a snapshot of the launch commit; the control's sha256 is the
    live file's. The two must be the same bytes, or no job can ever match the control."""
    import subprocess
    gs = _script()
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True)
    cfg = repo / "configs" / "ds.json"
    cfg.parent.mkdir()
    cfg.write_text('{"zero_optimization": {"stage": 1}}')
    subprocess.run([*git, "add", "."], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "ds"], cwd=repo, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    monkeypatch.setattr(grid, "ROOT", repo.resolve())
    ds = {"sft": str(cfg)}
    gs.committed_ds(ds, head)  # committed as is
    gs.committed_ds(None, head)
    cfg.write_text('{"zero_optimization": {"stage": 2}}')
    with pytest.raises(AssertionError, match="differs from commit"):
        gs.committed_ds(ds, head)
    new = repo / "configs" / "new.json"
    new.write_text("{}")
    with pytest.raises(AssertionError, match="not in commit"):
        gs.committed_ds({"sft": str(new)}, head)
    outside = tmp_path / "outside.json"  # open-instruct's pinned tree: not this repo's to check
    outside.write_text("{}")
    gs.committed_ds({"sft": str(outside)}, head)


def test_launch_env_refuses_the_cpu_count():
    """check_launcher ignores BEAKER_ASSIGNED_CPU_COUNT, so an explicit value could not
    be checked; the driver refuses it."""
    gs = _script()
    gs.check_launch_env(None)
    gs.check_launch_env(["NCCL_CUMEM_ENABLE=0"])
    with pytest.raises(AssertionError, match="CPU count"):
        gs.check_launch_env(["NCCL_CUMEM_ENABLE=0", "BEAKER_ASSIGNED_CPU_COUNT=48"])
    with pytest.raises(ValueError, match="not a launcher setting"):
        gs.check_launch_env(["HF_TOKEN=x"])


def test_status_shows_the_last_pass_summary_not_the_snapshot(tmp_path):
    gs = _script()
    assert gs.last_pass(tmp_path) is None
    gs._log(tmp_path, "launch: {}")
    assert gs.last_pass(tmp_path) is None
    gs._log(tmp_path, f"{gs.PASS_TAG}3 jobs of this user, 1 of this grid before this pass; budget 2; "
                      f"planned 1; submitted 1\nsubmitted sft x: ok\n{gs.SNAPSHOT_TAG} 5 6\nFAILED x")
    assert gs.last_pass(tmp_path).endswith("budget 2; planned 1; submitted 1")


def test_analysis_checks_each_models_provenance(g, tmp_path, analyze_008):
    """A directory holding another run is an error, never a model of the number; the
    models root is replaceable for the mock run only."""
    an = analyze_008
    c, root = g.cells()[0], tmp_path / "models"
    sel = an.selected(g, root=root)
    assert sel[c.model_id].sft == c.out("sft", root) and sel[c.model_id].k == c.k
    assert sel[c.model_id].base_ref == c.base.ref and sel[c.model_id][1] == "008"
    assert an.launchers({c.model_id: sel[c.model_id]}) == {c.model_id: None}  # no provenance
    _sft(c, root, **_prov_launch("sft", ZERO1, cpus=96))
    _dpo(c, root, **_prov_launch("dpo"))
    one = {c.model_id: sel[c.model_id]}
    got = an.launchers(one)[c.model_id]
    assert set(got) == {"sft", "dpo"} and got["sft"]["env"] == {"TOKENIZERS_PARALLELISM": "false"}
    other = g.cells()[1]  # another seed's run in this cell's directory
    _sft(c, root, **_prov_launch("sft", ZERO1), post_training_seed_k=other.k)
    with pytest.raises(AssertionError, match="seed"):
        an.launchers(one)
    _sft(c, root, **_prov_launch("sft", ZERO1), base_ref="allenai/x@y")
    with pytest.raises(AssertionError, match="base_ref"):
        an.launchers(one)
    _sft(c, root, **_prov_launch("sft", ZERO1))
    _dpo(c, root, **_prov_launch("dpo"), base_model="/elsewhere/model")
    with pytest.raises(AssertionError, match="base"):
        an.launchers(one)


def test_analysis_takes_the_main_recipe_from_one_source(tmp_path, monkeypatch, analyze_008):
    an = analyze_008
    g = grid.load(grid.ROOT / "data/manifests/grid_008.json")
    st = grid.load_stages("data/manifests/grid_008_stages.json", g)
    kill = an.selected(g, st, ["A", "B"])  # 024 KILL: 006's twins are the main recipe
    assert sum(v[1] == "006" for v in kill.values()) == 6 and len(kill) == 72 + 24 + 6
    go = an.selected(g, st, ["A0", "A", "B", "C"])  # GO: the rerun cells, seeds 1 and 2 only
    assert len(go) == 150 and all(v[1] == "008" for v in go.values())
    main = sorted(m for m, v in go.items() if v[0].recipe == "dclm-baseline")
    assert len(main) == 6 and not any(m.endswith(("seed3", "seed4")) for m in main)
    assert {v[0].recipe for v in go.values()} == {v[0].recipe for v in kill.values()}


def test_variant_grid_has_its_own_root_and_ids(tmp_path):
    p = tmp_path / "grid.json"
    p.write_text(json.dumps({**MAN, "prompt_id": "080", "variant": "swap080",
                             "sft_config": "configs/posttrain/sft_datadecide_eosbos_swap.yaml"}))
    v = grid.load(p)
    c = v.cells()[1]
    assert c.model_id == "dd-dclm-baseline-qc-10p-1b-large-aux-2-dpo-seed2-swap080"
    assert grid.models_root(v, False) == grid.MODELS_ROOT / "swap080"
    assert grid.models_root(v, True) == grid.MODELS_ROOT / "smoke" / "080" / "swap080"
    assert c.out("sft", grid.models_root(v, False)).parts[-4:] == (
        "swap080", "dd-dclm-baseline-qc-10p-1B-seed-large-aux-2", "sft", "seed2")


def test_reference_grid_paths_unchanged(g):
    assert grid.models_root(g, False) == grid.MODELS_ROOT
    assert grid.models_root(g, True) == grid.MODELS_ROOT / "smoke" / "008"
    assert g.variant is None and not g.cells()[0].model_id.endswith("swap080")


def test_off_reference_config_needs_a_variant(tmp_path):
    p = tmp_path / "grid.json"
    p.write_text(json.dumps({**MAN, "sft_config": "configs/posttrain/sft_datadecide_eosbos_swap.yaml"}))
    with pytest.raises(AssertionError, match="needs a variant"):
        grid.load(p)
