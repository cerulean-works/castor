"""024 equivalence rules (A = B trivially passes; a planted offset fails), checkpoint paths and
the cost parsers behind the speed-up."""

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from castor import grid

spec = importlib.util.spec_from_file_location(
    "equiv_024", Path(__file__).parents[1] / "scripts/equiv_024.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def _state(seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return {"w": torch.randn(50, 20, generator=g, dtype=torch.bfloat16) * scale,
            "b": torch.randn(20, generator=g)}


def test_l2_is_fp32_global_norm():
    x, y = _state(0), _state(1)
    want = torch.sqrt(sum(((x[k].float() - y[k].float()) ** 2).sum() for k in x)).item()
    assert abs(E.l2(x, y) - want) < 1e-4
    assert E.l2(x, x) == 0.0


def test_weight_rule():
    assert E.weight_rule({"A|R": 1.0, "A|B": 1.0, "R|B": 1.0})["pass"]
    assert E.weight_rule({"A|R": 1.25, "A|B": 0.5, "R|B": 1.0})["pass"]
    assert not E.weight_rule({"A|R": 1.0, "A|B": 1.3, "R|B": 1.0})["pass"]


def _matrices(n=400, k=20, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.5, 0.5, n)
    return lambda s: (np.random.default_rng(s).random((n, k)) < p[:, None]).astype(float), p


def test_behaviour_identical_a_and_b_passes():
    draw, _ = _matrices()
    r, b = draw(1), draw(2)
    res = E.behaviour_rule({"R": r, "B": b, "A": b.copy()}, n_boot=200)
    assert res["pass"] and res["diff_B_minus_A"] == 0


def test_behaviour_planted_offset_fails():
    draw, p = _matrices()
    r, b = draw(1), draw(2)
    rng = np.random.default_rng(9)
    perm = rng.permutation(len(p))
    a = draw(3)
    a[perm[:200]] = a[perm[:200]][::-1]  # scramble half the prompts: per-prompt signal lost
    res = E.behaviour_rule({"R": r, "B": b, "A": a}, n_boot=200)
    assert res["diff_B_minus_A"] > 0.01 and not res["pass"]


def test_level_rule():
    ok = E.level_rule({"s": {"R": 0.5, "B": 0.52, "A": 0.475}})
    assert ok["pass"] and ok["sources"]["s"]["band"] == 0.03
    wide = E.level_rule({"s": {"R": 0.5, "B": 0.56, "A": 0.45}})
    assert wide["pass"]  # band = |B - R| = 0.06
    assert not E.level_rule({"s": {"R": 0.5, "B": 0.51, "A": 0.46}})["pass"]


def test_verdict():
    p, f = {"pass": True}, {"pass": False}
    assert E.verdict(1.3, p, p, p) == "GO"
    assert E.verdict(1.29, p, p, p) == "KILL"
    assert E.verdict(2.0, p, f, p) == "KILL"


# ---- checkpoint paths and the cost numbers behind the GO/KILL speed-up ------------------------


def test_seed2_reference_is_006_default_s2():
    """R's and the seed-to-seed reference's checkpoints differ only in the post-training seed."""
    assert E.ckpt(E.SEED2, "dpo", 2) == E.M / E.GROUP / "dpo" / "seed2" / "model"
    assert E.ckpt(E.MODELS["R"][0], "dpo") == E.M / E.GROUP / "dpo" / "seed1" / "model"


def _sacct(monkeypatch, line):
    def run(cmd, **kw):
        assert cmd[0] == "sacct" and "AllocTRES" in cmd[-1]
        return subprocess.CompletedProcess(cmd, 0, stdout=line + "\n", stderr="")
    monkeypatch.setattr(grid.subprocess, "run", run)


def test_elapsed_hours():
    assert grid.elapsed_hours("00:30:00") == 0.5
    assert grid.elapsed_hours("1-02:00:36") == 26.01
    assert grid.elapsed_hours("05:24.500") == pytest.approx((5 * 60 + 24.5) / 3600)


def test_sacct_hours_is_elapsed_times_allocated_gpus(monkeypatch):
    _sacct(monkeypatch, "512|COMPLETED|ggpn07|1-01:30:00|0:0|billing=96,cpu=96,gres/gpu=8,node=1")
    c = E.sacct_hours("512")
    assert c["elapsed_h"] == 25.5 and c["gpus"] == 8 and c["h200_h"] == 204.0
    _sacct(monkeypatch, "513|COMPLETED|ggpn07|02:00:00|0:0|cpu=48,gres/gpu=4,node=1")
    assert E.sacct_hours("513")["h200_h"] == 8.0


def test_sacct_keeps_the_node_when_elapsed_is_unparsed(monkeypatch):
    """The grid survey needs a failed job's node; an odd Elapsed field must not turn the
    cell into an error. The GPU-hour consumer refuses it instead."""
    _sacct(monkeypatch, "515|COMPLETED|ggpn07|INVALID|0:0|cpu=96,gres/gpu=8,node=1")
    s = grid.sacct("515")
    assert s["node"] == "ggpn07" and s["elapsed_h"] is None and s["gpus"] == 8
    with pytest.raises(AssertionError, match="unparsed Elapsed"):
        E.sacct_hours("515")


def test_sacct_hours_refuses_an_unfinished_job(monkeypatch):
    _sacct(monkeypatch, "514|TIMEOUT|ggpn07|20:00:00|0:0|cpu=96,gres/gpu=8,node=1")
    with pytest.raises(AssertionError, match="514"):
        E.sacct_hours("514")


def _run(root, stage, started, finished, args=()):
    d = root / E.GROUP / stage / "seed1"
    d.mkdir(parents=True)
    (d / "provenance.json").write_text(json.dumps({
        "started": started, "finished": finished,
        "open_instruct": {"deepspeed_config": "x", "deepspeed_config_sha256": "y"},
        "launcher": {"env": {"A": "1"}}}))
    (d / "oi_args.txt").write_text("\n".join(args) + "\n")


def test_stage_hours_uses_the_job_gpu_count_and_the_offset(tmp_path):
    _run(tmp_path, "sft", "2026-10-02T10:00:00+0800", "2026-10-02T13:30:00+0800")
    # an offset change between the stamps (DST or another zone) is honoured, not ignored
    _run(tmp_path, "dpo", "2026-10-02T13:30:00+0800", "2026-10-02T06:00:00+0000")
    h = E.stage_hours(tmp_path, 4)
    assert h["sft"]["h200_h"] == 3.5 * 4 and h["dpo"]["h200_h"] == 0.5 * 4
    assert h["sft"]["launcher_env"] == {"A": "1"}


def test_oi_args_blanks_only_the_run_specific_values(tmp_path):
    args = ["--model_name_or_path", "/m/a", "--seed", "1", "--output_dir", "/o/a/oi",
            "--exp_name", "a_sft_seed1", "--learning_rate", "2e-5", ""]
    _run(tmp_path / "a", "sft", "", "", args)
    b = [x.replace("/a", "/b").replace("a_", "b_") for x in args]
    _run(tmp_path / "b", "sft", "", "", b)
    oa = E.oi_args(tmp_path / "a", "sft")
    assert oa == E.oi_args(tmp_path / "b", "sft")
    assert oa.count("<run>") == 3 and oa[oa.index("--seed") + 1] == "1"
    c = [x.replace("2e-5", "1e-5") for x in args]
    _run(tmp_path / "c", "sft", "", "", c)
    assert E.oi_args(tmp_path / "c", "sft") != oa
