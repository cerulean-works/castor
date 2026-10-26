"""The DeepSpeed config and launcher environment are explicit, copied next to
oi_args.txt and recorded (path, sha256, env) in every run's provenance; the default is the
current configuration and leaves the open-instruct argument list unchanged."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from castor import posttrain, sanityb

ROOT = Path(__file__).parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


pr = _load("posttrain_run")
sys.path.insert(0, str(ROOT / "scripts"))
L6 = _load("launch_006")
L24 = _load("launch_024")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A fake open-instruct tree with its two shipped configs, a dataset export, a config."""
    oi = tmp_path / "oi"
    for stage, rel in posttrain.OI_DS_CONFIG.items():
        (oi / rel).parent.mkdir(parents=True, exist_ok=True)
        (oi / rel).write_text(json.dumps({"zero_optimization": {"stage": 3 if stage == "sft" else 2}}))
    monkeypatch.setattr(posttrain, "OI_DIR", oi)
    data = tmp_path / "d.jsonl"
    data.write_text('{"messages": []}\n')
    data.with_suffix(".json").write_text(json.dumps({"sha256": posttrain.sha256_file(data)}))
    cfg = tmp_path / "sft.yaml"
    cfg.write_text(yaml.safe_dump({"dataset_mixer_list": [str(data), "1.0"],
                                   "per_device_train_batch_size": 2}))

    def run(cmd, **kw):
        out = posttrain.OI_COMMIT if cmd[:2] == ["git", "rev-parse"] else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(pr.subprocess, "run", run)
    monkeypatch.setattr(pr.provenance, "git_commit", lambda: "c0ffee")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "96")
    return tmp_path, cfg


def _args(tmp, cfg, **kw):
    a = {"stage": "sft", "config": str(cfg), "model": "allenai/x", "revision": None,
         "base_ref": None, "parent_base": None, "k": 1, "out": tmp / "run", "set": None,
         "job": "1", "ds_config": None, "launch_env": None}
    return argparse.Namespace(**(a | kw))


def test_default_is_the_current_config_and_is_recorded(env):
    tmp, cfg = env
    pr.render(_args(tmp, cfg))
    run = tmp / "run"
    src = posttrain.OI_DIR / posttrain.OI_DS_CONFIG["sft"]
    assert (run / "ds_config.json").read_bytes() == src.read_bytes()
    oi = json.loads((run / "provenance.json").read_text())["open_instruct"]
    # recorded as 006 recorded it: relative to the open-instruct tree
    assert oi["deepspeed_config"] == posttrain.OI_DS_CONFIG["sft"] and oi["deepspeed_config_default"]
    assert oi["deepspeed_config_tree"] == "open-instruct"
    assert oi["deepspeed_config_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    launcher = json.loads((run / "provenance.json").read_text())["launcher"]
    assert launcher["env"] == {"BEAKER_ASSIGNED_CPU_COUNT": "96", "TOKENIZERS_PARALLELISM": "false"}
    assert (run / "launch_env.sh").read_text() == (
        "export BEAKER_ASSIGNED_CPU_COUNT=96\nexport TOKENIZERS_PARALLELISM=false\n")


def test_explicit_config_and_env_change_nothing_but_themselves(env, tmp_path):
    tmp, cfg = env
    pr.render(_args(tmp, cfg, out=tmp / "a"))
    alt = tmp_path / "alt.json"
    alt.write_text(json.dumps({"zero_optimization": {"stage": 2}}))
    pr.render(_args(tmp, cfg, out=tmp / "b", ds_config=str(alt),
                    launch_env=["NCCL_CUMEM_ENABLE=0"]))
    a, b = tmp / "a", tmp / "b"
    assert (a / "oi_args.txt").read_text() == (b / "oi_args.txt").read_text().replace(str(b), str(a))
    assert (b / "ds_config.json").read_bytes() == alt.read_bytes()
    pb = json.loads((b / "provenance.json").read_text())
    assert pb["open_instruct"]["deepspeed_config"] == str(alt)
    assert pb["open_instruct"]["deepspeed_config_tree"] is None
    assert not pb["open_instruct"]["deepspeed_config_default"]
    assert pb["launcher"]["env"]["NCCL_CUMEM_ENABLE"] == "0"
    assert "export NCCL_CUMEM_ENABLE=0\n" in (b / "launch_env.sh").read_text()


def test_repo_config_is_recorded_repo_relative(env):
    tmp, cfg = env
    pr.render(_args(tmp, cfg, ds_config="configs/posttrain/ds/sft_zero1.json"))
    oi = json.loads((tmp / "run" / "provenance.json").read_text())["open_instruct"]
    assert oi["deepspeed_config"] == "configs/posttrain/ds/sft_zero1.json"
    assert oi["deepspeed_config_tree"] == "castor"


def test_oi_config_named_explicitly_is_recorded_like_the_default(env):
    """zero2 passes open-instruct's stage2 config by absolute path: the record is the same
    string DPO's default run records."""
    tmp, cfg = env
    pr.render(_args(tmp, cfg, out=tmp / "a", stage="sft",
                    ds_config=str(posttrain.OI_DIR / posttrain.OI_DS_CONFIG["dpo"])))
    oi = json.loads((tmp / "a" / "provenance.json").read_text())["open_instruct"]
    assert oi["deepspeed_config"] == posttrain.OI_DS_CONFIG["dpo"] == \
        "configs/ds_configs/stage2_accelerate.conf"


def test_relative_config_resolves_against_the_repo():
    assert posttrain.ds_config_path("sft", "configs/posttrain/ds/sft_zero1.json") == \
        ROOT / "configs/posttrain/ds/sft_zero1.json"
    assert posttrain.ds_config_path("dpo") == posttrain.OI_DIR / posttrain.OI_DS_CONFIG["dpo"]


def test_non_json_config_and_non_launcher_env_are_refused(env, tmp_path):
    tmp, cfg = env
    bad = tmp_path / "bad.conf"
    bad.write_text("stage: 2\n")
    with pytest.raises(json.JSONDecodeError):
        pr.render(_args(tmp, cfg, ds_config=str(bad)))
    with pytest.raises(ValueError, match="not a launcher setting"):
        pr.render(_args(tmp, cfg, launch_env=["PYTHONHASHSEED=3"]))
    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        pr.render(_args(tmp, cfg, launch_env=["NCCL_CUMEM_ENABLE"]))
    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        pr.render(_args(tmp, cfg, launch_env=["=0"]))


def test_empty_cpu_count_falls_back_to_12(env, monkeypatch):
    tmp, cfg = env
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "")
    pr.render(_args(tmp, cfg))
    env_ = json.loads((tmp / "run" / "provenance.json").read_text())["launcher"]["env"]
    assert env_["BEAKER_ASSIGNED_CPU_COUNT"] == "12"


def test_train_sh_refuses_a_relative_out():
    r = subprocess.run(["bash", str(ROOT / "shell/posttrain/train.sh"), "sft", "c.yaml", "m", "1",
                        "runs/x"], capture_output=True, text=True, check=False)
    assert r.returncode == 2 and "absolute" in r.stderr


def test_shell_quoting_of_env_values(env):
    tmp, cfg = env
    pr.render(_args(tmp, cfg, launch_env=["NCCL_P2P_LEVEL=a b"]))
    assert "export NCCL_P2P_LEVEL='a b'\n" in (tmp / "run" / "launch_env.sh").read_text()


# ---- launchers ------------------------------------------------------------------------------

@pytest.fixture
def nobase(monkeypatch):
    monkeypatch.setattr(sanityb, "check_base", lambda c: {})


def test_train_cmd_default_is_006s_command(nobase, tmp_path):
    cell = sanityb.Cell("default", 1)
    cmd = L6.train_cmd(cell, tmp_path, None)
    assert "--ds-config" not in cmd and "--launch-env" not in cmd
    assert cmd == L6.train_cmd(cell, tmp_path, None, None, None)


def test_train_cmd_places_configs_per_stage(nobase, tmp_path):
    cmd = L6.train_cmd(sanityb.Cell("default", 1), tmp_path, None, {"sft": "x.json"}, ["A=1"])
    sft, dpo = cmd.split(" && ")
    assert "--ds-config x.json" in sft and "--ds-config" not in dpo
    assert sft.endswith("--launch-env A=1") and dpo.endswith("--launch-env A=1")
    assert " && " not in L6.train_cmd(sanityb.Cell("default", 1), tmp_path, 5, stages=("sft",))


def test_launch_024_smokes_and_arms(nobase, monkeypatch):
    cur = L24.smoke_cmd("cur1")
    assert "max_train_steps=50" in cur and "--ds-config" not in cur and "train.sh dpo" not in cur
    assert f"--ds-config {L24.OI_ZERO2}" in L24.smoke_cmd("zero2")
    assert str(L24.smoke_root("cur1")) in cur and "/smoke/024/cur1/" in cur
    b = L24.arm_cmd("B")
    assert "--ds-config" not in b and "max_train_steps" not in b and "/posttrain/024/rep/" in b
    monkeypatch.setattr(L24, "ARM_A", "zero2")
    a = L24.arm_cmd("A")
    assert "--ds-config" in a and "/posttrain/024/cfgA/" in a and "train.sh dpo" in a
    # the open-instruct side of A and B is the same command apart from the output root
    assert a.replace(f" --ds-config {L24.OI_ZERO2}", "").replace("/cfgA/", "/rep/") == b


def test_launch_024_cells_are_006s_default_s1():
    for arm, c in L24.ARMS.items():
        assert (c.k, c.tag, c.base_ref, c.base_dir) == \
            (L24.BASE.k, L24.BASE.tag, L24.BASE.base_ref, L24.BASE.base_dir)
        assert c.out("dpo", L24.arm_root(arm)) == L24.BASE.out("dpo", L24.arm_root(arm))
    assert L24.ARMS["A"].model_id == "dd-dclm-1b-default-dpo-seed1-cfgA"


def test_launch_024_smoke_done_needs_every_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(L24, "smoke_root", lambda n: tmp_path / n)

    def prov(name, stage, finished):
        d = L24.BASE.out(stage, tmp_path / name)
        d.mkdir(parents=True)
        (d / "provenance.json").write_text(json.dumps({"finished": "t"} if finished else {}))

    prov("zero1-chain", "sft", True)
    assert not L24.smoke_done("zero1-chain")  # SFT finished, DPO never started
    prov("zero1-chain", "dpo", False)
    assert not L24.smoke_done("zero1-chain")  # DPO failed before finish
    (L24.BASE.out("dpo", tmp_path / "zero1-chain") / "provenance.json").write_text('{"finished": "t"}')
    assert L24.smoke_done("zero1-chain")
    prov("zero1", "sft", True)
    assert L24.smoke_done("zero1")  # SFT-only smoke


def test_candidate_configs_differ_only_in_scheduling():
    base = json.loads((posttrain.OI_DIR / posttrain.OI_DS_CONFIG["sft"]).read_text()) \
        if (posttrain.OI_DIR / posttrain.OI_DS_CONFIG["sft"]).exists() else None
    if base is None:
        pytest.skip("open-instruct tree not available")
    for f in (ROOT / "configs/posttrain/ds").glob("*.json"):
        c = json.loads(f.read_text())
        for k in ("bf16", "gradient_accumulation_steps", "gradient_clipping", "train_batch_size",
                  "train_micro_batch_size_per_gpu"):
            assert c[k] == base[k], (f.name, k)
        assert set(c) <= set(base), f.name  # no optimizer, scheduler, data_types, fp16, ...
