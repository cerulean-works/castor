"""Render one open-instruct run config and record its provenance (called by
shell/posttrain/train.sh; castor env, never open-instruct's).

    uv run --frozen python scripts/posttrain_run.py render --stage sft \
        --config configs/posttrain/sft_olmo2_1b.yaml --model allenai/OLMo-2-0425-1B --k 1 \
        --out /work/harrychang08/castor/models/posttrain/olmo2-1b/sft/seed1 [--set key=value]
    uv run --frozen python scripts/posttrain_run.py finish --out <same dir>

render writes <out>/oi_config.yaml (the versioned config + model, seed, output_dir, exp_name,
and any --set overrides, e.g. max_train_steps for smoke tests), the same as command-line flags
in <out>/oi_args.txt (one token per line), a copy of the DeepSpeed config the run uses
(<out>/ds_config.json; the stage default unless --ds-config names one), the launcher
environment (<out>/launch_env.sh, sourced by train.sh), and <out>/provenance.json with the
config's source path and sha256 and the environment. finetune.py and
dpo_tune_cache.py @ fc7748ea4 parse flags only (`parse_args_into_dataclasses`, finetune.py:934),
so the YAML is passed as flags, the way Ai2's own commands (docs/olmo2.md) are written.
finish links <out>/model to the checkpoint open-instruct saved (<out>/oi/<exp>__<seed>__<ts>)
and adds the end time and loss log location.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from castor import posttrain, provenance


def to_flags(cfg: dict) -> list[str]:
    """YAML config -> HfArgumentParser flags: bools as true/false, lists as repeated values."""
    out = []
    for k, v in cfg.items():
        assert v is not None, f"{k}: null cannot be passed as a flag; drop it from the config"
        out.append(f"--{k}")
        if isinstance(v, bool):
            out.append("true" if v else "false")
        elif isinstance(v, list):
            out += [str(x) for x in v]
        else:
            out.append(str(v))
    return out


def parent_lineage(model: str, revision: str | None, parent_base: str | None) -> dict:
    """The parent checkpoint and the parent's own base (a DPO run records both hops).

    A local parent is a castor post-training output (<run>/model), whose run provenance holds
    its base; an HF parent records no base, so it comes from --parent-base REPO@SHA."""
    run_prov = Path(model).parent / "provenance.json"
    if Path(model).exists() and run_prov.exists():
        assert parent_base is None, "a local parent's base is read from its provenance.json"
        p = json.loads(run_prov.read_text())
        return {"model": model, "revision": None, "run_provenance": str(run_prov),
                "stage": p["stage"], "oi_seed": p["oi_seed"], "checkpoint": p.get("checkpoint"),
                "git_commit": p["git_commit"],
                "base_model": p["base_model"], "base_revision": p["base_revision"]}
    base_model, base_rev = parent_base.split("@") if parent_base else (None, None)
    return {"model": model, "revision": revision, "base_model": base_model,
            "base_revision": base_rev, "base_source": "--parent-base" if parent_base else None}


def render(a) -> None:
    cfg = yaml.safe_load(Path(a.config).read_text())
    seed = posttrain.oi_seed(a.stage, a.k)
    run = {"model_name_or_path": a.model, "seed": seed, "output_dir": str(a.out / "oi"),
           "exp_name": f"{a.out.parent.parent.name}_{a.stage}_seed{a.k}"}
    clash = sorted(set(run) & set(cfg))
    assert not clash, f"run-level keys must not be in the versioned config: {clash}"
    if a.revision:
        assert cfg.get("model_revision", a.revision) == a.revision, "revision differs from config"
        run["model_revision"] = a.revision
    if a.base_ref:  # provenance only (never an open-instruct flag): the local base's HF identity
        src = json.loads((Path(a.model) / "provenance.json").read_text())["source"]
        assert src == a.base_ref, f"{a.model} holds {src}, not {a.base_ref}"
    prov_parent = parent_lineage(a.model, a.revision, a.parent_base)
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        run[k] = yaml.safe_load(v)
    ds_src = posttrain.ds_config_path(a.stage, a.ds_config)
    json.loads(ds_src.read_text())  # a DeepSpeed config is JSON; refuse anything else
    # set-but-empty counts as unset, as in ${SLURM_CPUS_PER_TASK:-12}
    env = posttrain.launch_env(int(os.environ.get("SLURM_CPUS_PER_TASK") or "12"), a.launch_env)
    ds_rec, ds_tree = posttrain.ds_config_record(ds_src)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "oi_config.yaml").write_text(yaml.safe_dump(cfg | run, sort_keys=False))
    (a.out / "oi_args.txt").write_text("\n".join(to_flags(cfg | run)) + "\n")
    shutil.copyfile(ds_src, a.out / "ds_config.json")
    (a.out / "launch_env.sh").write_text(
        "".join(f"export {k}={shlex.quote(v)}\n" for k, v in sorted(env.items())))
    oi_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=posttrain.OI_DIR,
                             capture_output=True, text=True, check=True).stdout.strip()
    # unmodified means no working-tree, staged or untracked change (git status, not
    # git diff, which ignores the index and new files)
    oi_dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                              cwd=posttrain.OI_DIR, capture_output=True, text=True,
                              check=True).stdout.splitlines()
    assert oi_head == posttrain.OI_COMMIT, f"open-instruct at {oi_head}, expected {posttrain.OI_COMMIT}"
    assert not oi_dirty, f"open-instruct tree is modified (D018): {oi_dirty[:5]}"
    src = Path(cfg["dataset_mixer_list"][0])
    data = json.loads(src.with_suffix(".json").read_text())
    assert data["sha256"] == posttrain.sha256_file(src), f"{src} differs from its export"
    prov = {"stage": a.stage, "post_training_seed_k": a.k, "oi_seed": seed,
            "base_model": a.model, "base_revision": a.revision or cfg.get("model_revision"),
            "base_ref": a.base_ref, "parent": prov_parent,
            "config": a.config, "config_sha256": posttrain.sha256_file(Path(a.config)),
            "overrides": run, "dataset": {k: data.get(k) for k in ("repo", "revision", "jsonl", "rows", "sha256")},
            "open_instruct": {"dir": str(posttrain.OI_DIR), "commit": oi_head,
                              "entry": posttrain.OI_ENTRY[a.stage],
                              "deepspeed_config": ds_rec, "deepspeed_config_tree": ds_tree,
                              "deepspeed_config_sha256": posttrain.sha256_file(ds_src),
                              "deepspeed_config_default": a.ds_config is None,
                              "lock": "configs/third_party/open-instruct.lock"},
            "launcher": {"script": "shell/posttrain/train.sh",
                         "script_sha256": posttrain.sha256_file(
                             posttrain.REPO_ROOT / "shell/posttrain/train.sh"),
                         "env": env},
            "slurm_job": a.job, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_commit": provenance.git_commit(), "command": provenance.command()}
    (a.out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(a.out / "oi_args.txt")


def finish(a) -> None:
    prov = json.loads((a.out / "provenance.json").read_text())
    runs = sorted((a.out / "oi").glob(f"{prov['overrides']['exp_name']}__{prov['oi_seed']}__*"))
    assert runs, f"no open-instruct output under {a.out / 'oi'}"
    # open-instruct names the run dir <exp>__<seed>__<int(time.time())> on every rank
    # (dpo_tune_cache.py:494, finetune.py:414), so ranks that cross a second boundary leave
    # extra empty dirs; exactly one dir holds the model the main process saved.
    saved = [r for r in runs if (r / "config.json").exists()]
    assert len(saved) == 1, f"expected one saved model among {runs}, found {saved}"
    ckpt = saved[0]
    link = a.out / "model"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(ckpt.relative_to(a.out))
    prov |= {"checkpoint": str(ckpt), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "tensorboard": sorted(str(p) for p in ckpt.rglob("events.out.tfevents*"))}
    (a.out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(ckpt)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--stage", choices=["sft", "dpo"], required=True)
    r.add_argument("--config", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--revision", default=None)
    r.add_argument("--base-ref", default=None,
                   help="REPO@revision the local --model was converted from (checked, recorded)")
    r.add_argument("--parent-base", default=None,
                   help="REPO@SHA an HF --model was itself trained from (provenance only)")
    r.add_argument("--k", type=int, required=True, help="post-training seed index (1 = Ai2's)")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--set", action="append", help="extra open-instruct key=value (smoke only)")
    r.add_argument("--job", default=None)
    r.add_argument("--ds-config", default=None,
                   help="DeepSpeed config (default: the stage's current one, posttrain.OI_DS_CONFIG)")
    r.add_argument("--launch-env", action="append",
                   help="KEY=VALUE exported to the launch (posttrain.LAUNCH_ENV_ALLOWED only)")
    f = sub.add_parser("finish")
    f.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    render(a) if a.cmd == "render" else finish(a)


if __name__ == "__main__":
    main()
