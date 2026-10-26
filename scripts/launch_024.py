"""Submit the runs/024 jobs: the SFT profile, the 50-step launcher-candidate smokes, and arms A and
B with their chained generation + judge_v1 jobs. Runs on the login node; every job goes through
shell/nchc/submit.sh pinned to HEAD (commit first).

    uv run --frozen python scripts/launch_024.py profile [--dry-run]
    uv run --frozen python scripts/launch_024.py smoke cur1 cur2 zero2 ... [--dry-run]
    uv run --frozen python scripts/launch_024.py arms [--dry-run]       # after the smokes
    uv run --frozen python scripts/launch_024.py genjudge [--dry-run]   # resubmit a lost gen+judge
    uv run --frozen python scripts/launch_024.py status

One base and one post-training seed throughout: the 006 cell `default-s1`
(`DataDecide-dclm-baseline-1B@step69369-seed-default`, SFT seed 1 / DPO seed 111) with the 006
configs unchanged, through launch_006's `train_cmd`. A launcher configuration (CANDIDATES) is only a
DeepSpeed config per stage and KEY=VALUE launcher environment; the open-instruct argument list is
the same for all of them (checked after the runs from each `oi_args.txt`). Smokes pass `--set
max_train_steps=50` (as launch_006's `--smoke 50`), SFT only unless named in SMOKE_DPO, on the dev
partition, under <models>/smoke/024/<name>/. Arms: A = the candidate in ARM_A, B = the current
configuration again (the identical-seed replicate), under <models>/024/<arm>/, model ids
dd-dclm-1b-default-dpo-seed1-{cfgA,rep}, generations under runs/<model-id>/024/. Skip-if-done is
launch_006's provenance rule (a smoke is done when every stage it runs has finished); a job name
already in `squeue --me` is never submitted twice. `arms` submits training + chained gen+judge for
an arm whose training is todo; `genjudge` submits gen+judge alone for an arm whose training is done
and whose generations are todo (the chained job failed or was cancelled).
"""

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from castor import posttrain, sanityb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_006 as L

PROMPT_ID = "024"
SMOKE_STEPS = 50
BASE = sanityb.Cell("default", 1)
OI_ZERO2 = str(posttrain.OI_DIR / posttrain.OI_DS_CONFIG["dpo"])  # open-instruct's stage2_accelerate.conf

# name -> launcher configuration: {"ds": {stage: path}, "env": [KEY=VALUE, ...]}; {} = current.
CANDIDATES: dict[str, dict] = {
    "cur1": {},  # smoke floor: the current configuration, twice, on different jobs/nodes/ports
    "cur2": {},
    "zero2": {"ds": {"sft": OI_ZERO2}},  # (a) open-instruct's shipped ZeRO-2 config, as DPO uses
    "zero1": {"ds": {"sft": "configs/posttrain/ds/sft_zero1.json"}},  # (b)
    "zero3t": {"ds": {"sft": "configs/posttrain/ds/sft_zero3_tuned.json"}},  # (c)
    # (d) environment only, on the current DeepSpeed config
    "env": {"env": ["NCCL_CUMEM_ENABLE=0", "CUDA_DEVICE_MAX_CONNECTIONS=8"]},
}
# the screen's pick (fastest candidate inside the smoke floor), and its 50 + 50 SFT -> DPO dry
# run under its own name
ARM_A: str | None = "zero1"
CANDIDATES["zero1-chain"] = CANDIDATES["zero1"]
SMOKE_DPO: set[str] = {"zero1-chain"}  # smokes that also run 50 DPO steps
PROFILE_DS = "configs/posttrain/ds/sft_zero3_profile.json"


@dataclass(frozen=True)
class Cell(sanityb.Cell):
    """The default-s1 cell (every sanityb.Cell attribute inherited) under another output root
    (`name`) and model id."""
    name: str = ""
    own_model_id: str = ""

    @property
    def model_id(self) -> str:
        return self.own_model_id


def cell(name: str, model_id: str = "") -> Cell:
    return Cell(BASE.pretrain_seed, BASE.k, name, model_id)


ARMS = {"A": cell("cfgA", "dd-dclm-1b-default-dpo-seed1-cfgA"),
        "B": cell("rep", "dd-dclm-1b-default-dpo-seed1-rep")}


def smoke_root(name: str) -> Path:
    return sanityb.MODELS_ROOT / "smoke" / PROMPT_ID / name


def arm_root(arm: str) -> Path:
    return sanityb.MODELS_ROOT / PROMPT_ID / ARMS[arm].name


def config_of(name: str) -> dict:
    assert name in CANDIDATES, f"unknown candidate {name}"
    return CANDIDATES[name]


def smoke_stages(name: str) -> tuple[str, ...]:
    return ("sft", "dpo") if name in SMOKE_DPO else ("sft",)


def smoke_cmd(name: str) -> str:
    c = config_of(name)
    return L.train_cmd(cell(name), smoke_root(name), SMOKE_STEPS, c.get("ds"), c.get("env"),
                       smoke_stages(name))


def profile_cmd() -> str:
    """Two 50-step SFT runs of the current configuration, back to back on one node: (1) with
    DeepSpeed's wall-clock breakdown (PROFILE_DS) and `nvidia-smi dmon` alongside, (2) the plain
    current config under Nsight Systems (CUDA kernels incl. NCCL)."""
    r1, r2 = smoke_root("profile-ds"), smoke_root("profile-nsys")
    ds = L.train_cmd(cell("profile-ds"), r1, SMOKE_STEPS, {"sft": PROFILE_DS}, None, ("sft",))
    plain = L.train_cmd(cell("profile-nsys"), r2, SMOKE_STEPS, None, None, ("sft",))
    nsys = ["nsys", "profile", "-t", "cuda,nvtx", "-s", "none", "--cpuctxsw=none",
            "--force-overwrite", "true", "-o", str(r2 / "nsys")]
    return (f"mkdir -p {r1} {r2} && (nvidia-smi dmon -s um -o T > {r1}/dmon.txt & echo $! > {r1}/dmon.pid)"
            f" && {ds}; kill $(cat {r1}/dmon.pid); {shlex.join(nsys)} {plain}")


def stage_done(root: Path, stage: str) -> bool:
    p = L._prov(BASE.out(stage, root) / "provenance.json")
    return bool(p and "finished" in p)


def smoke_done(name: str) -> bool:
    """Every stage the smoke runs has finished (SFT, and DPO for SMOKE_DPO)."""
    return all(stage_done(smoke_root(name), st) for st in smoke_stages(name))


def arm_cmd(arm: str) -> str:
    c = config_of(ARM_A) if arm == "A" else {}
    return L.train_cmd(ARMS[arm], arm_root(arm), None, c.get("ds"), c.get("env"))


def gj_cmd(arm: str) -> str:
    guard = ["uv", "run", "--frozen", "python", "scripts/launch_024.py", "trained", "--arm", arm]
    gj = ["bash", "shell/posttrain/gen_judge.sh", str(ARMS[arm].out("dpo", arm_root(arm)) / "model"),
          "-", ARMS[arm].model_id, PROMPT_ID]
    return shlex.join(guard) + " && " + shlex.join(gj)


OWN = ["scripts/launch_024.py", "scripts/launch_006.py", "scripts/posttrain_run.py",
       "castor/posttrain.py", "castor/sanityb.py", "shell/posttrain", "configs/posttrain"]


def head() -> str:
    """The commit every job runs (submit.sh --commit): the code must be committed."""
    r = subprocess.run(["git", "status", "--porcelain", "--", *OWN], cwd=L.REPO_ROOT,
                       capture_output=True, text=True, check=True)
    assert not r.stdout.strip(), f"commit first: {r.stdout}"
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=L.REPO_ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def submit(opts: list[str], cmd: str, dry: bool) -> str:
    """launch_006.submit pinned to HEAD; returns the Slurm job id ("<jobid>" on a dry run)."""
    return L.submit(opts, cmd, dry, commit=head())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["profile", "smoke", "arms", "genjudge", "status", "trained"])
    ap.add_argument("names", nargs="*", help="smoke: candidate names (CANDIDATES)")
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--dry-run", action="store_true", help="print sbatch scripts only")
    a = ap.parse_args()

    if a.phase == "trained":  # in-job guard of a chained gen + judge job
        st = L.status(ARMS[a.arm], arm_root(a.arm))
        if st != "done":
            sys.exit(f"arm {a.arm}: training {st}")
        return
    if a.phase == "status":
        for n in CANDIDATES:
            done = {st: stage_done(smoke_root(n), st) for st in smoke_stages(n)}
            print(f"smoke {n}\t" + "\t".join(f"{st}_done={v}" for st, v in done.items()))
        for arm, c in ARMS.items():
            st = L.status(c, arm_root(arm))
            gj = L.genjudge_status(c, arm_root(arm), PROMPT_ID) if st == "done" else "-"
            print(f"arm {arm}\t{c.model_id}\ttrain={st}\tgenjudge={gj}")
        return

    in_queue = L.queued()
    dev = ["--gpus", "8", "--dev", "--time", "02:00:00"]
    if a.phase == "profile":
        name = f"{PROMPT_ID}-profile"
        if name in in_queue or (stage_done(smoke_root("profile-ds"), "sft")
                                and stage_done(smoke_root("profile-nsys"), "sft")):
            print(f"skip {name}: queued or done")
            return
        submit([*dev, "--name", name], profile_cmd(), a.dry_run)
    elif a.phase == "smoke":
        assert a.names, "name the candidates to smoke"
        for n in a.names:
            config_of(n)
            name = f"{PROMPT_ID}-smoke-{n}"
            if name in in_queue or smoke_done(n):
                print(f"skip {n}: queued or done")
                continue
            submit([*dev, "--name", name], smoke_cmd(n), a.dry_run)
    elif a.phase == "genjudge":
        for arm, c in ARMS.items():
            gname = f"{PROMPT_ID}-gj-{arm}"
            if gname in in_queue or f"{PROMPT_ID}-train-{arm}" in in_queue:
                print(f"skip arm {arm}: queued or running")
                continue
            st = L.status(c, arm_root(arm))
            gj = L.genjudge_status(c, arm_root(arm), PROMPT_ID) if st == "done" else None
            if gj != "todo":
                print(f"skip arm {arm}: training {st}, generations {gj}"
                      + (" (from another checkpoint; move them aside by hand)" if gj == "stale"
                         else ""))
                continue
            submit(["--gpus", "1", "--time", "12:00:00", "--name", gname], gj_cmd(arm), a.dry_run)
    else:  # arms
        assert CANDIDATES.get(ARM_A), \
            "ARM_A is unset or the current configuration; fix it at the review gate"
        for arm, cell in ARMS.items():
            tname, gname = f"{PROMPT_ID}-train-{arm}", f"{PROMPT_ID}-gj-{arm}"
            if tname in in_queue or gname in in_queue:
                print(f"skip arm {arm}: queued or running")
                continue
            st = L.status(cell, arm_root(arm))
            if st != "todo":
                print(f"skip arm {arm}: training {st}"
                      + (" (gen+judge: the genjudge phase)" if st == "done" else ""))
                continue
            after = submit(["--gpus", "8", "--name", tname, "--time", "20:00:00"],
                              arm_cmd(arm), a.dry_run)
            submit(["--gpus", "1", "--time", "12:00:00", "--name", gname, "--after", after],
                      gj_cmd(arm), a.dry_run)


if __name__ == "__main__":
    main()
