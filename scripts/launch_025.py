"""Submit the runs/025 colocation jobs: N copies of the launch_024 50-step SFT smoke (cell
`default-s1`, through launch_006.train_cmd) started together inside one 8-GPU dev allocation by
shell/posttrain/colocate.sh, with or without a job-local MPS daemon. Runs on the login node;
every job goes through shell/nchc/submit.sh pinned to HEAD (commit first).

    uv run --frozen python scripts/launch_025.py colocate mps2-cur mps2-z1 ts2-z1 [--dry-run]
    uv run --frozen python scripts/launch_025.py status

A configuration (COLOCATE) is: MPS on/off, number of copies, and the launch_024 candidate whose
smoke each copy repeats ("cur1" = the default configuration, "zero1" = the ZeRO-1 SFT).
Copy i writes under <models>/smoke/025/<config>/c<i>/ and its log to
<models>/smoke/025/<config>/logs/c<i>.log; the open-instruct argument list is the launch_024
smoke's with only the output root changed (checked by scripts/colocate_025.py).
"""

import argparse
import shlex
import sys
from pathlib import Path

from castor import sanityb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_006 as L
import launch_024 as L24

PROMPT_ID = "025"
# name -> (mps, copies, launch_024 candidate repeated by every copy); (e): the 50 + 50 SFT -> DPO
# chain ("zero1-chain"), so the projection uses a measured DPO factor
COLOCATE: dict[str, tuple[bool, int, str]] = {
    "mps2-cur": (True, 2, "cur1"),   # (a)
    "mps2-z1": (True, 2, "zero1"),   # (b)
    "ts2-z1": (False, 2, "zero1"),   # (c) time-sliced (no MPS)
    "mps3-z1": (True, 3, "zero1"),   # (d) only if (b) reached >= 1.6x aggregate
    "mps2-z1-chain": (True, 2, "zero1-chain"),   # (e) candidates
    "ts2-z1-chain": (False, 2, "zero1-chain"),
    "mps3-z1-chain": (True, 3, "zero1-chain"),
}


def root(name: str) -> Path:
    return sanityb.MODELS_ROOT / "smoke" / PROMPT_ID / name


def copy_root(name: str, i: int) -> Path:
    return root(name) / f"c{i}"


def log_dir(name: str) -> Path:
    return root(name) / "logs"


def copy_cmd(name: str, i: int) -> str:
    """launch_024's smoke command for the candidate, with copy i's output root (same cell, steps,
    DeepSpeed config, launcher environment and stages)."""
    c24 = COLOCATE[name][2]
    cand = L24.config_of(c24)
    return L.train_cmd(L24.cell(name), copy_root(name, i), L24.SMOKE_STEPS, cand.get("ds"),
                       cand.get("env"), L24.smoke_stages(c24))


def job_cmd(name: str) -> list[str]:
    mps, n, _ = COLOCATE[name]
    return ["bash", "shell/posttrain/colocate.sh", "1" if mps else "0", str(log_dir(name)),
            *(copy_cmd(name, i) for i in range(1, n + 1))]


def done(name: str) -> bool:
    _, n, c24 = COLOCATE[name]
    return all(L24.stage_done(copy_root(name, i), st)
               for i in range(1, n + 1) for st in L24.smoke_stages(c24))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["colocate", "status"])
    ap.add_argument("names", nargs="*", help="colocate: configuration names (COLOCATE)")
    ap.add_argument("--dry-run", action="store_true", help="print sbatch scripts only")
    a = ap.parse_args()
    if a.phase == "status":
        for n in COLOCATE:
            print(f"{n}\tdone={done(n)}")
        return
    assert a.names, "name the configurations"
    in_queue = L.queued()
    for n in a.names:
        assert n in COLOCATE, f"unknown configuration {n}"
        jname = f"{PROMPT_ID}-coloc-{n}"
        if jname in in_queue or done(n) or log_dir(n).exists():
            print(f"skip {n}: queued, done or already attempted ({log_dir(n)})")
            continue
        L.submit(["--gpus", "8", "--dev", "--time", "01:00:00", "--name", jname],
                 shlex.join(job_cmd(n)), a.dry_run, commit=L24.head())


if __name__ == "__main__":
    main()
