"""Submit the runs/016 jobs through the launch_006 path: per ladder cell, one 8-GPU SFT → DPO
chain (`train.sh`, the `_eosbos` configs unchanged) and one 1-GPU gen + judge_v1 job
(`gen_judge.sh`), chained with `--after`. Runs on the login node; each job goes through
shell/nchc/submit.sh (snapshot of HEAD, so commit first).

    uv run --frozen python scripts/launch_016.py train --size 150M [--only 150M-default-s1 ...]
        [--ds-config configs/posttrain/ds/sft_zero1.json] [--launch-env KEY=VALUE ...]
        [--chain] [--max-queued 6] [--dry-run]
    uv run --frozen python scripts/launch_016.py genjudge --size 150M [--only ...] [--dry-run]
    # dry run: one twin per size, 50 SFT + 50 DPO steps, own output root
    # (<models>/smoke/016/<ds-config stem>/)
    uv run --frozen python scripts/launch_016.py train --size 150M --smoke 50 \
        --only 150M-default-s1 --dev --ds-config configs/posttrain/ds/sft_zero1.json

Launcher configuration: `--ds-config` (SFT DeepSpeed config, default `ladder.SFT_DS` = ZeRO-1)
and `--launch-env` go through `launch_006.train_cmd(..., ds_config=, launch_env=)`; `train.sh` →
`posttrain_run.py render` records path, sha256 and environment in every provenance. The launcher
refuses a request that does not resolve to the configuration of record
(`ladder.check_launcher_request`: same DeepSpeed bytes, same exported environment at both
stages), and a cell whose existing provenance is on another configuration is an error, never
"done" (`status`; castor.grid.check_launcher). `check-trained` re-checks the chain's
configuration inside the gen + judge job, so a twin trained on another one is never generated.

A wrapper, not a second launcher: the cell status, the base check and the training
command are `launch_006.status` / `launch_006.train_cmd` / `sanityb.check_base` applied to
`castor.ladder.Cell` (same interface as `sanityb.Cell`); the squeue read, the submit.sh call
and its job-id parse, and the generation status are `launch_006.queued` / `launch_006.submit`
(pinned to HEAD with `--commit`, dry runs executed with `run_dry`) /
`launch_006.genjudge_status`. What differs from launch_006.main: the cell list and sizes, job
names `016-…`, the smoke root `<models>/smoke/016/`, the generation directory
`runs/<model-id>/016/`, `--chain` (submit the gen+judge job right after its chain with `--after`;
the job first re-checks that training finished, so a failed chain never generates from a stale
or missing checkpoint), `--max-queued` (at most N 016 jobs in Slurm at once) and `--dry-run`
(executes `submit.sh --dry-run` on the real path's argv per job; launch_006.submit's `run_dry`).
"""

import argparse
import importlib.util
import shlex
from pathlib import Path

from castor import ladder, provenance

HEAD = ""  # set in main()

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("launch_006",
                                              REPO_ROOT / "scripts" / "launch_006.py")
L6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L6)


def genjudge_cmd(cell: ladder.Cell, root: Path) -> str:
    check = ["uv", "run", "--frozen", "python", "scripts/launch_016.py", "check-trained",
             "--size", cell.size, "--only", cell.tag]
    gj = ["bash", "shell/posttrain/gen_judge.sh", str(cell.out("dpo", root) / "model"), "-",
          cell.model_id, ladder.PROMPT_ID]
    return shlex.join(check) + " && " + shlex.join(gj)


def head() -> str:
    """The commit every job of this invocation runs (submit.sh --commit). `provenance.git_commit`
    returns HEAD only if every path submit.sh snapshots is clean, so the launcher's pin and the
    job's provenance.json agree."""
    return provenance.git_commit(REPO_ROOT)


DRY_JOBID = "0"  # a dry chain's --after: digits like a real id, so the printed sbatch script
# carries the real --dependency=afterany:<id> form (never submitted)


def submit(opts: list[str], cmd: str, dry: bool) -> str:
    """launch_006.submit pinned to this invocation's HEAD; returns the job id (DRY_JOBID dry).
    A dry run executes submit.sh --dry-run on the same argv (`run_dry`)."""
    assert HEAD, "HEAD not pinned: submit() is called from main() after head()"
    jid = L6.submit(opts, cmd, dry, commit=HEAD, run_dry=True)
    return DRY_JOBID if dry else jid


def genjudge_status(cell: ladder.Cell, root: Path) -> str:
    return L6.genjudge_status(cell, root, ladder.PROMPT_ID)


def status(cell: ladder.Cell, root: Path, want: ladder.Launcher | None = None) -> str:
    """launch_006.status (todo | started | done, lineage from provenance), plus: every
    provenance present was trained on the configuration of record (ladder.check_cell_launcher
    raises otherwise), so a chain from another configuration is never skipped as done nor
    built on. `want` = `ladder.Launcher()`, resolved once by main() for every cell."""
    st = L6.status(cell, root)
    if st != "todo":
        ladder.check_cell_launcher(cell, root, want=want)
    return st


def smoke_root(ds_config: str) -> Path:
    """Dry runs of one launcher configuration under <models>/smoke/016/<config stem>/."""
    return ladder.MODELS_ROOT / "smoke" / ladder.PROMPT_ID / Path(ds_config).stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["train", "genjudge", "check-trained"])
    ap.add_argument("--size", required=True, choices=sorted(ladder.SIZES))
    ap.add_argument("--only", nargs="*", help="cell tags, e.g. 150M-default-s1")
    ap.add_argument("--smoke", type=int, default=None, help="max_train_steps (dry run)")
    ap.add_argument("--dev", action="store_true", help="dev partition (4 h), smoke runs only")
    ap.add_argument("--chain", action="store_true", help="train: also submit gen+judge --after")
    ap.add_argument("--max-queued", type=int, default=6, help="cap on 016-* jobs in Slurm")
    ap.add_argument("--ds-config", default=ladder.SFT_DS,
                    help="SFT DeepSpeed config for every chain (train.sh --ds-config; must hash "
                         "to the ratified configuration, D067a); DPO keeps the stage default")
    ap.add_argument("--launch-env", action="append", default=None, metavar="KEY=VALUE",
                    help="launcher environment for train.sh (posttrain.LAUNCH_ENV_ALLOWED); must "
                         "resolve to the configuration of record")
    ap.add_argument("--dry-run", action="store_true",
                    help="submit nothing: run submit.sh --dry-run per job (git status, rev-parse "
                         "of the pinned HEAD, one sbatch script printed per job)")
    a = ap.parse_args()
    smoke = a.smoke is not None
    assert not smoke or a.smoke > 0, "--smoke takes a positive step count"
    assert not a.dev or smoke, "--dev (3.5 h) cannot hold a full SFT -> DPO chain; smoke only"
    assert not (smoke and a.phase != "train"), "smoke is for training only"
    assert not (smoke and a.chain), "smoke runs are not generated"
    cells = ladder.cells((a.size,))
    if a.only:
        unknown = set(a.only) - {c.tag for c in cells}
        assert not unknown, f"unknown cells {sorted(unknown)}"
        cells = [c for c in cells if c.tag in a.only]
    # the configuration of record, resolved once and pinned by content; the request must
    # resolve to it (ladder.check_launcher_request)
    want = ladder.Launcher()
    launcher = ladder.check_launcher_request(a.ds_config, a.launch_env, want)
    root = smoke_root(a.ds_config) if smoke else ladder.MODELS_ROOT

    if a.phase == "check-trained":  # run inside a chained gen+judge job, before generating
        for c in cells:
            st = status(c, root, want)  # raises if on another configuration
            assert st == "done", f"{c.tag}: training {st}, not generating"
            assert genjudge_status(c, root) != "stale", f"{c.tag}: stale generations"
        return

    global HEAD
    HEAD = head()
    in_queue = L6.queued()
    n_ours = sum(n.startswith("016-") for n in in_queue)
    for c in cells:
        name = f"016-{'smoke-' if smoke else ''}{a.phase}-{c.tag}"
        gj_name = f"016-genjudge-{c.tag}"
        if name in in_queue or (a.chain and gj_name in in_queue):
            print(f"skip {c.tag}: a job of this cell is queued or running")
            continue
        need = 2 if a.chain else 1
        if n_ours + need > a.max_queued:
            print(f"stop at {c.tag}: {n_ours} 016 jobs in Slurm, cap {a.max_queued}")
            break
        if a.phase == "train":
            st = status(c, root, want)
            if st != "todo":
                print(f"skip {c.tag}: {st} ({c.out('dpo', root)})")
                continue
            opts = ["--gpus", "8", "--name", name,
                    "--time", "03:30:00" if a.dev else "20:00:00", *(["--dev"] if a.dev else [])]
            cmd = L6.train_cmd(c, root, a.smoke, ds_config=launcher["ds_config"],
                               launch_env=launcher["launch_env"])
            jid = submit(opts, cmd, a.dry_run)
            n_ours += 1
            if a.chain:
                submit(["--gpus", "1", "--time", "12:00:00", "--name", gj_name,
                        "--after", jid], genjudge_cmd(c, root), a.dry_run)
                n_ours += 1
        else:
            if status(c, root, want) != "done":
                print(f"skip {c.tag}: training not finished")
                continue
            st = genjudge_status(c, root)
            if st != "todo":
                print(f"skip {c.tag}: generations {st}"
                      + (" (from another checkpoint; move them aside by hand)" if st == "stale"
                         else ""))
                continue
            submit(["--gpus", "1", "--time", "12:00:00", "--name", name],
                   genjudge_cmd(c, root), a.dry_run)
            n_ours += 1


if __name__ == "__main__":
    main()
