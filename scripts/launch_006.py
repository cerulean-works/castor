"""Submit the runs/006 jobs: 12 SFT → DPO training chains (8 GPUs each) and, once a chain has
finished, one generation + judge_v1 job per DPO twin (1 GPU). Runs on the login node; each job
goes through shell/nchc/submit.sh (snapshot of HEAD, so commit first).

    uv run --frozen python scripts/launch_006.py train [--only default-s1 ...] [--dry-run]
    uv run --frozen python scripts/launch_006.py genjudge [--only ...] [--dry-run]
    # dry run: one twin, 50 SFT + 50 DPO steps on the full data, own output root
    uv run --frozen python scripts/launch_006.py train --smoke 50 --only default-s1 --dev

Skip-if-done is keyed on provenance, never on the path alone: a training cell is done when
both provenances have `finished`, the SFT names this cell's REPO@revision (`base_ref`) and seed
and the DPO names that SFT and seed; a gen+judge cell when runs/<model-id>/006/provenance.json
covers the full panel (limit null), was generated from this twin's current DPO checkpoint,
and judge_v1 labels exist. A cell whose job is still in squeue (by job name), whose run is
started but unfinished, or whose generations come from another checkpoint is reported and
not resubmitted (inspect it by hand). Every chain checks its converted base against
REPO@revision first. Smoke runs (N > 0) pass `--set max_train_steps=N`, the only override,
and write under <models>/smoke/006/, never into a real run's directory; `--dev` is for smoke
runs only.
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from castor import sanityb

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMIT = REPO_ROOT / "shell" / "nchc" / "submit.sh"


def _prov(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def status(cell: sanityb.Cell, root: Path) -> str:
    """todo | started | done. Lineage is read from provenance, never from the path alone: the
    SFT names this cell's REPO@revision (checked against the converted copy at render time)
    and seed; the DPO names this SFT's model and the same seed."""
    sft = _prov(cell.out("sft", root) / "provenance.json")
    dpo = _prov(cell.out("dpo", root) / "provenance.json")
    if sft is None:
        assert dpo is None, f"{cell.tag}: DPO provenance without SFT"
        return "todo"
    where = cell.out("sft", root)
    assert sft["post_training_seed_k"] == cell.k, f"{where}: seed {sft['post_training_seed_k']}"
    assert sft["base_model"] == str(cell.base_dir), f"{where}: base {sft['base_model']}"
    assert sft.get("base_ref") == cell.base_ref, f"{where}: base_ref {sft.get('base_ref')}"
    if dpo is None:
        return "started"
    where = cell.out("dpo", root)
    assert dpo["post_training_seed_k"] == cell.k, f"{where}: seed {dpo['post_training_seed_k']}"
    assert dpo["base_model"] == str(cell.out("sft", root) / "model"), f"{where}: base differs"
    return "done" if "finished" in dpo and "finished" in sft else "started"


def launcher_args(stage: str, ds_config: dict[str, str] | None,
                  launch_env: list[str] | None) -> list[str]:
    """train.sh flags of a launcher configuration: an explicit DeepSpeed config per stage and
    KEY=VALUE environment settings. None/empty adds no flag (the default configuration)."""
    out = ["--ds-config", ds_config[stage]] if ds_config and stage in ds_config else []
    for kv in launch_env or []:
        out += ["--launch-env", kv]
    return out


def train_cmd(cell: sanityb.Cell, root: Path, smoke: int | None,
              ds_config: dict[str, str] | None = None, launch_env: list[str] | None = None,
              stages: tuple[str, ...] = ("sft", "dpo")) -> str:
    sanityb.check_base(cell)
    extra = ["--set", f"max_train_steps={smoke}"] if smoke is not None else []
    sft = ["bash", "shell/posttrain/train.sh", "sft", sanityb.SFT_CONFIG, str(cell.base_dir),
           str(cell.k), str(cell.out("sft", root)), "--base-ref", cell.base_ref, *extra,
           *launcher_args("sft", ds_config, launch_env)]
    dpo = ["bash", "shell/posttrain/train.sh", "dpo", sanityb.DPO_CONFIG,
           str(cell.out("sft", root) / "model"), str(cell.k), str(cell.out("dpo", root)), *extra,
           *launcher_args("dpo", ds_config, launch_env)]
    assert stages in (("sft", "dpo"), ("sft",)), stages
    return " && ".join(shlex.join(c) for c in ([sft, dpo] if "dpo" in stages else [sft]))


def genjudge_status(cell: sanityb.Cell, root: Path, prompt_id: str = sanityb.PROMPT_ID) -> str:
    """todo | done | stale. done = generations on the full panel (limit null) from this twin's
    current DPO checkpoint, and judge_v1 labels. stale = generations exist from another
    checkpoint (a retrained twin): gen_judge.sh would reuse them, so they are never resubmitted
    over; move them aside by hand. `prompt_id` names the generation directory."""
    out = sanityb.RUNS / cell.model_id / prompt_id
    gen = _prov(out / "provenance.json")
    if gen is None or gen.get("limit") is not None:
        return "todo"
    ckpt = str(Path(_prov(cell.out("dpo", root) / "provenance.json")["checkpoint"]).resolve())
    if gen.get("model_resolved") != ckpt:
        return "stale"
    return "done" if (out / "judge_v1" / "labels.jsonl").exists() else "todo"


def queued() -> set[str]:
    """Names of this user's pending or running Slurm jobs: a cell whose job is still queued has
    no provenance yet and must not be submitted twice."""
    r = subprocess.run(["squeue", "--me", "--noheader", "--format=%j"],
                       capture_output=True, text=True, check=True)
    return set(r.stdout.split())


def submit_argv(opts: list[str], cmd: str, dry: bool, commit: str | None = None) -> list[str]:
    """The submit.sh command line `submit` runs (one builder for the real and the dry path).
    `commit` None means unpinned; an empty string is a pin that was never
    set, not an unpinned submit, and is refused (submit.sh would silently fall back to the
    live tree)."""
    assert commit is None or commit, "empty --commit pin (HEAD not set)"
    return [str(SUBMIT), *(["--commit", commit] if commit else []), *opts,
            *(["--dry-run"] if dry else []), "--", "bash", "-c", cmd]


def submit(opts: list[str], cmd: str, dry: bool, commit: str | None = None,
           run_dry: bool = False) -> str:
    """submit.sh (pinned to `commit` when given); returns the Slurm job id
    ("<jobid>" on a dry run). A dry run prints the argv; with `run_dry` it also executes it
    (`submit.sh --dry-run`: options, `--after` and the pinned commit are parsed and the sbatch
    script printed, nothing submitted), so the argv checked is the one the real path runs."""
    argv = submit_argv(opts, cmd, dry, commit)
    print("$", shlex.join(argv), flush=True)  # before submit.sh's own output
    if dry:
        if run_dry:
            subprocess.run(argv, check=True, cwd=REPO_ROOT)
        return "<jobid>"
    r = subprocess.run(argv, check=False, cwd=REPO_ROOT, capture_output=True, text=True)
    print(r.stdout, end="")
    print(r.stderr, end="", file=sys.stderr)  # before raising: submit.sh's refusal says why
    if r.returncode:
        raise RuntimeError(f"submit.sh exited {r.returncode}: {r.stderr.strip()}")
    m = re.search(r"Submitted batch job (\d+)", r.stdout)
    assert m, f"no job id in submit.sh output: {r.stdout!r}"
    return m.group(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["train", "genjudge"])
    ap.add_argument("--only", nargs="*", help="cell tags, e.g. default-s1 large-aux-2-s3")
    ap.add_argument("--smoke", type=int, default=None, help="max_train_steps (dry run)")
    ap.add_argument("--dev", action="store_true", help="dev partition (4 h), smoke runs only")
    ap.add_argument("--dry-run", action="store_true", help="print sbatch scripts only")
    a = ap.parse_args()
    smoke = a.smoke is not None
    assert not smoke or a.smoke > 0, "--smoke takes a positive step count"
    assert not a.dev or smoke, "--dev (3.5 h) cannot hold a full SFT -> DPO chain; smoke only"
    assert not (smoke and a.phase == "genjudge"), "smoke is for training only"
    cells = sanityb.cells()
    if a.only:
        unknown = set(a.only) - {c.tag for c in cells}
        assert not unknown, f"unknown cells {sorted(unknown)}"
        cells = [c for c in cells if c.tag in a.only]
    root = sanityb.MODELS_ROOT / "smoke" / "006" if smoke else sanityb.MODELS_ROOT
    in_queue = queued()

    for c in cells:
        name = f"006-{'smoke-' if smoke else ''}{a.phase}-{c.tag}"
        if name in in_queue:
            print(f"skip {c.tag}: job {name} is queued or running")
            continue
        if a.phase == "train":
            st = status(c, root)
            if st != "todo":
                print(f"skip {c.tag}: {st} ({c.out('dpo', root)})")
                continue
            opts = ["--gpus", "8", "--name", name,
                    "--time", "03:30:00" if a.dev else "20:00:00", *(["--dev"] if a.dev else [])]
            submit(opts, train_cmd(c, root, a.smoke), a.dry_run)
        else:
            if status(c, root) != "done":
                print(f"skip {c.tag}: training not finished")
                continue
            st = genjudge_status(c, root)
            if st != "todo":
                print(f"skip {c.tag}: generations {st}"
                      + (" (from another checkpoint; move them aside by hand)" if st == "stale"
                         else ""))
                continue
            cmd = shlex.join(["bash", "shell/posttrain/gen_judge.sh",
                              str(c.out("dpo", root) / "model"), "-", c.model_id,
                              sanityb.PROMPT_ID])
            submit(["--gpus", "1", "--time", "12:00:00", "--name", name], cmd, a.dry_run)


if __name__ == "__main__":
    main()
