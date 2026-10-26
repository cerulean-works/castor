"""Submit the 4 post-training cells of runs/035: the launch_011 logic on the second same-seed
replicate base R2 (revision replicate2-i6198-d6198) in data/manifests/grid_035.json, seeds
{1, 2, 3, 4}, with SFT on the ZeRO-1 configuration. Runs on the login node; each job goes through
shell/nchc/submit.sh pinned to HEAD (commit first).

    uv run --frozen python scripts/launch_035.py submit [--only TAG ...] [--dry-run]
    uv run --frozen python scripts/launch_035.py status

Per cell, one SFT -> DPO training job (8 GPUs, `launch_006.train_cmd` with
ds_config = {"sft": SFT_DS}, DPO unchanged) and one generation + judge_v1 job (1 GPU,
`gen_judge.sh`, prompt id 035) chained on it with `--after`, guarded by
`launch_035.py trained --only TAG`. Skip-if-done and the queue check are launch_006's. The SFT
provenance records the DeepSpeed config path and sha256 (posttrain_run.py); the launcher refuses
unless SFT_DS hashes to SFT_DS_SHA256.

Differences from launch_011: the manifest, the prompt id and the base revision.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from castor import grid, posttrain, sanityb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_006 as L
import launch_009 as L9

MANIFEST = L.REPO_ROOT / "data" / "manifests" / "grid_035.json"
PROMPT_ID = "035"
SFT_DS = "configs/posttrain/ds/sft_zero1.json"
SFT_DS_SHA256 = "b615b6daa5ea9aa8e67d5b9055093b8581287439f84f73038730af2d6d70ab37"
DS_CONFIG = {"sft": SFT_DS}
REPLICATE = "castor/DataDecide-dclm-baseline-1B@replicate2-i6198-d6198"


def cells(manifest: Path = MANIFEST) -> list[L9.Cell]:
    """The 4 replicate cells (grid.checked_cells: prompt 035, sanityb's configs and seeds, the
    replicate base only), with sanityb.Cell's interface for launch_006's functions."""
    return [L9.Cell(c) for c in grid.checked_cells(manifest, PROMPT_ID, REPLICATE)]


def check_ds() -> None:
    sha = posttrain.sha256_file(posttrain.ds_config_path("sft", SFT_DS))
    assert sha == SFT_DS_SHA256, f"{SFT_DS} is {sha}, not the ratified configuration (D067a)"


def train_cmd(c: L9.Cell, root: Path) -> str:
    return L.train_cmd(c, root, None, DS_CONFIG)


def gj_cmd(c: L9.Cell, root: Path) -> str:
    guard = ["uv", "run", "--frozen", "python", "scripts/launch_035.py", "trained", "--only", c.tag]
    gj = ["bash", "shell/posttrain/gen_judge.sh", str(c.out("dpo", root) / "model"), "-",
          c.model_id, PROMPT_ID]
    return shlex.join(guard) + " && " + shlex.join(gj)


def submit(opts: list[str], cmd: str, dry: bool) -> str:
    """submit.sh pinned to HEAD (launch_009.head's check, plus this launcher and manifest)."""
    commit = L9.head()
    own = ["scripts/launch_035.py", str(MANIFEST), SFT_DS]
    r = subprocess.run(["git", "status", "--porcelain", "--", *own], cwd=L.REPO_ROOT,
                       capture_output=True, text=True, check=True)
    assert not r.stdout.strip(), f"commit first: {r.stdout}"
    return L.submit(opts, cmd, dry, commit)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["submit", "status", "trained"])
    ap.add_argument("--only", nargs="*", help="cell tags")
    ap.add_argument("--dry-run", action="store_true", help="print sbatch scripts only")
    a = ap.parse_args()
    cs = cells()
    if a.only:
        unknown = set(a.only) - {c.tag for c in cs}
        assert not unknown, f"unknown cells {sorted(unknown)}"
        cs = [c for c in cs if c.tag in a.only]
    root = sanityb.MODELS_ROOT

    if a.phase == "trained":  # in-job guard of the chained gen + judge job
        bad = {c.tag: L.status(c, root) for c in cs if L.status(c, root) != "done"}
        if bad:
            sys.exit(f"training not finished: {bad}")
        return
    if a.phase == "status":
        for c in cs:
            st = L.status(c, root)
            gj = L.genjudge_status(c, root, PROMPT_ID) if st == "done" else "-"
            print(f"{c.tag}\t{c.model_id}\ttrain={st}\tgenjudge={gj}")
        return

    check_ds()
    in_queue = L.queued()
    for c in cs:
        tname, gname = f"{PROMPT_ID}-train-{c.tag}", f"{PROMPT_ID}-gj-{c.tag}"
        if tname in in_queue or gname in in_queue:
            print(f"skip {c.tag}: a job of this cell is queued or running")
            continue
        st = L.status(c, root)
        after = None
        if st == "todo":
            opts = ["--gpus", "8", "--name", tname, "--time", "20:00:00"]
            after = submit(opts, train_cmd(c, root), a.dry_run)
        elif st != "done":
            print(f"skip {c.tag}: training {st} ({c.out('dpo', root)}); inspect by hand")
            continue
        elif (gst := L.genjudge_status(c, root, PROMPT_ID)) != "todo":
            print(f"skip {c.tag}: generations {gst}")
            continue
        opts = ["--gpus", "1", "--time", "12:00:00", "--name", gname,
                *(["--after", after] if after else [])]
        submit(opts, gj_cmd(c, root), a.dry_run)


if __name__ == "__main__":
    main()
