"""Submit the 8 post-training cells of runs/009: the launch_006 logic applied to the two 1B
stage-2 bases in data/manifests/grid_009.json, seeds {1, 2, 3, 4}. Runs on the login
node; each job goes through shell/nchc/submit.sh (snapshot of HEAD, so commit first).

    uv run --frozen python scripts/launch_009.py submit [--only TAG ...] [--dry-run]
    uv run --frozen python scripts/launch_009.py status

Per cell, one SFT -> DPO training job (8 GPUs, `launch_006.train_cmd`) and one generation + judge_v1
job (1 GPU, `gen_judge.sh`, prompt id 009) chained on it with `--after` (afterany). The gen + judge
job first runs `launch_009.py trained --only TAG --manifest-sha256 SHA` and exits non-zero without
generating unless the manifest is the one submitted, the cell's training is `done` and its
generation directory is `todo` (never stale or partial outputs, which gen_judge.sh would reuse).
Skip-if-done is the grid manifest's own rule (`grid.stage_state` / `train_state` / `genjudge_state`:
lineage, config and config sha256 from provenance, label row count), a cell whose training is `todo`
but whose generation directory exists is refused (outputs of an earlier checkpoint), and a job name
already in `squeue --me` is never submitted twice. Every job of one invocation is pinned to one
commit (`provenance.git_commit`: the whole code tree clean), computed once.

Only the cells' source differs from launch_006 (a grid manifest whose bases are converted
castor/ checkpoints); configs and post-training seeds are asserted equal to castor.sanityb's.
"""

import argparse
import hashlib
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from castor import evalset, grid, provenance, sanityb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_006 as L

MANIFEST = L.REPO_ROOT / "data" / "manifests" / "grid_009.json"
PROMPT_ID = "009"


@dataclass(frozen=True)
class Cell:
    """A grid.Cell with sanityb.Cell's interface, so launch_006's functions apply unchanged."""
    g: grid.Cell

    @property
    def k(self) -> int:
        return self.g.k

    @property
    def tag(self) -> str:
        return self.g.tag

    @property
    def base_ref(self) -> str:
        return self.g.base.ref

    @property
    def base_dir(self) -> Path:
        return self.g.base.base_dir

    @property
    def model_id(self) -> str:
        return self.g.model_id

    def out(self, stage: str, root: Path = sanityb.MODELS_ROOT) -> Path:
        return self.g.out(stage, root)


def cells(manifest: Path = MANIFEST) -> list[Cell]:
    """Raises (not `assert`, which -O strips) unless the manifest uses sanityb's configs/seeds."""
    g = grid.load(manifest)
    if g.prompt_id != PROMPT_ID:
        raise ValueError(f"{manifest}: prompt_id {g.prompt_id}")
    if (g.sft_config, g.dpo_config) != (sanityb.SFT_CONFIG, sanityb.DPO_CONFIG):
        raise ValueError(f"{manifest}: 009 must use 006's post-training configs")
    if g.posttrain_seeds != sanityb.POSTTRAIN_SEEDS:
        raise ValueError(f"{manifest}: posttrain_seeds {g.posttrain_seeds}")
    return [Cell(c) for c in g.cells()]


def manifest_sha(manifest: Path = MANIFEST) -> str:
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def train_st(c: Cell, root: Path) -> str:
    """todo | done | failed-sft | failed-dpo (grid.train_state; raises on another cell's or
    another config's provenance)."""
    return grid.train_state(grid.stage_state(c.g, root), in_queue=False)


def gj_st(c: Cell, root: Path, n_prompts: int) -> str:
    """todo | done | stale | failed (grid.genjudge_state: label rows counted). Training todo
    with a generation directory present is `stale`: outputs of an earlier checkpoint."""
    if train_st(c, root) != "done":
        return "stale" if c.g.gen_dir(sanityb.RUNS).exists() else "todo"
    return grid.genjudge_state(c.g, n_prompts, False, root, sanityb.RUNS)


def gj_cmd(c: Cell, root: Path, sha: str) -> str:
    guard = ["uv", "run", "--frozen", "python", "scripts/launch_009.py", "trained", "--only", c.tag,
             "--manifest-sha256", sha]
    gj = ["bash", "shell/posttrain/gen_judge.sh", str(c.out("dpo", root) / "model"), "-",
          c.model_id, PROMPT_ID]
    return shlex.join(guard) + " && " + shlex.join(gj)


def head() -> str:
    """The one commit every job of this invocation runs (submit.sh --commit), computed once in
    main(). `provenance.git_commit`: HEAD iff the whole code tree submit.sh snapshots is clean
    (castor/ scripts/ configs/ shell/ pyproject.toml uv.lock), so no
    uncommitted edit to code the jobs run is silently left out. The manifest lives under data/,
    which jobs read through a symlink to the live tree: it must be committed here, and the gen +
    judge guard checks its sha256 (`--manifest-sha256`)."""
    commit = provenance.git_commit(L.REPO_ROOT)
    r = subprocess.run(["git", "status", "--porcelain", "--", str(MANIFEST)], cwd=L.REPO_ROOT,
                       capture_output=True, text=True, check=True)
    if r.stdout.strip():
        raise RuntimeError(f"commit the manifest first: {r.stdout}")
    return commit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["submit", "status", "trained"])
    ap.add_argument("--only", nargs="*", help="cell tags")
    ap.add_argument("--dry-run", action="store_true", help="print sbatch scripts only")
    ap.add_argument("--manifest-sha256", help="trained: the manifest the job was submitted with")
    a = ap.parse_args()
    cs = cells()
    if a.only:
        unknown = set(a.only) - {c.tag for c in cs}
        assert not unknown, f"unknown cells {sorted(unknown)}"
        cs = [c for c in cs if c.tag in a.only]
    root = sanityb.MODELS_ROOT
    n_prompts = len(evalset.load_eval())

    if a.phase == "trained":  # in-job guard of the chained gen + judge job
        if a.manifest_sha256 != manifest_sha():
            sys.exit(f"{MANIFEST} changed since submission (sha256 {manifest_sha()})")
        bad = {}
        for c in cs:
            st = train_st(c, root)
            gst = gj_st(c, root, n_prompts) if st == "done" else "-"
            if st != "done" or gst != "todo":
                bad[c.tag] = f"train={st} genjudge={gst}"
        if bad:
            sys.exit(f"not generating: {bad}")
        return
    if a.phase == "status":
        for c in cs:
            st = train_st(c, root)
            print(f"{c.tag}\t{c.model_id}\ttrain={st}\tgenjudge={gj_st(c, root, n_prompts)}")
        return

    commit, sha = head(), manifest_sha()
    in_queue = L.queued()
    for c in cs:
        tname, gname = f"{PROMPT_ID}-train-{c.tag}", f"{PROMPT_ID}-gj-{c.tag}"
        if tname in in_queue or gname in in_queue:
            print(f"skip {c.tag}: a job of this cell is queued or running")
            continue
        st, gst = train_st(c, root), gj_st(c, root, n_prompts)
        if gst != "todo":
            print(f"skip {c.tag}: generations {gst} ({c.g.gen_dir(sanityb.RUNS)})"
                  + ("; move them aside by hand" if gst in ("stale", "failed") else ""))
            continue
        after = None
        if st == "todo":
            opts = ["--gpus", "8", "--name", tname, "--time", "20:00:00"]
            after = L.submit(opts, L.train_cmd(c, root, None), a.dry_run, commit=commit)
        elif st != "done":
            print(f"skip {c.tag}: training {st} ({c.out('dpo', root)}); inspect by hand")
            continue
        opts = ["--gpus", "1", "--time", "12:00:00", "--name", gname,
                *(["--after", after] if after else [])]
        try:
            L.submit(opts, gj_cmd(c, root, sha), a.dry_run, commit=commit)
        except RuntimeError:
            if after:
                print(f"{c.tag}: training job {after} is queued WITHOUT its gen + judge job; "
                      "rerun submit once it has finished", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
