"""Post-training grids over DataDecide bases: the cell list from a manifest, and each cell's
state read from provenance, Slurm and its log.

A manifest (`data/manifests/grid_<id>.json`, written by `scripts/grid.py manifest`) fixes
    prompt_id, sft_config, dpo_config, posttrain_seeds,
    bases: [{repo, revision[, base_dir][, posttrain_seeds]}][, variant]
(a base's own posttrain_seeds replace the grid's for that base; a `variant`, for a recipe other
than the reference one, puts the grid's models under <models root>/<variant>/ and appends
-<variant> to its model ids, so its cells never share a directory or an id with the reference
grid's) and one cell = base ×
post-training seed k (k → open-instruct SFT seed k, DPO seed 110 + k, castor.posttrain.oi_seed).
Every cell gets the same SFT → DPO chain (shell/posttrain/train.sh) and one gen + judge_v1 job
on its DPO model (shell/posttrain/gen_judge.sh).

Skip-if-done is keyed on provenance, never on a path alone. Training is done when both stages'
provenance has `finished`, the SFT names this cell's converted base (`base_model`), its
REPO@revision (`base_ref`), seed k and the current SFT config sha256, and the DPO names this
SFT's model, seed k and the current DPO config sha256; under a control, both also name the
control's launcher configuration (`check_launcher`). A provenance that names anything else is an
error (a directory holding another cell's run), never a skip. Gen + judge is done when the
generation provenance covers the full panel (limit null) from this cell's current DPO
checkpoint and labels.jsonl has one row per (prompt, sample incl. greedy).

States (training): todo · queued · running · done · failed-sft · failed-dpo. A stage with a
provenance without `finished` and no job in Slurm under its name has failed; so has a cell with
no provenance whose training the ledger says was submitted. A failed stage directory is moved
aside (never deleted) and resubmitted within the retry cap; a failed DPO resubmits the DPO stage
only. Running cells whose steady step rate (seconds per optimizer step after RATE_FROM steps)
exceeds SLOW_S_PER_STEP are flagged SLOW.
States (gen + judge): waiting (training not done) · todo · queued · running · done · stale
(generations from another checkpoint: never resubmitted over) · failed. A failed full-panel
gen + judge whose generation is complete (`generation_complete`: generate.py writes its
provenance last) reruns the judge stages only; a limited (dry-run) one reruns both.

Driver: `scripts/grid.py launch` writes a control file (<runs>/<prompt>/driver[-smoke]/
control.json: the manifest, the commit every job runs, the job budget schedule, retries, node
exclusions) and calls `refill`. Every job the driver submits calls `refill` when it starts and
when it ends, and a job Slurm kills submits a small refill job from its SIGTERM trap. `refill`
takes a lock (`lock`), surveys every cell, submits within the budget at the pinned commit,
retries failed cells (excluding the node they failed on) and rewrites status.txt / status.json.
A crontab line on the login node (`cron_line`, `refill` every 30 minutes) covers node failures;
it is removed once every enabled cell is labelled or the control is disabled.

The ledger (submissions.json) lists every job the driver submitted per cell and starting stage:
"<tag>|sft" (a full SFT -> DPO chain), "<tag>|dpo" (a DPO-only retry), "<tag>|gj". An entry is
written as "pending-<token>" before anything changes on disk or in Slurm and gets its job id
after sbatch answers; it is removed only if Slurm refused the job and the moved directories were
put back. A refill killed at any point therefore leaves at worst a pending entry, which counts
as an attempt: a cell never reruns uncounted. Retries used = entries − 1 for sft and gj, entries
for dpo.

Stages: an optional stages file (data/manifests/grid_<id>_stages.json) assigns every cell to the
first named stage whose post-training seeds and base repos match (repos null = every base;
`repos_from` = the `recipes` list of another manifest). Cells are submitted in stage order and
only for the stages the control enables; a grid without a stages file is one stage, "all".
"""

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from castor import posttrain, sanityb

ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = Path("/work/harrychang08/castor/models/posttrain")
RUNS = Path("/work/harrychang08/castor/runs")
K = 20  # sampled responses per prompt, plus one greedy pass
SLOW_S_PER_STEP = 4.0  # healthy SFT / DPO stages run at 1.6-1.9 s per step
RATE_FROM = 200  # steps: the rate is measured after warm-up and caching
EXCLUDE_FILE = ROOT / "shell" / "nchc" / "exclude_nodes"  # submit.sh reads the same file


def read_exclude(path: Path = EXCLUDE_FILE) -> str:
    """The default node exclusion (one non-comment line; the reasons are in the file). Read when
    a launch needs it, not at import: an analysis that never submits must not depend on it."""
    lines = [x.strip() for x in path.read_text().splitlines()
             if x.strip() and not x.startswith("#")]
    assert len(lines) == 1, f"{path}: expected one node list, got {lines}"
    return lines[0]


TRAIN_TIME = "20:00:00"  # a chain takes 8.2-8.4 h
GJ_TIME = "12:00:00"
STALL_S = 45 * 60  # a running job whose log has not changed for this long is flagged STALLED
LOCK_STALE_S = 600  # a refill takes seconds; a lock older than this belongs to a dead process
# allenai/… = released DataDecide checkpoints (step<N>-seed-<s>); castor/… = our own pretraining
# runs, converted by castor.localckpt with revision = the run's seed name
_REPO_RE = re.compile(r"^(?:allenai|castor)/DataDecide-(?P<recipe>.+)-(?P<size>\d+[MB])$")
_REV_RE = re.compile(r"^(?:(?P<step>step\d+)-seed-)?(?P<seed>.+)$")


@dataclass(frozen=True)
class Base:
    repo: str
    revision: str
    base_dir_override: str | None = None
    seeds: tuple[int, ...] | None = None  # this base's post-training seeds (default: the grid's)

    @property
    def recipe(self) -> str:
        return _REPO_RE.match(self.repo)["recipe"]

    @property
    def size(self) -> str:
        return _REPO_RE.match(self.repo)["size"]

    @property
    def pretrain_seed(self) -> str:
        return _REV_RE.match(self.revision)["seed"]

    @property
    def ref(self) -> str:
        return f"{self.repo}@{self.revision}"

    @property
    def base_dir(self) -> Path:
        if self.base_dir_override:
            return Path(self.base_dir_override)
        from castor import convert  # deferred: see check_base

        return convert.converted_dir(self.repo, self.revision)

    @property
    def group(self) -> str:
        return f"dd-{self.recipe}-{self.size}-seed-{self.pretrain_seed}"


@dataclass(frozen=True)
class Grid:
    prompt_id: str
    sft_config: str
    dpo_config: str
    posttrain_seeds: tuple[int, ...]
    bases: tuple[Base, ...]
    variant: str | None = None

    def cells(self) -> list["Cell"]:
        return [Cell(self, b, k) for b in self.bases
                for k in (b.seeds if b.seeds is not None else self.posttrain_seeds)]


@dataclass(frozen=True)
class Cell:
    grid: Grid
    base: Base
    k: int

    @property
    def tag(self) -> str:
        return f"{self.base.recipe}-{self.base.size}-{self.base.pretrain_seed}-s{self.k}"

    @property
    def model_id(self) -> str:
        b = self.base
        v = f"-{self.grid.variant}" if self.grid.variant else ""
        return f"dd-{b.recipe}-{b.size.lower()}-{b.pretrain_seed}-dpo-seed{self.k}{v}"

    def job_name(self, phase: str, smoke: bool = False) -> str:
        return f"{self.grid.prompt_id}-{'smoke-' if smoke else ''}{phase}-{self.tag}"

    def out(self, stage: str, root: Path = MODELS_ROOT) -> Path:
        return root / self.base.group / stage / f"seed{self.k}"

    def gen_dir(self, runs: Path = RUNS, smoke: bool = False) -> Path:
        return runs / self.model_id / (self.grid.prompt_id + ("-smoke" if smoke else ""))


@dataclass(frozen=True)
class Stage:
    name: str
    seeds: tuple[int, ...]
    repos: frozenset[str] | None  # None = every base

    def matches(self, cell: "Cell") -> bool:
        return cell.k in self.seeds and (self.repos is None or cell.base.repo in self.repos)


ALL_STAGE = Stage("all", (), None)  # a grid without a stages file


def load_stages(path: Path | str | None, g: "Grid") -> list[Stage]:
    """The stages of a grid, in submission order; every cell must fall in exactly one (the first
    that matches). No file = one stage "all"."""
    if path is None:
        return [ALL_STAGE]
    m = json.loads((ROOT / path).read_text())
    assert m["prompt_id"] == g.prompt_id, f"{path} is for prompt {m['prompt_id']}"
    out = []
    for s in m["stages"]:
        repos = s.get("repos")
        if "repos_from" in s:
            assert repos is None, f"stage {s['name']}: repos and repos_from"
            repos = json.loads((ROOT / s["repos_from"]).read_text())["recipes"]
        out.append(Stage(s["name"], tuple(s["seeds"]),
                         None if repos is None else frozenset(repos)))
    assert len({s.name for s in out}) == len(out), f"{path}: duplicate stage names"
    assert "all" not in {s.name for s in out}, "'all' is reserved for a grid without stages"
    known = {b.repo for b in g.bases}
    for s in out:
        assert s.repos is None or s.repos <= known, f"stage {s.name}: repos not in the grid"
    for c in g.cells():
        stage_of(c, out)
    return out


def stage_of(cell: "Cell", stages: list[Stage]) -> str:
    if stages == [ALL_STAGE]:
        return "all"
    for s in stages:
        if s.matches(cell):
            return s.name
    raise AssertionError(f"{cell.tag} is in no stage")


def is_driver_job(name: str, prompt_id: str) -> bool:
    """A job the grid driver submitted (training, gen + judge, smoke, trap refill): these, and
    no other job named <prompt>-…, count against the grid's budget."""
    return name == f"{prompt_id}-refill" or any(
        name.startswith(f"{prompt_id}-{p}-") for p in ("train", "gj", "smoke"))


def check_instrument(g: Grid, prompt_id: str) -> None:
    """Raise ValueError unless `g` is a `prompt_id` grid on the reference post-training recipe
    (sanityb's SFT and DPO configs). Post-training seeds are checked by the caller. Explicit
    raises, not asserts, so the check holds under any Python flags."""
    if g.prompt_id != prompt_id:
        raise ValueError(f"manifest prompt_id {g.prompt_id} != {prompt_id}")
    if (g.sft_config, g.dpo_config) != (sanityb.SFT_CONFIG, sanityb.DPO_CONFIG):
        raise ValueError(f"{prompt_id} must use 006's post-training configs")


def checked_cells(manifest: Path, prompt_id: str, base_ref: str) -> list[Cell]:
    """The cells of a one-base manifest on the reference instrument (check_instrument,
    sanityb.POSTTRAIN_SEEDS) with exactly the base `base_ref` (repo@revision) and no per-base
    override. Shared by the launchers and the analysis scripts, so both agree on the cell list,
    hence on model ids and output paths. Raises ValueError on any other manifest."""
    g = load(manifest)
    check_instrument(g, prompt_id)
    if g.posttrain_seeds != sanityb.POSTTRAIN_SEEDS:
        raise ValueError(f"posttrain_seeds {g.posttrain_seeds} != 006's {sanityb.POSTTRAIN_SEEDS}")
    if [b.ref for b in g.bases] != [base_ref]:
        raise ValueError(f"bases {[b.ref for b in g.bases]} != [{base_ref}]")
    # the per-base overrides `cells()` honours would silently change the cell list or the
    # weights trained from, past the grid-level checks above
    (b,) = g.bases
    if b.seeds is not None:
        raise ValueError(f"{base_ref}: base-level posttrain_seeds {b.seeds}")
    if b.base_dir_override is not None:
        raise ValueError(f"{base_ref}: base_dir override {b.base_dir_override}")
    return g.cells()


def models_root(g: Grid, smoke: bool) -> Path:
    """Where a grid's training runs live: a smoke under smoke/<prompt>, a variant under its name."""
    root = MODELS_ROOT / "smoke" / g.prompt_id if smoke else MODELS_ROOT
    return root / g.variant if g.variant else root


def load(manifest: Path) -> Grid:
    m = json.loads(Path(manifest).read_text())
    bases = tuple(Base(b["repo"], b["revision"], b.get("base_dir"),
                       tuple(b["posttrain_seeds"]) if "posttrain_seeds" in b else None)
                  for b in m["bases"])
    assert len({b.ref for b in bases}) == len(bases), "duplicate base in manifest"
    variant = m.get("variant")
    assert variant is None or re.fullmatch(r"[a-z0-9]+", variant), f"variant {variant!r}"
    # another recipe's runs would otherwise land in the reference grid's directories
    assert variant or (m["sft_config"], m["dpo_config"]) == (sanityb.SFT_CONFIG,
                                                             sanityb.DPO_CONFIG), \
        f"{manifest}: a grid off the reference configs needs a variant"
    g = Grid(m["prompt_id"], m["sft_config"], m["dpo_config"], tuple(m["posttrain_seeds"]),
             bases, variant)
    ids = [c.model_id for c in g.cells()]
    assert len(set(ids)) == len(ids), "model ids collide"
    return g


def check_base(base: Base) -> dict:
    """The converted copy must say it is REPO@revision, converted by today's converter."""
    # castor.convert imports torch + transformers (~2 s), which the CPU-only analyses and the
    # login-node launchers that import this module never need. The import is deferred here, not
    # inside convert.py, because convert.py's sha256 gates the reuse of every converted
    # checkpoint (converter_sha256): editing that file would invalidate all of them.
    from castor import convert

    prov = json.loads((base.base_dir / "provenance.json").read_text())
    assert prov["source"] == base.ref, f"{base.base_dir}: {prov['source']} != {base.ref}"
    assert prov["converter_sha256"] == convert.converter_sha256(), \
        f"{base.base_dir} was converted by different code"
    return prov


# ---- state from provenance ------------------------------------------------------------------


def _prov(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _config_sha(config: str) -> str:
    return posttrain.sha256_file(ROOT / config)


def launcher_of(prov: dict, stage: str) -> dict:
    """The launcher configuration a run was trained on: its DeepSpeed config sha256 and its
    exported environment without the CPU count (which follows the job's allocation). A
    provenance written before commit 335ecd2 records neither: train.sh then always used the
    stage's open-instruct default and exported LAUNCH_ENV_DEFAULTS, so that is what it reads as
    (its recorded DeepSpeed path must be the default)."""
    oi = prov["open_instruct"]
    if "deepspeed_config_sha256" not in oi:  # before 335ecd2
        assert oi["deepspeed_config"] == posttrain.OI_DS_CONFIG[stage], \
            f"pre-024 provenance with a non-default DeepSpeed config {oi['deepspeed_config']}"
        return {"deepspeed_config_sha256": posttrain.sha256_file(
                    posttrain.OI_DIR / posttrain.OI_DS_CONFIG[stage]),
                "env": dict(posttrain.LAUNCH_ENV_DEFAULTS)}
    env = {k: v for k, v in prov["launcher"]["env"].items() if k != "BEAKER_ASSIGNED_CPU_COUNT"}
    return {"deepspeed_config_sha256": oi["deepspeed_config_sha256"], "env": env}


def expected_launcher(stage: str, launcher: dict) -> dict:
    """launcher_of's reading of a run launched with the control's `launcher` = {"ds_config":
    {stage: path} | None, "launch_env": [KEY=VALUE, ...] | None}, resolved as train.sh does."""
    ds = (launcher.get("ds_config") or {}).get(stage)
    env = posttrain.launch_env(0, launcher.get("launch_env"))
    env.pop("BEAKER_ASSIGNED_CPU_COUNT")
    return {"deepspeed_config_sha256": posttrain.sha256_file(posttrain.ds_config_path(stage, ds)),
            "env": env}


def resolve_launcher(launcher: dict | None) -> dict[str, dict] | None:
    """The control's launcher configuration resolved per stage, {stage: expected_launcher}: once
    per survey (it hashes the DeepSpeed config), not once per cell and stage. None = no check."""
    return None if launcher is None else {st: expected_launcher(st, launcher) for st in ("sft", "dpo")}


def check_launcher(prov: dict, stage: str, want: dict, where: Path) -> None:
    """A run's launcher configuration is the control's (launcher_of == `want` =
    resolve_launcher(...)[stage]): same DeepSpeed config bytes, same exported environment."""
    got = launcher_of(prov, stage)
    assert got == want, f"{where}: launcher configuration {got} is not the control's {want}"


def stage_state(cell: Cell, root: Path = MODELS_ROOT,
                launcher: dict[str, dict] | None = None) -> dict:
    """Per stage None (no provenance) | 'started' | 'finished', with the recorded Slurm jobs.
    Raises if a provenance belongs to another cell or another config, or (with `launcher` =
    resolve_launcher(the control's); check_launcher) was trained on another launcher config."""
    sft = _prov(cell.out("sft", root) / "provenance.json")
    dpo = _prov(cell.out("dpo", root) / "provenance.json")
    out = {"sft": None, "dpo": None, "jobs": {}}
    if sft is not None:
        w = cell.out("sft", root)
        assert sft["stage"] == "sft", f"{w}: stage {sft['stage']}"
        assert sft["post_training_seed_k"] == cell.k, f"{w}: seed {sft['post_training_seed_k']}"
        assert sft["base_model"] == str(cell.base.base_dir), f"{w}: base {sft['base_model']}"
        assert sft.get("base_ref") == cell.base.ref, f"{w}: base_ref {sft.get('base_ref')}"
        assert sft["config"] == cell.grid.sft_config, f"{w}: config {sft['config']}"
        assert sft["config_sha256"] == _config_sha(cell.grid.sft_config), \
            f"{w}: SFT config changed since this run"
        if launcher is not None:
            check_launcher(sft, "sft", launcher["sft"], w)
        out["sft"] = "finished" if "finished" in sft else "started"
        out["jobs"]["sft"] = sft.get("slurm_job")
    if dpo is not None:
        w = cell.out("dpo", root)
        assert sft is not None, f"{w}: DPO provenance without SFT"
        assert dpo["stage"] == "dpo", f"{w}: stage {dpo['stage']}"
        assert dpo["post_training_seed_k"] == cell.k, f"{w}: seed {dpo['post_training_seed_k']}"
        assert dpo["base_model"] == str(cell.out("sft", root) / "model"), f"{w}: base differs"
        assert dpo["config"] == cell.grid.dpo_config, f"{w}: config {dpo['config']}"
        assert dpo["config_sha256"] == _config_sha(cell.grid.dpo_config), \
            f"{w}: DPO config changed since this run"
        if launcher is not None:
            check_launcher(dpo, "dpo", launcher["dpo"], w)
        out["dpo"] = "finished" if "finished" in dpo else "started"
        out["jobs"]["dpo"] = dpo.get("slurm_job")
    return out


def train_state(st: dict, in_queue: bool) -> str:
    """todo | queued-or-running | done | failed-sft | failed-dpo (see module doc)."""
    if st["sft"] == "finished" and st["dpo"] == "finished":
        return "done"
    if in_queue:
        return "in-queue"
    if st["sft"] is None:
        return "todo"
    if st["sft"] == "started":
        return "failed-sft"
    return "failed-dpo"  # SFT finished; DPO started and not finished, or never started


def n_label_rows(n_prompts: int) -> int:
    return n_prompts * (K + 1)


def genjudge_state(cell: Cell, n_prompts: int, in_queue: bool, root: Path = MODELS_ROOT,
                   runs: Path = RUNS, limit: int | None = None,
                   cache: dict | None = None) -> str:
    """`limit` = the generation's prompt limit (None = the full panel; a number only for a
    dry run, whose outputs go to <prompt>-smoke). `cache` memoises label counts."""
    out = cell.gen_dir(runs, smoke=limit is not None)
    if in_queue:
        return "in-queue"
    gen = _prov(out / "provenance.json")
    labels = out / "judge_v1" / "labels.jsonl"
    if gen is not None and gen.get("limit") == limit:
        ckpt = str(Path(_prov(cell.out("dpo", root) / "provenance.json")["checkpoint"]).resolve())
        if gen.get("model_resolved") != ckpt:
            return "stale"
        if labels.exists() and count_lines(labels, cache) == n_label_rows(n_prompts):
            return "done"
        return "failed"
    return "failed" if out.exists() else "todo"


def generation_complete(cell: Cell, root: Path = MODELS_ROOT, runs: Path = RUNS,
                        limit: int | None = None) -> bool:
    """The generation (not the judge) is complete and from this cell's current DPO checkpoint:
    generate.py writes provenance.json after samples.jsonl and greedy.jsonl."""
    gen = _prov(cell.gen_dir(runs, smoke=limit is not None) / "provenance.json")
    if gen is None or gen.get("limit") != limit:
        return False
    ckpt = str(Path(_prov(cell.out("dpo", root) / "provenance.json")["checkpoint"]).resolve())
    return gen.get("model_resolved") == ckpt


def count_lines(path: Path, cache: dict | None = None) -> int:
    """Lines in a file; with `cache`, memoised on (path, size, mtime_ns) (labels are ≈ 57 MB)."""
    st = path.stat()
    key = f"{path}|{st.st_size}|{st.st_mtime_ns}"
    if cache is not None and key in cache:
        return cache[key]
    with open(path, "rb") as f:
        n = sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))
    if cache is not None:
        cache[key] = n
    return n


def attempts(path: Path) -> list[str]:
    """Earlier attempts moved aside next to `path` (`<name>.failed-<job>`), by job id."""
    return sorted(p.name.rsplit(".failed-", 1)[1] for p in path.parent.glob(f"{path.name}.failed-*"))


def write_atomic(path: Path, text: str) -> None:
    """Readers never see a half-written file (rename is atomic on one filesystem)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


LEDGER = "submissions.json"
PHASES = ("sft", "dpo", "gj")


def read_ledger(d: Path) -> dict[str, list[str]]:
    """The driver's submission ledger (module doc): {"<tag>|<sft|dpo|gj>": [job ids or
    "pending-<token>"]}."""
    p = d / LEDGER
    return json.loads(p.read_text()) if p.exists() else {}


def ledger_pending(d: Path, key: str) -> str:
    """Write-ahead: append a pending entry before submitting; returns it (call under the lock)."""
    assert key.rsplit("|", 1)[1] in PHASES, key
    entry = f"pending-{uuid.uuid4().hex[:12]}"
    led = read_ledger(d)
    led.setdefault(key, []).append(entry)
    write_atomic(d / LEDGER, json.dumps(led, indent=1) + "\n")
    return entry


def ledger_resolve(d: Path, key: str, entry: str, job: str | None) -> None:
    """Replace a pending entry by its job id, or remove it (job None: Slurm refused, nothing
    changed). Call under the lock."""
    led = read_ledger(d)
    v = led.get(key, [])
    i = v.index(entry)
    if job is None:
        del v[i]
    else:
        v[i] = job
    led = {k: x for k, x in led.items() if x}
    write_atomic(d / LEDGER, json.dumps(led, indent=1) + "\n")


def retries_used(ledger: dict, tag: str, phase: str) -> int:
    n = len(ledger.get(f"{tag}|{phase}", []))
    return n if phase == "dpo" else max(n - 1, 0)


def last_job(ledger: dict, *keys: str) -> str | None:
    """The newest submitted job id under the first key that has one (pending entries skipped)."""
    for k in keys:
        ids = [j for j in ledger.get(k, []) if not j.startswith("pending-")]
        if ids:
            return ids[-1]
    return None


def move_aside(path: Path, job: str | None) -> Path | None:
    """Rename an unfinished run directory to `<name>.failed-<job>`; never deletes."""
    if not path.exists():
        return None
    dest = path.with_name(f"{path.name}.failed-{job or 'nojob'}")
    n = 2
    while dest.exists():
        dest = path.with_name(f"{path.name}.failed-{job or 'nojob'}-{n}")
        n += 1
    path.rename(dest)
    return dest


# ---- the control file, lock and budget ------------------------------------------------------


def driver_dir(prompt_id: str, smoke: bool, runs: Path = RUNS) -> Path:
    return runs / prompt_id / ("driver-smoke" if smoke else "driver")


def max_jobs_now(control: dict, now: datetime | None = None) -> int:
    """control["max_jobs"] = [[None, n0], ["2026-10-02T16:00:00", n1], ...]: the last step
    whose time has passed (local time, as Slurm prints it)."""
    now = now or datetime.now()  # noqa: DTZ005
    n = None
    for t, v in control["max_jobs"]:
        if t is None or datetime.fromisoformat(t) <= now:
            n = v
    assert n is not None, "max_jobs needs a first step with time null"
    return n


class LockLost(RuntimeError):
    """The driver lock was taken over while this holder worked."""


@contextmanager
def lock(d: Path, stale_s: float = LOCK_STALE_S, wait_s: float = 600):
    """Mutual exclusion across nodes on /work (mkdir is atomic). Yields `beat()`, which the holder
    calls while it works to refresh the lock's age. A lock not refreshed for stale_s is taken over
    (its holder died); otherwise wait up to wait_s, then raise TimeoutError. On release the lock is
    removed only if it still holds this holder's token, so a holder that was taken over never
    removes its successor's lock. `beat()` raises LockLost once the token is not this holder's,
    so call it before every change the lock protects: two waiters that both break the same stale
    lock (the age check and the rename are not atomic) then end with one holder, since only one
    token survives in `holder` and the other process stops at its next beat."""
    lk = d / "lock"
    token = uuid.uuid4().hex
    t0 = time.time()
    while True:
        try:
            lk.mkdir()
            break
        except FileExistsError:
            try:
                age = time.time() - lk.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > stale_s:
                try:  # kept, not deleted, so a takeover can be seen afterwards
                    os.rename(lk, d / f"lock.stale-{int(time.time())}-{os.getpid()}")
                except FileNotFoundError:
                    pass  # another process took it over first
                continue
            if time.time() - t0 > wait_s:
                raise TimeoutError(f"{lk} held for {age:.0f} s") from None
            time.sleep(2)

    def beat() -> None:
        try:
            ours = (lk / "holder").read_text().split()[0] == token
        except (FileNotFoundError, IndexError):
            ours = False
        if not ours:  # taken over (we stalled past stale_s, or lost a takeover race): stop
            raise LockLost(f"{lk} no longer holds this holder's token")
        os.utime(lk)

    try:
        (lk / "holder").write_text(f"{token} {os.uname().nodename} pid {os.getpid()} "
                                   f"job {os.environ.get('SLURM_JOB_ID', '-')}\n")
        beat()
        yield beat
    finally:
        try:
            mine = (lk / "holder").read_text().split()[0] == token
        except (FileNotFoundError, IndexError):
            mine = False
        if mine:
            for f in lk.iterdir():
                f.unlink()
            lk.rmdir()


# ---- cron on the login node ------------------------------------------------------------------


def cron_tag(prompt_id: str, smoke: bool) -> str:
    """The comment that marks this grid's crontab line."""
    return f"# castor-grid {prompt_id}{'-smoke' if smoke else ''}"


CRON_TOOLS = ("uv", "squeue", "sbatch", "sacct", "git", "crontab")  # what a cron refill runs


def cron_path(which=shutil.which) -> str:
    """PATH for the cron line, from the absolute paths of CRON_TOOLS resolved now (cron's own
    PATH is /usr/bin:/bin, and a login shell's is not guaranteed)."""
    found = {t: which(t) for t in CRON_TOOLS}
    missing = [t for t, p in found.items() if not p]
    assert not missing, f"not on PATH, the cron line needs their absolute paths: {missing}"
    return ":".join(dict.fromkeys([*(str(Path(p).parent) for p in found.values()),
                                   "/usr/bin", "/bin"]))


def cron_line(prompt_id: str, smoke: bool, control_path: Path, repo: Path, uv: str,
              path: str) -> str:
    """`path` = cron_path() at install time; `uv` its absolute path."""
    log = repo / "logs" / "nchc" / f"{prompt_id}{'-smoke' if smoke else ''}-cron.log"
    cmd = (f"export PATH={path} && cd {repo} && source shell/nchc/env.sh && {uv} run --frozen "
           f"python scripts/grid.py refill --control {control_path}")
    return f"*/30 * * * * /bin/bash -c '{cmd}' >> {log} 2>&1 {cron_tag(prompt_id, smoke)}"


def _crontab_lines() -> list[str]:
    r = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        if "no crontab" in r.stderr:
            return []
        raise RuntimeError(f"crontab -l failed: {r.stderr.strip()}")
    return r.stdout.splitlines()


def _crontab_write(lines: list[str]) -> None:
    subprocess.run(["crontab", "-"], input="".join(f"{x}\n" for x in lines), text=True,
                   check=True)


def install_cron(line: str, tag: str) -> None:
    """Put this grid's line (found by its tag) in the crontab, keeping every other line."""
    keep = [x for x in _crontab_lines() if not x.rstrip().endswith(tag)]
    _crontab_write([*keep, line])


def remove_cron(tag: str) -> bool:
    """Remove this grid's line; True if there was one."""
    lines = _crontab_lines()
    keep = [x for x in lines if not x.rstrip().endswith(tag)]
    if len(keep) == len(lines):
        return False
    _crontab_write(keep)
    return True


def labelled(rows: list[dict]) -> bool:
    """Every cell trained and judged (survey rows); False for no cells (every stage disabled is
    not a finished grid)."""
    return bool(rows) and all(r["train"] == "done" and r["genjudge"] == "done" for r in rows)


# ---- Slurm and logs -------------------------------------------------------------------------


def squeue_list() -> list[dict]:
    """Every job of this user: [{id, name, state, where}] (names need not be unique)."""
    r = subprocess.run(["squeue", "--me", "--noheader", "--format=%i|%j|%T|%R"],
                       capture_output=True, text=True, check=True)
    out = []
    for line in r.stdout.splitlines():
        jid, name, state, where = line.split("|", 3)
        out.append({"id": jid, "name": name, "state": state, "where": where})
    return out


def by_name(jobs: list[dict]) -> dict[str, dict]:
    """{name: job} for looking a cell's job up; with duplicate names a RUNNING job wins, then the
    newest. Count jobs from the list, never from this dict."""
    out: dict[str, dict] = {}
    for j in sorted(jobs, key=lambda j: (j["state"] == "RUNNING", int(j["id"].split("_")[0]))):
        out[j["name"]] = {k: j[k] for k in ("id", "state", "where")}
    return out


def squeue() -> dict[str, dict]:
    """This user's jobs by name: {name: {id, state, node_or_reason}} (see by_name)."""
    return by_name(squeue_list())


def sacct(job_id: str | None = None, name: str | None = None) -> dict:
    """The job's final record; by id, or the latest job with this name (since Sept 2026)."""
    if not job_id and not name:
        return {}
    sel = ["-j", str(job_id)] if job_id else ["--name", name, "-S", "2026-09-01"]
    r = subprocess.run(["sacct", *sel, "-X", "-n", "-P",
                        "-o", "JobID,State,NodeList,Elapsed,ExitCode,AllocTRES"],
                       capture_output=True, text=True, check=False)
    lines = r.stdout.strip().splitlines()
    if not lines:
        return {"job": job_id} if job_id else {}
    jid, state, node, elapsed, code, tres = lines[-1].split("|")
    gpus = [t.split("=", 1)[1] for t in tres.split(",") if t.startswith("gres/gpu=")]
    try:
        hours = elapsed_hours(elapsed)
    except (AssertionError, ValueError):
        hours = None  # the grid survey needs the node only
    return {"job": jid, "state": state, "node": node, "elapsed": elapsed, "exit": code,
            "elapsed_h": hours, "gpus": int(gpus[0]) if gpus else 0}


def elapsed_hours(elapsed: str) -> float:
    """Slurm's Elapsed ([D-]HH:MM:SS, or MM:SS[.ms] for short steps) in hours."""
    d, _, hms = elapsed.rpartition("-")
    parts = [float(x) for x in hms.split(":")]
    assert 2 <= len(parts) <= 3, f"unparsed elapsed {elapsed!r}"
    h, m, s = [0.0] * (3 - len(parts)) + parts
    return int(d or 0) * 24 + h + m / 60 + s / 3600


# open-instruct's log-line format at the pinned commit (`%(asctime)s - %(levelname)s -
# %(name)s - %(message)s` with its default date format). Some scripts keep their own regexes of
# the same format for extra fields; an edit here does not reach those
LOG_LINE_RE = re.compile(r"(\d\d/\d\d/\d{4} \d\d:\d\d:\d\d) - INFO - __main__ -\s+(.*)$",
                         re.MULTILINE)
_STEP_MSG_RE = re.compile(r"Step: (\d+),")


def log_time(ts: str) -> datetime:
    """A LOG_LINE_RE timestamp; the log clock is local and naive, only differences are used."""
    return datetime.strptime(ts, "%m/%d/%Y %H:%M:%S")  # noqa: DTZ007


def step_lines(log_text: str) -> list[tuple[int, datetime]]:
    """(step, time) of every `Step: N,` line of open-instruct's training log, in log order."""
    out = []
    for ts, msg in LOG_LINE_RE.findall(log_text):
        m = _STEP_MSG_RE.match(msg)
        if m:
            out.append((int(m.group(1)), log_time(ts)))
    return out


def step_rate(log_text: str, rate_from: int = RATE_FROM) -> dict | None:
    """Seconds per optimizer step of the current stage (the last run of increasing `Step: N`
    lines open-instruct logs), measured from step `rate_from` to the latest step. None before
    the stage reaches rate_from + 1 steps."""
    runs, cur, prev = [], [], 0
    for n, t in step_lines(log_text):
        if n <= prev and cur:
            runs.append(cur)
            cur = []
        cur.append((n, t))
        prev = n
    if cur:
        runs.append(cur)
    if not runs:
        return None
    stage = runs[-1]
    first = next(((n, t) for n, t in stage if n >= rate_from), None)
    last = stage[-1]
    if first is None or last[0] <= first[0]:
        return {"stage_index": len(runs), "step": last[0], "s_per_step": None}
    s = (last[1] - first[1]).total_seconds() / (last[0] - first[0])
    return {"stage_index": len(runs), "step": last[0], "s_per_step": s,
            "slow": s > SLOW_S_PER_STEP}


def main_repo() -> Path:
    """The main checkout, also when this code runs from a job's snapshot
    worktree: logs and the venv live there, not in the snapshot."""
    r = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                       cwd=ROOT, capture_output=True, text=True, check=True)
    return Path(r.stdout.strip()).parent


def log_path(name: str, job_id: str) -> Path:
    return main_repo() / "logs" / "nchc" / f"{name}-{job_id}.out"
