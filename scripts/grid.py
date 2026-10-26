"""Drive a post-training grid: write its manifest, report every cell's state, and keep the queue
filled within a job budget unattended.

    # the 008 manifest: every 1B recipe except the main one, seeds {1, 2}
    uv run --frozen python scripts/grid.py manifest --prompt-id 008 \
        --source data/manifests/datadecide_1B.json --drop-recipe dclm-baseline --seeds 1 2 \
        --out data/manifests/grid_008.json
    # see the grid (read-only)
    uv run --frozen python scripts/grid.py status --manifest data/manifests/grid_008.json
    # start (or change the budget of) the self-refilling grid, from the login node
    uv run --frozen python scripts/grid.py launch --manifest data/manifests/grid_008.json \
        --stages A B --max-jobs 12 --max-jobs-after 2026-10-02T16:00=16 [--max-retries 1] [--dry-run]
    # enable a further stage later (same pinned commit)
    uv run --frozen python scripts/grid.py stages --manifest data/manifests/grid_008.json --enable C
    # stop BEFORE any scancel of grid jobs (a cancelled job's trap refills and retries)
    uv run --frozen python scripts/grid.py stop --manifest data/manifests/grid_008.json
    # dry run: 50 SFT + 50 DPO steps, gen + judge on the first 40 prompts
    uv run --frozen python scripts/grid.py launch --manifest ... --smoke 50 --gj-limit 40 \
        --dev --stages A --only TAG [TAG ...] --max-jobs 1

`launch` writes the control file (castor/grid.py module doc) with the commit every job will run
(HEAD; uncommitted code is listed and not used) and the enabled stages (a grid with a stages file,
default `<manifest stem>_stages.json` beside the manifest, must name them), and runs one `refill`.
Each submitted job runs `refill` at its start and again at its end (`--ended-job $SLURM_JOB_ID`, so
its own slot counts as free), whatever its exit status; if Slurm kills it (time limit, scancel:
SIGTERM to the batch shell only, which `submit.sh --term-wait` forwards to the command; SIGKILL
after KillWait 60 s), at any point including during a refill, its trap submits a 1-GPU dev job
`<prompt>-refill` that runs `refill` after it has ended (with its own id as `--ended-job`;
`submit.sh --use-reserve` lets it take the account's last slot, which every other submit.sh job
leaves free). A node failure or a SIGKILL skips the trap. `refill` submits, in order: gen + judge
for cells whose training is done, failed cells up to `max_retries` retries per cell and stage (sft,
dpo, gj; write-ahead ledger `submissions.json`, castor/grid.py module doc; unfinished directory
moved to `<dir>.failed-<job>`, never deleted, and moved back if Slurm refuses the job; the failed
job's node joins the exclude list; a failed DPO reruns the DPO only; a failed full-panel gen + judge
with a complete generation reruns the judge only), then new training chains, each group in stage
order and only for enabled stages. It never exceeds the budget of this grid's driver jobs in Slurm
(running + pending; `grid.is_driver_job`: <prompt>-train-/-gj-/-smoke-/-refill, so an analysis job
named <prompt>-… does not count) nor the account cap of 20 (every job of the user) less one slot
kept for a trap's refill job; actions that fail before Slurm (a bad base) do not use the budget.
States and skip-if-done: castor/grid.py. To stop the grid, run `stop` before any scancel.

Cron (castor/grid.py module doc): `launch` (not --dry-run, not --no-cron, on the login node)
installs `*/30 … grid.py refill --control …  # castor-grid <prompt>` in the user's crontab and
records the host in the control; `stop` on that host removes it; the cron's own refill removes it
when every cell of the enabled stages is labelled or the control is disabled; `stages` reinstalls
it; a relaunch must run on the host of an earlier line. See it with `crontab -l` on the recorded
host; its output goes to logs/nchc/<prompt>-cron.log.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from castor import evalset, grid, posttrain, provenance

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_006  # train.sh flag builder (launcher_args)

SUBMIT = grid.ROOT / "shell" / "nchc" / "submit.sh"
ACCOUNT_CAP = 20  # jobs per user, running + pending
# refill leaves one slot of the cap free: a job Slurm kills still counts when its trap submits
# the later refill job, and sbatch refuses that job at the cap
TRAP_RESERVE = 1
DEV_TRAIN_TIME, DEV_GJ_TIME = "03:30:00", "01:00:00"


def cmd_manifest(a) -> None:
    src = json.loads(Path(a.source).read_text())
    own = {}
    for s in a.seeds_for:  # RECIPE=K,K,...: that recipe's bases get these seeds instead
        r, ks = s.split("=")
        own[r] = [int(k) for k in ks.split(",")]
    bases = [{"repo": e["repo"], "revision": e["revision"]} for e in src["entries"]
             if grid.Base(e["repo"], e["revision"]).recipe not in set(a.drop_recipe)]
    for b in bases:
        if (r := grid.Base(b["repo"], b["revision"]).recipe) in own:
            b["posttrain_seeds"] = own[r]
    assert set(own) <= {grid.Base(b["repo"], b["revision"]).recipe for b in bases}, \
        f"--seeds-for names a recipe with no base: {sorted(own)}"
    bases.sort(key=lambda b: (b["repo"], b["revision"]))
    m = {"prompt_id": a.prompt_id, "sft_config": a.sft_config, "dpo_config": a.dpo_config,
         "posttrain_seeds": a.seeds, "bases": bases,
         "source": a.source, "dropped_recipes": a.drop_recipe, "seeds_for": own,
         **({"variant": a.variant} if a.variant else {}), "command": provenance.command()}
    g_path = Path(a.out)
    g_path.write_text(json.dumps(m, indent=1) + "\n")
    g = grid.load(g_path)
    print(f"{g_path}: {len(g.bases)} bases, {len(g.cells())} cells (seeds {list(g.posttrain_seeds)}"
          f"{''.join(f'; {r}: {ks}' for r, ks in own.items())})")


# ---- job commands ---------------------------------------------------------------------------


def launcher(ctl: dict | None) -> dict | None:
    """The control's launcher configuration (castor.grid.check_launcher); a control written
    before --ds-config existed has the current one. None without a control (no check)."""
    if ctl is None:
        return None
    return {"ds_config": ctl.get("ds_config"), "launch_env": ctl.get("launch_env")}


def _train_cmd(cell: grid.Cell, root: Path, from_stage: str, smoke: int | None,
               launch: dict | None = None) -> str:
    """`launch` = launcher(ctl): its DeepSpeed config per stage and launcher environment are
    passed to train.sh unchanged (launch_006.launcher_args); None/empty = the default
    configuration."""
    grid.check_base(cell.base)
    launch = launch or {}
    ds, env = launch.get("ds_config"), launch.get("launch_env")
    extra = ["--set", f"max_train_steps={smoke}"] if smoke is not None else []
    sft = ["bash", "shell/posttrain/train.sh", "sft", cell.grid.sft_config,
           str(cell.base.base_dir), str(cell.k), str(cell.out("sft", root)),
           "--base-ref", cell.base.ref, *extra, *launch_006.launcher_args("sft", ds, env)]
    dpo = ["bash", "shell/posttrain/train.sh", "dpo", cell.grid.dpo_config,
           str(cell.out("sft", root) / "model"), str(cell.k), str(cell.out("dpo", root)), *extra,
           *launch_006.launcher_args("dpo", ds, env)]
    return shlex.join(dpo) if from_stage == "dpo" else shlex.join(sft) + " && " + shlex.join(dpo)


def _gj_cmd(cell: grid.Cell, root: Path, limit: int | None) -> str:
    out = cell.gen_dir(smoke=limit is not None)
    return shlex.join(["bash", "shell/posttrain/gen_judge.sh", str(cell.out("dpo", root) / "model"),
                       "-", cell.model_id, out.name, *([str(limit)] if limit else [])])


REFILL_JOB_TIME = "00:30:00"


TRAP_TAG = "TRAP:"  # driver.log lines a trap writes when its refill job is refused


def wrap(cmd: str, control_path: Path, ctl: dict, prompt_id: str) -> str:
    """The job's command: refill, work, refill with this job's slot free, keep the work's rc.
    A SIGTERM (Slurm time limit or scancel) at any point, both refills included, submits a
    refill job that starts after this one has ended. Each step runs in the background so the
    trap fires at once (bash defers a trap until a foreground child returns, and a refill can
    wait 600 s on the lock, longer than KillWait). A refill cut short by the job's end is safe:
    the ledger is written ahead (castor/grid.py module doc). If sbatch refuses the refill job
    (account cap), the trap says so in driver.log, which `status` shows as an attention item."""
    refill = shlex.join(["uv", "run", "--frozen", "python", "scripts/grid.py", "refill",
                         "--control", str(control_path)])
    repo = grid.main_repo()  # the snapshot may be going away with this job
    later = shlex.join([str(repo / "shell" / "nchc" / "submit.sh"), "--dev", "--gpus", "1",
                        "--time", REFILL_JOB_TIME, "--name", f"{prompt_id}-refill",
                        "--exclude", ctl["exclude"], "--commit", ctl["commit"], "--use-reserve"])
    # bash -c, so "$SLURM_JOB_ID" expands in the refill job itself: its own slot counts as free
    later_refill = shlex.join(["bash", "-c", f'{refill} --ended-job "$SLURM_JOB_ID"'])
    log = shlex.quote(str(control_path.parent / "driver.log"))
    refused = (f'echo "[$(date +%Y-%m-%dT%H:%M:%S) job $SLURM_JOB_ID] {TRAP_TAG} refill job '
               f'refused; run scripts/grid.py refill by hand" >> {log}')
    # a second TERM must not submit a second refill job: ignore TERM once the trap runs;
    # exit with the work's rc if it had finished (TERM during the end refill)
    on_term = (f'trap "" TERM; cd {shlex.quote(str(repo))} && '
               f'{later} --after "$SLURM_JOB_ID" -- {later_refill} || {refused}; '
               f'exit "${{rc:-143}}"')
    end_refill = f'{refill} --ended-job "$SLURM_JOB_ID"'
    return (f"trap {shlex.quote(on_term)} TERM; rc=; ( {refill} ) & wait $!; "
            f"( {cmd} ) & wait $!; rc=$?; ( {end_refill} ) & wait $!; exit $rc")


PASS_TAG = "Slurm: "  # first driver.log line of every completed (non-dry) refill pass
SNAPSHOT_TAG = "squeue snapshot:"  # driver.log line of every completed pass: the job ids it surveyed with
_TRAP_JOB_RE = re.compile(r"^\[\S+ job (\S+)\]")  # the trap line's prefix: [<stamp> job <id>]


def trap_failures(d: Path) -> list[str]:
    """driver.log lines of traps whose refill job sbatch refused and that no later refill pass has
    resolved. A pass resolves a trap once the trapped job is gone from its squeue snapshot: only
    then did its survey see that job's cell as failed and plan the retry. A pass whose snapshot
    predates the kill (the job still RUNNING or COMPLETING) leaves the item, whatever the order
    of the two log lines."""
    log = d / "driver.log"
    if not log.exists():
        return []
    out: list[str] = []
    for x in log.read_text(errors="replace").splitlines():
        if TRAP_TAG in x:
            out.append(x)
        elif f"] {SNAPSHOT_TAG}" in x:
            out = open_traps(out, set(x.split(SNAPSHOT_TAG, 1)[1].split()))
    return out


def last_pass(d: Path) -> str | None:
    """driver.log's summary line of the last completed refill pass (PASS_TAG: budget, planned,
    submitted); the lines after it are its submissions, snapshot and attention items."""
    log = d / "driver.log"
    if not log.exists():
        return None
    lines = [x for x in log.read_text(errors="replace").splitlines() if f"] {PASS_TAG}" in x]
    return lines[-1] if lines else None


def open_traps(traps: list[str], snapshot: set[str]) -> list[str]:
    """The trap lines a pass that surveyed with this squeue snapshot (job ids) leaves open: those
    whose job was still in Slurm (an unparsable line stays open rather than vanish)."""
    return [t for t in traps if not (m := _TRAP_JOB_RE.match(t)) or m[1] in snapshot]


# ---- survey ---------------------------------------------------------------------------------


def _running_info(name: str, job: dict, now: float) -> dict:
    """Step rate and log staleness of a running job."""
    log = grid.log_path(name, job["id"])
    if not log.exists():
        return {"log": None}
    info = {"log": str(log), "log_age_s": now - log.stat().st_mtime}
    info["stalled"] = info["log_age_s"] > grid.STALL_S
    return info


def survey(g: grid.Grid, cells: list[grid.Cell], root: Path, smoke: bool,
           jobs: dict[str, dict], gj_limit: int | None = None,
           cache: dict | None = None, ledger: dict | None = None,
           beat=lambda: None, stages: list[grid.Stage] | None = None,
           launch: dict | None = None) -> list[dict]:
    """`ledger` = the driver's submissions (grid.read_ledger); `beat` refreshes the driver lock;
    `launch` = launcher(ctl): a run on another launcher configuration is an error row."""
    ledger = ledger or {}
    stages = stages or [grid.ALL_STAGE]
    want = grid.resolve_launcher(launch)  # once: the control's configuration per stage
    n_prompts = len(evalset.load_eval())
    if gj_limit is not None:
        n_prompts = min(n_prompts, gj_limit)
    now = datetime.now().timestamp()  # noqa: DTZ005
    rows = []
    for c in cells:
        beat()
        tn, gn = c.job_name("train", smoke), c.job_name("gj", smoke)
        row = {"tag": c.tag, "model_id": c.model_id, "stage": grid.stage_of(c, stages),
               "train": "?", "genjudge": "waiting"}
        try:
            st = grid.stage_state(c, root, want)
            ts = grid.train_state(st, tn in jobs)
            if ts == "todo" and ledger.get(f"{c.tag}|sft"):
                ts = "failed-sft"  # submitted, died before any provenance (castor/grid.py doc)
            row["train"] = ts
            if tn in jobs:
                j = jobs[tn]
                row["train_job"] = j
                if j["state"] == "RUNNING":
                    row |= _running_info(tn, j, now)
                    if row.get("log"):
                        rate = grid.step_rate(Path(row["log"]).read_text(errors="replace"))
                        if rate:
                            rate["stage"] = "dpo" if st["sft"] == "finished" else "sft"
                        row["rate"] = rate
            elif ts.startswith("failed"):
                stage = ts.split("-")[1]
                keys = (f"{c.tag}|dpo", f"{c.tag}|sft") if stage == "dpo" else (f"{c.tag}|sft",)
                jid = (grid.last_job(ledger, *keys) or st["jobs"].get(stage)
                       or st["jobs"].get("sft"))
                # no job id (a pending-only ledger entry): no sacct by name, which can return
                # an earlier launch's or a smoke's job; the retry then excludes no extra node
                row["failed_job"] = grid.sacct(jid) if jid else {}
                row["attempts"] = ledger.get(keys[0], [])
                row["retries_used"] = grid.retries_used(ledger, c.tag, stage)
                row["moved_aside"] = _train_attempts(c, root)
            if ts == "done" and (not smoke or gj_limit is not None):
                gs = grid.genjudge_state(c, n_prompts, gn in jobs, root, limit=gj_limit,
                                         cache=cache)
                if gs == "todo" and ledger.get(f"{c.tag}|gj"):
                    gs = "failed"  # submitted, died before creating its directory
                row["genjudge"] = gs
                if gn in jobs:
                    row["gj_job"] = jobs[gn]
                    if jobs[gn]["state"] == "RUNNING":
                        row["gj_run"] = _running_info(gn, jobs[gn], now)
                elif gs == "failed":
                    jid = grid.last_job(ledger, f"{c.tag}|gj")
                    row["failed_job"] = grid.sacct(jid) if jid else {}
                    row["attempts"] = ledger.get(f"{c.tag}|gj", [])
                    row["retries_used"] = grid.retries_used(ledger, c.tag, "gj")
                    # gen_judge.sh regenerates whenever a limit is set: judge-only is full-panel
                    row["judge_only"] = (gj_limit is None
                                         and grid.generation_complete(c, root, limit=gj_limit))
        except Exception as e:  # noqa: BLE001 one broken cell must not stop the grid
            row["train"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return rows


def _train_attempts(c: grid.Cell, root: Path) -> list[str]:
    return sorted(set(grid.attempts(c.out("sft", root))) | set(grid.attempts(c.out("dpo", root))))


def attention(rows: list[dict]) -> list[str]:
    out = []
    for r in rows:
        rate = r.get("rate") or {}
        if rate.get("slow"):
            out.append(f"SLOW: {r['tag']} {rate['stage']} {rate['s_per_step']:.2f} s/step at step "
                       f"{rate['step']} on {r['train_job']['where']} (job {r['train_job']['id']})")
        for key in ("train_job", "gj_job"):
            run = r if key == "train_job" else r.get("gj_run", {})
            if r.get(key) and run.get("stalled"):
                out.append(f"STALLED: {r['tag']} job {r[key]['id']} log unchanged for "
                           f"{run['log_age_s'] / 60:.0f} min ({r[key]['where']})")
        if r["train"] == "error":
            out.append(f"ERROR: {r['tag']} {r['error']}")
        if r["train"].startswith("failed") or r["genjudge"] in ("failed", "stale"):
            out.append(f"FAILED: {r['tag']} train={r['train']} gj={r['genjudge']} "
                       f"attempts={r.get('attempts', [])} {r.get('failed_job', '')}")
    return out


STOP_NOTE = ("to stop the grid: `scripts/grid.py stop --manifest …` BEFORE any scancel of its jobs "
             "(a cancelled job's trap submits a refill, which retries the cell)")


def format_status(rows: list[dict], header: list[str], enabled: list[str] | None = None,
                  traps: list[str] | None = None, launched: bool = True) -> str:
    lines = [*header,
             (f"train: {dict(Counter(r['train'] for r in rows))} | "
              f"gen+judge: {dict(Counter(r['genjudge'] for r in rows))}")]
    names = list(dict.fromkeys(r.get("stage", "all") for r in rows))
    if names != ["all"]:
        for st in names:
            sub = [r for r in rows if r.get("stage", "all") == st]
            on = ("not launched" if not launched else
                  "enabled" if enabled is None or st in enabled else "disabled")
            lines.append(f"  stage {st} ({on}, {len(sub)} cells): "
                         f"train {dict(Counter(r['train'] for r in sub))} | "
                         f"gen+judge {dict(Counter(r['genjudge'] for r in sub))}")
    for r in rows:
        if r["train"] in ("done", "todo") and r["genjudge"] in ("waiting", "done", "todo"):
            continue
        j = r.get("gj_job") or r.get("train_job") or {}
        flag = ""
        rate = r.get("rate")
        if rate and rate.get("s_per_step") is not None:
            flag = f" {rate['stage']} step {rate['step']} {rate['s_per_step']:.2f} s/step"
        elif rate:
            flag = f" {rate['stage']} step {rate['step']}"
        where = f"{j.get('id', '')} {j.get('state', '')} {j.get('where', '')}".strip()
        lines.append(f"  {r['tag']:<52} train={r['train']:<10} gj={r['genjudge']:<8} {where}{flag}")
    att = attention(rows) + [f"TRAP REFILL REFUSED (grid may have stalled): {x}"
                             for x in traps or []]
    lines += att or ["no attention items"]
    lines.append(STOP_NOTE)
    return "\n".join(lines) + "\n"


# ---- control and refill ---------------------------------------------------------------------


def _control_path(manifest: Path, smoke: bool) -> Path:
    return grid.driver_dir(grid.load(manifest).prompt_id, smoke) / "control.json"


def _log(d: Path, msg: str) -> None:
    who = f"job {os.environ['SLURM_JOB_ID']}" if "SLURM_JOB_ID" in os.environ else \
        f"{os.uname().nodename} pid {os.getpid()}"
    stamp = datetime.now().isoformat(timespec="seconds")  # noqa: DTZ005
    with open(d / "driver.log", "a") as f:
        f.writelines(f"[{stamp} {who}] {line}\n" for line in msg.splitlines())


def submit(opts: list[str], cmd: str, dry: bool) -> str:
    argv = [str(SUBMIT), *opts, *(["--dry-run"] if dry else []), "--", "bash", "-c", cmd]
    print("$", shlex.join(argv))
    if dry:
        return "dry-run"
    r = subprocess.run(argv, cwd=grid.ROOT, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"submit failed ({r.returncode}): {r.stderr.strip()[-400:]}")
    return r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""


def order_cells(cells: list[grid.Cell], stages: list[grid.Stage],
                enabled: list[str] | None) -> list[grid.Cell]:
    """The cells of the enabled stages (None = all), in stage order, manifest order within."""
    rank = {s.name: i for i, s in enumerate(stages)}
    keep = [c for c in cells if enabled is None or grid.stage_of(c, stages) in enabled]
    return sorted(keep, key=lambda c: rank[grid.stage_of(c, stages)])  # stable


def plan(cells: list[grid.Cell], rows: dict, max_retries: int) -> list[tuple]:
    """`cells` = order_cells(...): only enabled stages, in stage order. Every action, not cut to
    the budget: refill stops once the budget's worth has been submitted, so a cell that fails
    before Slurm (a bad base, an OSError) does not hold a slot."""
    out = []
    for c in cells:  # gen + judge first: 1 GPU, unblocks the analysis
        r = rows[c.tag]
        if r["genjudge"] == "todo":
            out.append(("gj", c, None))
        elif r["genjudge"] == "failed" and r.get("retries_used", 0) < max_retries:
            out.append(("gj-retry", c, None))
    for c in cells:
        r = rows[c.tag]
        if r["train"].startswith("failed") and r.get("retries_used", 0) < max_retries:
            out.append(("retry", c, r["train"].split("-")[1]))
    out += [("train", c, "sft") for c in cells if rows[c.tag]["train"] == "todo"]
    return out


def _job_id(submit_out: str) -> str:
    """sbatch prints 'Submitted batch job <id>'."""
    return submit_out.split()[-1] if submit_out else "?"


def refill(ctl: dict, control_path: Path, ended_job: str | None = None,
           dry: bool = False, beat=lambda: None) -> str:
    """One pass: survey, submit within budget, write status. Returns the status text. Run under
    the driver lock (`beat` refreshes it) except as a dry run."""
    manifest = grid.ROOT / ctl["manifest"]
    assert posttrain.sha256_file(manifest) == ctl["manifest_sha256"], f"{manifest} changed"
    check_ds(ctl)
    g = grid.load(manifest)
    stages, enabled = _stages(ctl, g)
    smoke = ctl["smoke"] is not None
    root = grid.models_root(g, smoke)
    cells = [c for c in g.cells() if not ctl["only"] or c.tag in ctl["only"]]
    d = control_path.parent
    cache_path = d / "label_counts.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    all_jobs = [j for j in grid.squeue_list() if j["id"] != ended_job]
    snapshot = {j["id"] for j in all_jobs}  # this pass's view of Slurm; resolves earlier traps
    jobs = grid.by_name(all_jobs)
    mine = sum(grid.is_driver_job(j["name"], g.prompt_id) for j in all_jobs)
    max_jobs = grid.max_jobs_now(ctl)
    budget = max(min(max_jobs - mine, ACCOUNT_CAP - TRAP_RESERVE - len(all_jobs)), 0)
    ledger = grid.read_ledger(d)
    rows = {r["tag"]: r for r in survey(g, cells, root, smoke, jobs, ctl["gj_limit"], cache,
                                        ledger, beat, stages, launcher(ctl))}
    actions = (plan(order_cells(cells, stages, enabled), rows, ctl["max_retries"])
               if ctl["enabled"] else [])
    done, errors = [], []
    sent = 0  # submissions that reached Slurm, including one whose ledger update then failed
    for kind, c, stage in actions:
        if sent >= budget:
            break
        beat()
        r = rows[c.tag]
        excl = ctl["exclude"]
        node = (r.get("failed_job") or {}).get("node")
        if kind in ("retry", "gj-retry") and node and node not in ("None", "None assigned"):
            excl = f"{excl},{node}"
        key = f"{c.tag}|{'gj' if kind.startswith('gj') else stage}"
        moved: list[tuple[Path, Path]] = []  # (original, moved-aside), undone if Slurm refuses
        pending, submitted = None, False
        try:
            # the command first: a bad base fails here, before anything changes
            if kind.startswith("gj"):
                opts = ["--gpus", "1", "--time", DEV_GJ_TIME if ctl["dev"] else grid.GJ_TIME,
                        *(["--dev"] if ctl["dev"] else []), "--name", c.job_name("gj", smoke)]
                cmd = _gj_cmd(c, root, ctl["gj_limit"])
            else:
                opts = ["--gpus", "8", "--time", DEV_TRAIN_TIME if ctl["dev"] else grid.TRAIN_TIME,
                        *(["--dev"] if ctl["dev"] else []), "--name", c.job_name("train", smoke)]
                cmd = _train_cmd(c, root, stage, ctl["smoke"], launcher(ctl))
            if not dry:
                # write-ahead: from here on a refill killed mid-way still counts this attempt
                pending = grid.ledger_pending(d, key)
            if kind == "gj-retry" and not dry:
                gdir = c.gen_dir(smoke=ctl["gj_limit"] is not None)
                # a complete generation is kept: gen_judge.sh then reruns the judge only
                aside = gdir / "judge_v1" if r.get("judge_only") else gdir
                moved.append((aside, grid.move_aside(aside, r["failed_job"].get("job"))))
            if kind == "retry" and not dry:
                job = r["failed_job"].get("job")
                for st in ("dpo", "sft") if stage == "sft" else ("dpo",):
                    moved.append((c.out(st, root), grid.move_aside(c.out(st, root), job)))
            beat()
            # --term-wait: Slurm signals the batch shell only; it forwards TERM to wrap()'s trap
            # --ending-job: submit.sh's 19-slot check skips the job this refill counts as free
            out = submit([*opts, "--exclude", excl, "--commit", ctl["commit"], "--term-wait",
                          *(["--ending-job", ended_job] if ended_job else [])],
                         wrap(cmd, control_path, ctl, g.prompt_id), dry)
            submitted = True
            sent += 1
            if pending:
                beat()
                grid.ledger_resolve(d, key, pending, _job_id(out))
            done.append(f"{kind} {c.tag}: {out}")
        except grid.LockLost:
            raise  # not ours any more: change nothing (a pending entry stays, counted)
        except Exception as e:  # noqa: BLE001 logged; the other cells still submit
            if not submitted:
                beat()
                for orig, aside in reversed(moved):
                    if aside is not None and not orig.exists():
                        aside.rename(orig)  # nothing ran: the attempt is not used up
                if pending:
                    grid.ledger_resolve(d, key, pending, None)
            errors.append(f"{kind} {c.tag}: {type(e).__name__}: {e}"
                          + (" (job submitted; its ledger entry stays pending)" if submitted
                             else ""))
            if isinstance(e, RuntimeError) and "submit failed" in str(e):
                break  # Slurm refused (cap, outage): stop this pass, the next refill retries
    now = datetime.now().isoformat(timespec="seconds")  # noqa: DTZ005
    header = [(f"grid {g.prompt_id}{' SMOKE' if smoke else ''} status at {now} "
               f"(refill{' dry-run' if dry else ''}; ended job {ended_job or '-'})"),
              (f"control: enabled={ctl['enabled']} stages={'all' if enabled is None else enabled} "
               f"commit={ctl['commit'][:10]} max_jobs now={max_jobs} "
               f"retries={ctl['max_retries']} exclude={ctl['exclude']} "
               f"launcher={launcher_text(ctl)}"),
              (f"{PASS_TAG}{len(all_jobs)} jobs of this user, {mine} of this grid before this pass; "
               f"budget {budget}; planned {len(actions)}; submitted {sent}"),
              *[f"submitted {x}" for x in done], *[f"SUBMIT-ERROR {x}" for x in errors]]
    # the traps this pass leaves open: the same answer `status` reads from driver.log afterwards
    traps = open_traps(trap_failures(d), snapshot)
    text = format_status(list(rows.values()), header, enabled, traps)
    if not dry:
        beat()  # a holder taken over must not write over its successor's status
        grid.write_atomic(d / "status.txt", text)
        grid.write_atomic(d / "status.json", json.dumps(list(rows.values()), indent=1, default=str))
        grid.write_atomic(cache_path, json.dumps(cache))
        ids = " ".join(sorted(snapshot, key=lambda s: (len(s), s))) or "-"
        _log(d, "\n".join(header[2:]) + f"\n{SNAPSHOT_TAG} {ids}\n"
             + "\n".join(attention(list(rows.values()))))
        finished = grid.labelled([rows[c.tag] for c in order_cells(cells, stages, enabled)])
        if (not ctl["enabled"] or finished) and _cron_off(
                ctl, d, "grid labelled" if ctl["enabled"] else "control disabled"):
            beat()
            grid.write_atomic(control_path, json.dumps(ctl, indent=1) + "\n")
    return text


def _host() -> str:
    return os.uname().nodename


def _cron_on(ctl: dict, path: Path) -> None:
    """Install this grid's crontab line on this host and record it in ctl (caller writes ctl)."""
    if "SLURM_JOB_ID" in os.environ:  # a job's node is not the login node
        print("cron: not installed from inside a Slurm job")
        return
    cron_path = grid.cron_path()  # asserts every tool the refill runs resolves to a path
    g = grid.load(grid.ROOT / ctl["manifest"])
    smoke = ctl["smoke"] is not None
    line = grid.cron_line(g.prompt_id, smoke, path.resolve(), grid.main_repo(),
                          shutil.which("uv"), cron_path)
    grid.install_cron(line, grid.cron_tag(g.prompt_id, smoke))
    ctl["cron"] = {"host": _host(), "tag": grid.cron_tag(g.prompt_id, smoke), "line": line}
    ctl.pop("cron_removed", None)
    _log(path.parent, f"cron installed on {_host()}: {line}")
    print(f"cron installed on {_host()}: {line}")


def _cron_off(ctl: dict, d: Path, why: str) -> bool:
    """Remove this grid's crontab line if this process runs on the host that holds it, and clear
    ctl["cron"]. Returns True if ctl changed (the caller writes it)."""
    cron = ctl.get("cron")
    if not cron:
        return False
    if _host() != cron["host"]:
        if "SLURM_JOB_ID" not in os.environ:  # a person, on another login node
            print(f"cron: the line is on {cron['host']}; run this there (or `ssh "
                  f"{cron['host']} crontab -e`) to remove it")
        return False
    if grid.remove_cron(cron["tag"]):
        _log(d, f"cron removed on {cron['host']} ({why})")
        print(f"cron removed on {cron['host']} ({why})")
    ctl["cron"] = None  # removed now, or already gone by hand: nothing left to record
    ctl["cron_removed"] = {"host": cron["host"], "why": why}  # `stages --enable` reinstalls
    return True


def _relaunch_cron(old: dict | None, ctl: dict, path: Path, no_cron: bool) -> None:
    """The new control's cron (caller writes ctl). An earlier launch's line can be removed or
    replaced only on its host, and the new control must not lose track of it: elsewhere, refuse."""
    if old and old.get("cron"):
        assert _host() == old["cron"]["host"], (
            f"the grid's cron line is on {old['cron']['host']}: relaunch there, or run `stop` "
            "there first")
        if no_cron:
            _cron_off(old, path.parent, "relaunch with --no-cron")
    if not no_cron:
        _cron_on(ctl, path)  # same tag: replaces an earlier line on this host


def _stages(ctl: dict, g: grid.Grid, check: bool = True) -> tuple[list[grid.Stage], list[str] | None]:
    """The control's stages (file checked against its sha256 unless `check` is False: the
    read-only status reports a change instead) and the enabled names (None = all, for a grid
    without a stages file)."""
    path = ctl.get("stages_file")
    if path is None:
        return [grid.ALL_STAGE], None
    if check:
        assert not _stages_changed(ctl), f"{path} changed"
    stages = grid.load_stages(path, g)
    enabled = list(ctl["stages_enabled"])
    assert set(enabled) <= {s.name for s in stages}, f"unknown stages {enabled}"
    return stages, enabled


def _stages_changed(ctl: dict) -> bool:
    sf = ctl.get("stages_file")
    return bool(sf) and posttrain.sha256_file(grid.ROOT / sf) != ctl["stages_sha256"]


def ds_files(ds: dict[str, str] | None) -> dict[str, str] | None:
    """sha256 of each explicit DeepSpeed config (stage -> sha), as resolved by train.sh."""
    return {st: posttrain.sha256_file(posttrain.ds_config_path(st, p))
            for st, p in ds.items()} if ds else None


def _ds_changed(ctl: dict) -> bool:
    return ds_files(ctl.get("ds_config")) != ctl.get("ds_config_sha256")


def check_ds(ctl: dict) -> None:
    """A DeepSpeed config named at launch must not have changed since: every job of the grid
    trains on the same file (the pinned commit's copy; this checks the live tree's)."""
    assert not _ds_changed(ctl), f"DeepSpeed config {ctl.get('ds_config')} changed since launch"


def changed_since_launch(ctl: dict) -> list[str]:
    """`status` warnings: the control's files that changed since launch. refill refuses to run
    (check_ds, _stages) until each is restored or the grid is relaunched."""
    out = []
    if _stages_changed(ctl):
        out.append(ctl["stages_file"])
    if _ds_changed(ctl):
        out.append(f"DeepSpeed config {ctl['ds_config']}")
    return [f"WARNING: {x} changed since launch; refills refuse to run until it is restored or "
            "the grid is relaunched" for x in out]


def committed_ds(ds: dict[str, str] | None, head: str) -> None:
    """A --ds-config under this repo must be committed as is: every job renders from a snapshot
    of `head` (submit.sh --commit), so the control's sha256 (the live file's, ds_files) must be
    that copy's, or no job could ever match it (check_launcher) and the job-end refill, run from
    the snapshot, would refuse (check_ds) while the cron refill, run from the live tree, passes."""
    for st, p in (ds or {}).items():
        path = posttrain.ds_config_path(st, p).resolve()
        if not path.is_relative_to(grid.ROOT):
            continue  # open-instruct's pinned tree
        rel = path.relative_to(grid.ROOT).as_posix()
        r = subprocess.run(["git", "show", f"{head}:{rel}"], cwd=grid.ROOT, capture_output=True,
                           check=False)
        assert r.returncode == 0, f"--ds-config {p} is not in commit {head[:10]}: commit it first"
        assert hashlib.sha256(r.stdout).hexdigest() == posttrain.sha256_file(path), \
            f"--ds-config {p} differs from commit {head[:10]}, which the jobs run: commit it first"


def check_launch_env(env: list[str] | None) -> None:
    """--launch-env settings the grid driver accepts: posttrain.LAUNCH_ENV_ALLOWED, less the CPU
    count, which follows each job's allocation and which check_launcher ignores (launcher_of):
    an explicit value would be honoured by train.sh and invisible to the done check."""
    posttrain.launch_env(12, env)  # raises on a setting outside LAUNCH_ENV_ALLOWED
    for kv in env or []:
        assert not kv.startswith("BEAKER_ASSIGNED_CPU_COUNT="), \
            "--launch-env BEAKER_ASSIGNED_CPU_COUNT: the CPU count follows the job's allocation"


def launcher_text(ctl: dict) -> str:
    ds, env = ctl.get("ds_config"), ctl.get("launch_env")
    if not ds and not env:
        return "current"
    return " ".join([*(f"{k}:{v}" for k, v in (ds or {}).items()), *(env or [])])


def _default_stages_file(manifest: Path) -> str | None:
    p = Path(manifest).with_name(Path(manifest).stem + "_stages.json")
    return str(p) if (grid.ROOT / p).exists() else None


def cmd_refill(a) -> None:
    with grid.lock(a.control.parent) as beat:
        ctl = json.loads(a.control.read_text())  # under the lock: stop/launch may have changed it
        if not ctl["enabled"]:
            print(f"{a.control}: disabled")
        print(refill(ctl, a.control, a.ended_job, beat=beat))


def cmd_launch(a) -> None:
    g = grid.load(a.manifest)
    smoke = a.smoke is not None
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=grid.ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()
    if dirty := provenance.dirty_code():
        print(f"note: uncommitted code is NOT used; jobs run commit {head[:10]}: {dirty[:5]}")
    steps = [[None, a.max_jobs]]
    for s in a.max_jobs_after:
        t, n = s.split("=")
        steps.append([datetime.fromisoformat(t).isoformat(), int(n)])
    only = sorted(a.only) if a.only else None
    if only:
        unknown = set(only) - {c.tag for c in g.cells()}
        assert not unknown, f"unknown cells {sorted(unknown)}"
    sfile = a.stages_file if a.stages_file is not None else _default_stages_file(a.manifest)
    if sfile in ("", "none"):
        sfile = None
    stage_ctl = {"stages_file": None, "stages_sha256": None, "stages_enabled": None}
    if sfile is not None:
        names = [s.name for s in grid.load_stages(sfile, g)]
        assert a.stages, f"{sfile} defines stages {names}: name the enabled ones with --stages"
        assert set(a.stages) <= set(names), f"unknown stages {sorted(set(a.stages) - set(names))}"
        stage_ctl = {"stages_file": sfile,
                     "stages_sha256": posttrain.sha256_file(grid.ROOT / sfile),
                     "stages_enabled": [n for n in names if n in a.stages]}
    else:
        assert not a.stages, "--stages needs a stages file"
    ds = {"sft": a.ds_config} if a.ds_config else None
    env = list(a.launch_env) if a.launch_env else None
    if ds:  # refused here, before any job: the same checks render applies in each job
        json.loads(posttrain.ds_config_path("sft", a.ds_config).read_text())
        committed_ds(ds, head)
    check_launch_env(env)
    ctl = {"enabled": True, "manifest": str(Path(a.manifest)),
           "manifest_sha256": posttrain.sha256_file(grid.ROOT / a.manifest), **stage_ctl,
           "commit": head, "max_jobs": steps, "max_retries": a.max_retries,
           "exclude": a.exclude if a.exclude is not None else grid.read_exclude(),
           "smoke": a.smoke, "gj_limit": a.gj_limit, "dev": a.dev,
           "ds_config": ds, "ds_config_sha256": ds_files(ds), "launch_env": env,
           "only": only, "written": datetime.now().isoformat(timespec="seconds"),  # noqa: DTZ005
           "command": provenance.command(), "cron": None}
    path = _control_path(a.manifest, smoke)
    if a.dry_run:
        print(refill(ctl, path, dry=True))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with grid.lock(path.parent) as beat:
        beat()
        old = json.loads(path.read_text()) if path.exists() else None
        _relaunch_cron(old, ctl, path, a.no_cron)  # may refuse (cron line on another host)
        if old is not None:  # logged only once the replacement is going ahead
            _log(path.parent, f"control replaced; was {json.dumps(old)}")
        beat()
        grid.write_atomic(path, json.dumps(ctl, indent=1) + "\n")
        _log(path.parent, f"launch: {json.dumps(ctl)}")
        print(refill(ctl, path, beat=beat))


def cmd_stages(a) -> None:
    """Enable or disable stages of a launched grid; the pinned commit and everything else in the
    control stay as they are. Then one refill."""
    path = _control_path(a.manifest, a.smoke is not None)
    with grid.lock(path.parent) as beat:
        ctl = json.loads(path.read_text())
        assert ctl.get("stages_file"), f"{path}: this grid has no stages"
        names = [s.name for s in grid.load_stages(ctl["stages_file"],
                                                  grid.load(grid.ROOT / ctl["manifest"]))]
        want = (set(ctl["stages_enabled"]) | set(a.enable)) - set(a.disable)
        assert want <= set(names), f"unknown stages {sorted(want - set(names))}"
        was = ctl["stages_enabled"]
        ctl["stages_enabled"] = [n for n in names if n in want]
        # the line, or the one a finished grid's last refill removed (not a --no-cron launch)
        prev = ctl.get("cron") or ctl.get("cron_removed")
        if ctl["enabled"] and prev:
            if _host() == prev["host"]:
                _cron_on(ctl, path)
            else:
                print(f"cron: run `stages` on {prev['host']} to reinstall the line")
        beat()
        grid.write_atomic(path, json.dumps(ctl, indent=1) + "\n")
        _log(path.parent, f"stages: {was} -> {ctl['stages_enabled']} ({provenance.command()})")
        print(refill(ctl, path, beat=beat) if ctl["enabled"] else f"{path}: disabled")


def cmd_stop(a) -> None:
    path = _control_path(a.manifest, a.smoke is not None)
    with grid.lock(path.parent) as beat:
        ctl = json.loads(path.read_text())
        ctl["enabled"] = False
        beat()
        grid.write_atomic(path, json.dumps(ctl, indent=1) + "\n")
        _log(path.parent, "stop: refills disabled (running and queued jobs are untouched)")
        if _cron_off(ctl, path.parent, "stop"):
            beat()
            grid.write_atomic(path, json.dumps(ctl, indent=1) + "\n")
    print(f"{path}: disabled; jobs already in Slurm are untouched")


def cmd_status(a) -> None:
    g = grid.load(a.manifest)
    smoke = a.smoke is not None
    path = _control_path(a.manifest, smoke)
    ctl = json.loads(path.read_text()) if path.exists() else None
    root = grid.models_root(g, smoke)
    cells = [c for c in g.cells() if not a.only or c.tag in a.only]
    now = datetime.now().isoformat(timespec="seconds")  # noqa: DTZ005
    header = [f"grid {g.prompt_id}{' SMOKE' if smoke else ''} live status {now}"]
    if ctl:
        header.append(f"control: enabled={ctl['enabled']} commit={ctl['commit'][:10]} "
                      f"max_jobs now={grid.max_jobs_now(ctl)} retries={ctl['max_retries']} "
                      f"launcher={launcher_text(ctl)}")
        cron = ctl.get("cron")
        gone = ctl.get("cron_removed")
        header.append(f"cron: line on {cron['host']} (see `crontab -l` there)" if cron else
                      f"cron: removed on {gone['host']} ({gone['why']})" if gone else "cron: none")
        header.append(f"last refill pass: {last_pass(path.parent) or '-'}")
    else:
        header.append("control: none (grid not launched)")
    if ctl:
        stages, enabled = _stages(ctl, g, check=False)
        header.append(f"enabled stages: {'all' if enabled is None else enabled}")
        header.extend(changed_since_launch(ctl))
    else:
        stages, enabled = grid.load_stages(_default_stages_file(a.manifest), g), None
    rows = survey(g, cells, root, smoke, grid.squeue(), ctl["gj_limit"] if ctl else None,
                  ledger=grid.read_ledger(path.parent), stages=stages, launch=launcher(ctl))
    print(format_status(rows, header, enabled, trap_failures(path.parent),
                        launched=ctl is not None), end="")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1, default=str) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--prompt-id", required=True)
    m.add_argument("--source", required=True, help="a download manifest (data/manifests/)")
    m.add_argument("--drop-recipe", nargs="*", default=[])
    m.add_argument("--seeds", nargs="+", type=int, required=True)
    m.add_argument("--seeds-for", nargs="*", default=[], metavar="RECIPE=K,K,...",
                   help="that recipe's bases get these post-training seeds (D063c)")
    m.add_argument("--sft-config", default="configs/posttrain/sft_datadecide_eosbos.yaml")
    m.add_argument("--dpo-config", default="configs/posttrain/dpo_datadecide_eosbos.yaml")
    m.add_argument("--variant", default=None,
                   help="a non-reference recipe: own models root and model-id suffix")
    m.add_argument("--out", required=True)
    for name in ("status", "launch", "stop", "stages"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--smoke", type=int, default=None, help="max_train_steps (dry run)")
        if name in ("status", "launch"):
            p.add_argument("--only", nargs="*", help="cell tags")
    sub.choices["status"].add_argument("--json", type=Path, default=None)
    lp = sub.choices["launch"]
    lp.add_argument("--max-jobs", type=int, default=12, help="this grid's jobs in Slurm")
    lp.add_argument("--max-jobs-after", nargs="*", default=[], metavar="ISOTIME=N",
                    help="raise or lower the budget from that local time on")
    lp.add_argument("--max-retries", type=int, default=1,
                    help="per failed cell and stage (sft, dpo, gen + judge)")
    lp.add_argument("--exclude", default=None,
                    help="Slurm node list; default: shell/nchc/exclude_nodes (read at launch)")
    lp.add_argument("--ds-config", default=None,
                    help="SFT DeepSpeed config passed to train.sh for every cell (prompt 024; "
                         "default: the current configuration)")
    lp.add_argument("--launch-env", action="append", default=None, metavar="KEY=VALUE",
                    help="launcher environment passed to train.sh (posttrain.LAUNCH_ENV_ALLOWED)")
    lp.add_argument("--gj-limit", type=int, default=None, help="first N prompts (dry run only)")
    lp.add_argument("--dev", action="store_true", help="dev partition, smoke runs only")
    lp.add_argument("--dry-run", action="store_true", help="print sbatch scripts only")
    lp.add_argument("--no-cron", action="store_true",
                    help="do not install the login node's refill line (tests, probes)")
    lp.add_argument("--stages-file", default=None,
                    help="default: <manifest stem>_stages.json if it exists; 'none' = no stages")
    lp.add_argument("--stages", nargs="*", default=[], help="enabled stages (in the file's order)")
    sp = sub.choices["stages"]
    sp.add_argument("--enable", nargs="*", default=[])
    sp.add_argument("--disable", nargs="*", default=[])
    r = sub.add_parser("refill", help="run by the grid's jobs; see module doc")
    r.add_argument("--control", type=Path, required=True)
    r.add_argument("--ended-job", default=None)
    a = ap.parse_args()
    if a.cmd == "manifest":
        return cmd_manifest(a)
    if a.cmd == "refill":
        return cmd_refill(a)
    assert a.smoke is None or a.smoke > 0, "--smoke takes a positive step count"
    if a.cmd == "launch":
        assert not a.dev or a.smoke, "--dev (≤ 4 h) cannot hold a full SFT -> DPO chain; smoke only"
        assert a.gj_limit is None or a.smoke, "--gj-limit is for the smoke dry run only"
        return cmd_launch(a)
    if a.cmd == "stages":
        return cmd_stages(a)
    return cmd_stop(a) if a.cmd == "stop" else cmd_status(a)


if __name__ == "__main__":
    sys.exit(main())
