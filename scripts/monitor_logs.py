"""Live monitor of the grid's train and gen+judge logs: one static page, no daemon, no cron.

    uv run --frozen python scripts/monitor_logs.py --out /home/harrychang08/castor/runs/monitor/index.html
    watch -n 600 uv run --frozen python scripts/monitor_logs.py --out …     # refresh on the login node

Chains are `logs/nchc/<prompt>-(train|gj|genjudge)-<tag>-<jobid>.out` for the prompts given
(default 008 and 016). A retried cell writes a second segment with the same job name; segments
are concatenated in job-id order and a step repeated by a later segment wins. Train logs:
open-instruct's `Step: N, LR, Loss[, TPS]` lines after an `open-instruct <sha> | <stage> |`
header, with tqdm's `N/TOTAL [` prefix for the stage length. gen+judge logs: vLLM's
`Processed prompts: … N/TOTAL [` bars and submit.sh's final `[ts] done`. The SLOW and STALLED
flags are grid.step_rate / grid.STALL_S, as in `grid.py status`. Tooling only: nothing here
feeds a reported number.

TensorBoard: `--tb <absolute dir>` also writes every train chain as one run per cell,
`<dir>/<prompt>/<base>/<pretraining seed>-s<k>` (e.g. `008/dclm-baseline-1B/default-s1`), a
retried stage going to `…-s<k>-r2`. Tags: sft/{loss,s_per_step,lr,tokens_per_s},
dpo/{loss,s_per_step,lr}. `sft/loss` is open-instruct's reduce_loss=sum (a sum over the step's
label tokens), so bases are not comparable on it, and the log gives no per-step label-token
count. tensorboard is not in uv.lock; it runs as uv's ephemeral overlay:

    watch -n 600 uv run --frozen --with tensorboard python scripts/monitor_logs.py \
        --out /home/harrychang08/castor/runs/monitor/index.html --tb /work/harrychang08/castor/runs/monitor/tb2
    uvx tensorboard --logdir /work/harrychang08/castor/runs/monitor/tb2 --port 16006 --host 127.0.0.1

W&B: `--wandb ENTITY/PROJECT` with `--tb` mirrors the same runs to a W&B project (overlay
`--with wandb`). Training scalars only, never generations. Run local files go to `<tb>/../wandb`.

    watch -n 600 uv run --frozen --with tensorboard --with wandb python scripts/monitor_logs.py \
        --out /home/harrychang08/castor/runs/monitor/index.html \
        --tb /work/harrychang08/castor/runs/monitor/tb2 --wandb <entity>/castor
"""

import argparse
import html
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from castor import grid

NAME_RE = re.compile(r"^(\d{3})-(train|gj|genjudge)-(.+)-(\d+)\.out$")
HEADER = re.compile(r"open-instruct \w+ \| (sft|dpo) \|")
STEP = re.compile(r"(?:(\d+)/(\d+) \[[^\]]*\])?(\d\d)/(\d\d)/(\d{4}) (\d\d):(\d\d):(\d\d) - INFO - "
                  r"__main__ -\s+Step: (\d+), LR: ([0-9.eE+-]+), Loss: ([0-9.eE+-]+|nan)(?:, TPS: ([0-9.eE+-]+))?")
BAR = re.compile(r"Processed prompts:.*?(\d+)/(\d+) \[")
TAG = re.compile(r"^(?:(?P<recipe>.+)-)?(?P<size>\d+[MB])-(?P<seed>default|(?:small|large)-aux-\d+)-s(?P<k>\d+)$")
DONE = re.compile(r"^\[[0-9T:+-]+\] done\s*$", re.MULTILINE)
POINTS = 200  # bins per series on the page
COLORS = {"loss": "#2a78d6", "rate": "#eb6834"}  # dataviz reference palette, slots 1-2
STATUS = {"SLOW": "#ec835a", "STALLED": "#d03b3b", "FAILED": "#d03b3b", "PENDING": "#fab219"}


def parse_train(text: str) -> dict[str, dict]:
    """stage -> {"total": tqdm's stage length or None, "steps": {step: (loss, datetime)},
    "raw": {step: (lr, TPS or None)}}; lines before the first header are ignored; a repeated step's last occurrence wins."""
    out: dict[str, dict] = {}
    stage = None
    for line in text.splitlines():  # tqdm's \r is a line break too
        if h := HEADER.search(line):
            stage = h.group(1)
            out.setdefault(stage, {"total": None, "steps": {}, "raw": {}})
            continue
        if stage and (m := STEP.search(line)):
            _, tot, mo, d, y, hh, mm, ss, n, lr, loss, tps = m.groups()
            if tot:
                out[stage]["total"] = int(tot)
            ts = datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss))  # noqa: DTZ001
            out[stage]["steps"][int(n)] = (float(loss), ts)
            out[stage]["raw"][int(n)] = (float(lr), float(tps) if tps else None)
    return out


def parse_gj(text: str) -> dict:
    """{"bars": [(n, total)] (last state of each progress bar), "done": submit.sh's done line}."""
    bars: list[tuple[int, int]] = []
    for m in BAR.finditer(text):  # tqdm rewrites the bar on \r: a new bar starts when it resets
        n, tot = int(m.group(1)), int(m.group(2))
        if bars and tot == bars[-1][1] and n >= bars[-1][0]:
            bars[-1] = (n, tot)
        else:
            bars.append((n, tot))
    return {"bars": bars, "done": bool(DONE.search(text))}


def merge(parsed: list[dict[str, dict]]) -> dict[str, dict]:
    """Segments in job-id order: steps of a stage merged, later segments win."""
    out: dict[str, dict] = {}
    for p in parsed:
        for stage, d in p.items():
            o = out.setdefault(stage, {"total": None, "steps": {}})
            o["total"] = d["total"] or o["total"]
            o["steps"].update(d["steps"])
    return out


def binned(steps: dict[int, tuple[float, datetime]], k: int = POINTS) -> list[tuple]:
    """[(step, mean loss, s/step)] over k equal-width bins; s/step from the bin's first and last
    timestamp (None in a bin with one step or a clock jump backwards)."""
    xs = sorted(steps)
    if not xs:
        return []
    w = max(1, -(-len(xs) // k))
    out = []
    for i in range(0, len(xs), w):
        chunk = xs[i:i + w]
        vals = [steps[s][0] for s in chunk if steps[s][0] == steps[s][0]]  # drop nan
        dt = (steps[chunk[-1]][1] - steps[chunk[0]][1]).total_seconds()
        ds = chunk[-1] - chunk[0]
        rate = dt / ds if ds > 0 and dt >= 0 else None
        out.append((chunk[-1], sum(vals) / len(vals) if vals else None, rate))
    return out


def run_name(name: str) -> str:
    """Train job name -> TensorBoard run `<prompt>/<base>/<pretraining seed>-s<k>`. A tag without a
    data recipe (`150M-default-s1`) takes its base from ladder.SIZES; an unknown tag stays whole."""
    prompt, _, tag = name.split("-", 2)
    if not (m := TAG.match(tag)):
        return f"{prompt}/{tag}"
    if m["recipe"]:
        base = f"{m['recipe']}-{m['size']}"
    else:
        from castor import ladder

        base = ladder.SIZES[m["size"]][0].split("DataDecide-", 1)[1]
    return f"{prompt}/{base}/{m['seed']}-s{m['k']}"


def stage_points(r: dict, last: int) -> list[tuple[int, datetime, dict]]:
    """[(step, time, {scalar: value})] after step `last`, in step order: lr and tokens_per_s per step;
    loss and s_per_step on bins of width ceil(stage total / POINTS) steps (the page's bins once the
    stage is complete), only once a bin is complete."""
    top, total = max(r["steps"]), r["total"]
    w = max(1, -(-(total or 0) // POINTS))
    ends = [e for e in range(w, top + 1, w) if e > last]
    if total and top >= total and total % w:
        ends.append(total)  # the stage's short last bin
    pts = {s: (r["steps"][s][1], dict(zip(("lr", "tokens_per_s"), r["raw"][s], strict=True)))
           for s in sorted(r["steps"]) if s > last}
    for e in ends:
        chunk = {s: r["steps"][s] for s in range(e - w + 1, e + 1) if s in r["steps"]}
        for s, y, v in binned(chunk, k=1):
            pts.setdefault(s, (chunk[s][1], {}))[1].update(loss=y, s_per_step=v)
    return [(s, ts, {k: v for k, v in vals.items() if v is not None}) for s, (ts, vals) in sorted(pts.items())]


class TBSink:
    """TensorBoard event files in the run dir; marker last_step.json."""
    mark = "last_step.json"

    def write(self, d: Path, run: str, stage: str, pts: list) -> None:
        from torch.utils.tensorboard import SummaryWriter  # only with --tb, from the overlay

        wr = SummaryWriter(str(d))
        for s, ts, vals in pts:
            for k, v in vals.items():
                wr.add_scalar(f"{stage}/{k}", v, s, walltime=ts.timestamp())
        wr.close()


class WandbSink:
    """The same scalars as one W&B run per TensorBoard run, resumed by a fixed id so each pass
    appends; x axis per stage is `<stage>/step` (define_metric), so DPO restarting at step 1 after SFT
    is not dropped. Run = `<prompt>/<base>/<seed>-s<k>`, group `<prompt>/<base>`; marker wandb_step.json.
    Only training scalars leave the cluster, never generations."""
    mark = "wandb_step.json"

    def __init__(self, target: str, local_dir: Path):
        self.entity, _, self.project = target.rpartition("/")
        self.local_dir = local_dir

    def write(self, d: Path, run: str, stage: str, pts: list) -> None:
        import wandb  # only with --wandb, from the overlay

        prompt, *rest = run.split("/")
        self.local_dir.mkdir(parents=True, exist_ok=True)
        r = wandb.init(entity=self.entity or None, project=self.project, id=re.sub(r"[^\w-]", "_", run),
                       name=run, group="/".join([prompt, *rest[:1]]), job_type=prompt, resume="allow",
                       config={"prompt": prompt, "base": rest[0] if rest else None, "cell": run},
                       dir=str(self.local_dir), settings=wandb.Settings(silent=True, console="off", x_disable_stats=True))
        r.define_metric(f"{stage}/step")
        r.define_metric(f"{stage}/*", step_metric=f"{stage}/step")
        for s, _, vals in pts:
            r.log({f"{stage}/step": s} | {f"{stage}/{k}": v for k, v in vals.items()})
        r.finish()


def write_tb(name: str, logs: list[Path], tb: Path, sinks: list | None = None) -> int:
    """One run per cell, <tb>/<run_name>, tags <stage>/<scalar> (stage_points); a segment whose stage
    restarts at or below that stage's last step (a retry) goes to <run>-r2, -r3, ... Each sink keeps
    its own marker <run>/<sink.mark> ({stage: step}), so each pass appends only newer steps and a sink
    added later (W&B) backfills from step 0. Returns the number of scalars written over all sinks."""
    sinks = sinks if sinks is not None else [TBSink()]
    run = run_name(name)
    runs: dict[str, list[dict]] = {}
    for p in logs:
        for stage, d in parse_train(p.read_text(errors="replace")).items():
            rs = runs.setdefault(stage, [])
            if not d["steps"]:
                continue
            if not rs or min(d["steps"]) <= max(rs[-1]["steps"]):
                rs.append({"total": None, "steps": {}, "raw": {}})
            rs[-1]["total"] = d["total"] or rs[-1]["total"]
            rs[-1]["steps"].update(d["steps"])
            rs[-1]["raw"].update(d["raw"])
    n = 0
    for stage, rs in runs.items():
        for i, r in enumerate(rs):
            name_i = run if i == 0 else f"{run}-r{i + 1}"
            d = tb / name_i
            for sink in sinks:
                mark = d / sink.mark
                marks = json.loads(mark.read_text()) if mark.exists() else {}
                last, top = marks.get(stage, 0), max(r["steps"])
                if top <= last:
                    continue
                pts = stage_points(r, last)
                d.mkdir(parents=True, exist_ok=True)
                sink.write(d, name_i, stage, pts)
                n += sum(len(v) for _, _, v in pts)
                mark.write_text(json.dumps(marks | {stage: top}))
    return n


def chains(log_dir: Path, prompts: set[str]) -> dict[str, list[Path]]:
    """job name -> its log segments in job-id order."""
    out: dict[str, list[Path]] = defaultdict(list)
    for p in log_dir.iterdir():
        m = NAME_RE.match(p.name)
        if m and m.group(1) in prompts:
            out[f"{m.group(1)}-{m.group(2)}-{m.group(3)}"].append(p)
    return {k: sorted(v, key=lambda p: int(NAME_RE.match(p.name).group(4))) for k, v in out.items()}


def slurm(prompts: set[str], since: str) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """({job name: squeue jobs}, {job id: sacct state}); both empty without Slurm."""
    queue: dict[str, list[dict]] = defaultdict(list)
    ended: dict[str, str] = {}
    try:
        for j in grid.squeue_list():
            if j["name"].split("-")[0] in prompts:
                queue[j["name"]].append(j)
        r = subprocess.run(["sacct", "-X", "-n", "-P", "-S", since, "-o", "JobID,State"],
                           capture_output=True, text=True, check=True)
        for line in r.stdout.splitlines():
            jid, state = line.split("|")[:2]
            ended[jid] = state.split()[0]
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return queue, ended


def train_row(name: str, logs: list[Path], jobs: list[dict], ended: dict[str, str],
              now: datetime) -> dict:
    texts = [p.read_text(errors="replace") for p in logs]
    stages = merge([parse_train(t) for t in texts])
    cur = next((s for s in ("dpo", "sft") if stages.get(s, {}).get("steps")), None)  # header, no step yet: not current
    row = job_row(name, logs, jobs, ended, now)
    row["stages"] = {s: binned(d["steps"]) for s, d in stages.items()}
    if cur:
        st = stages[cur]
        last = max(st["steps"])
        rate = grid.step_rate("\n".join(texts)) if row["state"] == "RUNNING" else None
        spp = (rate or {}).get("s_per_step")
        row |= {"stage": cur, "step": last, "total": st["total"], "loss": st["steps"][last][0],
                "s_per_step": spp, "eta_h": None}
        if rate and rate.get("slow"):
            row["flags"].append("SLOW")
        if spp and st["total"] and row["state"] == "RUNNING":
            row["eta_h"] = (st["total"] - last) * spp / 3600
    return row


def gj_row(name: str, logs: list[Path], jobs: list[dict], ended: dict[str, str],
           now: datetime) -> dict:
    row = job_row(name, logs, jobs, ended, now)
    g = parse_gj(logs[-1].read_text(errors="replace"))
    row["bars"], row["done"] = g["bars"], g["done"]
    return row


def job_row(name: str, logs: list[Path], jobs: list[dict], ended: dict[str, str],
            now: datetime) -> dict:
    """Slurm state of the chain's latest job (squeue first, then sacct), log age, flags."""
    jid = NAME_RE.match(logs[-1].name).group(4) if logs else None
    live = {j["id"]: j for j in jobs}
    pending = [j for j in jobs if j["state"] == "PENDING"]
    if jid in live:
        state, where = live[jid]["state"], live[jid]["where"]
    elif pending:
        state, where, jid = "PENDING", pending[-1]["where"], pending[-1]["id"]
    else:
        state, where = ended.get(jid, "ended?"), ""
    age = (now - datetime.fromtimestamp(logs[-1].stat().st_mtime)).total_seconds() if logs else None  # noqa: DTZ006
    flags = []
    if state == "RUNNING" and age is not None and age > grid.STALL_S:
        flags.append("STALLED")
    if state in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"):
        flags.append("FAILED")
    if state == "PENDING":
        flags.append("PENDING")
    return {"name": name, "job": jid, "state": state, "where": where, "log_age_s": age,
            "flags": flags, "segments": len(logs), "logs": [str(p) for p in logs]}


def summary(row: dict) -> str:
    j = f"{row['job'] or '-'} {row['state']} {row['where']}".strip()
    seg = f" ({row['segments']} segments)" if row["segments"] > 1 else ""
    if "bars" in row:
        b = row["bars"]
        prog = ("done" if row["done"] else f"bar {len(b)} {b[-1][0]}/{b[-1][1]}" if b else "loading")
        body = f"gj {prog}"
    elif row.get("stage"):
        tot = f"/{row['total']}" if row["total"] else ""
        spp = f" {row['s_per_step']:.2f} s/step" if row.get("s_per_step") else ""
        eta = f" ETA {row['eta_h']:.1f} h" if row.get("eta_h") is not None else ""
        body = f"{row['stage']} step {row['step']}{tot} loss {row['loss']:.4g}{spp}{eta}"
    else:
        body = "no step yet"
    flags = " ".join(row["flags"])
    return f"{row['name']:<48} {body:<52} {j}{seg} {flags}".rstrip()


PW, PH, PAD = 300, 120, (44, 8, 10, 26)  # panel width, height; left, right, top, bottom


def svg(series: list[tuple[int, float]], color: str, ylabel: str, total: int | None) -> str:
    """One polyline panel; the y range is the 2nd-98th percentile so a spike does not flatten it."""
    pts = [(x, y) for x, y in series if y is not None]
    if not pts:
        return f'<svg width="{PW}" height="{PH}"><text x="{PW / 2}" y="{PH / 2}" class="mu">no {ylabel}</text></svg>'
    left, right, top, bottom = PAD
    w, h = PW - left - right, PH - top - bottom
    ys = sorted(y for _, y in pts)
    lo, hi = ys[int(0.02 * (len(ys) - 1))], ys[int(0.98 * (len(ys) - 1))]
    hi = max(hi, lo + 1e-9)
    x0, x1 = pts[0][0], max(total or 0, pts[-1][0], pts[0][0] + 1)

    def fx(x):
        return left + (x - x0) / (x1 - x0) * w

    def fy(y):
        return top + (hi - min(max(y, lo), hi)) / (hi - lo) * h

    out = [f'<svg width="{PW}" height="{PH}" class="p"><title>{ylabel}</title>']
    for t in (lo, (lo + hi) / 2, hi):
        out.append(f'<line x1="{left}" x2="{left + w}" y1="{fy(t):.1f}" y2="{fy(t):.1f}" class="g"/>'
                   f'<text x="{left - 3}" y="{fy(t) + 3:.1f}" text-anchor="end" class="mu">{t:.4g}</text>')
    out.append(f'<text x="{left}" y="{PH - 8}" class="mu">{x0}</text>'
               f'<text x="{left + w}" y="{PH - 8}" text-anchor="end" class="mu">{x1}</text>'
               f'<text x="{left + w / 2}" y="{PH - 8}" text-anchor="middle" class="mu">{ylabel}</text>')
    line = " ".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in pts)
    out.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2"/></svg>')
    return "".join(out)


def page(rows: list[dict], now: datetime, log_dir: Path) -> str:
    css = (":root{--bg:#fcfcfb;--fg:#0b0b0b;--mu:#52514e;--g:#e6e5e1}"
           "@media(prefers-color-scheme:dark){:root{--bg:#1a1a19;--fg:#fff;--mu:#c3c2b7;--g:#383835}}"
           "body{background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif;margin:16px}"
           "pre{white-space:pre-wrap}.c{border-top:1px solid var(--g);padding:6px 0}"
           ".mu{fill:var(--mu);color:var(--mu);font-size:10px}.g{stroke:var(--g);stroke-width:.5}"
           "svg{vertical-align:top;margin-right:8px}.f{padding:1px 6px;border-radius:4px;color:#fff;margin-left:6px}")
    out = [f"<!doctype html><meta charset=utf-8><title>CASTOR monitor</title><style>{css}</style>",
           (f"<h2>CASTOR train / gen+judge monitor</h2><p class=mu>{now:%Y-%m-%d %H:%M:%S} · "
            f"{len(rows)} chains · logs {html.escape(str(log_dir))} · refresh: "
            "<code>watch -n 600 uv run --frozen python scripts/monitor_logs.py --out …</code></p>"),
           "<pre>" + html.escape("\n".join(summary(r) for r in rows)) + "</pre>"]
    for r in rows:
        flags = "".join(f'<span class=f style="background:{STATUS[f]}">{f}</span>' for f in r["flags"])
        out.append(f'<div class=c id="{r["name"]}"><b>{html.escape(r["name"])}</b> '
                   f'<span class=mu>{html.escape(summary(r)[49:])}</span>{flags}<br>')
        for stage, pts in r.get("stages", {}).items():
            tot = r["total"] if stage == r.get("stage") else None
            out.append(f'<span class=mu>{stage}</span> ' + svg([(s, y) for s, y, _ in pts], COLORS["loss"], "loss", tot)
                       + svg([(s, v) for s, _, v in pts], COLORS["rate"], "s/step", tot))
        out.append("</div>")
    return "\n".join(out) + "\n"


def collect(log_dir: Path, prompts: set[str], now: datetime) -> list[dict]:
    ch = chains(log_dir, prompts)
    oldest = min((p.stat().st_mtime for v in ch.values() for p in v), default=now.timestamp())
    since = (datetime.fromtimestamp(oldest) - timedelta(days=7)).strftime("%Y-%m-%d")  # noqa: DTZ006
    queue, ended = slurm(prompts, since)
    rows = []
    for name in sorted(set(ch) | set(queue)):
        logs, jobs = ch.get(name, []), queue.get(name, [])
        kind = name.split("-")[1]
        if not logs:
            rows.append(job_row(name, logs, jobs, ended, now))
        elif kind == "train":
            rows.append(train_row(name, logs, jobs, ended, now))
        else:
            rows.append(gj_row(name, logs, jobs, ended, now))
    rank = {"RUNNING": 1, "PENDING": 2}
    rows.sort(key=lambda r: (not r["flags"] or r["flags"] == ["PENDING"], rank.get(r["state"], 3), r["name"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="absolute path of the HTML page (omit: stdout summary only)")
    ap.add_argument("--prompts", default="008,016")
    ap.add_argument("--tb", type=Path, help="absolute dir for TensorBoard runs (031; needs --with tensorboard)")
    ap.add_argument("--wandb", metavar="ENTITY/PROJECT",
                    help="also mirror the --tb runs to W&B (034; needs --tb, --with wandb and `wandb login`)")
    ap.add_argument("--log-dir", type=Path, default=grid.main_repo() / "logs/nchc")
    a = ap.parse_args()
    if a.wandb and not a.tb:
        ap.error("--wandb needs --tb (its markers live in the TensorBoard run dirs)")
    now = datetime.now()  # noqa: DTZ005
    rows = collect(a.log_dir, set(a.prompts.split(",")), now)
    print("\n".join(summary(r) for r in rows) or "no chains")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(page(rows, now, a.log_dir))
        print(f"page: {a.out} ({a.out.stat().st_size / 1e6:.2f} MB)")
    if a.tb:
        ch = chains(a.log_dir, set(a.prompts.split(",")))
        sinks = [TBSink()] + ([WandbSink(a.wandb, a.tb.parent / "wandb")] if a.wandb else [])
        n = sum(write_tb(k, v, a.tb, sinks) for k, v in ch.items() if k.split("-")[1] == "train")
        print(f"tensorboard{' + wandb ' + a.wandb if a.wandb else ''}: {a.tb} ({n} new scalars)")


if __name__ == "__main__":
    main()
