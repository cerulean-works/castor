"""The training profile and the launcher-candidate smoke table, from the job logs and run
directories (no GPU). Screen rule, fixed before data: with F1, F2 the per-step SFT
losses of the two current-config smokes and d_F = max_t |F1 - F2|, candidate C passes iff
max_t |C - F1| <= 1.25 d_F and max_t |C - F2| <= 1.25 d_F.

    uv run --frozen python scripts/smokes_024.py --out $PWD/runs/024
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from castor import grid, posttrain, provenance

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "logs" / "nchc"
SMOKES = Path("/work/harrychang08/castor/models/posttrain/smoke/024")
RUN = "dd-dclm-1B-seed-default/sft/seed1"
FLOOR = ("cur1", "cur2")
FACTOR = 1.25
STEP = re.compile(r"(\d\d/\d\d/\d{4} \d\d:\d\d:\d\d) - INFO - __main__ -\s+Step: (\d+), LR: \S+, Loss: ([\d.]+), TPS: ([\d.]+)")


def steps(text: str) -> dict[int, dict]:
    """Step -> {t (epoch s), loss, tps} from open-instruct's per-step log lines (first stage only)."""
    out = {}
    for m in STEP.finditer(text):
        s = int(m.group(2))
        if s in out:  # a second training stage in the same log (never for SFT-only smokes)
            break
        out[s] = {"t": time.mktime(time.strptime(m.group(1), "%m/%d/%Y %H:%M:%S")),
                  "loss": float(m.group(3)), "tps": float(m.group(4))}
    return out


def rate(st: dict[int, dict], a: int = 10, b: int = 50) -> float:
    """Seconds per optimizer step over steps a -> b (1-s log resolution)."""
    return (st[b]["t"] - st[a]["t"]) / (b - a)


def screen(c: list[float], f1: list[float], f2: list[float]) -> dict:
    d_f = max(abs(x - y) for x, y in zip(f1, f2, strict=True))
    d1 = max(abs(x - y) for x, y in zip(c, f1, strict=True))
    d2 = max(abs(x - y) for x, y in zip(c, f2, strict=True))
    fm = [(x + y) / 2 for x, y in zip(f1, f2, strict=True)]
    signed = [x - y for x, y in zip(c, fm, strict=True)]
    return {"d_floor": d_f, "d_cur1": d1, "d_cur2": d2, "bound": FACTOR * d_f,
            "pass": d1 <= FACTOR * d_f and d2 <= FACTOR * d_f,
            # descriptive, not part of the rule: systematic offset from the floor mean
            "mean_signed_vs_floor_mean": statistics.mean(signed),
            "n_positive_of_48": sum(x > 0 for x in signed[2:])}


def log_of(name: str) -> Path:
    logs = sorted(LOGS.glob(f"024-smoke-{name}-[0-9]*.out"))
    assert len(logs) == 1, f"{name}: {logs}"
    return logs[0]


def job_info(log: Path) -> dict:
    s = grid.sacct(log.stem.rsplit("-", 1)[1])
    return {k: s[k] for k in ("job", "elapsed", "node", "state")}


def run_files(name: str) -> dict:
    d = SMOKES / name / RUN
    p = json.loads((d / "provenance.json").read_text())
    return {"oi_args": (d / "oi_args.txt").read_text().replace(str(SMOKES / name), "<root>"),
            "ds_config": json.loads((d / "ds_config.json").read_text()),
            "ds_sha256": posttrain.sha256_file(d / "ds_config.json"),
            "ds_path": p["open_instruct"]["deepspeed_config"], "env": p["launcher"]["env"],
            "git_commit": p["git_commit"]}


def ds_diff(a: dict, b: dict, prefix: str = "") -> dict:
    out = {}
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if isinstance(x, dict) and isinstance(y, dict):
            out |= ds_diff(x, y, f"{prefix}{k}.")
        elif x != y:
            out[f"{prefix}{k}"] = [x, y]
    return out


def candidates(names: list[str]) -> dict:
    st = {n: steps(log_of(n).read_text()) for n in names}
    for n, s in st.items():
        assert sorted(s) == list(range(1, 51)), f"{n}: steps {sorted(s)[:3]}..{len(s)}"
    loss = {n: [st[n][t]["loss"] for t in range(1, 51)] for n in names}
    ref = run_files(FLOOR[0])
    rate_cur = statistics.mean(rate(st[f]) for f in FLOOR)
    out = {}
    for n in names:
        rf = run_files(n)
        out[n] = {"s_per_step_10_50": rate(st[n]), "speedup_vs_cur": rate_cur / rate(st[n]),
                  "tps_step50": st[n][50]["tps"], "loss": loss[n],
                  "screen": screen(loss[n], loss[FLOOR[0]], loss[FLOOR[1]]),
                  "oi_args_identical": rf["oi_args"] == ref["oi_args"],
                  "ds_diff_vs_cur": ds_diff(ref["ds_config"], rf["ds_config"]),
                  "env_diff_vs_cur": {k: [ref["env"].get(k), v] for k, v in rf["env"].items()
                                      if ref["env"].get(k) != v},
                  "ds_path": rf["ds_path"], "ds_sha256": rf["ds_sha256"],
                  "run_git_commit": rf["git_commit"], **job_info(log_of(n))}
    return out


def profile() -> dict:
    """DeepSpeed wall-clock breakdown (rank 0, steps 10-50 of run 1) and nvidia-smi dmon."""
    logs = sorted(LOGS.glob("024-profile-*.out"))
    assert len(logs) == 1, logs
    text = logs[0].read_text()
    br = re.findall(r"fwd_microstep: ([\d.]+) \| bwd_microstep: ([\d.]+) \| bwd_inner_microstep: "
                    r"([\d.]+) \| bwd_allreduce_microstep: ([\d.]+) \| step_microstep: ([\d.]+)", text)
    cols = ["fwd", "bwd", "bwd_inner", "bwd_allreduce", "step"]
    rows = [dict(zip(cols, map(float, r), strict=True)) for r in br]
    tail = rows[len(rows) // 5:]  # drop warm-up (first 10 of 50 steps)
    st = steps(text)
    # nvidia-smi dmon (same node clock as the log) over steps 10 -> 50
    t10, t50 = (time.strftime("%H:%M:%S", time.localtime(st[k]["t"])) for k in (10, 50))
    sm: dict[int, list[float]] = {}
    for line in (SMOKES / "profile-ds" / "dmon.txt").read_text().splitlines():
        f = line.split()
        if line.startswith("#") or len(f) < 3 or not t10 <= f[0] <= t50:
            continue
        sm.setdefault(int(f[1]), []).append(float(f[2]))
    return {"log": str(logs[0]), "n_breakdown_lines": len(rows),
            "dmon_sm_util_pct_steps_10_50": {g: statistics.mean(v) for g, v in sorted(sm.items())},
            "breakdown_ms_median": {c: statistics.median(r[c] for r in tail) for c in cols},
            "s_per_step_10_50_with_timers": rate(st), "micro_steps_per_step": 8}


SFT_CACHE = Path("/work/harrychang08/castor/cache/open_instruct_datasets/6edaf45195")  # in every smoke log


def padding(cache: Path = SFT_CACHE, seed: int = 0) -> dict:
    """Padding of a per-device micro-batch (2 rows padded to the longer, DataCollatorForSeq2Seq
    padding="longest"): random pairs of the cached tokenized SFT set."""
    import datasets
    import numpy as np

    d = datasets.load_from_disk(str(cache))
    n_tok = np.array(d.map(lambda b: {"n": [len(x) for x in b["input_ids"]]}, batched=True,
                           num_proc=8, remove_columns=d.column_names)["n"])
    perm = np.random.default_rng(seed).permutation(len(n_tok))
    pairs = n_tok[perm[: len(n_tok) // 2 * 2]].reshape(-1, 2)
    return {"cache": str(cache), "rows": len(n_tok), "len_mean": float(n_tok.mean()),
            "len_median": float(np.median(n_tok)), "micro_batch": 2,
            "tokens_real_per_microbatch": float(pairs.sum(1).mean()),
            "tokens_padded_per_microbatch": float((2 * pairs.max(1)).mean()),
            "padding_fraction": float(1 - pairs.sum() / (2 * pairs.max(1)).sum())}


NSYS = SMOKES / "profile-nsys" / "nsys.sqlite"  # nsys stats export of the profile job's report


def _union(iv: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _length(iv: list[tuple[int, int]]) -> int:
    return sum(b - a for a, b in iv)


def _intersect(x: list[tuple[int, int]], y: list[tuple[int, int]]) -> int:
    i = j = tot = 0
    while i < len(x) and j < len(y):
        a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        tot += max(0, b - a)
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return tot


def category(name: str) -> str:
    n = name.lower()
    if "nccl" in n:
        return "nccl"
    if "flash" in n or "fmha" in n:
        return "attention"
    if any(k in n for k in ("gemm", "cutlass", "nvjet", "sm90_xmma", "cublas", "matmul")):
        return "gemm"
    if "adam" in n or "multi_tensor" in n:
        return "optimizer"
    return "other"


def nsys_profile(db: Path = NSYS, trim: float = 0.2) -> dict:
    """Per GPU over the training window (span of attention kernels, first `trim` dropped as
    warm-up): busy time by kernel category, communication not hidden under compute, and idle."""
    import sqlite3

    con = sqlite3.connect(db)
    rows = con.execute("SELECT k.deviceId, k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                       "JOIN StringIds s ON k.shortName = s.id").fetchall()
    per_dev: dict[int, dict] = {}
    for dev in sorted({r[0] for r in rows}):
        ks = [(a, b, category(n)) for d, a, b, n in rows if d == dev]
        att = sorted(a for a, b, c in ks if c == "attention")
        t0 = att[0] + trim * (att[-1] - att[0])
        t1 = max(b for a, b, c in ks if c == "attention")
        ks = [(max(a, t0), min(b, t1), c) for a, b, c in ks if b > t0 and a < t1]
        cats = {c: _union([(a, b) for a, b, cc in ks if cc == c]) for c in
                ("gemm", "attention", "optimizer", "other", "nccl")}
        compute = _union([iv for c, v in cats.items() if c != "nccl" for iv in v])
        busy = _union(compute + cats["nccl"])
        w = t1 - t0
        per_dev[dev] = {"window_s": w / 1e9,
                        **{f"{c}_frac": _length(v) / w for c, v in cats.items()},
                        "compute_frac": _length(compute) / w,
                        "nccl_exposed_frac": (_length(cats["nccl"]) - _intersect(cats["nccl"], compute)) / w,
                        "idle_frac": 1 - _length(busy) / w}
    keys = next(iter(per_dev.values())).keys()
    return {"db": str(db), "trim": trim, "per_gpu": per_dev,
            "mean": {k: statistics.mean(v[k] for v in per_dev.values()) for k in keys}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--names", nargs="*", default=["cur1", "cur2", "zero2", "zero1", "zero3t", "env"])
    a = ap.parse_args()
    assert a.out.is_absolute()
    meta = {"git_commit": provenance.git_commit(), "command": provenance.command()}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "candidates.json").write_text(json.dumps(candidates(a.names) | {"_meta": meta}, indent=1) + "\n")
    prof = profile()
    p = a.out / "profile.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    p.write_text(json.dumps(old | {"deepspeed_breakdown": prof, "padding": padding(), "nsys": nsys_profile(),
                                   "_meta": meta}, indent=1) + "\n")


if __name__ == "__main__":
    main()
