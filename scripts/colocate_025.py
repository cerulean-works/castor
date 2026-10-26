"""Co-location probe: the table from colocate.sh's per-copy logs (no GPU). Rules, fixed before
data: per-copy s/step over steps 10 -> 50 (µs arrival stamps), aggregate x =
sum_i (1/s_i) / (1/s_ref); overlap aggregate over W = [max_i t_i(10), min_i t_i(50)]; the go rule
uses the smaller; 024's smoke screen on each SFT copy's losses; projection
17.04/X_sft + 12.74/X_dpo + 1.77 < 0.7 * 31.57.

    uv run --frozen python scripts/colocate_025.py --out $PWD/runs/025/probe2
"""

import argparse
import itertools
import json
import re
import statistics
import sys
from pathlib import Path

from castor import grid, posttrain, provenance

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_025 as L25
import smokes_024 as S24

STEP = re.compile(r"^(\d+\.\d+) .*?(\d\d/\d\d/\d{4} \d\d:\d\d:\d\d) - INFO - __main__ -\s+Step: (\d+), "
                  r"LR: \S+, Loss: ([\d.]+)")
TRANSPORT = re.compile(r"NCCL INFO Channel \d+/\d+ : \d+\[\d+\] -> \d+\[\d+\] (?:\[\w+\] )?via (\S+)")
A, B = 10, 50
# 024's arm A (ZeRO-1 SFT) by stage, H200-h, and the rule's bound (stage split from provenance
# sums to 29.78; the arm's sacct total 29.80 is the cost of record)
SFT_H, DPO_H, GJ_H, TWIN_H = 17.04, 12.74, 1.77, 29.80 + 1.77
BOUND = 0.7 * TWIN_H
MIN_OVERLAP_S = 10.0  # an overlap window shorter than this gives no co-location credit (rule cannot fire)


def stages(text: str) -> list[dict[int, dict]]:
    """Per training stage (a step count restarting at 1 opens a new one): step -> {t: µs arrival
    stamp in epoch s, t1: open-instruct's own 1-s stamp, loss}."""
    import time
    out: list[dict[int, dict]] = []
    for line in text.split("\n"):  # not splitlines(): tqdm's \r would split a stamped line
        m = STEP.match(line)
        if not m:
            continue
        s = int(m.group(3))
        if not out or s in out[-1]:
            out.append({})
        out[-1][s] = {"t": float(m.group(1)), "loss": float(m.group(4)),
                      "t1": time.mktime(time.strptime(m.group(2), "%m/%d/%Y %H:%M:%S"))}
    return out


def read_log(p: Path) -> str:
    """newline="": read_text() would turn tqdm's \r into \n and cut the stamp off a step line."""
    with open(p, newline="") as f:
        return f.read()


def rate(st: dict[int, dict], key: str = "t") -> float:
    return (st[B][key] - st[A][key]) / (B - A)


def steps_done(st: dict[int, dict], t: float) -> float:
    """Steps completed by time t, linear between logged steps (A..B only)."""
    ks = sorted(k for k in st if A <= k <= B)
    if t <= st[ks[0]]["t"]:
        return float(ks[0])
    for k0, k1 in itertools.pairwise(ks):
        t0, t1 = st[k0]["t"], st[k1]["t"]
        if t <= t1:
            return k0 + (k1 - k0) * (t - t0) / (t1 - t0) if t1 > t0 else float(k1)
    return float(ks[-1])


def aggregate(copies: list[dict[int, dict]], s_ref: float) -> dict:
    """Per-copy-rate aggregate and the overlap aggregate (both relative to one job at s_ref)."""
    per = [rate(c) for c in copies]
    agg = sum(1 / s for s in per) * s_ref
    lo, hi = max(c[A]["t"] for c in copies), min(c[B]["t"] for c in copies)
    w = hi - lo
    if w >= MIN_OVERLAP_S:
        ov = sum(steps_done(c, hi) - steps_done(c, lo) for c in copies) / w * s_ref
    else:
        ov = None
    frac = [max(0.0, w) / (c[B]["t"] - c[A]["t"]) for c in copies]
    return {"s_per_step": per, "aggregate_x": agg, "overlap_window_s": w,
            "overlap_fraction_per_copy": frac, "overlap_aggregate_x": ov,
            "rule_x": min(agg, ov) if ov is not None else None,
            "start_offsets_step10_s": [c[A]["t"] - lo for c in copies]}


def projection(x_sft: float | None, x_dpo: float | None) -> dict:
    if x_sft is None:
        return {"cost_h": None, "bound_h": BOUND, "fires": False}
    cost = SFT_H / x_sft + DPO_H / (x_dpo or 1.0) + GJ_H
    return {"x_sft": x_sft, "x_dpo": x_dpo or 1.0, "x_dpo_measured": x_dpo is not None,
            "cost_h": cost, "ratio_vs_zero1": cost / TWIN_H, "bound_h": BOUND,
            "fires": cost < BOUND}


def peak_mem(smi: Path) -> dict[int, float]:
    """GPU index -> max memory.used (MiB) over the job (nvidia-smi -l 1)."""
    out: dict[int, float] = {}
    for line in smi.read_text().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 4 or not f[1].isdigit():
            continue
        out[int(f[1])] = max(out.get(int(f[1]), 0.0), float(f[2].split()[0]))
    return out


def transports(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in TRANSPORT.finditer(text):
        out[m.group(1)] = out.get(m.group(1), 0) + 1
    return out


def ref_stages(name: str) -> list[dict[int, dict]]:
    """024's single-job smoke, same parser semantics (its log has no µs stamps: t = 1-s stamp)."""
    out = []
    import time
    for line in S24.log_of(name).read_text().splitlines():
        m = re.search(r"(\d\d/\d\d/\d{4} \d\d:\d\d:\d\d) - INFO - __main__ -\s+Step: (\d+), LR: \S+, "
                      r"Loss: ([\d.]+)", line)
        if not m:
            continue
        s = int(m.group(2))
        if not out or s in out[-1]:
            out.append({})
        t = time.mktime(time.strptime(m.group(1), "%m/%d/%Y %H:%M:%S"))
        out[-1][s] = {"t": t, "t1": t, "loss": float(m.group(3))}
    return out


def run_files(d: Path, root: Path) -> dict:
    p = json.loads((d / "provenance.json").read_text())
    return {"oi_args": (d / "oi_args.txt").read_text().replace(str(root), "<root>"),
            "ds_sha256": posttrain.sha256_file(d / "ds_config.json"), "env": p["launcher"]["env"],
            "git_commit": p["git_commit"], "finished": "finished" in p}


def config(name: str) -> dict | None:
    mps, n, c24 = L25.COLOCATE[name]
    logs = L25.log_dir(name)
    if not (logs / "env.txt").exists():
        return None
    texts = [read_log(logs / f"c{i}.log") for i in range(1, n + 1)]
    sts = [stages(t) for t in texts]
    rcs = [(logs / f"c{i}.rc").read_text().strip() if (logs / f"c{i}.rc").exists() else "missing"
           for i in range(1, n + 1)]
    env = (logs / "env.txt").read_text()
    job = re.search(r"job (\d+)", env).group(1)
    ref = ref_stages(c24)
    f1, f2 = (ref_stages(f)[0] for f in S24.FLOOR)
    s_ref_sft = (statistics.mean(rate(r[0], "t1") for r in (ref_stages("cur1"), ref_stages("cur2")))
                 if c24 == "cur1" else rate(ref[0], "t1"))
    out: dict = {"mps": mps, "copies": n, "candidate_024": c24, "rc": rcs, "env_txt": env,
                 "peak_mem_mib": peak_mem(logs / "smi.csv"), **grid.sacct(job)}
    complete = all(len(s) >= len(L25.L24.smoke_stages(c24)) and all(sorted(st) == list(range(1, 51))
                   for st in s) for s in sts) and all(r == "0" for r in rcs)
    out["complete"] = complete
    out["transport"] = [transports(t) for t in texts]
    for k, stg in enumerate(L25.L24.smoke_stages(c24)):
        cs = [s[k] for s in sts if len(s) > k and B in s[k] and A in s[k]]
        if len(cs) != n:
            out[stg] = {"incomplete": True}
            continue
        s_ref = s_ref_sft if stg == "sft" else rate(ref[1], "t1")
        o = {"s_ref": s_ref, **aggregate(cs, s_ref),
             "s_per_step_1s_stamps": [rate(c, "t1") for c in cs],
             "loss": [[c[t]["loss"] for t in range(1, 51)] for c in cs]}
        if stg == "sft":
            fl1, fl2 = ([f[t]["loss"] for t in range(1, 51)] for f in (f1, f2))
            o["screen"] = [S24.screen(lo, fl1, fl2) for lo in o["loss"]]
            o["screen_all_pass"] = all(s["pass"] for s in o["screen"])
        else:
            rl = [ref[1][t]["loss"] for t in range(1, 51)]
            o["dpo_max_abs_vs_024_chain"] = [max(abs(x - y) for x, y in zip(lo, rl, strict=True))
                                             for lo in o["loss"]]
        out[stg] = o
    # byte-identity of the argument list / DS config / launcher env with 024's smoke of the candidate
    ref_root = S24.SMOKES / c24
    ident = {}
    for stg in L25.L24.smoke_stages(c24):
        sub = L25.L24.BASE.out(stg, ref_root).relative_to(ref_root)
        r = run_files(ref_root / sub, ref_root)
        cp = [run_files(L25.copy_root(name, i) / sub, L25.copy_root(name, i)) for i in range(1, n + 1)]
        ident[stg] = {"oi_args_identical": all(c["oi_args"] == r["oi_args"] for c in cp),
                      "ds_identical": all(c["ds_sha256"] == r["ds_sha256"] for c in cp),
                      "env_identical": all(c["env"] == r["env"] for c in cp),
                      "ds_sha256": r["ds_sha256"], "run_git_commit": sorted({c["git_commit"] for c in cp})}
    out["identity_vs_024"] = ident
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert a.out.is_absolute()
    res = {n: c for n in L25.COLOCATE if (c := config(n)) is not None}
    rule = {}
    for n, c in res.items():
        if c["candidate_024"] == "cur1" or "-chain" in n:
            continue
        ok = c["complete"] and c["sft"].get("screen_all_pass", False)
        chain = res.get(f"{n}-chain")
        x_dpo = chain["dpo"].get("rule_x") if chain and chain.get("complete") else None
        rule[n] = {"eligible": ok, **projection(c["sft"].get("rule_x") if ok else None, x_dpo)}
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {"script": "scripts/colocate_025.py", "git_commit": provenance.git_commit(),
            "command": provenance.command(), "bound_h": BOUND}
    (a.out / "colocate.json").write_text(json.dumps({"configs": res, "rule": rule, "_meta": meta},
                                                    indent=1) + "\n")
    for n, c in res.items():
        for stg in ("sft", "dpo"):
            o = c.get(stg)
            if not o or o.get("incomplete"):
                continue
            print(f"{n}\t{stg}\tper-copy {[round(s, 3) for s in o["s_per_step"]]}\tagg {o['aggregate_x']:.2f}\t"
                  f"overlap {o['overlap_aggregate_x']}\tW {o['overlap_window_s']:.1f}s\t"
                  f"screen {o.get('screen_all_pass')}")
    print(json.dumps(rule, indent=1))


if __name__ == "__main__":
    main()
