"""The measured cost per ladder twin from one 50-step dry run per size, in H200-hours and SU at
60 per H200-hour, beside the earlier labelled projection. Pure CPU; reads the dry runs' Slurm
logs only.

    uv run --frozen python scripts/smoke_cost_016.py --log 150M=logs/nchc/<150M log> \\
        --log 300M=logs/nchc/<300M log> --out /abs/path/runs/016/smoke_cost_<config>.json

Method: the steady step time of each stage is the mean over steps 10 → 50 of the dry run
(timestamps of open-instruct's `Step: N` lines); the DPO reference-logprob caching is the
interval from DPO's `Num examples` line to its step 1; a twin is SFT_STEPS SFT steps + DPO_STEPS
DPO steps (the data and the effective batch of 128 are the same at every size) + caching +
START_H start-up and saves, × 8 GPUs, + GEN_JUDGE_H200H generation and judging (1B's measured
value as an upper bound; the judge does not depend on size). The projection it is compared with
is the ZeRO-3 SFT hours ÷ 3.124.
"""

import argparse
import json
import math
import re
import time
from collections.abc import Mapping
from pathlib import Path

from castor import grid, ladder, provenance

SFT_EXAMPLES, DPO_PAIRS, BATCH = 853_840, 370_845, 128
SFT_STEPS = 2 * math.ceil(SFT_EXAMPLES / BATCH)  # 2 epochs = 13,342
DPO_STEPS = math.ceil(DPO_PAIRS / BATCH)  # 1 epoch = 2,898
START_H = 0.03  # start-up and saves per twin, measured
GEN_JUDGE_H200H = 1.83  # 1B measured, upper bound at smaller sizes
GPUS = 8
SU_PER_H200H = 60  # NCHC billing rate
RATE_STEPS = (10, 50)
PROJECTION_D067C = {"150M": 17.37, "300M": 22.69}  # H200-h per twin, labelled projection
TWINS_PER_SIZE = len(ladder.PRETRAIN_SEEDS) * len(ladder.POSTTRAIN_SEEDS)  # 12
N_TWINS = TWINS_PER_SIZE * len(ladder.SIZES)  # 24: the ladder
TOTAL_H200H, TOTAL_SU = f"total_{N_TWINS}_h200h", f"total_{N_TWINS}_su_at_60"  # one spelling
assert set(PROJECTION_D067C) == set(ladder.SIZES)

_HDR = re.compile(r"^open-instruct (\w+) \| (sft|dpo) \| (\d+) GPUs \| ds (\w+) \|")


def parse_log(text: str) -> dict:
    """Per stage: the step timestamps, the `Num examples` timestamp, the DeepSpeed sha prefix
    and GPU count from train.sh's header line, the loss at steps 1 and 50."""
    stages, cur = {}, None
    for line in text.splitlines():
        h = _HDR.match(line)
        if h:
            cur = h.group(2)
            stages[cur] = {"oi_commit": h.group(1), "gpus": int(h.group(3)), "ds": h.group(4),
                           "steps": {}, "loss": {}, "num_examples_at": None}
            continue
        m = grid.LOG_LINE_RE.search(line)  # the one copy of open-instruct's log-line format
        if not m or cur is None:
            continue
        ts, msg = grid.log_time(m.group(1)), m.group(2)
        if msg.startswith("Num examples"):
            stages[cur]["num_examples_at"] = ts
        s = re.match(r"Step: (\d+), LR: \S+, Loss: (\S+)", msg)
        if s:
            stages[cur]["steps"][int(s.group(1))] = ts
            stages[cur]["loss"][int(s.group(1))] = float(s.group(2).rstrip(","))
    assert set(stages) == {"sft", "dpo"}, f"stages found: {sorted(stages)}"
    return stages


def step_time(st: dict, a: int = RATE_STEPS[0], b: int = RATE_STEPS[1]) -> float:
    """Steady seconds per step: (t_b − t_a) / (b − a)."""
    assert a in st["steps"] and b in st["steps"], f"steps {a}, {b} not both in the log"
    return (st["steps"][b] - st["steps"][a]).total_seconds() / (b - a)


def caching_s(dpo: dict) -> float:
    assert dpo["num_examples_at"] is not None and 1 in dpo["steps"], "DPO caching interval"
    return (dpo["steps"][1] - dpo["num_examples_at"]).total_seconds()


def cost(stages: dict, projection: float | None) -> dict:
    """The per-twin cost from one dry run's stages."""
    assert stages["sft"]["gpus"] == stages["dpo"]["gpus"] == GPUS, "8 GPUs per job (D059a)"
    s_sft, s_dpo, cache = step_time(stages["sft"]), step_time(stages["dpo"]), caching_s(
        stages["dpo"])
    sft_h = SFT_STEPS * s_sft / 3600
    dpo_h = DPO_STEPS * s_dpo / 3600 + cache / 3600
    train = GPUS * (sft_h + dpo_h + START_H)
    all_in = train + GEN_JUDGE_H200H
    return {"sft_s_per_step": s_sft, "dpo_s_per_step": s_dpo, "dpo_caching_min": cache / 60,
            "sft_h": sft_h, "dpo_h_incl_caching": dpo_h, "start_h": START_H,
            "train_h200h": train, "gen_judge_h200h": GEN_JUDGE_H200H, "all_in_h200h": all_in,
            "su_at_60": SU_PER_H200H * all_in,
            "projection_d067c_h200h": projection,
            "measured_over_projection": None if projection is None else all_in / projection,
            "ds_sha_prefix": {st: stages[st]["ds"] for st in ("sft", "dpo")},
            "loss_step1_step50": {st: (stages[st]["loss"].get(1), stages[st]["loss"].get(50))
                                  for st in ("sft", "dpo")},
            "steps_logged": {st: max(stages[st]["steps"]) for st in ("sft", "dpo")}}


def check_sizes(got: Mapping[str, object], what: str) -> None:
    """One log per ladder size, no more and no fewer: the TOTAL_H200H / TOTAL_SU keys are 12
    twins x every size of the ladder, so a missing or misspelt size is an error, not a smaller
    or null total. Applied by `log_paths` before any log is read and by `summarise`. `what`
    names the input the caller received ("--log", "logs")."""
    assert set(got) == set(ladder.SIZES), \
        f"{what}: one log per ladder size {sorted(ladder.SIZES)} is required, got {sorted(got)}"


def summarise(logs: dict[str, str]) -> dict:
    """The per-size costs and the ladder total from {size: log text} (`check_sizes`)."""
    check_sizes(logs, "logs")
    res = {"method": "notes/016 § Measured cost: steady s/step over steps 10-50, DPO caching "
                     "from 'Num examples' to step 1, x 13,342 SFT + 2,898 DPO steps + 0.03 h, "
                     "x 8 GPUs, + 1.83 H200-h gen + judge; SU at 60",
           "sizes": {z: cost(parse_log(logs[z]), PROJECTION_D067C[z]) for z in logs}}
    res[TOTAL_H200H] = sum(TWINS_PER_SIZE * r["all_in_h200h"] for r in res["sizes"].values())
    res[TOTAL_SU] = SU_PER_H200H * res[TOTAL_H200H]
    return res


def log_paths(args: list[str]) -> dict[str, str]:
    """`--log SIZE=PATH` values -> {size: path}, one per ladder size (`check_sizes`); a value
    without '=', a SIZE that is not a ladder size, an empty PATH, a size given twice or a size
    missing is refused here, before any log is read."""
    out = {}
    for kv in args:
        z, sep, path = (part.strip() for part in kv.partition("="))
        assert sep and path and z in ladder.SIZES, \
            f"--log wants SIZE=PATH with SIZE in {sorted(ladder.SIZES)}, got {kv!r}"
        assert z not in out, f"--log: size {z} given twice"
        out[z] = path
    check_sizes(out, "--log")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", required=True, metavar="SIZE=PATH")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute"
    t0 = time.time()
    paths = log_paths(a.log)
    res = summarise({z: Path(p).read_text() for z, p in paths.items()}) | {"logs": paths}
    res.update({"script": "scripts/smoke_cost_016.py", "git_commit": provenance.git_commit(),
                "command": provenance.command(), "seconds": time.time() - t0})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    for z, r in res["sizes"].items():
        print(f"{z}: SFT {r['sft_s_per_step']:.3f} s/step, DPO {r['dpo_s_per_step']:.3f} s/step, "
              f"caching {r['dpo_caching_min']:.1f} min -> {r['all_in_h200h']:.1f} H200-h "
              f"({r['su_at_60']:.0f} SU); projection {r['projection_d067c_h200h']}")
    print(f"{N_TWINS} twins: {res[TOTAL_H200H]:.0f} H200-h = {res[TOTAL_SU]:.0f} SU")


if __name__ == "__main__":
    main()
