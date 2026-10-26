"""The recipe grid's operational record: the per-cell manifest with provenance, the
judge-instrument pins with the rows cut to fit the judge context, and the measured cost. CPU;
reads provenance, labels and sacct only (no label value enters any statistic here).

    uv run --frozen python scripts/report_008.py --manifest data/manifests/grid_008.json \
        --out runs/008/cells.json

Per cell (156): model id, recipe, pretraining seed, post-training seed, stage, base ref; SFT
(config sha, DeepSpeed config sha, open-instruct commit, git commit, job), DPO (config sha, git
commit, job), generation and labels git commits; label rows, rows with `truncated_to_fit` and of
those the rows whose WildGuard or Qwen3Guard label is JUDGE-ERROR; every Slurm job of the cell
from the driver's ledger (`submissions.json`, including `.pre040` keys) with its sacct state,
elapsed and GPUs. Totals: H200-hours by kind (train, gen+judge, refill) and by final state; SU at
60 SU per H200-hour.
"""

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from castor import grid, provenance
from castor.judge.common import read_jsonl

SU_PER_H200_H = 60  # NCHC billing rate
DRIVER = grid.RUNS / "008" / "driver"


def elapsed_s(text: str) -> int:
    """sacct Elapsed ([D-]HH:MM:SS) -> seconds."""
    days, _, hms = text.rpartition("-")
    h, m, s = (int(x) for x in hms.split(":"))
    return int(days or 0) * 86400 + h * 3600 + m * 60 + s


def gpus(tres: str) -> int:
    """sacct AllocTRES -> number of GPUs (0 if none)."""
    for part in tres.split(","):
        if part.startswith("gres/gpu="):
            return int(part.split("=")[1])
    return 0


def sacct(start: str) -> dict[str, dict]:
    out = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-S", start, "-E", "now",
         "-o", "JobID,JobName%80,State,Elapsed,AllocTRES,NodeList"],
        check=True, capture_output=True, text=True).stdout
    jobs = {}
    for line in out.splitlines():
        jid, name, state, el, tres, node = line.split("|")
        jobs[jid] = {"name": name, "state": state.split()[0], "elapsed_s": elapsed_s(el),
                     "gpus": gpus(tres), "node": node}
    return jobs


def label_scan(path: Path) -> dict:
    n = cut = cut_err = 0
    for r in read_jsonl(path):
        n += 1
        if r.get("truncated_to_fit"):
            cut += 1
            cut_err += "JUDGE-ERROR" in (r.get("wg_label"), r.get("q3g_label"))
    return {"rows": n, "truncated_to_fit": cut, "judge_error_on_cut": cut_err}


def _json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--stages-file", default="data/manifests/grid_008_stages.json")
    ap.add_argument("--start", default="2026-10-08T11:00")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    g = grid.load(a.manifest)
    stages = grid.load_stages(a.stages_file, g)
    ledger = json.loads((DRIVER / "submissions.json").read_text())
    jobs = sacct(a.start)
    cells, used = [], set()
    for c in g.cells():
        sft, dpo = _json(c.out("sft") / "provenance.json"), _json(c.out("dpo") / "provenance.json")
        run = c.gen_dir()
        gen, lab = _json(run / "provenance.json"), _json(run / "judge_v1" / "labels.meta.json")
        cj = []
        for k, ids in sorted(ledger.items()):
            if k.split("|")[0] == c.tag:
                for j in ids:
                    used.add(j)
                    cj.append({"key": k, "job": j, **jobs.get(j, {"state": "UNKNOWN"})})
        cells.append({
            "model_id": c.model_id, "tag": c.tag, "recipe": c.base.recipe,
            "pretrain_seed": c.base.pretrain_seed, "posttrain_seed": c.k,
            "stage": grid.stage_of(c, stages), "base_ref": c.base.ref,
            "sft": {k: sft.get(k) for k in ("config_sha256", "git_commit", "slurm_job", "finished")}
            | {"ds_config_sha256": (sft.get("open_instruct") or {}).get("deepspeed_config_sha256"),
               "open_instruct": (sft.get("open_instruct") or {}).get("commit")},
            "dpo": {k: dpo.get(k) for k in ("config_sha256", "git_commit", "slurm_job", "finished")},
            "generation_commit": gen.get("git_commit"), "labels_commit": lab.get("git_commit"),
            "labels": label_scan(run / "judge_v1" / "labels.jsonl"), "jobs": cj})
    refills = {j: v for j, v in jobs.items() if v["name"].startswith("008-refill")}
    hours = defaultdict(float)
    for c in cells:
        for j in c["jobs"]:
            if "elapsed_s" in j:
                kind = "train" if j["key"].endswith("|sft") else "genjudge"
                h = j["elapsed_s"] * j["gpus"] / 3600
                hours[kind] += h
                hours[f"{kind}:{j['state']}"] += h
    hours["refill"] = sum(v["elapsed_s"] * v["gpus"] / 3600 for v in refills.values())
    total = hours["train"] + hours["genjudge"] + hours["refill"]
    record = {
        "n_cells": len(cells), "cells": cells,
        "cost": {"h200_hours": dict(sorted(hours.items())), "total_h200_hours": total,
                 "su": total * SU_PER_H200_H, "su_per_h200_hour": SU_PER_H200_H,
                 "n_refill_jobs": len(refills),
                 "job_states": dict(sorted(_count(j["state"] for c in cells for j in c["jobs"]).items()))},
        "unledgered_008_jobs": sorted(j for j, v in jobs.items() if v["name"].startswith("008-")
                                      and j not in used and j not in refills),
        "git_commit": provenance.git_commit(), "command": provenance.command(),
        "script": "scripts/report_008.py"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["cost"], indent=1))


def _count(xs) -> dict:
    d = defaultdict(int)
    for x in xs:
        d[x] += 1
    return d


if __name__ == "__main__":
    main()
