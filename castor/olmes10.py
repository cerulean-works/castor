"""OLMES-10 macro average from an oe-eval output directory, as DataDecide reports it.

DataDecide's `olmes_10_macro_avg` is the plain mean of 10 per-task scores (checked against the
released rows of DataDecide-eval-results@9919b5a0: equal to 1e-16 for the 150M DCLM seeds).
Per-task score = oe-eval's primary metric, except MMLU, whose released score is `acc_raw`
(not OLMES's acc_per_char), aggregated over the 57 subtasks as `mmlu_agg` says.
"""

import json
from pathlib import Path

TASKS9 = ["arc_challenge", "arc_easy", "boolq", "csqa", "hellaswag", "openbookqa", "piqa",
          "socialiqa", "winogrande"]
N_MMLU = 57


def load_metrics(out_dir: Path) -> dict[str, dict]:
    """task_name -> the oe-eval metrics.json of that task."""
    res = {}
    for p in sorted(out_dir.glob("task-*-metrics.json")):
        m = json.loads(p.read_text())
        assert m["task_name"] not in res, f"duplicate task {m['task_name']} in {out_dir}"
        res[m["task_name"]] = m
    return res


def mmlu_score(ms: dict[str, dict], agg: str = "macro") -> float:
    subs = {k: v for k, v in ms.items() if k.startswith("mmlu_")}
    if len(subs) != N_MMLU:
        raise ValueError(f"expected {N_MMLU} MMLU subtasks, found {len(subs)}")
    if agg == "macro":
        return sum(v["metrics"]["acc_raw"] for v in subs.values()) / N_MMLU
    if agg == "micro":
        n = sum(v["num_instances"] for v in subs.values())
        return sum(v["metrics"]["acc_raw"] * v["num_instances"] for v in subs.values()) / n
    raise ValueError(agg)


def olmes10(out_dir: Path, mmlu_agg: str = "macro") -> dict:
    ms = load_metrics(out_dir)
    missing = [t for t in TASKS9 if t not in ms]
    if missing:
        raise ValueError(f"missing tasks {missing} in {out_dir}")
    scores = {t: ms[t]["metrics"]["primary_score"] for t in TASKS9}
    scores["mmlu"] = mmlu_score(ms, mmlu_agg)
    return {
        "scores": scores,
        "olmes_10_macro_avg": sum(scores.values()) / len(scores),
        "mmlu_agg": mmlu_agg,
        "num_instances": {t: m["num_instances"] for t, m in ms.items()},
        "task_hash": {t: m["task_hash"] for t, m in ms.items()},
        "model_config": next(iter(ms.values()))["model_config"],
    }
