"""Parse OLMo trainer console logs: in-loop held-out evaluations per step.

OLMo (ccabb54e, no W&B) prints metrics only to the console: a `[step=N/M]` line after every
train step, then, on eval steps, a block per evaluator whose lines read
`    eval/<set>-validation/Perplexity=<value>`, formatted by `Trainer.log_metrics_to_console`
(2 decimals for values in (10, 100], 3 in (1, 10], 1 in (100, 1000]; thousands separators above).
A block belongs to the last step printed before it. A resumed run repeats steps; the last
occurrence wins.
"""

import re
from pathlib import Path

STEP = re.compile(r"\[step=(\d+)/\d+\]")
PPL = re.compile(r"^\s+eval/([\w.-]+)-validation/(Perplexity|CrossEntropyLoss)=([\d,.eE+-]+)\s*$")


def in_loop_evals(logs: list[Path]) -> dict[int, dict[str, dict[str, float]]]:
    """step -> set -> {"Perplexity": v, "CrossEntropyLoss": v}, over logs in order."""
    out: dict[int, dict[str, dict[str, float]]] = {}
    for log in logs:
        step = None
        for line in log.read_text(errors="replace").splitlines():
            if m := STEP.search(line):
                step = int(m.group(1))
            elif (m := PPL.match(line)) and step is not None:
                name, metric, val = m.groups()
                out.setdefault(step, {}).setdefault(name, {})[metric] = float(val.replace(",", ""))
    return out
