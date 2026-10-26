"""Count the generation rows whose judge input exceeds a judge's limit
(max_model_len - max_new_tokens), per cell of the 008 and 016 runs. Tokenizers only, no weights.

    uv run --frozen python scripts/scan_judge_inputs.py --out runs/040/scan.json

A cell is `<run root>/<model id>/<prompt>/` with greedy.jsonl and/or samples.jsonl; `judged`
= judge_v1/labels.jsonl exists. Every stage's input is built by the stage's own live builder
(`raw_ids`, the uncut input that fit_response wraps), so a count is the number of rows
fit_response cuts. The coherence stage sends only
rows no rule caught to the LLM; both counts are reported (`all`, `rule_none`). Only rows
whose input *could* exceed the limit are tokenized: these byte-level / byte-fallback
tokenizers give at most one token per UTF-8 byte (+2 for a leading dummy prefix;
tests/test_judge_truncation.py pins this), so a formatted input of at most `limit - 2`
bytes fits. A partial last line (a generation still being written) is counted, not parsed.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from castor import provenance
from castor.judge import STAGES, coherence

RUNS = Path("/work/harrychang08/castor/runs")
PROMPTS = ("008", "016")
FILES = ("greedy.jsonl", "samples.jsonl")
BYTE_SLACK = 2  # tokens a stage's input may exceed its UTF-8 byte count by
_judges: dict = {}


def judges() -> dict:
    """Each stage's input builder over its own tokenizer (no vLLM engine)."""
    if not _judges:
        from transformers import AutoTokenizer

        for name, (cls, model, rev, max_len, max_new) in STAGES.items():
            j = object.__new__(cls)
            j.clf = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(model, revision=rev),
                                    limit=max_len - max_new)
            _judges[name] = j
    return _judges


def frame_bytes(j) -> int:
    """UTF-8 bytes of the formatted input with empty prompt and response."""
    return len(j.format_text("", "").encode())


def may_exceed(frame: int, limit: int, prompt: str, response: str) -> bool:
    """False when the input is short enough in bytes that it cannot exceed `limit` tokens."""
    return len(prompt.encode()) + len(response.encode()) + frame > limit - BYTE_SLACK


def scan_cell(cell: str) -> dict:
    d = Path(cell)
    js = judges()
    frames = {n: frame_bytes(j) for n, j in js.items()}
    res = {"judged": (d / "judge_v1" / "labels.jsonl").exists(), "n_rows": 0,
           "n_partial_lines": 0, "files": [],
           "stages": {n: {"limit": j.clf.limit, "n_tokenized": 0, "max_tokens_tokenized": None,
                          "n_over": 0, "n_over_rule_none": 0, "over": []}
                      for n, j in js.items()}}
    for fname in FILES:
        path = d / fname
        if not path.exists():
            continue
        res["files"].append(fname)
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    res["n_partial_lines"] += 1
                    continue
                res["n_rows"] += 1
                for n, j in js.items():
                    s = res["stages"][n]
                    if not may_exceed(frames[n], s["limit"], r["prompt"], r["response"]):
                        continue
                    s["n_tokenized"] += 1
                    k = len(j.raw_ids(r["prompt"], r["response"]))
                    s["max_tokens_tokenized"] = max(s["max_tokens_tokenized"] or 0, k)
                    if k > s["limit"]:
                        rule = coherence.rule_check(r["prompt"], r["response"],
                                                    r["n_tokens"])["rule"]
                        ids, cut = j.input_ids(r["prompt"], r["response"])
                        s["n_over"] += 1
                        s["n_over_rule_none"] += rule is None
                        s["over"].append({"prompt_id": r["prompt_id"], "sample": r["sample"],
                                          "n_tokens_input": k, "coherence_rule": rule,
                                          "n_tokens_after_fit": len(ids), "cut": cut,
                                          "response_chars": len(r["response"])})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    commit, cmd = provenance.git_commit(), provenance.command()
    cells = sorted({str(p.parent) for pr in PROMPTS for f in FILES
                    for p in Path(args.runs).glob(f"*/{pr}/{f}")})
    out = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for cell, r in zip(cells, ex.map(scan_cell, cells)):
            out[str(Path(cell).relative_to(args.runs))] = r
            print(cell, r["n_rows"], {n: s["n_over"] for n, s in r["stages"].items()},
                  file=sys.stderr, flush=True)
    summary = {
        "n_cells": len(out), "n_judged": sum(r["judged"] for r in out.values()),
        "judged_cells_with_over": sorted(c for c, r in out.items() if r["judged"] and any(
            s["n_over"] for s in r["stages"].values())),
        "unjudged_cells_with_over": sorted(c for c, r in out.items() if not r["judged"] and any(
            s["n_over"] for s in r["stages"].values())),
        "n_over_by_stage": {n: sum(r["stages"][n]["n_over"] for r in out.values())
                            for n in STAGES},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"summary": summary, "cells": out, "script": "scripts/scan_judge_inputs.py",
         "git_commit": commit, "command": cmd}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
