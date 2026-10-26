"""Run one stage of the instrument (judge_v1) over a generations file.

    uv run --frozen python scripts/judge.py --stage coherence  --generations G --out DIR
    uv run --frozen python scripts/judge.py --stage wildguard  --generations G --out DIR
    uv run --frozen python scripts/judge.py --stage qwen3guard --generations G --out DIR
    uv run --frozen python scripts/judge.py --stage labels     --generations G --out DIR

One stage per process (each loads its own vLLM engine on one GPU). Writes DIR/<stage>.jsonl
and DIR/<stage>.meta.json (model@revision, template hash, throughput, git commit, command).
`labels` joins coherence + wildguard (+ qwen3guard if present) into DIR/labels.jsonl with the
label from each judge (`wg_*` primary, `q3g_*` cross-check) and their refusal agreement.
"""

import argparse
import json
import time
from pathlib import Path

from castor import provenance
from castor.judge import JUDGE_VERSION, coherence, labels
from castor.judge.common import INPUT_TRUNCATION, JUDGE_V1_PATCH, key, read_jsonl, write_jsonl

JUDGED = ("wildguard", "qwen3guard")


def run_model_stage(stage: str, rows: list[dict]) -> tuple[list[dict], dict]:
    t_load = time.perf_counter()
    if stage == "coherence":
        judge = coherence.ValidityLLM()
        load_s = time.perf_counter() - t_load
        t0 = time.perf_counter()
        out = coherence.run(rows, judge)
        n_judged = sum("llm_valid" in o for o in out)
        meta = {"rules": {"MIN_TOKENS": coherence.MIN_TOKENS, "MAX_REP": coherence.MAX_REP,
                          "MIN_NGRAMS": coherence.MIN_NGRAMS,
                          "MAX_NONLATIN": coherence.MAX_NONLATIN,
                          "MIN_LETTERS": coherence.MIN_LETTERS, "MAX_ECHO": coherence.MAX_ECHO,
                          "MIN_ECHO_NGRAMS": coherence.MIN_ECHO_NGRAMS},
                "validity_llm": {**judge.meta, "gates_coherent": False}}
    else:
        if stage == "wildguard":
            from castor.judge.wildguard import WildGuard as J
        else:
            from castor.judge.qwen3guard import Qwen3Guard as J
        judge = J()
        load_s = time.perf_counter() - t_load
        t0 = time.perf_counter()
        out = judge.judge(rows)
        n_judged = len(out)
        meta = dict(judge.meta)  # never write run fields into a judge's own meta
    dt = time.perf_counter() - t0
    # input_truncation and judge_v1_patch describe the instrument (pipeval.instrument);
    # n_truncated_to_fit and throughput describe the run (pipeval.RUN_FIELDS).
    meta["input_truncation"] = INPUT_TRUNCATION
    meta["judge_v1_patch"] = JUDGE_V1_PATCH
    meta["n_truncated_to_fit"] = sum(bool(o.get("truncated_to_fit")) for o in out)
    meta["throughput"] = {"n_rows": len(rows), "n_llm_judged": n_judged,
                          "seconds": round(dt, 2), "load_seconds": round(load_s, 1),
                          "llm_judgments_per_s": round(n_judged / dt, 2) if dt else None,
                          "rows_per_s": round(len(rows) / dt, 2) if dt else None}
    return out, meta


def compose_labels(out_dir: Path, gens: dict) -> tuple[list[dict], dict]:
    coh = {key(r): r for r in read_jsonl(out_dir / "coherence.jsonl")}
    stage_rows = {"coherence": coh}  # every stage's own rows, for the truncation flag
    per = {}
    for j in JUDGED:
        path = out_dir / f"{j}.jsonl"
        if path.exists():
            stage_rows[j] = judged = {key(r): r for r in read_jsonl(path)}
            per[j] = {key(r): r for r in labels.compose_rows(coh, judged, gens)}
    if "wildguard" not in per:
        raise FileNotFoundError("labels need wildguard.jsonl (the primary judge)")
    rows = []
    for k in gens:
        r = {"model": k[0], "prompt_id": k[1], "sample": k[2],
             "coherent": coh[k]["coherent"], "coherence_rule": coh[k]["rule"],
             "llm_invalid": coh[k].get("llm_invalid", False), "p_valid": coh[k].get("p_valid"),
             # the response was cut to fit a judge's context in at least one stage
             "truncated_to_fit": any(s[k].get("truncated_to_fit", False)
                                     for s in stage_rows.values())}
        for j, pre in (("wildguard", "wg"), ("qwen3guard", "q3g")):
            if j in per:
                x = per[j][k]
                r.update({f"{pre}_{f}": x[f] for f in ("label", "partial", "vulnerable",
                                                     "harmful", "vulnerable_incl")})
        if "qwen3guard" in per:
            a, b = r["wg_label"], r["q3g_label"]
            r["refusal_agree"] = None if not r["coherent"] or labels.JUDGE_ERROR in (a, b) \
                else (a == labels.REFUSAL) == (b == labels.REFUSAL)
        rows.append(r)
    return rows, {"judges": sorted(per)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["coherence", *JUDGED, "labels"], required=True)
    ap.add_argument("--generations", required=True, nargs="+",
                    help="generation JSONL file(s); rows are keyed by (model, prompt_id, sample)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    commit, cmd = provenance.git_commit(), provenance.command()

    rows = [r for g in args.generations for r in read_jsonl(g)]
    gens = {key(r): r for r in rows}
    if len(gens) != len(rows):
        raise ValueError("duplicate (model, prompt_id, sample) keys in generations")
    out_dir = Path(args.out)
    if args.stage == "labels":
        out, meta = compose_labels(out_dir, gens)
    else:
        out, meta = run_model_stage(args.stage, rows)
    write_jsonl(out_dir / f"{args.stage}.jsonl", out)
    meta.update({"stage": args.stage, "judge_version": JUDGE_VERSION,
                 "generations": args.generations, "n": len(out), "git_commit": commit,
                 "command": cmd, "script": "scripts/judge.py"})
    (out_dir / f"{args.stage}.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in meta if k in ("stage", "n", "throughput")}))


if __name__ == "__main__":
    main()
