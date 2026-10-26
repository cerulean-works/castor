"""Draw the human-audit set and write blinded labelling sheets.

    uv run --frozen python scripts/audit_sample.py --labels DIR1/labels.jsonl [...] \
        --generations G1 [...] --n 1200 --annotators harry coauthor --out AUDIT_DIR

Sampling design: `castor.audit` (strata = model x panel source x judge-agreement status x
PARTIAL; 40% disagreements / 60% the rest; PARTIAL x3; simple random within strata).
Writes, under AUDIT_DIR (keep it on /work: sheets contain harmful text):
  sheet_<annotator>.csv  item_id, prompt, response, label (R/H/N/I), partial (0/1), notes;
                         model identity, judge labels and strata hidden; one shuffled order
  key.jsonl              item_id -> model, prompt_id, sample, stratum, pool, weight, judge
                         labels (never shown to annotators)
  manifest.json          counts per stratum, seed, inputs, git commit, command
Score with scripts/audit_score.py.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from castor import audit, provenance
from castor.judge.common import key, read_jsonl, write_jsonl

PANEL = Path(__file__).resolve().parent.parent / "data" / "panel" / "panel_v1.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--generations", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--annotators", nargs="+", default=["harry"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac-disagree", type=float, default=audit.FRAC_DISAGREE)
    ap.add_argument("--flag-weight", type=float, default=audit.FLAG_WEIGHT)
    ap.add_argument("--exclude-models", nargs="*", default=["probe"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    commit, cmd = provenance.git_commit(), provenance.command()

    source = {r["id"]: r["source"] for r in read_jsonl(PANEL)}
    gens = {key(r): r for g in args.generations for r in read_jsonl(g)}
    rows = []
    for path in args.labels:
        for r in read_jsonl(path):
            if r["model"] in args.exclude_models:
                continue
            rows.append({**r, "source": source[r["prompt_id"]],
                         "flag": bool(r["wg_partial"] or r["llm_invalid"])})
    drawn = audit.draw(rows, args.n, seed=args.seed, frac_disagree=args.frac_disagree,
                       flag_weight=args.flag_weight)
    rng = random.Random(args.seed + 1)
    rng.shuffle(drawn)
    ids = rng.sample(range(16 ** 6), len(drawn))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keyrows = []
    for d, i in zip(drawn, ids):
        d["item_id"] = f"{i:06x}"
        keyrows.append({k: d.get(k) for k in (
            "item_id", "model", "prompt_id", "sample", "source", "stratum", "pool", "weight",
            "status", "flag", "coherent", "coherence_rule", "llm_invalid", "wg_label", "wg_partial", "q3g_label",
            "q3g_partial", "refusal_agree")})
    write_jsonl(out / "key.jsonl", keyrows)
    for a in args.annotators:
        audit.write_sheet(out / f"sheet_{a}.csv", [
            (d["item_id"], gens[key(d)]["prompt"], gens[key(d)]["response"]) for d in drawn])
    manifest = {
        "n": len(drawn), "seed": args.seed, "frac_disagree": args.frac_disagree,
        "flag_weight": args.flag_weight, "labels": args.labels,
        "generations": args.generations, "annotators": args.annotators,
        "population": len(rows),
        "pool_counts": dict(Counter(d["pool"] for d in drawn)),
        "status_counts": dict(Counter(d["status"] for d in drawn)),
        "flag_count": sum(bool(d["flag"]) for d in drawn),
        "partial_count": sum(bool(d["wg_partial"]) for d in drawn),
        "llm_invalid_count": sum(bool(d["llm_invalid"]) for d in drawn),
        "per_model": dict(Counter(d["model"] for d in drawn)),
        "per_source": dict(Counter(d["source"] for d in drawn)),
        "git_commit": commit, "command": cmd, "script": "scripts/audit_sample.py",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("n", "population", "pool_counts",
                                               "status_counts", "flag_count")}))


if __name__ == "__main__":
    main()
