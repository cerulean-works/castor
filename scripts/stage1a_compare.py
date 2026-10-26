"""Does our same-seed 150M run track the released run's in-loop perplexity?

Pre-registered rule: at every evaluated step, for each of the 11 held-out sets (all 11 must be
parsed), our in-loop perplexity is within 1% relative of the released in-loop value (no
checkpoint was evaluated twice by Ai2, so no released between-evaluation tolerance exists).
Ours: parsed from the run's OLMo console logs (same evaluator, same world size 8, same seed ->
the same eval sequences). Released: DataDecide-ppl-results@c4a9fa36, DCLM-Baseline 150M default.

    uv run --frozen python scripts/stage1a_compare.py \
        --logs logs/nchc/003-1a-150M-s6198-*.out --out /work/harrychang08/castor/runs/003/stage1a
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from castor import provenance
from castor.olmolog import in_loop_evals

PPL = "/work/harrychang08/castor/data/datadecide_evals/ppl_results.parquet"
TOL = 0.01
N_SETS = 11
JOBID = re.compile(r"-(\d+)\.out$")


def log_order(logs: list[Path]) -> list[Path]:
    """Logs in job order (olmolog: the last occurrence of a repeated step wins). Slurm logs are
    `<name>-<jobid>.out`; the job id is compared as a number, so 999999 sorts before 1000000."""
    def key(p: Path) -> int:
        m = JOBID.search(p.name)
        assert m, f"log name without a job id: {p}"
        return int(m.group(1))
    return sorted(logs, key=key)


def compare(ours: dict, rel: pd.DataFrame) -> tuple[pd.DataFrame, list[int], dict[int, list[str]]]:
    """Per (step, set) relative difference of our in-loop perplexity vs the reference.
    Returns (rows, reference steps absent from ours, steps whose parsed sets are not exactly the
    reference's 11 held-out sets -> the sets that differ). Any missing or incomplete step fails."""
    sets = sorted(c.removeprefix("eval/").removesuffix("-validation/Perplexity")
                  for c in rel.columns if c.endswith("-validation/Perplexity"))
    assert len(sets) == N_SETS, sets
    rows, incomplete = [], {}
    for step, got in sorted(ours.items()):
        if float(step) not in rel.index:
            continue
        diff = sorted(set(sets) ^ {k for k, v in got.items() if "Perplexity" in v})
        if diff:
            incomplete[int(step)] = diff
        for name in sets:
            if name in got and "Perplexity" in got[name]:
                v, r = got[name]["Perplexity"], rel.loc[float(step), f"eval/{name}-validation/Perplexity"]
                rows.append({"step": step, "set": name, "ours": v, "released": r, "rel_diff": v / r - 1})
    df = pd.DataFrame(rows, columns=["step", "set", "ours", "released", "rel_diff"])
    missing = sorted(set(rel.index.astype(int)) - set(df.step))
    return df, missing, incomplete


def passes(df: pd.DataFrame, missing: list[int], incomplete: dict) -> bool:
    return len(df) > 0 and bool((df.rel_diff.abs() <= TOL).all()) and not missing and not incomplete


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", type=Path, required=True)
    ap.add_argument("--ppl", default=PPL)
    ap.add_argument("--params", default="150M")
    ap.add_argument("--data", default="DCLM-Baseline")
    ap.add_argument("--seed", default="default")
    ap.add_argument("--reference-logs", nargs="+", type=Path, default=None,
                    help="diagnosis: compare against another of our runs instead of Ai2's")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    ours = in_loop_evals(log_order(a.logs))
    if a.reference_logs:
        ref = in_loop_evals(log_order(a.reference_logs))
        rel = pd.DataFrame({float(st): {f"eval/{k}-validation/Perplexity": v["Perplexity"]
                                        for k, v in sets.items()} for st, sets in ref.items()}).T
        rel = rel[rel.index <= max(ours)]
    else:
        pp = pd.read_parquet(a.ppl)
        rel = pp[(pp.params == a.params) & (pp.data == a.data) & (pp.seed == a.seed)].set_index("step")
    df, missing, incomplete = compare(ours, rel)
    worst = df.loc[df.rel_diff.abs().idxmax()].to_dict() if len(df) else None
    summary = {
        "n_steps_compared": int(df.step.nunique()), "n_values": len(df),
        "released_steps_not_in_ours": missing, "steps_with_set_mismatch": incomplete,
        "max_abs_rel_diff": float(df.rel_diff.abs().max()), "worst": worst,
        "per_step_max_abs_rel": df.groupby("step").rel_diff.apply(lambda x: x.abs().max()).to_dict(),
        "tolerance": TOL, "pass": passes(df, missing, incomplete),
        "logs": [str(p) for p in log_order(a.logs)],
        "reference": [str(p) for p in a.reference_logs] if a.reference_logs else f"{a.ppl} {a.params} {a.data} {a.seed}",
        "git_commit": provenance.git_commit(),
        "command": provenance.command(),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "stage1a_trajectory.tsv", sep="\t", index=False)
    (a.out / "stage1a_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ["n_steps_compared", "max_abs_rel_diff", "pass",
                                              "released_steps_not_in_ours",
                                              "steps_with_set_mismatch"]}, default=str))


if __name__ == "__main__":
    main()
