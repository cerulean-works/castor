"""Score 080's swap-arm precision audit: judge_v1 refusal precision and recall, beside 051's.

    uv run --frozen python scripts/audit_score_080.py --audit AUDIT_DIR \
        --sheets AUDIT_DIR/sheet_harry.csv AUDIT_DIR/sheet_coauthor.csv \
        --adjudicated AUDIT_DIR/sheet_adjudicate.csv --out /abs/.../score_080.json
    uv run --frozen python scripts/audit_score_080.py --audit AUDIT_DIR --placeholder \
        --out /abs/.../score_080_DRYRUN.json

051's judge_error block (scripts/audit_correct_008.py) on a recipe-keyed draw of any size, with
no grid and no level correction (out of scope for 080's audit, 092). Gold is
scripts/audit_score.py's; nothing is computed unless every key item has gold (D043c). Statistic:
`castor.audit008.rates` on the precision population (sampled, judge-coherent items), pooled and
per source; Rao-Wu bootstrap within the audit strata, one-item strata collapsed into their
recipe x source cell (`castor.auditgate`), as 051. On 008's audit with 051's sheets it gives
051's judge_error points (`--comparator` checks them).

--placeholder: gold drawn at random (audit_correct_008.placeholder_gold); never a result.
"""

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

from castor import audit008 as A8
from castor import auditgate, provenance
from castor.judge.common import read_jsonl

ROOT = Path(__file__).resolve().parent.parent
COMPARATOR = ROOT / "runs" / "008" / "audit_correct.json"
KEYS = ("judge_rate", "precision", "human_rate", "recall", "fp_share", "fn_share")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stat(items: list[dict]) -> dict:
    """Pooled rates and per-source precision and recall, design-weighted."""
    out = {f"pooled:{k}": v for k, v in A8.rates(items).items() if k in KEYS}
    for s in A8.SOURCES:
        r = A8.rates([x for x in items if x["source"] == s])
        out.update({f"source:{s}:{k}": r[k] for k in KEYS})
    return out


def score(items: list[dict], n_boot: int, seed: int) -> dict:
    boot = auditgate.stratified_bootstrap(items, stat, n_boot, seed,
                                          collapse=auditgate.source_cell)
    pop = A8.population(items)
    return {
        "pooled": {k: boot[f"pooled:{k}"] for k in KEYS},
        "per_source": {s: {**{k: boot[f"source:{s}:{k}"] for k in KEYS},
                           "counts": A8.rates([x for x in pop if x["source"] == s])}
                       for s in A8.SOURCES},
        "counts": A8.rates(pop),
        "n_gold": len(items), "n_population": len(pop),
        "n_greedy_excluded": sum(x["sample"] < 0 for x in items),
    }


def comparator(path: Path) -> dict:
    """051's pooled precision and recall (runs/008/audit_correct.json judge_error)."""
    d = json.loads(path.read_text())
    je = d["judge_error"]
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "git_commit": d["git_commit"],
            **{k: je[k] for k in KEYS},
            **{f"recall:{s}": je[f"recall:{s}"] for s in A8.SOURCES},
            "precision_per_source": d["factors"]["per_source"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--sheets", nargs=2)
    ap.add_argument("--adjudicated")
    ap.add_argument("--placeholder", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000, help="audit bootstrap draws (015, 051)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--comparator", type=Path, default=COMPARATOR)
    ap.add_argument("--check-comparator", action="store_true",
                    help="require the points to equal the comparator's (run on 008's audit)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if not a.out.is_absolute():
        raise SystemExit("--out must be absolute (submit.sh snapshots are deleted)")
    if not (a.placeholder or (a.sheets and a.adjudicated)):
        raise SystemExit("--sheets and --adjudicated needed")
    commit, cmd, t0 = provenance.git_commit(), provenance.command(), time.time()

    keyrows = {r["item_id"]: r for r in read_jsonl(a.audit / "key.jsonl")}
    if a.placeholder:
        gold, _ = _load("audit_correct_008").placeholder_gold(keyrows, a.seed)
        info = {"unresolved": 0}
    else:
        S = _load("audit_score")
        sheets = [S.read_sheet(s) for s in a.sheets]
        gold, info = S.gold_labels(sheets, S.read_sheet(a.adjudicated))
        auditgate.check_complete(keyrows, sheets, gold, info)
    items = [{**keyrows[i], "gold": g["label"], "gold_partial": g["partial"]}
             for i, g in gold.items()]
    if {x["source"] for x in items} - set(A8.SOURCES):
        raise SystemExit("an item outside the two adversarial-harmful sources")

    out = {"placeholder": a.placeholder, "n_items": len(keyrows), **info,
           **score(items, a.n_boot, a.seed),
           "recipes": sorted({x["model"] for x in items}),
           "models": sorted({x.get("model_id", x["model"]) for x in items}),
           "comparator_051": comparator(a.comparator),
           "n_boot": a.n_boot, "seed": a.seed, "sheets": a.sheets, "adjudicated": a.adjudicated,
           "script": "scripts/audit_score_080.py", "git_commit": commit, "command": cmd}
    if a.check_comparator:
        c = out["comparator_051"]
        for k in KEYS:
            if not math.isclose(out["pooled"][k]["point"], c[k]["point"], abs_tol=1e-12):
                raise SystemExit(f"comparator: {k} {out['pooled'][k]['point']} != {c[k]['point']}")
        out["comparator_points_reproduced"] = True
    out["seconds"] = time.time() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"placeholder": a.placeholder,
                      "precision": out["pooled"]["precision"], "recall": out["pooled"]["recall"],
                      "per_source_precision": {s: v["precision"]
                                               for s, v in out["per_source"].items()}},
                     default=float))


if __name__ == "__main__":
    main()
