"""006's primary contrast (r_MZ, r_U, MZ - U; adversarial-harmful, wg_vulnerable) recomputed on
precision-corrected per-prompt rates (2606.25487 p.6: rate x precision).

    uv run --frozen python scripts/audit_correct_006.py --report AUDIT_DIR/report_015.json \
        --out /abs/runs/006/audit/015/correct_006.json
    uv run --frozen python scripts/audit_correct_006.py --placeholder --out /abs/...dry.json

Variants (each sample's 0/1 outcome times a factor, so per-prompt rates are scaled):
  none        factor 1: must reproduce runs/006/signal.json exactly (asserted), same n_boot/seed
  per_model   factor = the model's design-weighted precision on the subset (report_015
              `adv:<model>:precision`). A per-model constant leaves Pearson r, split-half
              reliability and every twin correlation invariant; no prompt bootstrap.
  per_source  factor = precision per panel source pooled over the 12 twins
              (`adv_twins_source:<src>:precision`); prompts from different sources are scaled
              differently, so the correlations can move.
--placeholder uses fixed made-up factors (per_model 0.9, per_source 0.8 / 0.95) to exercise
the code; its output is never a result.
Also reported per variant: each twin's mean corrected vulnerability.
Every factor must be finite and in (0, 1], and every panel source of the subset must have one
(a NaN factor would silently drop that source's prompts).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from castor import evalset, provenance, reliability, sanityb
from castor.judge.common import read_jsonl

SIGNAL = Path(__file__).resolve().parent.parent / "runs" / "006" / "signal.json"
FIELD = "wg_vulnerable"


def scaled(ys: dict[str, np.ndarray], factor: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """factor[m]: per-prompt multiplier (len n_prompts)."""
    return {m: y * factor[m][:, None] for m, y in ys.items()}


def check_factors(factors: dict[str, float], what: str, expected: list[str]) -> None:
    missing = sorted(set(expected) - set(factors))
    bad = {k: v for k, v in factors.items()
           if not (isinstance(v, (int, float)) and math.isfinite(v) and 0 < v <= 1)}
    if missing or bad:
        raise SystemExit(f"{what}: no factor for {missing}; not finite in (0, 1]: {bad}")


def summarise(ys: dict, info: dict, n_boot: int, seed: int, ci: bool = True) -> dict:
    tw = reliability.twin_correlations(ys, info)
    return {"levels": tw["levels"],
            "ci_prompts": reliability.bootstrap_prompts(ys, info, n_boot, seed) if ci else None,
            "mean_vulnerability": {m: float(np.nanmean(reliability.per_prompt_p(y)))
                                   for m, y in ys.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--report", type=Path)
    g.add_argument("--placeholder", action="store_true")
    ap.add_argument("--signal", type=Path, default=SIGNAL)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (submit.sh snapshots are deleted)"
    commit, cmd, t0 = provenance.git_commit(), provenance.command(), time.time()
    sig = json.loads(a.signal.read_text())
    n_boot, seed = sig["n_boot"], sig["seed"]

    panel = evalset.load_eval()
    source = {r["id"]: r["source"] for r in panel}
    ids = sanityb.subset_ids(panel, sanityb.PRIMARY_SUBSET)
    src = np.array([source[i] for i in ids])
    cells = {c.model_id: c for c in sanityb.cells()}
    info = {m: c.info for m, c in cells.items()}
    ys = {m: sanityb.outcome_matrix(read_jsonl(sanityb.RUNS / m / sanityb.PROMPT_ID / "judge_v1"
                                               / "labels.jsonl"), ids, FIELD)[0]
          for m in cells}

    if a.placeholder:
        pm = dict.fromkeys(cells, 0.9)
        ps = {s: {"wildjailbreak": 0.8}.get(s, 0.95) for s in sorted(set(src))}
    else:
        rep = json.loads(a.report.read_text())
        assert rep.get("unresolved", 0) == 0, "report built on an unadjudicated audit"
        b = rep["bootstrap"]
        pm = {m: b[k]["point"] for m in cells if (k := f"adv:{m}:precision") in b}
        ps = {s: b[k]["point"] for s in sorted(set(src))
              if (k := f"adv_twins_source:{s}:precision") in b}
    check_factors(pm, "per_model", list(cells))
    check_factors(ps, "per_source", sorted(set(src)))
    n = len(ids)
    factors = {"none": {m: np.ones(n) for m in cells},
               "per_model": {m: np.full(n, pm[m]) for m in cells},
               "per_source": {m: np.array([ps[s] for s in src]) for m in cells}}
    res = {v: summarise(scaled(ys, f), info, n_boot, seed, ci=v != "per_model")
           for v, f in factors.items()}

    # the uncorrected variant must reproduce 006's primary contrast bit for bit
    ref = sig[sanityb.PRIMARY_SUBSET]["primary"]
    for lvl in ("MZ", "U"):
        for k in ("r", "r_corrected"):
            assert res["none"]["levels"][lvl][k] == ref["levels"][lvl][k], (lvl, k)
    for k in ("MZ_minus_U", "MZ_minus_U_corrected"):
        assert res["none"]["ci_prompts"][k] == ref["ci_prompts"][k], k

    out = {"variants": res, "factors": {"per_model": pm, "per_source": ps},
           "placeholder": a.placeholder, "n_prompts": n, "field": FIELD,
           "subset": sanityb.PRIMARY_SUBSET, "report": str(a.report) if a.report else None,
           "signal": str(a.signal), "n_boot": n_boot, "seed": seed,
           "script": "scripts/audit_correct_006.py", "git_commit": commit, "command": cmd,
           "seconds": time.time() - t0}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({v: {"r_MZ": r["levels"]["MZ"]["r"], "r_U": r["levels"]["U"]["r"],
                          "MZ_minus_U_corrected": (r["ci_prompts"] or {}).get(
                              "MZ_minus_U_corrected")}
                      for v, r in res.items()}, default=float))


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    main()
