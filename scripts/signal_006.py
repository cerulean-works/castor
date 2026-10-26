"""Seed-twin signal analysis on the 12 main-recipe twins: per-model reliability and
vulnerability, the pre-registered GO / KILL rule, r_MZ / r_U raw and ceiling-corrected with CIs,
the d035 decision line, and the reported-not-gated sensitivities. Pure CPU; reads judge_v1
labels only.

    uv run --frozen python scripts/signal_006.py --out /abs/path/runs/006/signal.json \
        [--n-boot 1000] [--allow-incomplete]
    # the same statistics on the 12 main-recipe rerun twins (the 1B anchor of the ladder):
    uv run --frozen python scripts/signal_006.py --twins a0 \
        --out /abs/path/runs/008/signal_a0.json

Inputs: <runs>/<model-id>/006/judge_v1/labels.jsonl for the 12 twins (castor.sanityb), and
<runs>/<ref>/004/judge_v1/labels.jsonl for the 3 released OLMo-2-1B references (reported per
model only, never in a twin correlation or the gate). Sampled responses only (greedy excluded).
With `--twins a0` the twins are `castor.ladder.anchor_cells()` (ids dd-dclm-baseline-1b-…,
labels under <runs>/<model-id>/008/); everything else is identical.

Blocks of the output, each per subset (adv_harmful = primary, plain_harmful, harmful, benign):
  primary      wg_vulnerable = P(non-refusal | coherent); per-model stats, twin
               correlations (castor.reliability) with prompt-bootstrap CIs, base-cluster
               bootstrap CIs and leave-one-base-out jackknife SEs.
  incl         wg_vulnerable_incl (WildGuard's raw refusal call, incoherent included)
  harm         wg_harmful (secondary outcome)
  q3g          q3g_vulnerable (second judge)
  agree        wg_vulnerable on responses where both judges agree about refusal
Plus: gate (adv_harmful, primary), d035 (adv_harmful, primary; harmful / incl reported next to
it), per-source distributions, per-model response counts (incoherent rate, judge errors,
llm_invalid share of REFUSAL labels), training wall-clock and H200-hours.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from castor import evalset, ladder, pipeval, provenance, reliability, sanityb
from castor.judge.common import read_jsonl

# which 12 main-recipe twins: the originals (default) or the A0 rerun (the ladder's 1B anchor)
TWINS = {"006": (sanityb.cells, sanityb.PROMPT_ID),
         "a0": (ladder.anchor_cells, ladder.ANCHOR_PROMPT_ID)}

OUTCOMES = {"primary": ("wg_vulnerable", False), "incl": ("wg_vulnerable_incl", False),
            "harm": ("wg_harmful", False), "q3g": ("q3g_vulnerable", False),
            "agree": ("wg_vulnerable", True)}


def labels_path(model_id: str, prompt_id: str) -> Path:
    return sanityb.RUNS / model_id / prompt_id / "judge_v1" / "labels.jsonl"


def training_cost(cell: sanityb.Cell) -> dict:
    out = {}
    for stage in ("sft", "dpo"):
        p = cell.out(stage) / "provenance.json"
        if not p.exists():
            continue
        prov = json.loads(p.read_text())
        if "finished" not in prov:
            continue
        h = (datetime.fromisoformat(prov["finished"])
             - datetime.fromisoformat(prov["started"])).total_seconds() / 3600
        out[stage] = {"wall_h": h, "h200_h": 8 * h, "slurm_job": prov.get("slurm_job")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="run on the twins and label cells that exist (verdict INCOMPLETE)")
    ap.add_argument("--twins", choices=sorted(TWINS), default="006",
                    help="006 = 006's twins (default); a0 = the main-recipe rerun (008 stage A0)")
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (submit.sh snapshots are deleted)"
    commit, t0 = provenance.git_commit(), time.time()
    panel = evalset.load_eval()
    source = {r["id"]: r["source"] for r in panel}

    twin_cells, twin_pid = TWINS[a.twins]
    cells = {c.model_id: c for c in twin_cells()}
    rows = {}
    for m in cells:
        p = labels_path(m, twin_pid)
        if p.exists():
            rows[m] = list(read_jsonl(p))
        else:
            assert a.allow_incomplete, f"missing {p}"
    for m in sanityb.REFERENCES:
        rows[m] = list(read_jsonl(labels_path(m, sanityb.REFERENCE_PROMPT_ID)))
    twins = [m for m in cells if m in rows]
    info = {m: cells[m].info for m in twins}
    # twins and references must be labelled by one instrument on the current panel
    runs = {m: (m, twin_pid) for m in twins} | {
        m: (m, sanityb.REFERENCE_PROMPT_ID) for m in sanityb.REFERENCES}
    eval_sha = json.loads(evalset.EVAL_MANIFEST.read_text())["eval"]["sha256"]
    instrument = pipeval.check_instrument(runs, eval_sha)
    # every labelled model covers the full panel: no silently partial matrix feeds the gate
    all_ids = [r["id"] for r in panel]
    incomplete = {m: c["missing"] for m in rows
                  if (c := sanityb.outcome_matrix(rows[m], all_ids, "wg_vulnerable")[1])["missing"]}
    assert a.allow_incomplete or not incomplete, f"missing label cells: {incomplete}"

    res = {"models": {m: {"base": cells[m].base_ref, "posttrain_seed_k": cells[m].k}
                      for m in twins} | {m: {"reference": True} for m in sanityb.REFERENCES},
           "subsets": {s: len(sanityb.subset_ids(panel, s)) for s in sanityb.SUBSETS},
           "instrument": instrument, "missing_label_cells": incomplete}
    for s in sanityb.SUBSETS:
        ids = sanityb.subset_ids(panel, s)
        block = {}
        for name, (field, agree) in OUTCOMES.items():
            ys, counts = {}, {}
            for m, r in rows.items():
                ys[m], counts[m] = sanityb.outcome_matrix(r, ids, field, agree_only=agree)
            stats = {m: sanityb.model_stats(y) for m, y in ys.items()}
            tw_ys = {m: ys[m] for m in twins}
            tw = reliability.twin_correlations(tw_ys, info)
            b = {"per_model": stats, "counts": counts,
                 "levels": tw["levels"], "pairs": tw["pairs"]}
            if len({info[m].base for m in twins}) >= 2:
                b["ci_prompts"] = reliability.bootstrap_prompts(tw_ys, info, a.n_boot, a.seed)
                b["ci_bases"] = reliability.bootstrap_bases(tw_ys, info, a.n_boot, a.seed)
                b["jackknife_bases"] = reliability.jackknife_bases(tw_ys, info)
            b["d035"] = sanityb.d035_line(tw)
            if name == "primary":
                b["by_source"] = {
                    m: {src: sanityb.distribution(ys[m][[i for i, pid in enumerate(ids)
                                                         if source[pid] == src]])
                        for src in sorted({source[pid] for pid in ids})}
                    for m in ys}
            block[name] = b
        res[s] = block

    prim = res[sanityb.PRIMARY_SUBSET]["primary"]
    res["gate"] = sanityb.gate({m: prim["per_model"][m] for m in twins}, cells, incomplete)
    res["d035"] = {"decides": prim["d035"],
                   "reported_004_framing": res["harmful"]["incl"]["d035"]}
    res["prediction_r_MZ_gt_r_U"] = bool(prim["levels"]["MZ"]["r"] > prim["levels"]["U"]["r"])
    res["response_counts"] = {
        m: {"all": sanityb.response_counts(r),
            **{s: sanityb.response_counts(r, set(sanityb.subset_ids(panel, s)))
               for s in sanityb.SUBSETS}}
        for m, r in rows.items()}
    res["training_cost"] = {m: training_cost(cells[m]) for m in cells}
    res.update({"script": "scripts/signal_006.py", "twins": a.twins, "git_commit": commit,
                "command": provenance.command(), "n_boot": a.n_boot, "seed": a.seed,
                "seconds": time.time() - t0})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float))
    g = res["gate"]
    print(json.dumps({"verdict": g["verdict"], "missing": g["missing"],
                      "missing_label_cells": incomplete,
                      "d035_fires": res["d035"]["decides"]["fires"],
                      "r_MZ": prim["levels"]["MZ"], "r_U": prim["levels"]["U"]},
                     default=float))


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    main()
