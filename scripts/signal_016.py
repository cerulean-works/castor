"""Size-ladder analysis: per size (150M, 300M) the signal_006.py statistics on the 12 twins of
that size, the per-size gate, the pre-stated secondaries, and the three-point trend with the 1B
main-recipe twins. Pure CPU; reads judge_v1 labels only.

    uv run --frozen python scripts/signal_016.py --out /abs/path/runs/016/signal.json \
        [--anchor-1b runs/008/signal_a0.json] [--n-boot 1000] [--allow-incomplete]

Inputs: <runs>/<model-id>/016/judge_v1/labels.jsonl for the 24 ladder twins (castor.ladder),
<runs>/<model-id>/008/judge_v1/labels.jsonl for the 12 1B twins of the main-recipe rerun
(`ladder.anchor_cells`), and the released OLMo-2-1B-DPO's 004 labels (descriptive). Sampled
responses only.

One launcher configuration: every twin of every size must have been trained on
`ladder.LAUNCHER` (SFT on ZeRO-1, DPO default), read from its provenance
(`ladder.check_cell_launcher`). The original 1B twins (ZeRO-3) are therefore not the 1B point:
`--anchor-1b` must be the A0 twins' signal file (`scripts/signal_006.py --twins a0`, default
`runs/008/signal_a0.json`); `runs/006/signal.json` is refused by its model ids
(`ladder.read_anchor`).

Per size, as signal_006.py (same functions: castor.sanityb, castor.reliability), for each subset
(adv_harmful = primary, plain_harmful, harmful, benign) × outcome (primary, incl, harm, q3g,
agree): per-model stats, twin correlations, prompt- and base-cluster bootstrap CIs, base
jackknife; per-source distributions; the gate (adv_harmful × primary); response counts;
training cost. Pre-stated secondaries: the variance of per-prompt P and the shares P = 0 / P = 1
per model and subset; on adv_harmful × primary, the contrast per source, the template residual
on HumanJailbreaks, top-decile transfer and the OLMo-2-1B-DPO reference.
Trend (pre-registered, castor.ladder): corrected gap per size and every pairwise difference
under one shared prompt resample, the rule, and a check that the recomputed 1B gap equals the
anchor signal file's.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from castor import evalset, gridstats, ladder, pipeval, provenance, reliability, sanityb
from castor.judge.common import read_jsonl

OUTCOMES = {"primary": ("wg_vulnerable", False), "incl": ("wg_vulnerable_incl", False),
            "harm": ("wg_harmful", False), "q3g": ("q3g_vulnerable", False),
            "agree": ("wg_vulnerable", True)}  # = signal_006.OUTCOMES
REFERENCE = "olmo2-1b-dpo-ai2"
HJ_SOURCE, WJ_SOURCE = "harmbench_humanjb", "wildjailbreak"


def labels_path(model_id: str, prompt_id: str) -> Path:
    return sanityb.RUNS / model_id / prompt_id / "judge_v1" / "labels.jsonl"


def training_cost(cell) -> dict:
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


def spread(y: np.ndarray) -> dict:
    """Pre-stated secondary: variance of per-prompt P and the shares at 0 and 1."""
    p = reliability.per_prompt_p(y)
    p = p[np.isfinite(p)]
    return {"var_P": float(p.var()) if len(p) else float("nan"),
            "share_0": float(np.mean(p == 0)) if len(p) else float("nan"),
            "share_1": float(np.mean(p == 1)) if len(p) else float("nan")}


def twin_block(ys: dict, info: dict, n_boot: int, seed: int) -> dict:
    tw = reliability.twin_correlations(ys, info)
    return {"levels": tw["levels"], "pairs": tw["pairs"], "reliability": tw["reliability"],
            "gap_corrected": ladder.gap(tw["levels"]),
            "ci_prompts": reliability.bootstrap_prompts(ys, info, n_boot, seed),
            "ci_bases": reliability.bootstrap_bases(ys, info, n_boot, seed),
            "jackknife_bases": reliability.jackknife_bases(ys, info)}


def d042(ys: dict, twins: list[str], info: dict, ids: list[str], source: dict, sid: dict,
         y_ref: np.ndarray, n_boot: int, seed: int) -> dict:
    """The four pre-stated secondaries on one size's twins (adv_harmful × primary), as
    analyze_008."""
    infos = [info[m] for m in twins]
    P, Po, Pe = gridstats.halves(ys, twins)
    rel = gridstats.reliabilities(Po, Pe)
    sec = {}
    for src in (WJ_SOURCE, HJ_SOURCE):  # (2) the contrast per source
        cols = [i for i, pid in enumerate(ids) if source[pid] == src]
        sec[f"source_{src}"] = twin_block({m: ys[m][cols] for m in twins}, info, n_boot, seed)
    cols = [i for i, pid in enumerate(ids) if source[pid] == HJ_SOURCE]  # (1) template residual
    beh = np.array([sid[ids[i]].split("::")[0] for i in cols])
    tpl = np.array([sid[ids[i]].split("::")[1] for i in cols])
    Rp, Ro, Re = gridstats.residualise(P[:, cols], Po[:, cols], Pe[:, cols], beh, tpl)
    rrel = gridstats.reliabilities(Ro, Re)
    sec["template_residual"] = {
        "n_prompts": len(cols), "n_behaviours": len(set(beh)), "n_templates": len(set(tpl)),
        "levels": gridstats.levels(Rp, rrel, infos),
        "reliability": dict(zip(twins, rrel.tolist())),
        "ci_prompts_conditional_on_fit": gridstats.prompt_bootstrap(Rp, Ro, Re, infos, n_boot,
                                                                    seed)}
    sec["top_decile_transfer"] = gridstats.top_decile_transfer(P, Po, Pe, infos)  # (3)
    rel_ref = reliability.split_half(y_ref)["reliability"]  # (4), descriptive
    sec["reference_olmo2_1b_dpo"] = {
        "reliability_ref": rel_ref,
        "per_model": gridstats.reference_correlation(P, rel, reliability.per_prompt_p(y_ref),
                                                     rel_ref, twins)}
    return sec


def size_result(cells: dict, rows: dict, panel: list[dict], ref_rows: list[dict],
                incomplete: dict, n_boot: int, seed: int) -> tuple[dict, dict, list[str]]:
    """Everything signal_006.py computes, for one size's 12 twins. Returns (block, primary ys,
    the prompt ids naming the primary matrices' rows)."""
    source = {r["id"]: r["source"] for r in panel}
    sid = {r["id"]: r["source_id"] for r in panel}
    twins = [m for m in cells if m in rows]
    info = {m: cells[m].info for m in twins}
    res = {"models": {m: {"base": cells[m].base_ref, "posttrain_seed_k": cells[m].k}
                      for m in twins},
           "missing_label_cells": {m: v for m, v in incomplete.items() if m in cells}}
    prim_ys, prim_ids = None, None
    for s in sanityb.SUBSETS:
        ids = sanityb.subset_ids(panel, s)
        block = {}
        for name, (field, agree) in OUTCOMES.items():
            ys, counts = {}, {}
            for m in twins:
                ys[m], counts[m] = sanityb.outcome_matrix(rows[m], ids, field, agree_only=agree)
            stats = {m: sanityb.model_stats(y) | spread(y) for m, y in ys.items()}
            b = {"per_model": stats, "counts": counts}
            if len({info[m].base for m in twins}) >= 2:
                b |= twin_block(ys, info, n_boot, seed)
            else:
                tw = reliability.twin_correlations(ys, info)
                b |= {"levels": tw["levels"], "pairs": tw["pairs"],
                      "reliability": tw["reliability"]}
            b["d035"] = sanityb.d035_line(b)  # reported only; not a ladder rule
            if name == "primary":
                b["by_source"] = {
                    m: {src: sanityb.distribution(ys[m][[i for i, pid in enumerate(ids)
                                                         if source[pid] == src]])
                        for src in sorted({source[pid] for pid in ids})}
                    for m in ys}
            block[name] = b
            if s == sanityb.PRIMARY_SUBSET and name == "primary":
                prim_ys, prim_ids = ys, ids
                if len(twins) == len(cells):
                    y_ref, _ = sanityb.outcome_matrix(ref_rows, ids, field)
                    res["secondary_D042"] = d042(ys, twins, info, ids, source, sid, y_ref,
                                                 n_boot, seed)
        res[s] = block
    prim = res[sanityb.PRIMARY_SUBSET]["primary"]
    res["gate"] = sanityb.gate({m: prim["per_model"][m] for m in twins}, cells,
                               res["missing_label_cells"])
    res["response_counts"] = {
        m: {"all": sanityb.response_counts(rows[m]),
            **{s: sanityb.response_counts(rows[m], set(sanityb.subset_ids(panel, s)))
               for s in sanityb.SUBSETS}}
        for m in twins}
    res["training_cost"] = {m: training_cost(cells[m]) for m in cells}
    return res, prim_ys, prim_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="0 = per-size blocks only (CPU smoke); the trend rule is then undecided")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="run on the twins and label cells that exist, with or without the "
                         "anchor file (verdicts INCOMPLETE)")
    ap.add_argument("--anchor-1b", type=Path, default=ladder.ANCHOR_SIGNAL,
                    help="the A0 twins' signal file of record (signal_006.py --twins a0); the "
                         "recomputed 1B gap must equal its gap")
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (submit.sh snapshots are deleted)"
    assert a.n_boot >= 0, "--n-boot must be non-negative"
    commit, t0 = provenance.git_commit(), time.time()
    sizes = {z: {c.model_id: c for c in ladder.cells((z,))} for z in ladder.SIZES}
    sizes["1B"] = {c.model_id: c for c in ladder.anchor_cells()}
    pid = {z: ladder.PROMPT_ID for z in ladder.SIZES} | {"1B": ladder.ANCHOR_PROMPT_ID}
    # the anchor file is read first: a wrong file (its twins are not the A0 cells) fails here,
    # as does its absence without --allow-incomplete, not after the per-size statistics
    # (ladder.read_anchor / anchor_block own that policy; a ladder.Anchor is the proof that
    # the twins were checked, so nothing below checks them again)
    anchor = ladder.read_anchor(list(sizes["1B"].values()), a.anchor_1b, a.allow_incomplete)
    panel = evalset.load_eval()
    all_ids = [r["id"] for r in panel]

    # one launcher configuration for every twin of every size; a twin
    # without a finished chain is only tolerated under --allow-incomplete
    want = ladder.Launcher()  # resolved once (two file hashes)
    launchers = {}
    for z, cells in sizes.items():
        for m, c in cells.items():
            got = ladder.check_cell_launcher(c, ladder.MODELS_ROOT, want=want)
            assert set(got) == {"sft", "dpo"} or a.allow_incomplete, \
                f"{m}: provenance for {sorted(got)} only (chain not finished)"
            launchers[m] = got
    rows = {}
    for z, cells in sizes.items():
        for m in cells:
            p = labels_path(m, pid[z])
            if p.exists():
                rows[m] = list(read_jsonl(p))
            else:
                assert a.allow_incomplete, f"missing {p}"
    ref_rows = list(read_jsonl(labels_path(REFERENCE, sanityb.REFERENCE_PROMPT_ID)))
    # every size and the reference must be labelled by one instrument on the current panel
    runs = {m: (m, pid[z]) for z, cells in sizes.items() for m in cells if m in rows} | {
        REFERENCE: (REFERENCE, sanityb.REFERENCE_PROMPT_ID)}
    eval_sha = json.loads(evalset.EVAL_MANIFEST.read_text())["eval"]["sha256"]
    instrument = pipeval.check_instrument(runs, eval_sha)
    incomplete = {m: c["missing"] for m in rows
                  if (c := sanityb.outcome_matrix(rows[m], all_ids, "wg_vulnerable")[1])["missing"]}
    assert a.allow_incomplete or not incomplete, f"missing label cells: {incomplete}"

    res = {"subsets": {s: len(sanityb.subset_ids(panel, s)) for s in sanityb.SUBSETS},
           "instrument": instrument, "sizes": {}}
    prim_ys, prim_ids = {}, {}
    for z, cells in sizes.items():
        have = {m: rows[m] for m in cells if m in rows}
        r, prim_ys[z], prim_ids[z] = size_result(cells, have, panel, ref_rows, incomplete,
                                                 a.n_boot, a.seed)
        res["sizes"][z] = r
        print(z, r["gate"]["verdict"],
              {k: r[sanityb.PRIMARY_SUBSET]["primary"]["levels"][k] for k in ("MZ", "U")})

    # every twin labelled and no label cell missing; otherwise the rule is INCOMPLETE, not
    # 'fails' (a size with missing cells cannot pass its gate)
    complete_size = {z: len(prim_ys[z]) == len(sizes[z]) and not (set(sizes[z]) & set(incomplete))
                     for z in sizes}
    complete = all(complete_size.values())
    # 1B anchor: the recomputed gap must be the A0 signal file's (same labels, same functions);
    # skipped and recorded only under --allow-incomplete (ladder.anchor_block), and then the
    # rule is INCOMPLETE, never a verdict on an unverified anchor
    lv1b = res["sizes"]["1B"][sanityb.PRIMARY_SUBSET]["primary"]["levels"]
    res["anchor_1B"] = ladder.anchor_block(anchor, ladder.gap(lv1b), complete_size["1B"],
                                           a.allow_incomplete)
    res["launcher"] = {"configuration": ladder.LAUNCHER, "expected": want.as_dict(),
                       "per_model": launchers}

    passed = {z: res["sizes"][z]["gate"]["verdict"] == "GO" for z in sizes}
    info = {z: {m: sizes[z][m].info for m in prim_ys[z]} for z in sizes}
    boot = ladder.paired_size_bootstrap({z: prim_ys[z] for z in ladder.TREND},
                                        {z: prim_ids[z] for z in ladder.TREND},
                                        {z: info[z] for z in ladder.TREND}, a.n_boot, a.seed)
    res["trend"] = {"bootstrap": boot, "gates_passed": passed,
                    "rule": ladder.trend_rule(boot, passed, labels_complete=complete,
                                              anchor_checked=res["anchor_1B"]["checked"])}
    res.update({"script": "scripts/signal_016.py", "git_commit": commit,
                "command": provenance.command(), "n_boot": a.n_boot, "seed": a.seed,
                "seconds": time.time() - t0})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"gates": passed, "trend": res["trend"]["rule"],
                      "gaps": {k: v for k, v in boot.items() if k.startswith(("gap", "diff"))}},
                     default=float))


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    main()
