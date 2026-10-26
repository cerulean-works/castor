"""The ZeRO-1-vs-ZeRO-3 configuration effect on the n = 12 same-cell pairs (006's twins vs their
A0 reruns in 008), under 024's three criteria. A statement, not a gate. CPU only (weights are
read in fp32).

    uv run --frozen python scripts/config_effect_027.py --out /abs/runs/027/config_effect.json
        [--n-boot 2000] [--seed 0] [--weight-stages dpo sft]

Models (24): 006's 12 (<runs>/dd-dclm-1b-<seed>-dpo-seed<k>/006/judge_v1/labels.jsonl, SFT on
ZeRO-3) and 008's 12 A0 cells (<runs>/dd-dclm-baseline-1b-<seed>-dpo-seed<k>/008/judge_v1/
labels.jsonl, SFT on ZeRO-1, asserted on every SFT provenance). One instrument across all 24.
1. Behaviour (castor.confeff): per-prompt corrected r on adv_harmful x wg_vulnerable for every
   within-base pair, by kind (same_cell_cross = the 12 configuration pairs; MZ within each
   configuration; MZ across), the contrasts, and a prompt bootstrap (all 24 resampled jointly).
   Point summaries for the other subsets (no bootstrap).
2. Weights (024's rule): global fp32 L2 of every within-base pair after DPO (the rule) and SFT
   (reported), each same-cell distance as a ratio to 024's identical-seed floor (runs/024/
   equiv.json, R|B) against the 1.25 bound; the within- and cross-configuration seed-to-seed
   distances as the reference.
3. Levels (024's rule): per panel source the pooled refusal rate (pipeval.refusal_rate,
   wg_vulnerable_incl), A0 - 006 per cell, against the 3-point band.
Also: per-model mean vulnerability (adv_harmful) A0 vs 006, and a read-only copy of 009's
fifth-base rows (runs/009/factorial_fifth.json) that this effect is reported beside.
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

from castor import confeff, evalset, pipeval, provenance, reliability, sanityb
from castor.judge.common import read_jsonl

ROOT = Path(__file__).resolve().parent.parent
EQUIV_024 = ROOT / "runs" / "024" / "equiv.json"
FIFTH_009 = ROOT / "runs" / "009" / "factorial_fifth.json"
SUBSET, FIELD, LEVEL_FIELD = "adv_harmful", "wg_vulnerable", "wg_vulnerable_incl"
FIFTH_ROWS = ("RX", "RY", "XY", "DR", "R")  # within our hardware, plus the rung-2 pair and R's MZ


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def models() -> dict[str, dict]:
    """id -> {info, prompt_id, cell, dirs}: 006's twins (zero3) and the A0 reruns (zero1)."""
    F = _load("floor_011")
    out = {}
    for m, (i, pid, sft) in F.comparator_models("006").items():
        out[m] = {"info": confeff.CfgInfo(i.recipe, i.pretrain_seed, i.posttrain_seed, confeff.ZERO3),
                  "prompt_id": pid, "sft": sft, "dpo": sft.parent.parent / "dpo" / sft.name}
    for m, (i, pid, sft) in F.comparator_models("a0").items():
        out[m] = {"info": confeff.CfgInfo(i.recipe, i.pretrain_seed, i.posttrain_seed, confeff.ZERO1),
                  "prompt_id": pid, "sft": sft, "dpo": sft.parent.parent / "dpo" / sft.name}
        sha = F.sft_ds_sha(sft)
        assert sha == F.SFT_ZERO1_SHA256, f"{m}: SFT config {sha} is not D067a's"
    assert len(out) == 24
    return out


def same_cells(ms: dict[str, dict]) -> list[tuple[str, str, str]]:
    """(A0 id, 006 id, cell tag) for the 12 cells."""
    by = {}
    for m, v in ms.items():
        by.setdefault(v["info"].cell, {})[v["info"].config] = m
    return [(c[confeff.ZERO1], c[confeff.ZERO3], tag) for tag, c in sorted(by.items())]


def weight_rows(ms: dict[str, dict], stage: str, E, log=print) -> list[dict]:
    """Every within-base pair's global fp32 L2 (024's `l2`, computed on flattened vectors:
    confeff.flatten / l2_flat). At most two flattened vectors (~5 GB each) are held at a time:
    one anchor and one partner, the partner reloaded per pair, so a base costs 8 + 28 = 36
    checkpoint reads instead of 8 held at once. Dev jobs only."""
    rows = []
    for base in sorted({v["info"].base for v in ms.values()}):
        ids = sorted(m for m, v in ms.items() if v["info"].base == base)
        t0 = time.time()
        for i, a in enumerate(ids):
            va = confeff.flatten(E.load_state(ms[a][stage] / "model"))
            for b in ids[i + 1:]:
                vb = confeff.flatten(E.load_state(ms[b][stage] / "model"))
                kind = confeff.kind_of(ms[a]["info"], ms[b]["info"])
                rows.append({"a": a, "b": b, "kind": kind, "l2": confeff.l2_flat(va, vb)})
                del vb
            del va
        log(f"weights {stage} {base[1]}: {len(ids)} models, {len(ids) * (len(ids) + 1) // 2} reads, "
            f"{time.time() - t0:.0f} s")
    return rows


def fifth_base_rows() -> dict:
    f = json.loads(FIFTH_009.read_text())
    fb = f["fifth_base"]
    prim = fb[SUBSET]["primary"]
    return {"path": str(FIFTH_009.relative_to(ROOT)), "git_commit": f["git_commit"],
            "label": fb["label"],
            "by_base_pair": {k: prim["by_base_pair"][k] for k in FIFTH_ROWS},
            "ci_prompts": {k: prim["ci_prompts"].get(f"bp_{k}_corrected") for k in FIFTH_ROWS},
            "contrasts": {k: v for k, v in prim["ci_prompts"].items() if not k.startswith("bp_")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)  # as 011 / 024
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight-stages", nargs="*", default=["dpo", "sft"], choices=["dpo", "sft"])
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (submit.sh snapshots are deleted)"
    commit, t0 = provenance.git_commit(), time.time()
    E = _load("equiv_024")
    ms = models()
    info = {m: v["info"] for m, v in ms.items()}
    cells = same_cells(ms)
    assert len(cells) == 12, cells

    panel = evalset.load_eval()
    eval_sha = json.loads(evalset.EVAL_MANIFEST.read_text())["eval"]["sha256"]
    instrument = pipeval.check_instrument({m: (m, v["prompt_id"]) for m, v in ms.items()}, eval_sha)
    rows = {m: list(read_jsonl(sanityb.RUNS / m / v["prompt_id"] / "judge_v1" / "labels.jsonl"))
            for m, v in ms.items()}
    all_ids = [r["id"] for r in panel]
    incomplete = {m: c["missing"] for m in rows
                  if (c := sanityb.outcome_matrix(rows[m], all_ids, FIELD)[1])["missing"]}
    assert not incomplete, f"missing label cells: {incomplete}"

    stat = confeff.stat_for(info)
    behaviour = {}
    for s in sanityb.SUBSETS:
        ids = sanityb.subset_ids(panel, s)
        ys = {m: sanityb.outcome_matrix(r, ids, FIELD)[0] for m, r in rows.items()}
        res = confeff.analyze(ys, info)
        block = {"n_prompts": len(ids), "summary": stat(res),
                 "per_model": {m: sanityb.model_stats(y) for m, y in ys.items()},
                 "pairs": res["pairs"]}
        if s == SUBSET:
            block["ci_prompts"] = reliability.bootstrap_prompts(ys, info, a.n_boot, a.seed,
                                                                stat=stat, analyze=confeff.analyze)
        block["mean_shift"] = {tag: block["per_model"][a0]["mean_vulnerability"] - block["per_model"][m6]["mean_vulnerability"]
                               for a0, m6, tag in cells}
        behaviour[s] = block

    eq = json.loads(EQUIV_024.read_text())
    weights = {}
    for st in a.weight_stages:
        wr = weight_rows(ms, st, E)
        weights[st] = confeff.weight_table(wr, floor=eq["weights"][st]["floor"]) | {
            "floor_source": {"path": str(EQUIV_024.relative_to(ROOT)), "key": f"weights.{st}.floor",
                             "git_commit": eq["git_commit"], "pass_024": eq["weights"][st]["pass"]},
            "rows": wr}

    pp = {m: pipeval.per_prompt(pipeval.refusals_from_labels(r, LEVEL_FIELD)) for m, r in rows.items()}
    by_src = {}
    for r in panel:
        by_src.setdefault(r["source"], []).append(r["id"])
    rates = {s: {m: pipeval.refusal_rate(pp[m], i) for m in pp} for s, i in sorted(by_src.items())}
    levels = confeff.level_table(rates, cells) | {"rates": rates}

    res = {"models": {m: {"pretrain_seed": v["info"].pretrain_seed,
                          "posttrain_seed_k": int(v["info"].posttrain_seed),
                          "config": v["info"].config, "prompt_id": v["prompt_id"],
                          "sft_deepspeed_config": _load("floor_011").sft_ds_sha(v["sft"])}
                      for m, v in ms.items()},
           "cells": [{"a0": a0, "006": m6, "tag": tag} for a0, m6, tag in cells],
           "instrument": instrument, "subsets": {s: behaviour[s]["n_prompts"] for s in behaviour},
           "behaviour": behaviour, "weights": weights, "levels": levels,
           "reference_024": {"path": str(EQUIV_024.relative_to(ROOT)), "git_commit": eq["git_commit"],
                             "behaviour": {k: eq["behaviour"][k] for k in
                                           ("r_star", "diff_B_minus_A", "diff_ci")},
                             "weights": {st: {k: eq["weights"][st][k] for k in
                                              ("distances", "floor", "ratio_AR", "ratio_AB", "pass")}
                                         for st in ("dpo", "sft")},
                             "levels_max_abs_B_minus_R": max(abs(v["B_minus_R"]) for v in
                                                             eq["levels"]["sources"].values())},
           "fifth_base_009": fifth_base_rows(),
           "script": "scripts/config_effect_027.py", "git_commit": commit,
           "command": provenance.command(), "n_boot": a.n_boot, "seed": a.seed,
           "seconds": time.time() - t0}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    sm, ci = behaviour[SUBSET]["summary"], behaviour[SUBSET]["ci_prompts"]
    print(f"same-cell r* {sm['r_same_cell_cross_corrected']:.4f} "
          f"[{ci['r_same_cell_cross_corrected']['lo']:.4f}, {ci['r_same_cell_cross_corrected']['hi']:.4f}]; "
          f"MZ within {sm['r_MZ_within_corrected']:.4f}; config effect (MZ cross - within) "
          f"{sm['config_effect_corrected']:+.4f} [{ci['config_effect_corrected']['lo']:+.4f}, "
          f"{ci['config_effect_corrected']['hi']:+.4f}]")
    for st, w in weights.items():
        print(f"weights {st}: same-cell L2 ratio to floor {w['ratio_mean']:.3f} "
              f"[{w['ratio_min']:.3f}, {w['ratio_max']:.3f}], {w['n_within_bound']}/{w['n_same_cell']} within 1.25")
    print(f"levels: {levels['n_within']}/{levels['n_total']} within 3 pt, max |delta| "
          f"{100 * levels['max_abs_delta']:.2f} pt")


if __name__ == "__main__":
    main()
