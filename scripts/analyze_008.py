"""Analysis of the 25-recipe grid: the pre-registered primary contrast, its secondaries and four
additional analyses. Pure CPU; reads judge_v1 labels only.

    uv run --frozen python scripts/analyze_008.py --manifest data/manifests/grid_008.json \
        --stages A B [C] --out /abs/path/runs/008/grid.json [--n-boot 1000] \
        [--allow-incomplete] [--runs DIR]

--stages = the stages of data/manifests/grid_008_stages.json that ran; their cells are the
expected models. Without C, recipes other than the main 8 have one post-training seed: r_MZ and
MZ − U average over the 9 recipes with two seeds, r_U and r_R over all 25 (castor/gridstats.py
module doc), and the result is labelled a declared deviation from the two-seed design.

--runs and --models-root replace the runs root and the models root for the mock end-to-end
exercise only; every reported number uses the defaults.

Models (150): the cells with post-training seed 1 or 2 of the stages that ran
(<runs>/<model-id>/008/judge_v1/labels.jsonl). The main recipe comes from the grid's own rerun
cells (stage A0: dd-dclm-baseline-1b-<s>-dpo-seed<k>) when A0 ran, else from 006
(<runs>/dd-dclm-1b-<s>-dpo-seed<k>/006/…); never both. 25 recipes × 3 pretraining seeds × 2
post-training seeds, one instrument and one launcher configuration (both checked). Seeds 3 and 4
of the main recipe are not in any grid number (every recipe gets the same 2 seeds).
The released OLMo-2-1B DPO (004's labels) enters only additional analysis (4).

Blocks, per subset (adv_harmful = primary, plain_harmful, harmful, benign) × outcome (primary =
wg_vulnerable, incl, harm, q3g, agree):
  per_model    mean vulnerability, reliability, NaN share (castor.sanityb.model_stats)
  levels       per-recipe MZ / U / R raw and corrected; means over recipes; overall R
  ci_recipes   recipe bootstrap (25 recipes resampled), ci_prompts   prompt bootstrap
Only for adv_harmful × primary: prediction (the three pre-registered parts), variance
components of mean vulnerability (nested ANOVA), and additional analyses: (1) template residual
on HarmBench HumanJailbreaks, (2) the primary contrast per source, (3) top-decile transfer,
(4) correlation with olmo2-1b-dpo-ai2. The nested variance is also given for every subset ×
outcome.
"""

import argparse
import json
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from castor import evalset, grid, gridstats, pipeval, posttrain, provenance, reliability, sanityb
from castor.judge.common import read_jsonl

OUTCOMES = {"primary": ("wg_vulnerable", False), "incl": ("wg_vulnerable_incl", False),
            "harm": ("wg_harmful", False), "q3g": ("q3g_vulnerable", False),
            "agree": ("wg_vulnerable", True)}
GRID_SEEDS = (1, 2)  # the post-training seeds that enter the grid (main recipe included)
HJ_SOURCE, WJ_SOURCE = "harmbench_humanjb", "wildjailbreak"
REFERENCE = "olmo2-1b-dpo-ai2"


class Model(NamedTuple):
    info: reliability.ModelInfo
    prompt_id: str  # of its generations
    sft: Path
    dpo: Path
    base_ref: str  # REPO@revision its SFT must name (launchers)
    k: int  # post-training seed its SFT and DPO must name


def selected(g: grid.Grid, stages: list[grid.Stage] | None = None,
             ran: list[str] | None = None, root: Path = grid.MODELS_ROOT) -> dict[str, Model]:
    """model id -> Model: the cells with a GRID_SEED of the stages that ran (all cells without
    stages), and the main recipe's GRID_SEEDS from 006 only if no such cell is of the main recipe."""
    stages = stages or [grid.ALL_STAGE]
    cells = [c for c in g.cells() if c.k in GRID_SEEDS
             and (ran is None or grid.stage_of(c, stages) in ran)]
    out = {c.model_id: Model(reliability.ModelInfo(c.base.recipe, c.base.pretrain_seed, str(c.k)),
                             g.prompt_id, c.out("sft", root), c.out("dpo", root), c.base.ref, c.k)
           for c in cells}
    if not any(c.base.recipe == sanityb.RECIPE for c in cells):
        for c in sanityb.cells():
            if c.k in GRID_SEEDS:
                out[c.model_id] = Model(c.info, sanityb.PROMPT_ID, c.out("sft", root),
                                        c.out("dpo", root), c.base_ref, c.k)
    return out


def models(g: grid.Grid, stages: list[grid.Stage] | None = None,
           ran: list[str] | None = None) -> dict[str, tuple[reliability.ModelInfo, str]]:
    """model id -> (info, prompt id of its generations); see `selected`."""
    return {m: v[:2] for m, v in selected(g, stages, ran).items()}


def launchers(sel: dict[str, Model]) -> dict[str, dict | None]:
    """model id -> the launcher configuration (grid.launcher_of) of its SFT and DPO, None if a
    provenance is missing (a mock run). Each provenance must be this model's (stage, seed, base
    ref, DPO on this SFT) and finished; a directory holding another run is an error."""
    out = {}
    for m, v in sel.items():
        provs = {st: d / "provenance.json" for st, d in (("sft", v.sft), ("dpo", v.dpo))}
        if not all(p.exists() for p in provs.values()):
            out[m] = None
            continue
        sft, dpo = (json.loads(provs[st].read_text()) for st in ("sft", "dpo"))
        for st, prov in (("sft", sft), ("dpo", dpo)):
            w = provs[st]
            assert prov["stage"] == st, f"{w}: stage {prov['stage']}"
            assert prov["post_training_seed_k"] == v.k, f"{w}: seed {prov['post_training_seed_k']}"
            assert "finished" in prov, f"{w}: not finished"
        assert sft.get("base_ref") == v.base_ref, f"{provs['sft']}: base_ref {sft.get('base_ref')}"
        assert dpo["base_model"] == str(v.sft / "model"), f"{provs['dpo']}: base {dpo['base_model']}"
        out[m] = {st: grid.launcher_of(prov, st) for st, prov in (("sft", sft), ("dpo", dpo))}
    return out


def matrices(rows: list[dict], all_ids: list[str]) -> tuple[dict, dict]:
    """One model's label rows -> {outcome: (n_panel, K) matrix} on the full panel order, and the
    primary outcome's counts. Subsets are row slices of these (outcome_matrix places each
    (prompt, sample) by id, so slicing equals building on the subset's ids)."""
    out, counts = {}, None
    for oname, (field, agree) in OUTCOMES.items():
        y, c = sanityb.outcome_matrix(rows, all_ids, field, agree_only=agree)
        out[oname] = y
        counts = c if oname == "primary" else counts
    return out, counts


def variance(means: dict[str, float], infos: dict[str, reliability.ModelInfo]) -> dict:
    """Nested variance on a balanced design: every model if every recipe has the same number of
    post-training seeds; else seed 1 only (all recipes, the reading the prediction uses) and
    `two_seed_recipes` (None with a reason when fewer than 2 recipes have two seeds)."""
    n_k = gridstats.seeds_per_recipe(infos[m] for m in means)
    if len({len(v) for v in n_k.values()}) == 1:
        return gridstats.nested_variance(means, infos) | {"design": "all models"}
    s1 = {m: v for m, v in means.items() if infos[m].posttrain_seed == "1"}
    two = {m: v for m, v in means.items() if len(n_k[infos[m].recipe]) == 2}
    n_two = sum(len(v) == 2 for v in n_k.values())
    return gridstats.nested_variance(s1, infos) | {
        "design": "post-training seed 1 only (base level includes post-training-seed variance)",
        "two_seed_recipes": gridstats.nested_variance(two, infos) if n_two > 1 else None,
        "two_seed_recipes_skipped": None if n_two > 1 else
        f"nested ANOVA needs ≥ 2 recipes with two post-training seeds; {n_two} here"}


def fit(ys: dict, names: list[str]) -> tuple:
    """(P, P_odd, P_even, reliabilities) of these models, in `names` order."""
    P, Po, Pe = gridstats.halves(ys, names)
    return P, Po, Pe, gridstats.reliabilities(Po, Pe)


def block(ys: dict, names: list[str], infos: list, n_boot: int, seed: int,
          fitted: tuple | None = None) -> dict:
    """`fitted` = fit(ys, names), passed when the caller holds it already."""
    P, Po, Pe, rel = fit(ys, names) if fitted is None else fitted
    lv = gridstats.levels(P, rel, infos)
    return {"levels": lv, "reliability": dict(zip(names, rel.tolist())),
            "ci_recipes": gridstats.recipe_bootstrap(lv["per_recipe"], n_boot, seed),
            "ci_prompts": gridstats.prompt_bootstrap(P, Po, Pe, infos, n_boot, seed)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--runs", type=Path, default=grid.RUNS, help="mock runs only")
    ap.add_argument("--models-root", type=Path, default=grid.MODELS_ROOT, help="mock runs only")
    ap.add_argument("--stages", nargs="+", required=True, help="stages that ran (D049d)")
    ap.add_argument("--stages-file", default="data/manifests/grid_008_stages.json")
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (submit.sh snapshots are deleted)"
    commit, t0 = provenance.git_commit(), time.time()
    g = grid.load(a.manifest)
    panel = evalset.load_eval()
    source = {r["id"]: r["source"] for r in panel}
    sid = {r["id"]: r["source_id"] for r in panel}
    all_ids = [r["id"] for r in panel]

    stages = grid.load_stages(a.stages_file, g)
    assert set(a.stages) <= {s.name for s in stages}, f"unknown stages {a.stages}"
    sel = selected(g, stages, a.stages, a.models_root)
    mods = {m: v[:2] for m, v in sel.items()}
    mats, counts, missing = {}, {}, []
    for m, (_, pid) in [*mods.items(), (REFERENCE, ("", "004"))]:
        p = sanityb.labels_path(m, pid, a.runs)
        if p.exists():
            mats[m], counts[m] = matrices(list(read_jsonl(p)), all_ids)
        else:
            missing.append(m)
    assert a.allow_incomplete or not missing, f"missing labels: {missing[:5]} ({len(missing)})"
    assert REFERENCE in mats, f"{REFERENCE} labels missing"
    names = sorted(m for m in mods if m in mats)
    infos = [mods[m][0] for m in names]
    k_of = gridstats.seeds_per_recipe(infos)
    design = ("D016: two post-training seeds on every recipe"
              if all(len(v) == 2 for v in k_of.values()) else
              f"D035 variant, declared deviation from D016 for budget reasons (D049d): "
              f"{sum(len(v) == 2 for v in k_of.values())} recipes with two post-training "
              f"seeds, {len(k_of)} in all")
    # one instrument for every model (and the reference) on the current panel
    eval_sha = json.loads(evalset.EVAL_MANIFEST.read_text())["eval"]["sha256"]
    instrument = pipeval.check_instrument({m: (m, mods[m][1]) for m in names}
                                          | {REFERENCE: (REFERENCE, "004")}, eval_sha, a.runs)
    # one launcher configuration for every model of the grid
    launch = launchers({m: sel[m] for m in names})
    kinds = {json.dumps(v, sort_keys=True) for v in launch.values() if v is not None}
    assert len(kinds) <= 1, f"models trained on {len(kinds)} launcher configurations: {kinds}"
    no_prov = sorted(m for m, v in launch.items() if v is None)
    assert a.allow_incomplete or not no_prov, f"no training provenance: {no_prov[:5]}"
    incomplete = {m: c["missing"] for m, c in counts.items() if c["missing"]}
    assert a.allow_incomplete or not incomplete, f"missing label cells: {incomplete}"
    complete = not missing and not incomplete

    res = {"design": design, "stages": a.stages, "stages_file": a.stages_file,
           "stages_sha256": posttrain.sha256_file(grid.ROOT / a.stages_file),
           "n_models": len(names), "missing_models": missing, "missing_label_cells": incomplete,
           "complete": complete, "instrument": instrument,
           "launcher": {"configuration": json.loads(kinds.pop()) if kinds else None,
                        "models_without_training_provenance": no_prov},
           "models": {m: {"recipe": i.recipe, "pretrain_seed": i.pretrain_seed,
                          "posttrain_seed_k": int(i.posttrain_seed),
                          "labels": str(sanityb.labels_path(m, mods[m][1], a.runs))}
                      for m, i in zip(names, infos)},
           "subsets": {s: len(sanityb.subset_ids(panel, s)) for s in sanityb.SUBSETS}}
    for s in sanityb.SUBSETS:
        ids = sanityb.subset_ids(panel, s)
        pos = {pid: i for i, pid in enumerate(all_ids)}
        idx = np.array([pos[pid] for pid in ids])
        res[s] = {}
        for oname in OUTCOMES:
            ys = {m: mats[m][oname][idx] for m in [*names, REFERENCE]}
            stats = {m: sanityb.model_stats(y) for m, y in ys.items()}
            fitted = fit(ys, names)
            b = block(ys, names, infos, a.n_boot, a.seed, fitted)
            b["per_model"] = stats
            b["variance"] = variance({m: stats[m]["mean_vulnerability"] for m in names},
                                     dict(zip(names, infos))) if complete else None
            res[s][oname] = b
            if s == sanityb.PRIMARY_SUBSET and oname == "primary":
                P, Po, Pe, rel = fitted
                sec = {}
                # (2) the primary contrast per source
                for src in (WJ_SOURCE, HJ_SOURCE):
                    cols = [i for i, pid in enumerate(ids) if source[pid] == src]
                    sec[f"source_{src}"] = block({m: y[cols] for m, y in ys.items()},
                                                 names, infos, a.n_boot, a.seed)
                # (1) template residual on HumanJailbreaks (source_id = behaviour::template)
                cols = [i for i, pid in enumerate(ids) if source[pid] == HJ_SOURCE]
                beh = np.array([sid[ids[i]].split("::")[0] for i in cols])
                tpl = np.array([sid[ids[i]].split("::")[1] for i in cols])
                Rp, Ro, Re = gridstats.residualise(P[:, cols], Po[:, cols], Pe[:, cols], beh, tpl)
                rrel = gridstats.reliabilities(Ro, Re)
                rlv = gridstats.levels(Rp, rrel, infos)
                sec["template_residual"] = {
                    "n_prompts": len(cols), "n_behaviours": len(set(beh)),
                    "n_templates": len(set(tpl)), "levels": rlv,
                    "reliability": dict(zip(names, rrel.tolist())),
                    "ci_recipes": gridstats.recipe_bootstrap(rlv["per_recipe"], a.n_boot, a.seed),
                    "ci_prompts_conditional_on_fit": gridstats.prompt_bootstrap(
                        Rp, Ro, Re, infos, a.n_boot, a.seed)}
                # (3) top-decile transfer
                sec["top_decile_transfer"] = gridstats.top_decile_transfer(P, Po, Pe, infos)
                # (4) released OLMo-2-1B DPO, descriptive
                yr = ys[REFERENCE]
                rel_ref = reliability.split_half(yr)["reliability"]
                sec["reference_olmo2_1b_dpo"] = {
                    "reliability_ref": rel_ref,
                    "per_model": gridstats.reference_correlation(
                        P, rel, reliability.per_prompt_p(yr), rel_ref, names)}
                res["secondary_D042"] = sec
                if complete:
                    res["prediction"] = gridstats.prediction(b["levels"], b["variance"])

    prim = res[sanityb.PRIMARY_SUBSET]["primary"]
    res["primary_metric"] = {
        "MZ_minus_U_corrected_mean_over_recipes": prim["levels"]["MZ_minus_U_corrected"],
        "ci_recipes": prim["ci_recipes"]["MZ_minus_U_corrected"],
        "ci_prompts": prim["ci_prompts"]["MZ_minus_U_corrected"],
        "n_recipes": prim["levels"]["n_recipes"],
        "n_recipes_MZ": prim["levels"]["n_recipes_MZ"],
        "n_recipes_MZ_undefined": prim["levels"]["n_recipes_MZ_undefined"], "design": design}
    res.update({"script": "scripts/analyze_008.py", "git_commit": commit,
                "command": provenance.command(), "n_boot": a.n_boot, "seed": a.seed,
                "seconds": time.time() - t0})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"complete": complete, "n_models": len(names),
                      "primary": res["primary_metric"]}, default=float))


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    main()
