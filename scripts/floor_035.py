"""Rung 2 with a second same-seed replicate R2. The three same-seed base pairs R-D (read back from
runs/027 and asserted identical), R-R2 (same stack) and R2-D (different stack), their mean, the
pre-registered difference r(R-R2) - r(R-D) with its prompt-bootstrap CI and jackknife SE, the
frozen rule of floor_011.py on the three-pair mean, and the branch flag of the go / kill rule. CPU
only; judge_v1 labels only.

    uv run --frozen python scripts/floor_035.py --out /abs/path/runs/035/floor_r2.json \
        [--n-boot 2000] [--seed 0] [--stand-in-r2]

Models (20): the 12 A0 cells (runs/008 stage A0: released default / large-aux-2 / large-aux-3 x
post-training seeds 1-4, SFT ZeRO-1), the 4 replicate twins of runs/011, and the 4 R2 twins of
data/manifests/grid_035.json (<runs>/<model-id>/035/judge_v1/labels.jsonl). Otherwise as
`scripts/floor_011.py --comparator a0`: one instrument (pipeval.check_instrument), one SFT
configuration (sha asserted on all 20 provenances), the prompt bootstrap over all 20 models
jointly (so every difference carries its covariance), and the seed jackknife over the 12
same-seed models (D, R, R2). r_U, r_MZ and r(R-D) are recomputed and asserted equal to
runs/027/floor_a0.json (per-pair values do not depend on which other models are in the panel).

`--stand-in-r2` (dry run before R2 exists): the R2 model ids carry copies of the
replicate's labels and provenance, so every code path runs end to end on known answers
(r(R2-D) = r(R-D) exactly; R-R2 pairs the same model with itself for k = k'). The output is
marked `dry_run` and must be written to a path containing "dryrun"; no number in it is
reportable.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from castor import evalset, floor, grid, pipeval, provenance, reliability, sanityb
from castor.judge.common import read_jsonl

ROOT = Path(__file__).resolve().parent.parent
FLOOR_027 = ROOT / "runs" / "027" / "floor_a0.json"
MANIFEST_035 = ROOT / "data" / "manifests" / "grid_035.json"
PROMPT_ID = "035"
R2_BASE_REF = "castor/DataDecide-dclm-baseline-1B@replicate2-i6198-d6198"
R2_MODEL_IDS = tuple(f"dd-dclm-baseline-1b-replicate2-i6198-d6198-dpo-seed{k}" for k in (1, 2, 3, 4))


def r2_cells(manifest: Path = MANIFEST_035) -> list[grid.Cell]:
    """The 4 R2 cells: one base (replicate2-i6198-d6198), seeds 1-4. launch_035 calls this so
    the launcher and the analysis address the same cells."""
    cells = grid.checked_cells(manifest, PROMPT_ID, R2_BASE_REF)
    assert tuple(c.model_id for c in cells) == R2_MODEL_IDS, [c.model_id for c in cells]
    return cells


def check_against_027(f027: dict, res: dict, outcomes: dict, n_boot: int, seed: int,
                      atol: float = 1e-12) -> dict:
    """Assert that rung 2 (R-D), r_U and r_MZ of runs/027 are reproduced for every subset x
    outcome. With 027's `n_boot` and `seed` the bootstrap row draws are identical (they depend
    on the seed alone), so the r(R-D) prompt CIs must coincide too; otherwise that is only
    recorded. Returns the side-by-side."""
    pairs = {"r_rung2": "r_rung2_RD", "r_U_006": "r_U_006", "r_MZ_006": "r_MZ_006",
             "r_MZ_rep": "r_MZ_rep", "r_rep_other": "r_rep_other"}
    mismatches = []
    for s in sanityb.SUBSETS:
        for o in outcomes:
            a, b = f027[s][o]["summary"], res[s][o]["summary"]
            for k27, k35 in pairs.items():
                for sfx in ("", "_corrected"):
                    if not np.isclose(a[k27 + sfx], b[k35 + sfx], atol=atol, rtol=0):
                        mismatches.append((s, o, k27 + sfx, a[k27 + sfx], b[k35 + sfx]))
                if a[f"n_{k27[2:]}"] != b[f"n_{k35[2:]}"]:
                    mismatches.append((s, o, f"n_{k27[2:]}", a[f"n_{k27[2:]}"], b[f"n_{k35[2:]}"]))
    p27 = f027["prediction"]
    sm = res[sanityb.PRIMARY_SUBSET]["primary"]["summary"]
    for k27, k35 in (("r_rung2", "r_rung2_RD_corrected"), ("r_U", "r_U_006_corrected"),
                     ("r_MZ", "r_MZ_006_corrected")):
        if not np.isclose(p27[k27], sm[k35], atol=atol, rtol=0):
            mismatches.append(("prediction", k27, p27[k27], sm[k35]))
    assert not mismatches, f"027's values not reproduced: {mismatches}"
    ci27 = f027[sanityb.PRIMARY_SUBSET]["primary"]["ci_prompts"]["r_rung2_corrected"]
    ci35 = res[sanityb.PRIMARY_SUBSET]["primary"]["ci_prompts"]["r_rung2_RD_corrected"]
    ci_identical = bool(np.isclose(ci27["lo"], ci35["lo"], atol=atol, rtol=0)
                        and np.isclose(ci27["hi"], ci35["hi"], atol=atol, rtol=0))
    same_draws = (n_boot, seed) == (f027["n_boot"], f027["seed"])
    assert ci_identical or not same_draws, \
        f"same n_boot and seed as 027 but the r(R-D) prompt CI differs: 027 {ci27}, here {ci35}"
    return {"path": str(FLOOR_027.relative_to(ROOT)), "git_commit": f027["git_commit"],
            "n_boot": f027["n_boot"], "seed": f027["seed"],
            "prediction": p27, "point_values_identical": True,  # asserted above
            "ci_rung2_RD_027": ci27, "ci_rung2_RD_035": ci35,
            "ci_identical": ci_identical, "ci_identity_asserted": same_draws,
            "by_base_pair": f027["by_base_pair"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)  # pre-registered
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stand-in-r2", action="store_true",
                    help="DRY RUN: R2 = copies of the replicate's labels (R2 does not exist yet)")
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (submit.sh snapshots are deleted)"
    if a.stand_in_r2:
        assert "dryrun" in a.out.name, "a stand-in run must be written to a *dryrun* path"
    # only a stand-in dry run may run on uncommitted code
    commit, t0 = provenance.git_commit(allow_dirty=a.stand_in_r2), time.time()
    f027 = json.loads(FLOOR_027.read_text())
    # a record run uses 027's draws, so the R-D prompt CI is identical by construction
    assert a.stand_in_r2 or (a.n_boot, a.seed) == (f027["n_boot"], f027["seed"]), \
        f"a record run uses 027's draws (n_boot {f027['n_boot']}, seed {f027['seed']})"
    panel = evalset.load_eval()
    F11, L11 = provenance.load_script("floor_011"), provenance.load_script("launch_011")
    outcomes = F11.OUTCOMES

    # id -> (info, prompt id, SFT dir, labels source id)
    models: dict[str, tuple[reliability.ModelInfo, str, Path, str]] = {}
    for m, (info_m, pid, sft) in F11.comparator_models("a0").items():
        models[m] = (info_m, pid, sft, m)
    rep = {c.k: c for c in L11.cells()}
    for k, c in rep.items():
        models[c.model_id] = (reliability.ModelInfo(sanityb.RECIPE, floor.REPLICATE, str(k)),
                              L11.PROMPT_ID, c.out("sft"), c.model_id)
    for c in r2_cells():
        info_m = reliability.ModelInfo(sanityb.RECIPE, floor.REPLICATE2, str(c.k))
        if a.stand_in_r2:
            r = rep[c.k]
            models[c.model_id] = (info_m, L11.PROMPT_ID, r.out("sft"), r.model_id)
        else:
            models[c.model_id] = (info_m, PROMPT_ID, c.out("sft"), c.model_id)
    assert len(models) == 20, sorted(models)
    info = {m: v[0] for m, v in models.items()}
    rows = {}
    for m, (_, pid, _, src) in models.items():
        p = sanityb.labels_path(src, pid)
        assert p.exists(), f"missing {p}"
        rows[m] = list(read_jsonl(p))
    eval_sha = json.loads(evalset.EVAL_MANIFEST.read_text())["eval"]["sha256"]
    instrument = pipeval.check_instrument({m: (v[3], v[1]) for m, v in models.items()}, eval_sha)
    all_ids = [r["id"] for r in panel]
    incomplete = {m: c["missing"] for m in rows
                  if (c := sanityb.outcome_matrix(rows[m], all_ids, "wg_vulnerable")[1])["missing"]}
    assert not incomplete, f"missing label cells: {incomplete}"
    shas = {m: F11.sft_ds_sha(v[2]) for m, v in models.items()}
    assert set(shas.values()) == {F11.SFT_ZERO1_SHA256}, f"not one configuration: {shas}"

    same_seed = sorted(m for m, i in info.items() if i.pretrain_seed in floor.SAME_SEED)
    assert len(same_seed) == 12, same_seed
    stat = floor.stat_for_035(info)
    res = {"models": {m: {"pretrain_seed": i.pretrain_seed, "posttrain_seed_k": int(i.posttrain_seed),
                          "prompt_id": models[m][1], "labels_from": models[m][3],
                          "sft_deepspeed_config": shas[m]}
                      for m, i in info.items()},
           "instrument": instrument,
           "subsets": {s: len(sanityb.subset_ids(panel, s)) for s in sanityb.SUBSETS},
           "dry_run": ("R2 = copies of the replicate's labels (--stand-in-r2); no number here is "
                       "reportable") if a.stand_in_r2 else None}
    for s in sanityb.SUBSETS:
        ids = sanityb.subset_ids(panel, s)
        block = {}
        for name, (field, agree) in outcomes.items():
            ys = {m: sanityb.outcome_matrix(r, ids, field, agree_only=agree)[0]
                  for m, r in rows.items()}
            tw = reliability.twin_correlations(ys, info)
            block[name] = {
                "per_model": {m: sanityb.model_stats(y) for m, y in ys.items()},
                "summary": stat(tw),
                "pairs": [x | {"group": floor.group_of_035(info[x["a"]].pretrain_seed,
                                                           info[x["b"]].pretrain_seed)}
                          for x in tw["pairs"]],
                "ci_prompts": reliability.bootstrap_prompts(ys, info, a.n_boot, a.seed, stat=stat),
                "jackknife_seeds": floor.jackknife_models(ys, info, same_seed, stat=stat),
            }
        res[s] = block

    res["reference_027"] = check_against_027(f027, res, outcomes, a.n_boot, a.seed)
    prim = res[sanityb.PRIMARY_SUBSET]["primary"]
    sm, ci = prim["summary"], prim["ci_prompts"]
    d = ci["RR2_minus_RD_corrected"]
    res["prediction"] = floor.prediction_035(sm["r_rung2_RR2_corrected"], f027["prediction"]["r_rung2"],
                                             sm["RR2_minus_RD_corrected"], d["lo"], d["hi"])
    res["prediction"]["jackknife_se"] = prim["jackknife_seeds"]["RR2_minus_RD_corrected"]["se"]
    res["prediction"]["reference"] = "027's r(R-D) from runs/027/floor_a0.json:prediction.r_rung2"
    # both rows of the pre-registered branch, whichever applies (2a = R-R2, 2b = mean of R-D, R2-D)
    res["prediction"]["branch_rows"] = {
        row: {"estimate": sm[key], "ci_prompts": ci[key], "jackknife": prim["jackknife_seeds"][key]}
        for row, key in (("2a_same_stack_RR2", "r_rung2_RR2_corrected"),
                         ("2b_diff_stack_mean_RD_R2D", "rung2_diffstack_corrected"))}
    m3 = ci["rung2_mean3_corrected"]
    res["rule_mean3"] = floor.prediction(sm["rung2_mean3_corrected"], m3["lo"], m3["hi"],
                                         sm["r_U_006_corrected"], sm["r_MZ_006_corrected"])
    res["rule_mean3"]["reference"] = ("011's frozen rule on the three-pair mean (D081a); r_U / "
                                      "r_MZ = the A0 cells' own levels, identical to 027's")
    res["rule_mean3"]["pair_means"] = {g: sm[f"r_{g}_corrected"] for g in floor.RUNG2_PAIRS_035}
    res["by_base_pair"] = {s: {o: floor.by_base_pair(res[s][o]["pairs"], info, floor.group_of_035)
                               for o in outcomes} for s in sanityb.SUBSETS}
    res["comparator"] = {"name": "a0", "prompt_id": F11.A0_PROMPT_ID, "stage": F11.A0_STAGE,
                         "manifest": str(F11.A0_MANIFEST.relative_to(ROOT)),
                         "r2_manifest": str(MANIFEST_035.relative_to(ROOT))}
    res.update({"script": "scripts/floor_035.py", "git_commit": commit,
                "command": provenance.command(), "n_boot": a.n_boot, "seed": a.seed,
                "configuration_note": "all 20 SFT on ZeRO-1 (D067a): one configuration",
                "seconds": time.time() - t0})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    p, r = res["prediction"], res["rule_mean3"]
    print(f"{'DRY RUN (stand-in R2) ' if a.stand_in_r2 else ''}"
          f"R-D {sm['r_rung2_RD_corrected']:.4f}  R-R2 {sm['r_rung2_RR2_corrected']:.4f}  "
          f"R2-D {sm['r_rung2_R2D_corrected']:.4f}  mean3 {sm['rung2_mean3_corrected']:.4f} "
          f"[{m3['lo']:.4f}, {m3['hi']:.4f}]; RR2 - RD {p['RR2_minus_RD']:+.4f} "
          f"[{d['lo']:+.4f}, {d['hi']:+.4f}]; prediction holds: {p['holds']}; branch: {p['branch']}; "
          f"rule on mean3 holds: {r['holds']}")


if __name__ == "__main__":
    main()
