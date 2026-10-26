"""Rung 2 of the ladder: the same-seed replicate's twins against released `seed-default`'s, with
the MZ and U levels recomputed by the same code (castor.floor). CPU only; reads judge_v1 labels
(sampled responses; greedy excluded).

    uv run --frozen python scripts/floor_011.py --out /abs/path/runs/011/floor.json \
        [--n-boot 2000] [--seed 0] [--comparator 006|a0]

Models (16): the 12 released-base twins
(<runs>/dd-dclm-1b-<seed>-dpo-seed<k>/006/judge_v1/labels.jsonl) and the replicate's 4
(data/manifests/grid_011.json, <runs>/<model-id>/011/judge_v1/labels.jsonl), checked for one
instrument (pipeval.check_instrument). Each model's SFT DeepSpeed config sha256
is recorded: the replicate's SFT ran on ZeRO-1 and the 006 twins' on ZeRO-3, so the default
comparison mixes configurations.

`--comparator a0`: the 12 released-base models are the stage-A0 cells of runs/008 (same three
bases x post-training seeds 1-4, SFT on ZeRO-1), so all 16 share one configuration (asserted on
every SFT provenance). The frozen rule is then evaluated against r_U and r_MZ of the same run, not
runs/006/signal.json. The output adds the per-base-pair table (castor.floor.by_base_pair) and the
values of runs/011/floor.json for comparison.

Per subset (adv_harmful = primary, plain_harmful, harmful, benign) x outcome (primary =
wg_vulnerable, incl, harm, q3g, agree): pair-group means raw and corrected (castor.floor.summary),
prompt-bootstrap percentile CIs (all 16 models resampled jointly, so contrasts with the 006 levels
carry their covariance), and the leave-one-model-out seed jackknife over the 8 rung-2 models. The
frozen prediction is evaluated on adv_harmful x primary.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from castor import evalset, floor, grid, pipeval, provenance, reliability, sanityb
from castor.judge.common import read_jsonl

OUTCOMES = {"primary": ("wg_vulnerable", False), "incl": ("wg_vulnerable_incl", False),
            "harm": ("wg_harmful", False), "q3g": ("q3g_vulnerable", False),
            "agree": ("wg_vulnerable", True)}
ROOT = Path(__file__).resolve().parent.parent
SIGNAL_006 = ROOT / "runs" / "006" / "signal.json"
FLOOR_011 = ROOT / "runs" / "011" / "floor.json"  # mixed-configuration run
A0_MANIFEST = ROOT / "data" / "manifests" / "grid_008.json"
A0_STAGES = ROOT / "data" / "manifests" / "grid_008_stages.json"
A0_PROMPT_ID, A0_STAGE = "008", "A0"
SFT_ZERO1_SHA256 = "b615b6daa5ea9aa8e67d5b9055093b8581287439f84f73038730af2d6d70ab37"
Models = dict[str, tuple[reliability.ModelInfo, str, Path]]  # id -> (info, prompt id, SFT dir)


def _launch_011():
    return provenance.load_script("launch_011")


def a0_cells() -> list[grid.Cell]:
    """The 12 stage-A0 cells of runs/008: ZeRO-1 reruns of the 006 bases x seeds 1-4."""
    g = grid.load(A0_MANIFEST)
    grid.check_instrument(g, A0_PROMPT_ID)
    stages = grid.load_stages(A0_STAGES.relative_to(ROOT), g)
    cells = [c for c in g.cells() if grid.stage_of(c, stages) == A0_STAGE]
    want = {(c.base_ref, c.k) for c in sanityb.cells()}
    got = {(c.base.ref, c.k) for c in cells}
    if got != want:
        raise ValueError(f"A0 cells are not 006's bases x seeds: {sorted(got ^ want)}")
    return cells


def comparator_models(name: str) -> Models:
    """The 12 released-base models rung 2 is measured against: the 006 twins (ZeRO-3 SFT) or the
    008 A0 reruns of the same cells (ZeRO-1 SFT)."""
    models: Models = {}
    if name == "006":
        for c in sanityb.cells():
            models[c.model_id] = (c.info, sanityb.PROMPT_ID, c.out("sft"))
    elif name == "a0":
        for c in a0_cells():
            models[c.model_id] = (reliability.ModelInfo(sanityb.RECIPE, c.base.pretrain_seed,
                                                        str(c.k)), A0_PROMPT_ID, c.out("sft"))
    else:
        raise ValueError(name)
    assert len(models) == 12, sorted(models)
    return models


def sft_ds_sha(sft_dir: Path) -> str | None:
    p = sft_dir / "provenance.json"
    if not p.exists():
        return None
    oi = json.loads(p.read_text()).get("open_instruct", {})
    return oi.get("deepspeed_config_sha256") or oi.get("deepspeed_config")


def reference_011(info: dict[str, reliability.ModelInfo]) -> dict:
    """runs/011/floor.json read back for comparison: prediction, subset x outcome summaries and
    prompt CIs, and its per-base-pair table (model ids differ from A0's; bases x seeds coincide)."""
    f = json.loads(FLOOR_011.read_text())
    info011 = {m: reliability.ModelInfo(sanityb.RECIPE, i["pretrain_seed"], str(i["posttrain_seed_k"]))
               for m, i in f["models"].items()}
    assert sorted((i.pretrain_seed, i.posttrain_seed) for i in info011.values()) == \
        sorted((i.pretrain_seed, i.posttrain_seed) for i in info.values()), \
        "011's models are not this run's bases x seeds"
    return {"path": str(FLOOR_011.relative_to(ROOT)), "git_commit": f["git_commit"],
            "prediction": f["prediction"],
            "summary": {s: {o: f[s][o]["summary"] for o in OUTCOMES} for s in sanityb.SUBSETS},
            "ci_prompts": {s: {o: f[s][o]["ci_prompts"] for o in OUTCOMES} for s in sanityb.SUBSETS},
            "by_base_pair": {s: {o: floor.by_base_pair(f[s][o]["pairs"], info011) for o in OUTCOMES}
                             for s in sanityb.SUBSETS}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)  # pre-registered
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--comparator", choices=["006", "a0"], default="006",
                    help="the released-base twins: 006's (default, mixed configuration) or "
                         "008's A0 reruns (027, one configuration)")
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (submit.sh snapshots are deleted)"
    commit, t0 = provenance.git_commit(), time.time()
    panel = evalset.load_eval()
    L11 = _launch_011()

    models = comparator_models(a.comparator)
    for c in L11.cells():
        models[c.model_id] = (reliability.ModelInfo(sanityb.RECIPE, floor.REPLICATE, str(c.k)),
                              L11.PROMPT_ID, c.out("sft"))
    info = {m: v[0] for m, v in models.items()}
    rows = {}
    for m, (_, pid, _) in models.items():
        p = sanityb.labels_path(m, pid)
        assert p.exists(), f"missing {p}"
        rows[m] = list(read_jsonl(p))
    eval_sha = json.loads(evalset.EVAL_MANIFEST.read_text())["eval"]["sha256"]
    instrument = pipeval.check_instrument({m: (m, v[1]) for m, v in models.items()}, eval_sha)
    all_ids = [r["id"] for r in panel]
    incomplete = {m: c["missing"] for m in rows
                  if (c := sanityb.outcome_matrix(rows[m], all_ids, "wg_vulnerable")[1])["missing"]}
    assert not incomplete, f"missing label cells: {incomplete}"

    rung2_models = sorted(m for m, i in info.items()
                          if i.pretrain_seed in (floor.DEFAULT, floor.REPLICATE))
    stat = floor.stat_for(info)
    res = {"models": {m: {"pretrain_seed": i.pretrain_seed, "posttrain_seed_k": int(i.posttrain_seed),
                          "prompt_id": models[m][1], "sft_deepspeed_config": sft_ds_sha(models[m][2])}
                      for m, i in info.items()},
           "instrument": instrument,
           "subsets": {s: len(sanityb.subset_ids(panel, s)) for s in sanityb.SUBSETS}}
    for s in sanityb.SUBSETS:
        ids = sanityb.subset_ids(panel, s)
        block = {}
        for name, (field, agree) in OUTCOMES.items():
            ys = {m: sanityb.outcome_matrix(r, ids, field, agree_only=agree)[0]
                  for m, r in rows.items()}
            tw = reliability.twin_correlations(ys, info)
            block[name] = {
                "per_model": {m: sanityb.model_stats(y) for m, y in ys.items()},
                "summary": stat(tw),
                "pairs": [x | {"group": floor.group_of(info[x["a"]].pretrain_seed,
                                                       info[x["b"]].pretrain_seed)}
                          for x in tw["pairs"]],
                "ci_prompts": reliability.bootstrap_prompts(ys, info, a.n_boot, a.seed, stat=stat),
                "jackknife_seeds": floor.jackknife_models(ys, info, rung2_models),
            }
        res[s] = block

    prim = res[sanityb.PRIMARY_SUBSET]["primary"]
    sm = prim["summary"]
    if a.comparator == "006":
        lv006 = json.loads(SIGNAL_006.read_text())[sanityb.PRIMARY_SUBSET]["primary"]["levels"]
        r_u, r_mz = lv006["U"]["r_corrected"], lv006["MZ"]["r_corrected"]
        assert np.isclose(sm["r_U_006_corrected"], r_u, atol=1e-12), (sm["r_U_006_corrected"], r_u)
        assert np.isclose(sm["r_MZ_006_corrected"], r_mz, atol=1e-12), (sm["r_MZ_006_corrected"], r_mz)
        reference = str(SIGNAL_006.relative_to(ROOT))
        configuration_note = "replicate SFT on ZeRO-1 (D067a), 006 on ZeRO-3: mixed"
    else:
        # every SFT provenance must carry the ZeRO-1 sha; the *_006 keys hold the A0 cells
        shas = {m: res["models"][m]["sft_deepspeed_config"] for m in info}
        assert set(shas.values()) == {SFT_ZERO1_SHA256}, f"not one configuration: {shas}"
        r_u, r_mz = sm["r_U_006_corrected"], sm["r_MZ_006_corrected"]
        reference = "same run: the A0 cells' U_006 / MZ_006 groups (keys named after 011's layout)"
        configuration_note = "all 16 SFT on ZeRO-1 (D067a): one configuration"
    ci = prim["ci_prompts"]["r_rung2_corrected"]
    res["prediction"] = floor.prediction(sm["r_rung2_corrected"], ci["lo"], ci["hi"], r_u, r_mz)
    res["prediction"]["reference"] = reference
    if a.comparator != "006":
        res["comparator"] = {"name": a.comparator, "prompt_id": A0_PROMPT_ID, "stage": A0_STAGE,
                             "manifest": str(A0_MANIFEST.relative_to(ROOT))}
        res["by_base_pair"] = {s: {o: floor.by_base_pair(res[s][o]["pairs"], info)
                                   for o in OUTCOMES} for s in sanityb.SUBSETS}
        res["reference_011"] = reference_011(info)
    res.update({"script": "scripts/floor_011.py", "git_commit": commit,
                "command": provenance.command(), "n_boot": a.n_boot, "seed": a.seed,
                "configuration_note": configuration_note,
                "seconds": time.time() - t0})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    p = res["prediction"]
    print(f"rung2 corrected {p['r_rung2']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}] "
          f"(r_U {r_u:.4f}, r_MZ {r_mz:.4f}); prediction holds: {p['holds']}")


if __name__ == "__main__":
    main()
