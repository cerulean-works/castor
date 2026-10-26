"""The init × data-order factorial table: ceiling-corrected per-prompt twin correlations for the
four pair types (castor.factorial), with prompt-bootstrap CIs.

    uv run --frozen python scripts/factorial_009.py --manifest data/manifests/grid_009.json \
        --out /abs/path/runs/009/factorial.json [--n-boot 1000]

Models (16): 006's post-trained released DCLM 1B twins `seed-default` (D) and `seed-large-aux-2`
(A), seeds k = 1..4 (labels under runs/<id>/006/), and the 009 grid's twins of our two stage-2
runs (X, Y), seeds 1..4 (labels under runs/<id>/009/). The init and data seeds of X and Y come
from data/manifests/castor_twins_1B.json and are checked against each run's OLMo config
(`seed` = init, `data.seed` = order) and its revision name; D and A are 6198/6198 and 4/4
(step-0 hash match). Every label file must come from one judge_v1 instrument on the current eval
panel (castor.pipeval.check_instrument) and cover it fully, and all 16 models must share one SFT
deepspeed config (no mixed-configuration comparisons), recorded per model.

Blocks, as signal_006: `adv_harmful` (primary subset) × {primary, incl, harm, q3g, agree}; every
other subset × primary. Each: levels (both / init / order / neither: n pairs, raw and corrected r),
per base pair, per model reliability and stats, prompt-bootstrap CIs of castor.factorial.summary
and leave-one-base-out jackknife SEs (castor.factorial.jackknife_bases).
Sampled responses only (greedy excluded), as 006.

Fifth base (optional, default output unchanged): `--extra-grid data/manifests/grid_011.json
--allow-mixed-sft` adds 011's replicate R (init 6198 / data 6198, our hardware; labels under
runs/<id>/011/) and writes `fifth_base`: per subset × outcome as above (levels, per base pair,
reliability and pairs over the 20 models with `level_fifth`, R's per-model stats; no jackknife),
the R / X / Y within-hardware contrasts (castor.factorial.summary_fifth) and their
prompt-bootstrap CIs. R's pairs sit under `replicate` and `R_*` levels only, so the four 4-base
levels are unchanged. R's twins ran SFT on ZeRO-1, the 16 on ZeRO-3 (the label names the stages
read from the recorded configs), so these rows are mixed configuration (descriptive, appendix);
the flag is required for that and is recorded. Every fifth-base check runs before the 16-model
blocks, which are untouched.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from castor import evalset, factorial, grid, pipeval, posttrain, provenance, sanityb

TWINS_MANIFEST = provenance.ROOT / "data/manifests/castor_twins_1B.json"
RELEASED = {"D": ("default", 6198, 6198), "A": ("large-aux-2", 4, 4)}
SEED_NAMES = {"default": 6198, "aux2": 4}  # revision init-<i>-data-<d>
OUTCOMES = {"primary": ("wg_vulnerable", False), "incl": ("wg_vulnerable_incl", False),
            "harm": ("wg_harmful", False), "q3g": ("q3g_vulnerable", False),
            "agree": ("wg_vulnerable", True)}


def check_seeds(e: dict) -> tuple[int, int]:
    """(init, data) seeds of a twins-manifest entry, after checking them against the revision
    name and the OLMo config of the checkpoint it was converted from."""
    i, d = e["init_seed"], e["data_seed"]
    if e["revision"].startswith("replicate-"):  # replicate-i<i>-d<d> (011)
        _, ni, nd = e["revision"].split("-")
        named = (int(ni.removeprefix("i")), int(nd.removeprefix("d")))
    else:
        _, ni, _, nd = e["revision"].split("-")  # init-<i>-data-<d>
        named = (SEED_NAMES[ni], SEED_NAMES[nd])
    assert named == (i, d), f"revision name vs manifest seeds: {e}"
    cfg = yaml.safe_load((Path(e["source_checkpoint"]) / "config.yaml").read_text())
    # OLMo's data order falls back to `seed` when `data.seed` is null (olmo/data/__init__.py:108
    # at the pinned commit); the replicate leaves it unset
    data_seed = cfg["data"].get("seed")
    data_seed = cfg["seed"] if data_seed is None else data_seed
    assert (cfg["seed"], data_seed) == (i, d), \
        f"{e['revision']}: OLMo config seeds {cfg['seed']}, {data_seed} (effective) vs manifest"
    return i, d


def sft_ds_config(sft_dir: Path) -> str:
    """The SFT deepspeed config a model was trained with (as floor_011 records it)."""
    oi = json.loads((sft_dir / "provenance.json").read_text())["open_instruct"]
    return oi.get("deepspeed_config_sha256") or oi["deepspeed_config"]


def ds_describe(v: str) -> str:
    """A recorded SFT deepspeed config (sha256 or open-instruct path) named by its ZeRO stage,
    read from the file it resolves to; the raw value when no candidate file matches."""
    if Path(v).suffix:
        cands = [posttrain.ds_config_path("sft", v), posttrain.OI_DIR / v]
    else:
        cands = [*(posttrain.OI_DIR / "configs/ds_configs").glob("*"),
                 *(posttrain.REPO_ROOT / "configs/posttrain/ds").glob("*")]
        cands = [f for f in cands if f.suffix in (".json", ".conf") and f.is_file()
                 and posttrain.sha256_file(f) == v]
    for f in cands:
        if f.is_file():
            stage = json.loads(f.read_text()).get("zero_optimization", {}).get("stage")
            return f"ZeRO-{stage} ({f.name})"
    return v


def models(manifest: Path, twins_manifest: Path = TWINS_MANIFEST
           ) -> tuple[dict[str, factorial.Twin], dict[str, tuple[str, str]], dict, dict]:
    """model id -> Twin, model id -> (id, prompt id of its labels), model id -> base ref,
    model id -> its SFT output dir."""
    twins, runs, refs, sft = {}, {}, {}, {}
    for c in sanityb.cells():
        for letter, (seed, i, d) in RELEASED.items():
            if c.pretrain_seed == seed:
                twins[c.model_id] = factorial.Twin(letter, i, d, c.k)
                runs[c.model_id] = (c.model_id, sanityb.PROMPT_ID)
                refs[c.model_id] = c.base_ref
                sft[c.model_id] = c.out("sft")
    reg = {f"{e['repo']}@{e['revision']}": e
           for e in json.loads(twins_manifest.read_text())["entries"]}
    g = grid.load(manifest)
    for c in g.cells():
        i, d = check_seeds(reg[c.base.ref])
        # X = init default, data aux-2; Y = init aux-2, data default (castor.factorial)
        letter = {(6198, 4): "X", (4, 6198): "Y"}[(i, d)]
        twins[c.model_id] = factorial.Twin(letter, i, d, c.k)
        runs[c.model_id] = (c.model_id, g.prompt_id)
        refs[c.model_id] = c.base.ref
        sft[c.model_id] = c.out("sft")
    assert sorted({t.base for t in twins.values()}) == ["A", "D", "X", "Y"], twins
    assert len(twins) == 16, len(twins)
    return twins, runs, refs, sft


def extra_models(manifest: Path, twins_manifest: Path = TWINS_MANIFEST
                 ) -> tuple[dict[str, factorial.Twin], dict[str, tuple[str, str]], dict, dict]:
    """As `models`, for the fifth base: a grid manifest whose one base is a same-seed replicate
    of D (letter R)."""
    reg = {f"{e['repo']}@{e['revision']}": e
           for e in json.loads(twins_manifest.read_text())["entries"]}
    g = grid.load(manifest)
    twins, runs, refs, sft = {}, {}, {}, {}
    for c in g.cells():
        i, d = check_seeds(reg[c.base.ref])
        assert (i, d) == RELEASED["D"][1:], f"fifth base is not a replicate of D: {c.base.ref}"
        twins[c.model_id] = factorial.Twin(factorial.REPLICATE, i, d, c.k)
        runs[c.model_id] = (c.model_id, g.prompt_id)
        refs[c.model_id] = c.base.ref
        sft[c.model_id] = c.out("sft")
    assert len(twins) == 4 and len({r for r in refs.values()}) == 1, refs
    return twins, runs, refs, sft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-grid", type=Path, help="fifth base (D071a): grid_011.json")
    ap.add_argument("--allow-mixed-sft", action="store_true",
                    help="admit a fifth base whose SFT deepspeed config differs (recorded)")
    a = ap.parse_args()
    if a.allow_mixed_sft and not a.extra_grid:
        ap.error("--allow-mixed-sft only applies with --extra-grid")
    assert a.out.is_absolute(), "--out must be absolute (submit.sh snapshots are deleted)"
    commit, t0 = provenance.git_commit(), time.time()
    twins, runs, refs, sft = models(a.manifest)
    ds = {m: sft_ds_config(d) for m, d in sft.items()}
    assert len(set(ds.values())) == 1, f"mixed SFT deepspeed configs (D067a): {ds}"
    panel = evalset.load_eval()
    eval_sha = json.loads(evalset.EVAL_MANIFEST.read_text())["eval"]["sha256"]
    instrument = pipeval.check_instrument(runs, eval_sha)
    rows = {m: pipeval.load_rows(run) for m, run in runs.items()}
    all_ids = [r["id"] for r in panel]
    missing = {m: c["missing"] for m in rows
               if (c := sanityb.outcome_matrix(rows[m], all_ids, "wg_vulnerable")[1])["missing"]}
    assert not missing, f"missing label cells: {missing}"
    # every fifth-base check runs before the 16-model blocks, so a refusal costs no compute
    fifth = (prepare_fifth(a, panel, eval_sha, instrument, twins, rows, ds) if a.extra_grid
             else None)

    res = {"models": {m: {"base": refs[m], "letter": t.base, "init_seed": t.init_seed,
                          "data_seed": t.data_seed, "posttrain_seed_k": t.k,
                          "labels": f"{runs[m][0]}/{runs[m][1]}",
                          "sft_deepspeed_config": ds[m]} for m, t in twins.items()},
           "instrument": instrument,
           "subsets": {s: len(sanityb.subset_ids(panel, s)) for s in sanityb.SUBSETS}}
    for s in sanityb.SUBSETS:
        ids = sanityb.subset_ids(panel, s)
        block = {}
        for name, (field, agree) in OUTCOMES.items():
            if s != sanityb.PRIMARY_SUBSET and name != "primary":
                continue
            ys = {m: sanityb.outcome_matrix(r, ids, field, agree_only=agree)[0]
                  for m, r in rows.items()}
            fa = factorial.analyze(ys, twins)
            block[name] = {"levels": fa["levels"], "by_base_pair": fa["by_base_pair"],
                           "reliability": fa["reliability"], "pairs": fa["pairs"],
                           "per_model": {m: sanityb.model_stats(y) for m, y in ys.items()},
                           "ci_prompts": factorial.bootstrap_prompts(ys, twins, a.n_boot, a.seed),
                           "jackknife_bases": factorial.jackknife_bases(ys, twins)}
        res[s] = block
    if fifth:
        res["fifth_base"] = fifth_base(a, panel, twins, rows, fifth)
    res |= {"script": "scripts/factorial_009.py", "git_commit": commit,
            "extra_grid": str(a.extra_grid) if a.extra_grid else None,
            "allow_mixed_sft": a.allow_mixed_sft,
            "command": provenance.command(), "n_boot": a.n_boot, "seed": a.seed,
            "elapsed_s": round(time.time() - t0, 1)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    prim = res[sanityb.PRIMARY_SUBSET]["primary"]
    for lvl in factorial.LEVELS:
        ci = prim["ci_prompts"][f"r_{lvl}_corrected"]
        print(f"{lvl:8s} n={prim['levels'][lvl]['n_pairs']:3d} r={prim['levels'][lvl]['r']:.3f} "
              f"corr={ci['estimate']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]")



def prepare_fifth(a, panel, eval_sha, instrument, twins, rows, ds) -> dict:
    """The fifth base's models, checked (seeds, SFT config gate, instrument, full labels) and
    its label rows loaded, before any compute."""
    tw5, runs5, refs5, sft5 = extra_models(a.extra_grid)
    assert not set(tw5) & set(twins), f"fifth-base model ids collide: {set(tw5) & set(twins)}"
    ds5 = {m: sft_ds_config(d) for m, d in sft5.items()}
    assert len(set(ds5.values())) == 1, f"fifth base: mixed SFT configs among its twins: {ds5}"
    mixed = set(ds5.values()) != set(ds.values())
    assert not mixed or a.allow_mixed_sft, \
        f"fifth base SFT config {set(ds5.values())} != {set(ds.values())}: pass --allow-mixed-sft"
    # the 16 already match `instrument`, so R's four matching it is the 20-model check
    assert pipeval.check_instrument(runs5, eval_sha) == instrument, \
        "fifth base labels come from a different instrument than the 16 models'"
    rows5 = {m: pipeval.load_rows(run) for m, run in runs5.items()}
    all_ids = [r["id"] for r in panel]
    missing = {m: c["missing"] for m in rows5
               if (c := sanityb.outcome_matrix(rows5[m], all_ids, "wg_vulnerable")[1])["missing"]}
    assert not missing, f"missing label cells: {missing}"
    return {"twins": tw5, "runs": runs5, "refs": refs5, "ds": ds5, "rows": rows5,
            "mixed": mixed, "ds16": ds}


def fifth_base(a, panel, twins, rows, f) -> dict:
    tw, rows = twins | f["twins"], rows | f["rows"]
    ds5, ds = f["ds"], f["ds16"]
    r_cfg = ", ".join(ds_describe(v) for v in sorted(set(ds5.values())))
    daxy_cfg = ", ".join(ds_describe(v) for v in sorted(set(ds.values())))
    out = {"label": (f"mixed configuration (R: {r_cfg} SFT; D, A, X, Y: {daxy_cfg}) - "
                     "descriptive, appendix (D067a, D068a(iv))") if f["mixed"]
           else f"single configuration ({daxy_cfg})",
           "allow_mixed_sft": a.allow_mixed_sft, "mixed_sft": f["mixed"],
           "sft_deepspeed_configs": {"R": sorted(set(ds5.values())),
                                     "DAXY": sorted(set(ds.values()))},
           "extra_grid": str(a.extra_grid),
           "models": {m: {"base": f["refs"][m], "letter": t.base, "init_seed": t.init_seed,
                          "data_seed": t.data_seed, "posttrain_seed_k": t.k,
                          "labels": f"{f['runs'][m][0]}/{f['runs'][m][1]}",
                          "sft_deepspeed_config": ds5[m]} for m, t in f["twins"].items()}}
    for s in sanityb.SUBSETS:
        ids = sanityb.subset_ids(panel, s)
        block = {}
        for name, (field, agree) in OUTCOMES.items():
            if s != sanityb.PRIMARY_SUBSET and name != "primary":
                continue
            ys = {m: sanityb.outcome_matrix(r, ids, field, agree_only=agree)[0]
                  for m, r in rows.items()}
            fa = factorial.analyze_fifth(ys, tw)
            # LEVELS are the 4-base table's; R's pairs sit under replicate and R_* only
            block[name] = {"levels": fa["levels"], "by_base_pair": fa["by_base_pair"],
                           "reliability": fa["reliability"], "pairs": fa["pairs"],
                           "per_model": {m: sanityb.model_stats(ys[m]) for m in f["twins"]},
                           "summary": factorial.summary_fifth(fa),
                           "ci_prompts": factorial.bootstrap_prompts_fifth(ys, tw, a.n_boot,
                                                                           a.seed)}
        out[s] = block
    return out


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    main()
