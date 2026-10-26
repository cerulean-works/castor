"""Manifest of twins for the ladder_007 readout on new models, resolved through provenance.

    uv run --frozen python scripts/manifest_021.py --out data/manifests/mech_021.json

Every model is a post-trained (SFT -> DPO) DataDecide twin whose DPO checkpoint, generation and
judge_v1 labels exist; the entry is written only after its chain of provenance.json files agrees
(DPO parent = the SFT of the same cell, SFT base = the converted base, generation model = the DPO
checkpoint, full panel, labels row count). Format: mech_007.json's fields plus `size`,
`source_prompt`, `set` membership and (factorial set) `ancestry`.

Sets (the pair sets the ladder analyses):
  A      stage A of 008: post-training seed 1 on the 72 non-main bases (24 recipes x 3 seeds):
         the PRIMARY set, 72 pure different-seed (U) pairs, never the ladder_007 twins
  B      the 8 stage-B recipes with seeds 1 and 2 (48 models): MZ pairs for the ladder gap and
         the harm control
  A0     the 12 main-recipe reruns (3 bases x seeds 1-4): the ladder_007 bases re-run; not
         independent of the exploratory sample, reported separately
  F      the factorial (009 + 011 + A0): D = A0 default, A = A0 large-aux-2, X = init-default-
         data-aux2, Y = init-aux2-data-default, R = replicate-i6198-d6198; levels by
         castor.factorial.level_fifth (both / replicate / init / order / neither)
  S300M  the 300M twins of runs/016 (16 layers; the fixed sites exist)
  S150M  the 150M twins of runs/016 (12 layers: layer 13 does not exist; site map only)
"""

import argparse
import json
from pathlib import Path

from castor import factorial, grid, ladder, provenance
from castor.sanityb import K

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path("/work/harrychang08/castor/runs")
TWINS = ROOT / "data/manifests/castor_twins_1B.json"
N_PANEL = 6226

# ancestry of the released factorial bases (castor.factorial); castor's own runs are in TWINS
ANCESTRY = {("dclm-baseline", "default"): ("D", 6198, 6198),
            ("dclm-baseline", "large-aux-2"): ("A", 4, 4)}


def read(p: Path) -> dict:
    return json.loads(p.read_text())


def resolve(model_id: str, dpo_dir: Path, base_ref: str, base_dir: Path, recipe: str, size: str,
            pretrain_seed: str, k: int, prompt_id: str) -> dict:
    """One entry, after checking the DPO -> SFT -> base chain and the generation + labels."""
    ckpt = dpo_dir / "model"
    prov = read(dpo_dir / "provenance.json")
    assert prov["stage"] == "dpo" and prov["post_training_seed_k"] == k, (model_id, prov["stage"])
    parent = prov["parent"]
    assert parent["stage"] == "sft" and parent["oi_seed"] == k, (model_id, "SFT parent")
    assert Path(parent["base_model"]).resolve() == base_dir.resolve(), (
        model_id, parent["base_model"], str(base_dir))
    sft_prov = read(Path(parent["run_provenance"]))
    assert sft_prov["stage"] == "sft" and sft_prov["base_model"] == parent["base_model"]
    resolved = ckpt.resolve()
    assert resolved.is_dir() and (resolved / "config.json").exists(), (model_id, "no checkpoint")
    gen = RUNS / model_id / prompt_id
    gprov = read(gen / "provenance.json")
    assert gprov["model_id"] == model_id and Path(gprov["model"]).resolve() == resolved, (
        model_id, gprov["model"])
    assert gprov["limit"] is None and gprov["n_prompts"] == N_PANEL, (model_id, "partial panel")
    assert gprov["sampling"]["k"] == K
    labels = gen / "judge_v1" / "labels.jsonl"
    meta = read(gen / "judge_v1" / "labels.meta.json")
    assert meta["judge_version"] == "judge_v1" and meta["n"] == N_PANEL * (K + 1), (
        model_id, meta["n"])
    assert labels.exists()
    return {"model_id": model_id, "checkpoint": str(ckpt), "checkpoint_resolved": str(resolved),
            "base": base_ref, "base_converted": str(base_dir), "recipe": recipe, "size": size,
            "pretrain_seed": pretrain_seed, "posttrain_seed": str(k),
            "source_prompt": prompt_id, "generations": str(gen), "labels": str(labels),
            "provenance": {"dpo_commit": prov["git_commit"], "dpo_job": prov.get("slurm_job"),
                           "sft_commit": sft_prov["git_commit"],
                           "generation_commit": gprov["git_commit"],
                           "labels_commit": meta["git_commit"], "labels_n": meta["n"]}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/manifests/mech_021.json")
    a = ap.parse_args()
    commit = provenance.git_commit()
    models: dict[str, dict] = {}
    sets: dict[str, list[str]] = {s: [] for s in ("A", "B", "A0", "F", "S300M", "S150M")}
    ancestry: dict[str, dict] = {}

    def add(entry: dict, *into: str) -> None:
        assert entry["model_id"] not in models, entry["model_id"]
        models[entry["model_id"]] = entry
        for s in into:
            sets[s].append(entry["model_id"])

    # grid 008: stages A, B, A0 (stage C is not used)
    g8 = grid.load(ROOT / "data/manifests/grid_008.json")
    stages = grid.load_stages("data/manifests/grid_008_stages.json", g8)
    b_repos = next(s.repos for s in stages if s.name == "B")
    for c in g8.cells():
        st = grid.stage_of(c, stages)
        if st == "C":
            continue
        e = resolve(c.model_id, c.out("dpo"), c.base.ref, c.base.base_dir, c.base.recipe,
                    c.base.size, c.base.pretrain_seed, c.k, "008")
        if st == "A":
            add(e, "A", *(["B"] if c.base.repo in b_repos else []))
        elif st == "B":
            add(e, "B")
        else:
            assert st == "A0"
            key = (c.base.recipe, c.base.pretrain_seed)
            if key in ANCESTRY:
                tag, i, d = ANCESTRY[key]
                ancestry[c.model_id] = {"base_tag": tag, "init_seed": i, "data_seed": d}
                add(e, "A0", "F")
            else:
                add(e, "A0")
    # 009 (X, Y) and 011 (R): castor's own pretraining runs, seeds from the twins' registration
    twins = {t["revision"]: t for t in read(TWINS)["entries"]}
    tags = {"init-default-data-aux2": "X", "init-aux2-data-default": "Y",
            "replicate-i6198-d6198": factorial.REPLICATE}
    for pid in ("009", "011"):
        gg = grid.load(ROOT / f"data/manifests/grid_{pid}.json")
        for c in gg.cells():
            t = twins[c.base.revision]
            assert Path(t["local_path"]).resolve() == c.base.base_dir.resolve()
            e = resolve(c.model_id, c.out("dpo"), c.base.ref, c.base.base_dir, c.base.recipe,
                        c.base.size, c.base.pretrain_seed, c.k, pid)
            ancestry[c.model_id] = {"base_tag": tags[c.base.revision],
                                    "init_seed": t["init_seed"], "data_seed": t["data_seed"]}
            add(e, "F")
    # 016: the 150M and 300M main-recipe twins
    for c in ladder.cells():
        e = resolve(c.model_id, c.out("dpo"), c.base_ref, c.base_dir, ladder.RECIPE, c.size,
                    c.pretrain_seed, c.k, "016")
        add(e, f"S{c.size}")

    assert [len(sets[s]) for s in ("A", "B", "A0", "F", "S300M", "S150M")] == [72, 48, 12, 20,
                                                                                12, 12]
    out = {"prompt": "021", "source_prompts": ["008", "009", "011", "016"],
           "models": [models[m] for m in sorted(models)],
           "sets": {
               "A": {"models": sorted(sets["A"]), "levels": "pair_level", "role": "primary",
                     "why": "008 stage A: post-training seed 1 on the 72 non-main bases; 72 "
                            "pure different-seed pairs, 3 per recipe (D053d)"},
               "B": {"models": sorted(sets["B"]), "levels": "pair_level", "role": "secondary",
                     "why": "the 8 stage-B recipes with seeds 1 and 2: MZ pairs for the ladder "
                            "gap and the harm control"},
               "A0": {"models": sorted(sets["A0"]), "levels": "pair_level", "role": "secondary",
                      "why": "the 12 main-recipe reruns (008 A0): 007's bases in the second "
                             "set-up; not independent of the exploratory sample"},
               "F": {"models": sorted(sets["F"]), "levels": "factorial", "role": "secondary",
                     "ancestry": {m: ancestry[m] for m in sorted(sets["F"])},
                     "why": "009's X/Y and 011's R with A0's D/A: pure U = neither (X-Y, D-A); "
                            "init-only, order-only and same-seed (replicate) pairs reported "
                            "separately, never pooled (prompt 021)"},
               "S300M": {"models": sorted(sets["S300M"]), "levels": "pair_level",
                         "role": "secondary", "why": "016's 300M twins (16 layers)"},
               "S150M": {"models": sorted(sets["S150M"]), "levels": "pair_level",
                         "role": "secondary", "site_map_only": True,
                         "why": "016's 150M twins: 12 layers, layer 13 does not exist"}},
           "git_commit": commit, "command": provenance.command()}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(models)} models -> {a.out}; sets " + ", ".join(
        f"{s} {len(v['models'])}" for s, v in out["sets"].items()))


if __name__ == "__main__":
    main()
