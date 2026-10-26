"""Harm-direction control per data recipe, from the activations stored by runs/021. CPU work, one
dev slot per part (the panel slices need more than 20 GiB of RAM).

    uv run --frozen python scripts/harm_046.py --manifest data/manifests/mech_021.json \
        --acts-root /work/harrychang08/castor/activations/021 \
        --cells B:c4,B:dolma1_7 --out <absolute>/part_<name>.json [--n-boot-sec 200]
    uv run --frozen python scripts/harm_046.py --list-cells
    uv run --frozen python scripts/harm_046.py --merge <part>.json ... --out <abs>/runs/046/harm_per_recipe.json

Per model the instrument is that of scripts/ladder_007.py (Twin, readout, the 200 half-splits
with seed 2027, per-prompt P = judge_v1's `wg_vulnerable`, the adversarial-harmful
subset); the pair levels are MZ / U within a recipe (castor.reliability.pair_level). Per
cell (castor.harmcontrol.cells: one recipe; B / A / A0 / S300M models):

  layers.<13|11>   refusal readout at the layer's site ((13, −1) or (11, −2)) and the harm readout
                   (last instruction token, stored position −8, same layer) on the same prompts:
                   corrected r_MZ / r_U per readout, the gaps where MZ pairs exist, refusal − harm
                   differences, prompt-bootstrap CIs (castor.mechladder.two_readouts over
                   ladder_contrast's two ladders), per-pair rows
  harm_by_layer    the harm readout's corrected levels at every layer (point estimates)
  diagnosis.a      direction reliability (mean Spearman–Brown half-direction cosine) and the
                   unaligned cosines per pair level (MZ meaningful; U reported, not meaningful)
                   for the harm direction at both layers and the refusal direction at both sites
  diagnosis.b      leave-one-source-out: the harm direction re-estimated without one harmful
                   training source (advbench / malicious_instruct / tdc2023) and, as the control,
                   without the same number of random harmful prompts (3 seeds); levels per variant
  diagnosis.c      how much of each projection is the last instruction token: eta^2 of the harm (and
                   refusal) projection on the panel prompt's last-character class and on its last
                   token id, the corrected unrelated-pair level of the within-class part of each
                   projection, and the correlation of the harm projection with the activation norm
                   at (layer, −8)
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from castor import directions as Dn
from castor import evalset, provenance, reliability, sanityb
from castor import harmcontrol as HC
from castor import mechladder as ML

LD = provenance.load_script("ladder_007")
N_SPLITS, SPLIT_SEED = LD.N_SPLITS, LD.SPLIT_SEED
J_HARM = 0  # stored position −8, the last instruction token
SITES = {13: {"layer": 13, "position": -1}, 11: {"layer": 11, "position": -2}}
ROOT = Path(__file__).resolve().parent.parent
EXTRACTION = ROOT / "data" / "directions" / "extraction_v1.jsonl"
RANDOM_SEEDS = (1, 2, 3)


def stored_of(position: int) -> int:
    return LD.stored(position + Dn.N_POST)


def level_point(reads: dict, twins: dict, idx: np.ndarray, info: dict) -> dict:
    """Corrected levels of one readout on the subset (point estimate)."""
    return ML.ladder_summary(ML.ladder(N_SPLITS)(LD.packed(reads, twins, idx), info))


def readout_from(tw, layer: int, j: int, h_idx: np.ndarray, splits_sub) -> dict:
    """ladder_007's readout with the harmful training set restricted to rows h_idx."""
    h, b = tw.train_at(layer, j)
    h = h[h_idx]
    full = Dn.unit(Dn.mean_diff(h, b))
    d1, d2 = Dn.half_directions(h, b, splits_sub)
    x = tw.panel(j)[:, layer].double().numpy()
    return {"full": Dn.project(x, full), "h1": Dn.project(x, d1), "h2": Dn.project(x, d2),
            "dir": full, "d1": d1, "d2": d2}


def cosines(reads: dict, info: dict) -> dict:
    return ML.direction_cosines({m: r["dir"] for m, r in reads.items()},
                                {m: r["d1"] for m, r in reads.items()},
                                {m: r["d2"] for m, r in reads.items()}, info)


def within_level(reads: dict, twins: dict, idx: np.ndarray, info: dict, groups) -> dict:
    """Corrected levels of the within-group part of a readout (group means removed from the
    full and the half-direction projections, prompt by prompt)."""
    ys = {}
    for m, r in reads.items():
        a = ML.pack(r["full"][idx], r["h1"][idx], r["h2"][idx])
        ys[m] = np.concatenate([HC.demean_within(a, groups),
                                np.asarray(twins[m].y["wg_vulnerable"][idx], float)], axis=1)
    return ML.ladder_summary(ML.ladder(N_SPLITS)(ys, info))


def token_groups(panel: list[dict], ex_harm: list[dict], ex_safe: list[dict], tok) -> dict:
    """Last-character classes and last-token ids of the panel and the extraction prompts."""
    panel_char = [HC.last_char_class(r["prompt"]) for r in panel]
    panel_tok = []
    suffix = None
    for r in panel:
        ids = evalset.prompt_ids(tok, r["prompt"])
        tail = tuple(ids[-Dn.N_POST:])
        suffix = suffix or tail
        assert tail == suffix, "panel prompt without the common post-instruction tail"
        panel_tok.append(int(ids[-Dn.N_STORED]))
    ex = {}
    for name, rows in (("harmful_train", ex_harm), ("harmless_train", ex_safe)):
        chars = [HC.last_char_class(r["prompt"]) for r in rows]
        toks = [int(evalset.prompt_ids(tok, r["prompt"])[-Dn.N_STORED]) for r in rows]
        ex[name] = {"n": len(rows), "last_char_shares": HC.shares(chars),
                    "n_distinct_last_tokens": len(set(toks)),
                    "top_last_tokens": {str(t): c for t, c in
                                        sorted(((t, toks.count(t)) for t in set(toks)),
                                               key=lambda x: -x[1])[:5]}}
    return {"panel_last_char_shares": HC.shares(panel_char),
            "panel_n_distinct_last_tokens": len(set(panel_tok)),
            "extraction": ex, "_panel_char": panel_char, "_panel_tok": panel_tok}


def analyse_cell(cell: dict, twins: dict, adv: np.ndarray, splits, n_boot: int,
                 ex_sources: list[str], groups: dict) -> dict:
    info = {m: t.info for m, t in twins.items()}
    counts = ML.levels_of(list(twins), info)
    n_layer = next(iter(twins.values())).scores["n_layer"]
    out = {"set": cell["set"], "recipe": cell["recipe"], "size": cell["size"],
           "models": sorted(twins), "n_models": len(twins),
           "n_MZ": counts.get("MZ", 0), "n_U": counts.get("U", 0), "n_layer": n_layer}
    g_char = np.array(groups["_panel_char"])[adv]
    g_tok = np.array(groups["_panel_tok"])[adv]

    # the refusal readouts at the two sites
    ref = {la: {m: LD.readout(t, la, stored_of(SITES[la]["position"]), splits)
                for m, t in twins.items()} for la in HC.LAYERS}
    for la in HC.LAYERS:
        for t in twins.values():
            t.drop(stored_of(SITES[la]["position"]))

    # the harm readout at every layer (point levels), kept whole at the two layers
    harm, by_layer = {}, {}
    for layer in range(n_layer):
        if any(not np.any(Dn.mean_diff(*t.train_at(layer, J_HARM))) for t in twins.values()):
            by_layer[layer] = {"undefined": "difference of means is exactly zero"}
            continue
        r_h = {m: LD.readout(t, layer, J_HARM, splits) for m, t in twins.items()}
        pt = level_point(r_h, twins, adv, info)
        by_layer[layer] = {k: pt.get(k, float("nan")) for k in
                           ("r_MZ_corrected", "r_U_corrected", "MZ_minus_U_corrected",
                            "r_MZ", "r_U", "mean_reliability")}
        if layer in HC.LAYERS:
            harm[layer] = r_h
    out["harm_by_layer"] = by_layer
    norms = {la: {m: np.linalg.norm(t.panel(J_HARM)[:, la].double().numpy(), axis=1)[adv]
                  for m, t in twins.items()} for la in HC.LAYERS}

    an, st = ML.two_readouts(N_SPLITS, ML.ladder(N_SPLITS), ML.ladder_summary,
                             names=("refusal", "harm"))
    out["layers"] = {}
    for la in HC.LAYERS:
        ys2 = {m: ML.pack_two((ref[la][m]["full"][adv], ref[la][m]["h1"][adv],
                               ref[la][m]["h2"][adv]),
                              (harm[la][m]["full"][adv], harm[la][m]["h1"][adv],
                               harm[la][m]["h2"][adv]),
                              twins[m].y["wg_vulnerable"][adv]) for m in twins}
        res = an(ys2, info)
        point = st(res)
        entry = {"site": SITES[la], "harm_position": J_HARM - Dn.N_STORED,
                 "point": point,
                 "ci_prompts": reliability.bootstrap_prompts(ys2, info, n_boot, stat=st,
                                                             analyze=an),
                 "n_draws": n_boot,
                 "reliability": {k: res[k]["reliability"] for k in ("refusal", "harm")},
                 "pairs": {k: res[k]["pairs"] for k in ("refusal", "harm")}}
        # diagnosis (a)
        entry["diagnosis_a"] = {"harm": cosines(harm[la], info), "refusal": cosines(ref[la], info)}
        # diagnosis (b): leave-one-source-out and the size-matched random control
        loo = {}
        for src in HC.SOURCES:
            keep = HC.loo_keep(ex_sources, src)
            sp = Dn.half_splits(len(keep), 256, N_SPLITS, SPLIT_SEED)
            r_s = {m: readout_from(t, la, J_HARM, keep, sp) for m, t in twins.items()}
            pt = level_point(r_s, twins, adv, info)
            loo[src] = {"n_dropped": len(ex_sources) - len(keep), "n_kept": len(keep),
                        "harm_r_MZ_corrected": pt.get("r_MZ_corrected", float("nan")),
                        "harm_r_U_corrected": pt["r_U_corrected"],
                        "harm_MZ_minus_U_corrected": pt.get("MZ_minus_U_corrected", float("nan")),
                        "mean_reliability": pt["mean_reliability"],
                        "cos_to_full_direction": {m: float(np.dot(r_s[m]["dir"],
                                                                  harm[la][m]["dir"]))
                                                  for m in twins}}
            ctrl = []
            for seed in RANDOM_SEEDS:
                keep_r = HC.matched_random_keep(len(ex_sources), loo[src]["n_dropped"], seed)
                sp_r = Dn.half_splits(len(keep_r), 256, N_SPLITS, SPLIT_SEED)
                r_r = {m: readout_from(t, la, J_HARM, keep_r, sp_r) for m, t in twins.items()}
                pr = level_point(r_r, twins, adv, info)
                ctrl.append({"seed": seed, "harm_r_U_corrected": pr["r_U_corrected"],
                             "harm_r_MZ_corrected": pr.get("r_MZ_corrected", float("nan"))})
            loo[src]["random_drop_control"] = {
                "n_dropped": loo[src]["n_dropped"], "seeds": ctrl,
                "harm_r_U_corrected_mean": float(np.mean([c["harm_r_U_corrected"]
                                                          for c in ctrl]))}
        entry["diagnosis_b"] = loo
        # diagnosis (c): the last instruction token
        per_model = {}
        for m in twins:
            hp, rp = harm[la][m]["full"][adv], ref[la][m]["full"][adv]
            per_model[m] = {
                "eta2_harm_last_char": HC.eta_squared(hp, g_char),
                "eta2_harm_last_token": HC.eta_squared(hp, g_tok),
                "eta2_refusal_last_char": HC.eta_squared(rp, g_char),
                "eta2_refusal_last_token": HC.eta_squared(rp, g_tok),
                "r_harm_norm": reliability.corr(hp, norms[la][m]),
                "eta2_norm_last_char": HC.eta_squared(norms[la][m], g_char)}
        c = {"per_model": per_model,
             "mean": {k: float(np.nanmean([v[k] for v in per_model.values()]))
                      for k in next(iter(per_model.values()))}}
        for name, reads in (("harm", harm[la]), ("refusal", ref[la])):
            for gname, g in (("last_char", g_char), ("last_token", g_tok)):
                wl = within_level(reads, twins, adv, info, g)
                c[f"{name}_within_{gname}"] = {k: wl.get(k, float("nan")) for k in
                                               ("r_MZ_corrected", "r_U_corrected",
                                                "MZ_minus_U_corrected", "mean_reliability")}
        entry["diagnosis_c"] = c
        out["layers"][la] = entry
        print(f"  layer {la}: refusal r_U {point['refusal_r_U_corrected']:.3f} harm r_U "
              f"{point['harm_r_U_corrected']:.3f} refusal gap "
              f"{point.get('refusal_MZ_minus_U_corrected', float('nan')):.3f} harm gap "
              f"{point.get('harm_MZ_minus_U_corrected', float('nan')):.3f} "
              f"eta2(harm, last char) {c['mean']['eta2_harm_last_char']:.3f}", flush=True)
    for t in twins.values():
        t.drop(J_HARM)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/mech_021.json")
    ap.add_argument("--acts-root", type=Path,
                    default=Path("/work/harrychang08/castor/activations/021"))
    ap.add_argument("--cells", default="", help="comma list of cell ids (--list-cells)")
    ap.add_argument("--list-cells", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n-boot-sec", type=int, default=200)
    ap.add_argument("--merge", type=Path, nargs="+", default=None)
    ap.add_argument("--add-per-run", type=Path, default=None,
                    help="merged artifact: add summary.per_run (D096a re-read); every existing "
                         "key must stay bit-identical")
    a = ap.parse_args()
    man = json.loads(a.manifest.read_text())
    all_cells = HC.cells(man)
    if a.list_cells:
        for c, v in all_cells.items():
            print(c, v["size"], len(v["models"]))
        return
    assert a.out is not None and a.out.is_absolute(), "--out must be absolute (D036)"
    run_of = {m["model_id"]: m["pretrain_seed"] for m in man["models"]}
    if a.add_per_run:
        text = a.add_per_run.read_text()
        old = json.loads(text)
        assert json.dumps(old, indent=1, default=float) + "\n" == text, "artifact not canonical"
        assert "per_run" not in old["summary"], "per_run already present"
        fresh = json.loads(json.dumps(HC.summarise(old["cells"]), default=float))
        assert json.dumps(fresh, sort_keys=True) == json.dumps(old["summary"], sort_keys=True), \
            "summary does not recompute from the cells"
        new = json.loads(text)
        new["summary"]["per_run"] = HC.per_run(new["cells"], run_of)
        new["per_run_provenance"] = {"command": provenance.command(),
                                     "git_commit": provenance.git_commit(),
                                     "input_sha256": hashlib.sha256(text.encode()).hexdigest()}
        for k, v in old.items():
            if k != "summary":
                assert json.dumps(new[k], default=float) == json.dumps(v, default=float), k
        for k, v in old["summary"].items():
            assert json.dumps(new["summary"][k], default=float) == json.dumps(v, default=float), k
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(new, indent=1, default=float) + "\n")
        pr = new["summary"]["per_run"]
        print(f"per_run: {pr['n_token']} token / {pr['n_odd']} odd of {pr['n_runs']} -> {a.out}")
        return
    if a.merge:
        out = HC.merge(a.merge)
        out["summary"]["per_run"] = HC.per_run(out["cells"], run_of)
        out["merge_command"] = provenance.command()
        rows = [json.loads(line) for line in EXTRACTION.read_text().splitlines()]
        out["extraction_last_char"] = {
            f"{c}_{s}": HC.shares(HC.last_char_class(r["prompt"]) for r in rows
                                  if r["class"] == c and r["split"] == s)
            for c in ("harmful", "harmless") for s in ("train", "val")}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(f"merged {len(a.merge)} parts, {len(out['cells'])} cells -> {a.out}")
        return

    commit = provenance.git_commit()
    t0 = time.time()
    wanted = [c for c in a.cells.split(",") if c]
    assert wanted and all(c in all_cells for c in wanted), f"unknown cells in {wanted}"
    by_id = {m["model_id"]: m for m in man["models"]}
    panel = evalset.load_eval()
    ids = [r["id"] for r in panel]
    adv = np.array([i for i, r in enumerate(panel) if sanityb.SUBSETS["adv_harmful"](r)])
    ex_rows = [json.loads(line) for line in EXTRACTION.read_text().splitlines()]
    ex_harm = [r for r in ex_rows if r["class"] == "harmful" and r["split"] == "train"]
    ex_safe = [r for r in ex_rows if r["class"] == "harmless" and r["split"] == "train"]
    ex_sources = [r["source"] for r in ex_harm]
    assert len(ex_harm) == len(ex_safe) == 256
    from transformers import AutoTokenizer

    tok_dir = Path(by_id[all_cells[wanted[0]]["models"][0]]["checkpoint"])
    tok = AutoTokenizer.from_pretrained(tok_dir)
    groups = token_groups(panel, ex_harm, ex_safe, tok)
    splits = Dn.half_splits(256, 256, N_SPLITS, SPLIT_SEED)
    out = {"prompt": "046", "manifest": str(a.manifest),
           "manifest_sha256": hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
           "extraction_sha256": hashlib.sha256(EXTRACTION.read_bytes()).hexdigest(),
           "n_splits": N_SPLITS, "split_seed": SPLIT_SEED, "n_boot_sec": a.n_boot_sec,
           "layers": list(HC.LAYERS), "harm_position": J_HARM - Dn.N_STORED,
           "sites": {str(k): v for k, v in SITES.items()},
           "n_prompts": {"panel": len(panel), "adv_harmful": len(adv)},
           "token_groups": {k: v for k, v in groups.items() if not k.startswith("_")},
           "tokenizer": {"path": str(tok_dir), "tokenizer_json_sha256": hashlib.sha256(
               (tok_dir / "tokenizer.json").read_bytes()).hexdigest()},
           "cells": {}, "inputs": {}}
    for c in wanted:
        cell = all_cells[c]
        t1 = time.time()
        print(f"cell {c}: {len(cell['models'])} models", flush=True)
        twins = {m: LD.Twin(by_id[m], a.acts_root, ids) for m in cell["models"]}
        assert len({t.train["harmful"].shape[0] for t in twins.values()} | {256}) == 1
        res = analyse_cell(cell, twins, adv, splits, a.n_boot_sec, ex_sources, groups)
        res["seconds"] = time.time() - t1
        out["cells"][c] = res
        out["inputs"] |= {m: {"scores_sha256": hashlib.sha256(
            (a.acts_root / m / "scores.json").read_bytes()).hexdigest(),
            "extract_commit": t.scores["git_commit"], "labels": by_id[m]["labels"]}
            for m, t in twins.items()}
        del twins
        print(f"cell {c} done in {res['seconds']:.0f}s", flush=True)
    out |= {"seconds": time.time() - t0, "git_commit": commit, "command": provenance.command()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"done in {out['seconds']:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
