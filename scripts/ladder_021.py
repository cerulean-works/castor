"""The ladder_007 readout on new twins at the fixed site (13, −1) against the ladder_007 common
site (11, −2), on the same pairs and prompts. CPU.

    uv run --frozen python scripts/ladder_021.py --manifest data/manifests/mech_021.json \
        --acts-root /work/harrychang08/castor/activations/021 --set A \
        --site 13,-1 --compare-site 11,-2 --out <absolute>/ladder_A_main.json \
        [--blocks main,sites] [--n-boot 1000] [--n-boot-sec 200]
    uv run --frozen python scripts/ladder_021.py --merge <part>.json ... --out runs/021/ladder.json

Per model the instrument is that of scripts/ladder_007.py (Twin, readout, the 200 half-splits
with seed 2027, per-prompt P from judge_v1's `wg_vulnerable`, the adversarial-harmful subset).
Here the site is fixed instead of selected, two sites are read on the same pairs, and the new
statistic is the paired difference of the two links under one prompt resample
(castor.mechladder.two_readouts).

A `--set` names a pair set of the manifest (`sets`): its models and its level rule:
  pair_level  MZ (same base) / U (same recipe, other pretraining seed); R pairs dropped
  factorial   castor.factorial.level_fifth on the manifest's `ancestry`: both / replicate /
              init / order / neither

Output blocks (aggregate statistics only; activations stay on /work):
  link         two-site link per level (corrected for different-base levels, raw for same-base),
               the paired difference, prompt-bootstrap CIs, per-pair rows, the frozen reading at
               each site, the recipe (or base) jackknife, and the two-part rule
  per_recipe   the same per recipe (sets with several recipes), with the spread across recipes
  ladder       two-site projection ladder per level (gap MZ − U where MZ pairs exist)
  harm_control ladder_007's secondary 6 at each site's layer (sets with MZ and U pairs); the
               harm-direction link at each layer otherwise
  validity     projection vs P per twin at the fixed site
  cosines      unaligned direction cosines at the fixed site (pair_level sets)
  site_map     gap and link at every (layer, position), point estimates (block "sites")
"""

import argparse
import hashlib
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from castor import directions as Dn
from castor import evalset, factorial, provenance, reliability, sanityb
from castor import mechladder as ML

LD = provenance.load_script("ladder_007")
N_SPLITS, SPLIT_SEED = LD.N_SPLITS, LD.SPLIT_SEED
J_HARM = 0  # stored position −8, the last instruction token
BLOCKS = ("main", "sites")
BLOCK_KEYS = {"main": ("link", "per_recipe", "ladder", "harm_control", "validity", "cosines"),
              "sites": ("site_map",)}
SHARED = ("prompt", "set", "site", "compare_site", "n_splits", "split_seed", "n_boot",
          "n_boot_sec", "models", "panel_every", "levels", "n_prompts", "inputs", "manifest",
          "manifest_sha256", "git_commit")


def parse_site(s: str) -> dict:
    """"13,-1" -> layer 13, position −1 (pos_index 6, stored index 7)."""
    layer, position = (int(x) for x in s.split(","))
    assert -Dn.N_POST <= position <= -1, f"position {position}: post-instruction only"
    p = position + Dn.N_POST
    return {"layer": layer, "position": position, "pos_index": p, "stored": LD.stored(p)}


class Levels:
    """A set's pair levels: the per-model info objects, the level function and the level lists
    the ladder and the link summarise."""

    def __init__(self, spec: dict, models: dict[str, dict]):
        self.rule = spec["levels"]
        if self.rule == "pair_level":
            self.info = {m: reliability.ModelInfo(s["recipe"], s["pretrain_seed"],
                                                  s["posttrain_seed"]) for m, s in models.items()}
            self.level_of = reliability.pair_level
            self.ladder_levels = ("MZ", "U")
            self.corrected, self.raw = ("U",), ("MZ",)
            self.recipe = {m: s["recipe"] for m, s in models.items()}
        elif self.rule == "factorial":
            anc = spec["ancestry"]
            self.info = {m: factorial.Twin(anc[m]["base_tag"], anc[m]["init_seed"],
                                           anc[m]["data_seed"], int(s["posttrain_seed"]))
                         for m, s in models.items()}
            self.level_of = factorial.level_fifth
            self.ladder_levels = tuple(factorial.FIFTH_LEVELS)
            self.corrected = tuple(x for x in factorial.FIFTH_LEVELS if x != "both")
            self.raw = ("both",)
            self.recipe = {m: anc[m]["base_tag"] for m in models}
        else:
            raise ValueError(self.rule)
        self.counts: dict[str, int] = {}
        for a, b in combinations(sorted(models), 2):
            lvl = self.level_of(self.info[a], self.info[b])
            if lvl:
                self.counts[lvl] = self.counts.get(lvl, 0) + 1
        self.has_mz_u = self.rule == "pair_level" and all(self.counts.get(k, 0) > 0
                                                          for k in ("MZ", "U"))
        wanted = set(self.ladder_levels) | set(self.corrected) | set(self.raw)
        self.pairs = [(a, b) for a, b in combinations(sorted(models), 2)
                      if self.level_of(self.info[a], self.info[b]) in wanted]

    def restrict(self, an):
        """analyze() over this set's levelled pairs only (R pairs of a 72-model set are never
        computed), on whatever subset of models a bootstrap or jackknife passes."""
        def analyze(ys, info, method="pearson", pairs=None):
            if pairs is None:
                pairs = [(a, b) for a, b in self.pairs if a in ys and b in ys]
            return an(ys, info, method, pairs)
        return analyze

    def ladder(self):
        return self.restrict(ML.ladder(N_SPLITS, self.level_of, self.ladder_levels))

    def link(self):
        return self.restrict(ML.link(N_SPLITS, self.level_of, self.corrected, self.raw))


def two_site(lv: Levels, which: str):
    one = lv.link() if which == "link" else lv.ladder()
    st = ML.link_summary if which == "link" else ML.ladder_summary
    return ML.two_readouts(N_SPLITS, one, st)


def pack_sites(reads_s: dict, reads_c: dict, twins: dict, idx: np.ndarray) -> dict:
    return {m: ML.pack_two((reads_s[m]["full"][idx], reads_s[m]["h1"][idx], reads_s[m]["h2"][idx]),
                           (reads_c[m]["full"][idx], reads_c[m]["h1"][idx], reads_c[m]["h2"][idx]),
                           twins[m].y["wg_vulnerable"][idx]) for m in twins}


def pair_rows(res: dict) -> list[dict]:
    """Per-pair rows of a two-site link result: both sites and the difference."""
    by = {(r["a"], r["b"]): r for r in res["compare"]["pairs"]}
    out = []
    for r in res["site"]["pairs"]:
        c = by[(r["a"], r["b"])]
        row = {"a": r["a"], "b": r["b"], "level": r["level"], "link_site": r["link"],
               "link_compare": c["link"], "diff_link": r["link"] - c["link"]}
        if "link_corrected" in r:
            row |= {"link_site_corrected": r["link_corrected"],
                    "link_compare_corrected": c["link_corrected"],
                    "diff_link_corrected": r["link_corrected"] - c["link_corrected"],
                    "rel_dP_site": r["rel_dP"], "rel_dproj_site": r["rel_dproj"],
                    "rel_dP_compare": c["rel_dP"], "rel_dproj_compare": c["rel_dproj"]}
        out.append(row)
    return out


def per_recipe_gaps(res2: dict, recipe: dict[str, str]) -> dict:
    """Corrected MZ − U gap of the refusal and of the harm readout per recipe, from the per-pair
    rows of one ladder_contrast result (both models of a pair share the recipe)."""
    out = {}
    for rc in sorted(set(recipe.values())):
        row = {}
        for which in ("refusal", "harm"):
            lv = {}
            for level in ("MZ", "U"):
                sel = [r["r_corrected"] for r in res2[which]["pairs"]
                       if r["level"] == level and recipe[r["a"]] == rc]
                lv[level] = float(np.nanmean(sel)) if sel else float("nan")
                row[f"n_{level}"] = len(sel)
            row[f"{which}_r_MZ_corrected"], row[f"{which}_r_U_corrected"] = lv["MZ"], lv["U"]
            row[f"{which}_MZ_minus_U_corrected"] = lv["MZ"] - lv["U"]
        row["gap_refusal_minus_harm_corrected"] = (row["refusal_MZ_minus_U_corrected"]
                                                   - row["harm_MZ_minus_U_corrected"])
        out[rc] = row
    return out


def spread(values: list[float]) -> dict:
    v = np.array([x for x in values if np.isfinite(x)], float)
    return {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1
            else float("nan"), "min": float(v.min()), "median": float(np.median(v)),
            "max": float(v.max()),
            "n_runs_through": int((v >= 0.5).sum()), "n_partly": int(((v >= 0.3) & (v < 0.5)).sum()),
            "n_elsewhere": int((v < 0.3).sum())}


def merge(parts: list[Path], out_path: Path) -> None:
    """Join the block files of one run (one set: blocks main + sites) or of several sets into
    `sets[<set>]`; shared fields of one set must agree (same manifest, commit, sites)."""
    ds = [json.loads(p.read_text()) for p in parts]
    sets: dict[str, dict] = {}
    for p, d in zip(parts, ds):
        s = sets.setdefault(d["set"], {"blocks": [], "parts": {}})
        for k in SHARED:
            if k in s:
                assert s[k] == d[k], f"{p}: {k} differs within set {d['set']}"
            s[k] = d[k]
        for b in d["blocks"]:
            assert b not in s["blocks"], f"{d['set']}: block {b} twice"
            s["blocks"].append(b)
            for k in BLOCK_KEYS[b]:
                if k in d:
                    s[k] = d[k]
        s["parts"][str(p)] = {"seconds": d["seconds"], "command": d["command"],
                              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    for s in sets.values():
        s["blocks"] = sorted(s["blocks"])
    out = {"prompt": "021", "sets": dict(sorted(sets.items())),
           "merge_command": provenance.command()}
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"merged {len(parts)} parts, sets {sorted(sets)} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--acts-root", type=Path)
    ap.add_argument("--set", dest="set_name")
    ap.add_argument("--site", default="13,-1", help="LAYER,POSITION; the fixed site (D053d)")
    ap.add_argument("--compare-site", default="11,-2", help="007's common site")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-boot-sec", type=int, default=200)
    ap.add_argument("--panel-every", type=int, default=1, help="every K-th prompt (smoke)")
    ap.add_argument("--allow-subset", action="store_true", help="smoke: models present only")
    ap.add_argument("--blocks", default=",".join(BLOCKS))
    ap.add_argument("--merge", type=Path, nargs="+", default=None)
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (D036)"
    if a.merge:
        merge(a.merge, a.out)
        return
    blocks = set(a.blocks.split(","))
    assert blocks <= set(BLOCKS), f"unknown blocks {blocks - set(BLOCKS)}"
    commit = provenance.git_commit()
    t0 = time.time()
    S, C = parse_site(a.site), parse_site(a.compare_site)
    assert (S["layer"], S["pos_index"]) != (C["layer"], C["pos_index"])

    man = json.loads(a.manifest.read_text())
    spec = man["sets"][a.set_name]
    by_id = {s["model_id"]: s for s in man["models"]}
    specs = [by_id[m] for m in spec["models"]
             if (a.acts_root / m / "scores.json").exists() or not a.allow_subset]
    assert len(specs) >= 2, "a set needs at least two models present"
    panel = evalset.load_eval()[:: a.panel_every]
    ids = [r["id"] for r in panel]
    twins = {s["model_id"]: LD.Twin(s, a.acts_root, ids) for s in specs}
    n_layer = {t.scores["n_layer"] for t in twins.values()}
    assert len(n_layer) == 1, f"mixed depths in one set: {n_layer}"
    n_layer = n_layer.pop()
    lv = Levels(spec, {m: by_id[m] for m in twins})
    info = lv.info
    subset_idx = {k: np.array([i for i, r in enumerate(panel) if sanityb.SUBSETS[k](r)])
                  for k in LD.SUBSETS if k != "all"}
    adv = subset_idx["adv_harmful"]
    first = next(iter(twins.values()))
    splits = Dn.half_splits(len(first.train["harmful"]), len(first.train["harmless"]),
                            N_SPLITS, SPLIT_SEED)
    fixed_defined = S["layer"] < n_layer and C["layer"] < n_layer
    out = {"prompt": "021", "set": a.set_name, "site": S, "compare_site": C,
           "n_splits": N_SPLITS, "split_seed": SPLIT_SEED, "n_boot": a.n_boot,
           "n_boot_sec": a.n_boot_sec, "models": sorted(twins), "panel_every": a.panel_every,
           "levels": {"rule": lv.rule, "counts": lv.counts, "n_layer": n_layer,
                      "fixed_sites_defined": fixed_defined},
           "n_prompts": {k: len(v) for k, v in subset_idx.items()}}
    primary_boot = a.n_boot if spec.get("role") == "primary" else a.n_boot_sec

    def site_reads(site: dict) -> dict:
        r = {m: LD.readout(t, site["layer"], site["stored"], splits) for m, t in twins.items()}
        for t in twins.values():
            t.drop(site["stored"])
        return r

    if "main" in blocks and not fixed_defined:
        print(f"set {a.set_name}: {n_layer} layers, the fixed sites are undefined; site map only")
        out["link"] = {"undefined": f"layer {max(S['layer'], C['layer'])} does not exist in a "
                                    f"{n_layer}-layer model"}
    if "main" in blocks and fixed_defined:
        reads_s, reads_c = site_reads(S), site_reads(C)
        ys2 = pack_sites(reads_s, reads_c, twins, adv)

        an_l, st_l = two_site(lv, "link")
        res = an_l(ys2, info)
        point = st_l(res)
        ci = reliability.bootstrap_prompts(ys2, info, primary_boot, stat=st_l, analyze=an_l)
        link = {"point": point, "ci_prompts": ci, "n_draws": primary_boot,
                "pairs": pair_rows(res),
                "reading": {k: ML.link_reading(point.get(f"{k}_link_U_corrected", float("nan")))
                            for k in ("site", "compare")}}
        groups = lv.recipe
        if len(set(groups.values())) > 1:
            link["jackknife_recipes" if lv.rule == "pair_level" else "jackknife_bases"] = \
                ML.jackknife_groups(ys2, info, groups, st_l, an_l)
        if lv.rule == "pair_level" and lv.counts.get("U", 0) > 0:
            link["prediction_d053d"] = ML.prediction_021(ci["site_link_U_corrected"],
                                                        ci["diff_link_U_corrected"])
            link["prediction_d053d"]["confirmatory"] = spec.get("role") == "primary"
        out["link"] = link
        print(f"link site {point.get('site_link_U_corrected')} compare "
              f"{point.get('compare_link_U_corrected')} diff {point.get('diff_link_U_corrected')}",
              flush=True)

        recipes = sorted(set(lv.recipe.values()))
        if lv.rule == "pair_level" and len(recipes) > 1:
            per = {}
            for rc in recipes:
                sub = {m: ys2[m] for m in twins if lv.recipe[m] == rc}
                if ML.levels_of(list(sub), info).get("U", 0) == 0:
                    continue
                pt = st_l(an_l(sub, info))
                per[rc] = {"n_models": len(sub), "point": pt,
                           "ci_prompts": reliability.bootstrap_prompts(sub, info, a.n_boot_sec,
                                                                       stat=st_l, analyze=an_l)}
                print(f"recipe {rc}: site {pt['site_link_U_corrected']:.3f} compare "
                      f"{pt['compare_link_U_corrected']:.3f}", flush=True)
            out["per_recipe"] = {
                "recipes": per,
                "spread_site": spread([v["point"]["site_link_U_corrected"] for v in per.values()]),
                "spread_compare": spread([v["point"]["compare_link_U_corrected"]
                                          for v in per.values()]),
                "n_recipes_diff_positive": int(sum(v["point"]["diff_link_U_corrected"] > 0
                                                   for v in per.values())),
                "n_recipes_site_ci_above_0.3": int(sum(
                    v["ci_prompts"]["site_link_U_corrected"]["lo"] > 0.3 for v in per.values()))}

        an_g, st_g = two_site(lv, "ladder")
        res_g = an_g(ys2, info)
        out["ladder"] = {"point": st_g(res_g),
                         "ci_prompts": reliability.bootstrap_prompts(ys2, info, a.n_boot_sec,
                                                                     stat=st_g, analyze=an_g),
                         "reliability_site": res_g["site"]["reliability"],
                         "reliability_compare": res_g["compare"]["reliability"],
                         "pairs_site": res_g["site"]["pairs"],
                         "pairs_compare": res_g["compare"]["pairs"]}

        # harm-direction control at each site's layer; only the harm link without MZ pairs
        hc = {}
        for name, site, reads in (("site", S, reads_s), ("compare", C, reads_c)):
            r_h = {m: LD.readout(t, site["layer"], J_HARM, splits) for m, t in twins.items()}
            for t in twins.values():
                t.drop(J_HARM)
            entry = {"layer": site["layer"], "harm_stored_index": J_HARM,
                     "harm_position": J_HARM - Dn.N_STORED}
            y_h = LD.packed(r_h, twins, adv)
            an_hl = lv.link()
            entry["harm_link"] = {"point": ML.link_summary(an_hl(y_h, info)),
                                  "ci_prompts": reliability.bootstrap_prompts(
                                      y_h, info, a.n_boot_sec, stat=ML.link_summary,
                                      analyze=an_hl)}
            if lv.has_mz_u:
                y_2 = {m: ML.pack_two((reads[m]["full"][adv], reads[m]["h1"][adv],
                                       reads[m]["h2"][adv]),
                                      (r_h[m]["full"][adv], r_h[m]["h1"][adv], r_h[m]["h2"][adv]),
                                      twins[m].y["wg_vulnerable"][adv]) for m in twins}
                an2, st2 = lv.restrict(ML.ladder_contrast(N_SPLITS)), ML.ladder_contrast_summary
                res2 = an2(y_2, info)
                entry["contrast"] = {"point": st2(res2),
                                     "reliability_harm": res2["harm"]["reliability"],
                                     "ci_prompts": reliability.bootstrap_prompts(
                                         y_2, info, a.n_boot_sec, stat=st2, analyze=an2),
                                     "pairs_refusal": res2["refusal"]["pairs"],
                                     "pairs_harm": res2["harm"]["pairs"],
                                     "per_recipe": per_recipe_gaps(res2, lv.recipe)}
            hc[name] = entry
        out["harm_control"] = hc

        out["validity"] = {m: ML.validity(ML.pack(r["full"][adv], r["h1"][adv], r["h2"][adv],
                                                  twins[m].y["wg_vulnerable"][adv]), N_SPLITS)
                           for m, r in reads_s.items()}
        if lv.rule == "pair_level":
            out["cosines"] = ML.direction_cosines({m: r["dir"] for m, r in reads_s.items()},
                                                  {m: r["d1"] for m, r in reads_s.items()},
                                                  {m: r["d2"] for m, r in reads_s.items()}, info)
        del reads_s, reads_c, ys2

    if "sites" in blocks:
        an_g, an_l = lv.ladder(), lv.link()
        smap = []
        for p in range(Dn.N_POST):
            j = LD.stored(p)
            for layer in range(n_layer):
                row = {"layer": layer, "pos_index": p, "position": p - Dn.N_POST,
                       "fixed": (layer, p) == (S["layer"], S["pos_index"]),
                       "compare": (layer, p) == (C["layer"], C["pos_index"])}
                if any(not np.any(Dn.mean_diff(*t.train_at(layer, j))) for t in twins.values()):
                    smap.append(row | {"undefined": "difference of means is exactly zero"})
                    continue
                rs = {m: LD.readout(t, layer, j, splits) for m, t in twins.items()}
                ys_s = LD.packed(rs, twins, adv)
                lad = ML.ladder_summary(an_g(ys_s, info))
                lk = ML.link_summary(an_l(ys_s, info))
                row |= {k: v for k, v in lad.items() if k.startswith("r_") or "MZ_minus_U" in k
                        or k == "mean_reliability"}
                row |= {k: v for k, v in lk.items() if k.startswith("link_")}
                smap.append(row)
            for t in twins.values():
                t.drop(j)
            print(f"site map: position {p - Dn.N_POST} done", flush=True)
        out["site_map"] = smap

    out["inputs"] = {m: {"scores_sha256": hashlib.sha256(
        (a.acts_root / m / "scores.json").read_bytes()).hexdigest(),
        "extract_commit": t.scores["git_commit"], "labels": by_id[m]["labels"]}
        for m, t in twins.items()}
    out |= {"blocks": sorted(blocks), "manifest": str(a.manifest),
            "manifest_sha256": hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
            "seconds": time.time() - t0, "git_commit": commit, "command": provenance.command()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"done in {out['seconds']:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
