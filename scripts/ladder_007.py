"""Projection ladder (primary), twin-difference link, secondaries. CPU.

    uv run --frozen python scripts/ladder_007.py --models data/manifests/mech_007.json \
        --acts-root /work/harrychang08/castor/activations/007 \
        --selection runs/007/selection.json [--selection-filtered runs/007/selection_filtered.json] \
        [--fp32 /work/harrychang08/castor/activations/007-fp32/<model-id>] \
        --out /home/harrychang08/castor/runs/007/ladder.json [--n-boot 1000] [--n-boot-sec 200]

Per model: its own unit direction at the common (layer, position) of selection.json, from the
256 + 256 training prompts; S = 200 random half-splits (128 + 128 each; one set of splits for
every model, numpy default_rng(2027)) give the half-direction projections that measure the
projection's split-half reliability (castor.mechladder). Per-prompt P comes from the judge_v1
labels of runs/006 (`wg_vulnerable`, K = 20, greedy excluded; castor.sanityb.outcome_matrix).

Output blocks (aggregate statistics only; activations stay on /work):
  primary      adv_harmful: ladder (corrected r_MZ − r_U) with prompt bootstrap, base-cluster
               bootstrap and base jackknife (castor.reliability, the 006 code path), per pair
  link         adv_harmful: link per U pair corrected, MZ raw; same three uncertainty blocks;
               the frozen reading
  validity     per twin: projection vs P on adv_harmful and harmful (secondary 1)
  layer_curve  ladder and link at every layer, common position (secondary 2)
  subsets      ladder on plain_harmful, harmful, benign and all prompts (pooled harmful +
               benign: inflated by between-class variance) (secondary 3)
  cosines      direction cosines, no alignment (secondary 4)
  sensitivity  own_site (per-twin selection), cosine readout, arditi_filter (the site chosen
               with Arditi's per-model validation filter, same directions), q3g link (5)
  harm_control harm-direction ladder at the last instruction token, common layer; corrected
               (MZ − U)_refusal − (MZ − U)_harm with the prompt bootstrap (secondary 6)
  site_map     primary gap and link at every (layer, position); a-priori sites with intervals
               (secondary 7)
  fp32         bf16 vs float32 projections on one twin
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from castor import directions as Dn
from castor import evalset, provenance, reliability, sanityb
from castor import mechladder as ML
from castor.judge.common import read_jsonl

N_SPLITS = 200
SPLIT_SEED = 2027
SUBSETS = ("adv_harmful", "plain_harmful", "harmful", "benign", "all")
APRIORI_LAYERS = (7, 8, 9, 10)  # 2608.25390: OLMo-2-0425-1B, best at layers 7–10 of 16


def stored(pos_index: int) -> int:
    """Post-instruction candidate index (0 … 6 = positions −7 … −1) -> stored index (0 … 7 =
    positions −8 … −1)."""
    return pos_index + (Dn.N_STORED - Dn.N_POST)


class Twin:
    def __init__(self, spec: dict, acts_root: Path, ids: list[str]):
        self.id = spec["model_id"]
        self.info = reliability.ModelInfo(spec["recipe"], spec["pretrain_seed"],
                                          spec["posttrain_seed"])
        d = acts_root / self.id
        self.scores = json.loads((d / "scores.json").read_text())
        with safe_open(str(d / "extraction.safetensors"), "pt") as f:
            self.train = {c: f.get_tensor(f"acts_{c}_train") for c in ("harmful", "harmless")}
        self.panel_path = d / "panel.safetensors"
        with safe_open(str(self.panel_path), "pt") as f:
            pids = json.loads(f.metadata()["ids"])
        assert pids == ids, f"{self.id}: panel activations do not match the evaluated panel"
        self._cache: dict[int, torch.Tensor] = {}
        rows = list(read_jsonl(Path(spec["labels"])))
        self.y = {f: sanityb.outcome_matrix(rows, pids, f)[0]
                  for f in ("wg_vulnerable", "q3g_vulnerable")}
        self.n = len(pids)

    def panel(self, j: int) -> torch.Tensor:
        """(n, L, d) bf16 panel activations at stored position j (cached)."""
        if j not in self._cache:
            with safe_open(str(self.panel_path), "pt") as f:
                self._cache[j] = f.get_slice("acts")[:, :, j:j + 1, :][:, :, 0, :].clone()
        return self._cache[j]

    def drop(self, j: int) -> None:
        self._cache.pop(j, None)

    def train_at(self, layer: int, j: int):
        h = self.train["harmful"][:, layer, j].double().numpy()
        b = self.train["harmless"][:, layer, j].double().numpy()
        return h, b


def readout(tw: Twin, layer: int, j: int, splits, cosine: bool = False) -> dict:
    """Full and half-direction readouts of every panel prompt at one site."""
    h, b = tw.train_at(layer, j)
    full = Dn.unit(Dn.mean_diff(h, b))
    d1, d2 = Dn.half_directions(h, b, splits)
    x = tw.panel(j)[:, layer].double().numpy()
    f = Dn.cosine_readout if cosine else Dn.project
    return {"full": f(x, full), "h1": f(x, d1), "h2": f(x, d2), "dir": full, "d1": d1, "d2": d2}


def packed(reads: dict, twins: dict, idx: np.ndarray, field: str = "wg_vulnerable") -> dict:
    return {m: ML.pack(r["full"][idx], r["h1"][idx], r["h2"][idx], twins[m].y[field][idx])
            for m, r in reads.items()}


def ladder_block(ys, info, n_boot, full: bool) -> dict:
    an, st = ML.ladder(N_SPLITS), ML.ladder_summary
    res = an(ys, info)
    out = {"point": st(res), "reliability": res["reliability"],
           "ci_prompts": reliability.bootstrap_prompts(ys, info, n_boot, stat=st, analyze=an)}
    if full:
        out["pairs"] = res["pairs"]
        out["ci_bases"] = reliability.bootstrap_bases(ys, info, n_boot, stat=st, analyze=an)
        out["jackknife_bases"] = reliability.jackknife_bases(ys, info, stat=st, analyze=an)
    return out


def link_block(ys, info, n_boot, full: bool) -> dict:
    an, st = ML.link(N_SPLITS), ML.link_summary
    res = an(ys, info)
    point = st(res)
    out = {"point": point, "reading": ML.link_reading(point["link_U_corrected"]),
           "ci_prompts": reliability.bootstrap_prompts(ys, info, n_boot, stat=st, analyze=an)}
    if full:
        out["pairs"] = res["pairs"]
        out["ci_bases"] = reliability.bootstrap_bases(ys, info, n_boot, stat=st, analyze=an)
        out["jackknife_bases"] = reliability.jackknife_bases(ys, info, stat=st, analyze=an)
    return out


BLOCKS = ("main", "sens", "layers", "sites")
BLOCK_KEYS = {"main": ("primary", "link", "validity", "projection_means", "subsets", "cosines",
                       "harm_control", "fp32"),
              "sens": ("sensitivity",), "layers": ("layer_curve",), "sites": ("site_map",)}
SHARED = ("prompt", "site", "n_splits", "split_seed", "n_boot", "n_boot_sec", "models",
          "panel_every", "levels", "n_prompts", "inputs", "selection", "selection_sha256",
          "git_commit")


def merge(parts: list[Path], out_path: Path) -> None:
    """Join the block files of one run: shared fields must agree (same selection, inputs,
    code commit); every block appears exactly once."""
    ds = [json.loads(p.read_text()) for p in parts]
    out = {k: ds[0][k] for k in SHARED}
    seen: list[str] = []
    for p, d in zip(parts, ds):
        for k in SHARED:
            assert d[k] == out[k], f"{p}: {k} differs from {parts[0]}"
        for b in d["blocks"]:
            assert b not in seen, f"block {b} twice"
            seen.append(b)
            for k in BLOCK_KEYS[b]:
                if k in d:
                    out[k] = d[k]
    assert sorted(seen) == sorted(BLOCKS), f"blocks present: {seen}"
    out |= {"blocks": sorted(seen),
            "parts": {str(p): {"seconds": d["seconds"], "command": d["command"],
                               "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p, d in zip(parts, ds)},
            "merge_command": provenance.command()}
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"merged {len(parts)} parts -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--acts-root", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--selection-filtered", type=Path, default=None)
    ap.add_argument("--fp32", type=Path, default=None, help="float32 extraction dir, one twin")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-boot-sec", type=int, default=200, help="prompt draws, secondaries")
    ap.add_argument("--panel-every", type=int, default=1, help="every K-th prompt (smoke)")
    ap.add_argument("--allow-subset", action="store_true", help="smoke: models present only")
    ap.add_argument("--blocks", default=",".join(BLOCKS),
                    help=f"comma list of {BLOCKS}; parts run as parallel jobs, then --merge")
    ap.add_argument("--merge", type=Path, nargs="+", default=None,
                    help="part files of one run (same selection, commit) -> --out")
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (D036)"
    if a.merge:
        merge(a.merge, a.out)
        return
    blocks = set(a.blocks.split(","))
    assert blocks <= set(BLOCKS), f"unknown blocks {blocks - set(BLOCKS)}"
    commit = provenance.git_commit()
    t0 = time.time()

    sel = json.loads(a.selection.read_text())
    L, P = sel["common"]["layer"], sel["common"]["pos_index"]
    J = stored(P)
    panel = evalset.load_eval()[:: a.panel_every]
    ids = [r["id"] for r in panel]
    specs = [s for s in json.loads(a.models.read_text())["models"]
             if (a.acts_root / s["model_id"] / "scores.json").exists() or not a.allow_subset]
    assert sorted(s["model_id"] for s in specs) == sel["models"], "selection / manifest mismatch"
    twins = {s["model_id"]: Twin(s, a.acts_root, ids) for s in specs}
    info = {m: t.info for m, t in twins.items()}
    subset_idx = {k: np.array([i for i, r in enumerate(panel) if sanityb.SUBSETS[k](r)])
                  for k in SUBSETS if k != "all"}
    subset_idx["all"] = np.arange(len(panel))
    splits = Dn.half_splits(len(next(iter(twins.values())).train["harmful"]),
                            len(next(iter(twins.values())).train["harmless"]),
                            N_SPLITS, SPLIT_SEED)
    out = {"prompt": "007", "site": {"layer": L, "pos_index": P, "position": P - Dn.N_POST},
           "n_splits": N_SPLITS, "split_seed": SPLIT_SEED, "n_boot": a.n_boot,
           "n_boot_sec": a.n_boot_sec, "models": sorted(twins), "panel_every": a.panel_every,
           "levels": ML.levels_of(list(twins), info),
           "n_prompts": {k: len(v) for k, v in subset_idx.items()}}

    reads = {m: readout(t, L, J, splits) for m, t in twins.items()}
    adv = subset_idx["adv_harmful"]
    if "main" in blocks:
        ys = packed(reads, twins, adv)
        out["primary"] = ladder_block(ys, info, a.n_boot, full=True)
        out["link"] = link_block(ys, info, a.n_boot, full=True)
        print(f"primary {out['primary']['point']}  link {out['link']['point']}", flush=True)

    if "main" in blocks:
        # 1. validity per twin
        out["validity"] = {k: {m: ML.validity(ML.pack(r["full"][subset_idx[k]], r["h1"][subset_idx[k]],
                                                      r["h2"][subset_idx[k]],
                                                      twins[m].y["wg_vulnerable"][subset_idx[k]]),
                                              N_SPLITS) for m, r in reads.items()}
                           for k in ("adv_harmful", "harmful")}
        out["projection_means"] = {m: {k: float(r["full"][v].mean()) for k, v in subset_idx.items()}
                                   for m, r in reads.items()}

    if "layers" in blocks:
        # 2. layer curve at the common position
        curve = {}
        for layer in range(next(iter(twins.values())).scores["n_layer"]):
            if any(not np.any(Dn.mean_diff(*t.train_at(layer, J))) for t in twins.values()):
                # e.g. layer 0 at a template position: the same token embedding for every prompt
                curve[layer] = {"undefined": "difference of means is exactly zero"}
                print(f"layer {layer}: undefined (zero difference of means)", flush=True)
                continue
            rl = reads if layer == L else {m: readout(t, layer, J, splits) for m, t in twins.items()}
            y_l = packed(rl, twins, adv)
            curve[layer] = {"ladder": ladder_block(y_l, info, a.n_boot_sec, full=False),
                            "link": link_block(y_l, info, a.n_boot_sec, full=False)}
            print(f"layer {layer}: {curve[layer]['ladder']['point']['MZ_minus_U_corrected']:.4f} "
                  f"link {curve[layer]['link']['point']['link_U_corrected']:.4f}", flush=True)
        out["layer_curve"] = curve

    if "main" in blocks:
        # 3. subsets
        out["subsets"] = {k: ladder_block(packed(reads, twins, subset_idx[k]), info, a.n_boot_sec,
                                          full=False)
                          for k in ("plain_harmful", "harmful", "benign", "all")}
        out["subsets"]["all"]["note"] = "pooled harmful + benign: inflated by between-class variance"

        # 4. cosines without alignment
        out["cosines"] = ML.direction_cosines({m: r["dir"] for m, r in reads.items()},
                                              {m: r["d1"] for m, r in reads.items()},
                                              {m: r["d2"] for m, r in reads.items()}, info)

    if "sens" in blocks:
        # 5. sensitivities
        sens = {}
        own = {m: sel["own"][m] for m in twins}
        if all(v is not None for v in own.values()):
            r_own = {m: readout(twins[m], v["layer"], stored(v["pos_index"]), splits)
                     for m, v in own.items()}
            y_o = packed(r_own, twins, adv)
            sens["own_site"] = {"sites": own,
                                "ladder": ladder_block(y_o, info, a.n_boot_sec, full=False),
                                "link": link_block(y_o, info, a.n_boot_sec, full=False)}
        else:
            sens["own_site"] = {"sites": own, "skipped": "a model has no passing candidate"}
        r_cos = {m: readout(t, L, J, splits, cosine=True) for m, t in twins.items()}
        y_c = packed(r_cos, twins, adv)
        sens["cosine_readout"] = {"ladder": ladder_block(y_c, info, a.n_boot_sec, full=False),
                                  "link": link_block(y_c, info, a.n_boot_sec, full=False)}
        if a.selection_filtered:
            # Arditi's per-model filter on the validation prompts changes the scores, hence the
            # site (select_007.py --filtered); the directions stay the ones those scores were
            # computed for (all 256 + 256 training prompts), read at that site.
            self_f = json.loads(a.selection_filtered.read_text())
            assert self_f["filtered"] and self_f["models"] == sel["models"]
            Lf, Jf = self_f["common"]["layer"], stored(self_f["common"]["pos_index"])
            r_f = reads if (Lf, Jf) == (L, J) else {m: readout(t, Lf, Jf, splits)
                                                    for m, t in twins.items()}
            y_f = packed(r_f, twins, adv)
            sens["arditi_filter"] = {"site": self_f["common"], "n_val_kept": self_f["n_kept"],
                                     "same_site_as_primary": (Lf, Jf) == (L, J),
                                     "ladder": ladder_block(y_f, info, a.n_boot_sec, full=False),
                                     "link": link_block(y_f, info, a.n_boot_sec, full=False)}
        y_q = packed(reads, twins, adv, field="q3g_vulnerable")
        sens["q3g_link"] = link_block(y_q, info, a.n_boot_sec, full=False)
        out["sensitivity"] = sens

    if "main" in blocks:
        # 6. harm-direction control ladder: a harmful-vs-harmless difference-of-means direction
        # read at the last instruction token (stored position −8) at the common layer (2507.11878)
        J_HARM = 0
        r_h = {m: readout(t, L, J_HARM, splits) for m, t in twins.items()}
        y_2 = {m: ML.pack_two((reads[m]["full"][adv], reads[m]["h1"][adv], reads[m]["h2"][adv]),
                              (r_h[m]["full"][adv], r_h[m]["h1"][adv], r_h[m]["h2"][adv]),
                              twins[m].y["wg_vulnerable"][adv]) for m in twins}
        an2, st2 = ML.ladder_contrast(N_SPLITS), ML.ladder_contrast_summary
        res2 = an2(y_2, info)
        out["harm_control"] = {
            "site": {"layer": L, "stored_index": J_HARM, "position": J_HARM - Dn.N_STORED},
            "point": st2(res2), "reliability_harm": res2["harm"]["reliability"],
            "ci_prompts": reliability.bootstrap_prompts(y_2, info, a.n_boot, stat=st2, analyze=an2),
            "harm_link": link_block(packed(r_h, twins, adv), info, a.n_boot_sec, full=False)}
        print(f"harm control {out['harm_control']['point']['gap_refusal_minus_harm_corrected']:.4f}",
              flush=True)
        for t in twins.values():
            t.drop(J_HARM)

    if "sites" in blocks:
        # 7. site map: the primary gap and the link at every candidate, point estimates;
        # prompt-bootstrap intervals at the a-priori sites (position −1, layers 7–10:
        # 2507.11878, 2608.25390); the common site is the primary block above
        apriori = [(layer, Dn.N_POST - 1) for layer in APRIORI_LAYERS]
        smap = []
        for p in range(Dn.N_POST):
            j = stored(p)
            for layer in range(next(iter(twins.values())).scores["n_layer"]):
                row = {"layer": layer, "pos_index": p, "position": p - Dn.N_POST,
                       "common": (layer, p) == (L, P), "apriori": (layer, p) in apriori}
                if any(not np.any(Dn.mean_diff(*t.train_at(layer, j))) for t in twins.values()):
                    smap.append(row | {"undefined": "difference of means is exactly zero"})
                    continue
                rs = reads if (layer, j) == (L, J) else {m: readout(t, layer, j, splits)
                                                         for m, t in twins.items()}
                ys_s = packed(rs, twins, adv)
                row |= ML.site_point(ys_s, info, N_SPLITS)
                if row["apriori"]:
                    row["ladder"] = ladder_block(ys_s, info, a.n_boot_sec, full=False)
                    row["link"] = link_block(ys_s, info, a.n_boot_sec, full=False)
                smap.append(row)
            if j != J:
                for t in twins.values():
                    t.drop(j)
            print(f"site map: position {p - Dn.N_POST} done", flush=True)
        out["site_map"] = smap

    # float32 projections on one twin
    if "main" in blocks and a.fp32:
        s32 = json.loads((a.fp32 / "scores.json").read_text())
        assert s32["dtype"] == "float32"
        m = s32["model_id"]
        with safe_open(str(a.fp32 / "extraction.safetensors"), "pt") as f:
            h = f.get_tensor("acts_harmful_train")[:, L, J].double().numpy()
            b = f.get_tensor("acts_harmless_train")[:, L, J].double().numpy()
        with safe_open(str(a.fp32 / "panel.safetensors"), "pt") as f:
            x = f.get_slice("acts")[:, L:L + 1, J:J + 1, :][:, 0, 0, :].double().numpy()
        p32 = Dn.project(x, Dn.unit(Dn.mean_diff(h, b)))
        p16 = reads[m]["full"]
        out["fp32"] = {"model_id": m, "r_all": reliability.corr(p16, p32),
                       "r_adv_harmful": reliability.corr(p16[adv], p32[adv]),
                       "max_abs_diff": float(np.max(np.abs(p16 - p32))),
                       "sd_projection": float(np.std(p32))}

    out["inputs"] = {m: {"scores_sha256": hashlib.sha256(
        (a.acts_root / m / "scores.json").read_bytes()).hexdigest(),
        "extract_commit": t.scores["git_commit"]} for m, t in twins.items()}
    out |= {"blocks": sorted(blocks), "selection": str(a.selection),
            "selection_sha256": hashlib.sha256(a.selection.read_bytes()).hexdigest(),
            "seconds": time.time() - t0, "git_commit": commit, "command": provenance.command()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"done in {out['seconds']:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
