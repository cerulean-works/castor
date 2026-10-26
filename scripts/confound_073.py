"""073: the refusal-direction readout at (13, -1) under within-last-token centring, and a
refusal-logit control at the same site. One 1-GPU dev job per set; the GPU only
applies each model's final RMSNorm and unembedding to the stored activations (the logit lens).

    uv run --frozen python scripts/confound_073.py --set A0 --out <absolute>/part_A0.json \
        [--n-boot 200] [--max-models N] [--panel-every K]
    uv run --frozen python scripts/confound_073.py --check-lens <model-id> --out <abs>.json
    uv run --frozen python scripts/confound_073.py --merge <part>.json ... --out <abs>/runs/073/confound.json

Per model everything is 007's instrument unchanged (scripts/ladder_007.py: Twin, readout, the
200 half-splits with seed 2027, per-prompt P = judge_v1's `wg_vulnerable`, the adversarial-
harmful subset) at 021's fixed site (13, -1); the last-token groups are 046's (scripts/harm_046.py
token_groups: the panel prompt's last instruction token id). The statistics are
castor.confound073's: the link family per U pair (raw, centred, projection-only centred; logit
link; partial link), the MZ - U gap of raw and centred projections (sets with MZ pairs), 200
prompt draws, per recipe and the recipe jackknife on the 72-pair set.

The logit readout: Arditi's refusal score (castor.directions.refusal_score: log-odds of the
model's own refusal-token set, `scores.json` `refusal_tokens`) of the logit lens at (13, -1):
x -> RMSNorm(x; the model's final norm weight, eps from its config) -> lm_head, in float32.
`--check-lens` runs one model forward on a few panel prompts and compares (a) its layer-13 input
at the last position with the stored activation and (b) the lens applied to the final norm's
input with the model's own last-position refusal score.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from castor import confound073 as C
from castor import directions as Dn
from castor import evalset, provenance, reliability, sanityb
from castor import mechladder as ML

LD = provenance.load_script("ladder_007")
HM = provenance.load_script("harm_046")
N_SPLITS, SPLIT_SEED = LD.N_SPLITS, LD.SPLIT_SEED
SITE = {"layer": 13, "position": -1}
J_SITE = LD.stored(SITE["position"] + Dn.N_POST)  # stored index 7
ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifests" / "mech_021.json"
ACTS = Path("/work/harrychang08/castor/activations/021")
SETS = ("A0", "A")


def lens_weights(checkpoint: Path) -> dict:
    """The final RMSNorm weight, its eps and the unembedding, from the checkpoint."""
    import torch

    cfg = json.loads((checkpoint / "config.json").read_text())
    assert not cfg.get("tie_word_embeddings", False)
    path = checkpoint / "pytorch_model.bin"
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    w = {"norm": sd["model.norm.weight"].float().clone(),
         "lm_head": sd["lm_head.weight"].float().clone(), "eps": float(cfg["rms_norm_eps"])}
    del sd
    st = path.stat()
    w["source"] = {"path": str(path), "bytes": st.st_size, "mtime": st.st_mtime,
                   "sha256_norm_lm_head": hashlib.sha256(
                       w["norm"].numpy().tobytes() + w["lm_head"].numpy().tobytes()).hexdigest()}
    return w


def lens_scores(x, w: dict, refusal_toks, batch: int = 1024) -> np.ndarray:
    """(n, d) residual-stream activations -> (n,) refusal log-odds of the logit lens."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm, head = w["norm"].to(dev), w["lm_head"].to(dev)
    out = []
    for i in range(0, x.shape[0], batch):
        h = x[i:i + batch].to(dev, torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + w["eps"]) * norm
        out.append(Dn.refusal_score(h @ head.T, refusal_toks).cpu().numpy())
    return np.concatenate(out)


def check_lens(spec: dict, out_path: Path, n_prompts: int = 16) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    panel = evalset.load_eval()
    ids = [r["id"] for r in panel]
    tw = LD.Twin(spec, ACTS, ids)
    ck = Path(spec["checkpoint"])
    tok = AutoTokenizer.from_pretrained(ck)
    model = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.bfloat16).cuda().eval()
    w = lens_weights(ck)
    rows = []
    captured = {}
    hk = model.model.norm.register_forward_pre_hook(
        lambda mod, args: captured.__setitem__("pre", args[0].detach()))
    stored = tw.panel(J_SITE)[:, SITE["layer"]]
    for i in range(0, len(panel), len(panel) // n_prompts)[:n_prompts]:
        t = torch.tensor([evalset.prompt_ids(tok, panel[i]["prompt"])]).cuda()
        with torch.no_grad():
            o = model(t, output_hidden_states=True)
        h13 = o.hidden_states[SITE["layer"]][0, -1].float().cpu()
        s = stored[i].float()
        final = float(Dn.refusal_score(o.logits[0, -1:].float(), tw.scores["refusal_tokens"])[0])
        lens_final = float(lens_scores(captured["pre"][0, -1:].float().cpu(), w,
                                       tw.scores["refusal_tokens"])[0])
        lens13_live = float(lens_scores(h13[None], w, tw.scores["refusal_tokens"])[0])
        lens13_stored = float(lens_scores(s[None], w, tw.scores["refusal_tokens"])[0])
        rows.append({"row": i, "cos_h13_stored": float(torch.nn.functional.cosine_similarity(
            h13, s, dim=0)), "rel_err_h13": float((h13 - s).norm() / s.norm()),
            "refusal_score_model": final, "refusal_score_lens_on_final_norm_input": lens_final,
            "lens13_live": lens13_live, "lens13_stored": lens13_stored})
    hk.remove()
    res = {"model": spec["model_id"], "rows": rows,
           "max_abs_final_vs_lens": max(abs(r["refusal_score_model"]
                                            - r["refusal_score_lens_on_final_norm_input"])
                                        for r in rows),
           "min_cos_h13": min(r["cos_h13_stored"] for r in rows),
           "max_abs_lens13_live_vs_stored": max(abs(r["lens13_live"] - r["lens13_stored"])
                                                for r in rows),
           "git_commit": provenance.git_commit(), "command": provenance.command()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


def run_set(set_name: str, out_path: Path, n_boot: int, max_models: int | None,
            panel_every: int, boot: str = "prompts") -> None:
    commit = provenance.git_commit()
    t0 = time.time()
    man = json.loads(MANIFEST.read_text())
    by_id = {m["model_id"]: m for m in man["models"]}
    models = sorted(man["sets"][set_name]["models"])[:max_models]
    panel = evalset.load_eval()
    ids = [r["id"] for r in panel]
    adv = np.array([i for i, r in enumerate(panel) if sanityb.SUBSETS["adv_harmful"](r)])
    adv = adv[::panel_every]
    ex_rows = [json.loads(line) for line in HM.EXTRACTION.read_text().splitlines()]
    ex_harm = [r for r in ex_rows if r["class"] == "harmful" and r["split"] == "train"]
    ex_safe = [r for r in ex_rows if r["class"] == "harmless" and r["split"] == "train"]
    from transformers import AutoTokenizer

    tok_dir = Path(by_id[models[0]]["checkpoint"])
    tok = AutoTokenizer.from_pretrained(tok_dir)
    groups = HM.token_groups(panel, ex_harm, ex_safe, tok)
    g_tok = np.array(groups["_panel_tok"])[adv]
    splits = Dn.half_splits(256, 256, N_SPLITS, SPLIT_SEED)
    ys, info, recipe, inputs, lens_corr = {}, {}, {}, {}, {}
    for m in models:
        t1 = time.time()
        spec = by_id[m]
        tw = LD.Twin(spec, ACTS, ids)
        r = LD.readout(tw, SITE["layer"], J_SITE, splits)
        w = lens_weights(Path(spec["checkpoint"]))
        toks = tw.scores["refusal_tokens"]
        logit = lens_scores(tw.panel(J_SITE)[:, SITE["layer"]], w, toks)
        # how close the layer-13 lens is to the model's own decision on the extraction prompts
        lens_tr = np.concatenate([lens_scores(tw.train[c][:, SITE["layer"], J_SITE], w, toks)
                                  for c in ("harmful", "harmless")])
        final_tr = np.concatenate([tw.scores["baseline_refusal_scores"][f"{c}_train"]
                                   for c in ("harmful", "harmless")])
        lens_corr[m] = {"r_lens13_final_train": reliability.corr(lens_tr, final_tr),
                        "n": len(final_tr)}
        ys[m] = C.pack((r["full"][adv], r["h1"][adv], r["h2"][adv]), logit[adv], g_tok,
                       tw.y["wg_vulnerable"][adv])
        info[m] = tw.info
        recipe[m] = spec["recipe"]
        inputs[m] = {"scores_sha256": hashlib.sha256((ACTS / m / "scores.json").read_bytes())
                     .hexdigest(), "extract_commit": tw.scores["git_commit"],
                     "labels": spec["labels"], "refusal_tokens": toks, "lens": w["source"]}
        del tw, w
        print(f"{m}: {time.time() - t1:.0f}s", flush=True)
    gap = any(reliability.pair_level(info[a], info[b]) == "MZ" for a in ys for b in ys if a < b)
    an = C.analyze(N_SPLITS, gap, recipe if set_name == "A" else None)
    res = an(ys, info)
    point = C.summary(res)
    print("point:", {k: round(v, 4) for k, v in point.items()
                     if not k.startswith("group:") and isinstance(v, float)}, flush=True)
    t2 = time.time()
    if boot == "clusters":  # declared addition (notes/073 § 5): whole last-token groups
        ci = reliability.bootstrap_prompt_clusters(ys, info, g_tok.tolist(), n_boot,
                                                   stat=C.summary, analyze=an)
    else:
        ci = reliability.bootstrap_prompts(ys, info, n_boot, stat=C.summary, analyze=an)
    print(f"bootstrap ({boot}) {n_boot} draws: {time.time() - t2:.0f}s", flush=True)
    if boot == "clusters":
        out = {"prompt": "073", "set": set_name, "boot": boot, "n_boot": n_boot,
               "panel_every": panel_every, "models": models, "point": point,
               "ci_token_clusters": ci, "n_clusters": len(set(g_tok.tolist())),
               "seconds": time.time() - t0, "git_commit": commit,
               "command": provenance.command()}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(f"done in {out['seconds']:.0f}s -> {out_path}")
        return
    out = {"prompt": "073", "set": set_name, "site": SITE, "n_splits": N_SPLITS,
           "split_seed": SPLIT_SEED, "n_boot": n_boot, "panel_every": panel_every,
           "models": models, "n_prompts": len(adv), "n_last_token_groups": len(set(g_tok)),
           "point": point, "ci_prompts": ci, "pairs": res["pairs"],
           "validity": {m: C.validity(ys[m], N_SPLITS) for m in models},
           "lens_vs_final": lens_corr, "inputs": inputs,
           "tokenizer": {"path": str(tok_dir), "tokenizer_json_sha256": hashlib.sha256(
               (tok_dir / "tokenizer.json").read_bytes()).hexdigest()}}
    if set_name == "A":
        out["jackknife_recipes"] = ML.jackknife_groups(ys, info, recipe, C.summary_pooled, an)
    out |= {"seconds": time.time() - t0, "git_commit": commit, "command": provenance.command()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"done in {out['seconds']:.0f}s -> {out_path}")


def checks(ds: dict, tol: float = 1e-9) -> dict:
    """The uncentred statistics must be 021's and the A0 centred gap point 046's (same instrument,
    same stored activations)."""
    l21 = json.loads((ROOT / "runs/021/ladder.json").read_text())["sets"]
    h46 = json.loads((ROOT / "runs/046/harm_per_recipe.json").read_text())
    c46 = h46["cells"]["A0:dclm-baseline"]["layers"]["13"]
    pairs = [
        ("A raw link = 021", ds["A"]["point"]["raw_link_corrected"],
         l21["A"]["link"]["point"]["site_link_U_corrected"]),
        ("A0 raw link = 021", ds["A0"]["point"]["raw_link_corrected"],
         l21["A0"]["link"]["point"]["site_link_U_corrected"]),
        ("A0 raw gap = 021", ds["A0"]["point"]["ladder_raw_MZ_minus_U_corrected"],
         l21["A0"]["ladder"]["point"]["site_MZ_minus_U_corrected"]),
        ("A0 centred gap = 046", ds["A0"]["point"]["ladder_centred_MZ_minus_U_corrected"],
         c46["diagnosis_c"]["refusal_within_last_token"]["MZ_minus_U_corrected"]),
        ("A0 centred r_U = 046", ds["A0"]["point"]["ladder_centred_r_U_corrected"],
         c46["diagnosis_c"]["refusal_within_last_token"]["r_U_corrected"]),
    ]
    out = {name: {"073": a, "reference": b, "abs_diff": abs(a - b)} for name, a, b in pairs}
    bad = [k for k, v in out.items() if not v["abs_diff"] < tol]
    assert not bad, f"073 does not reproduce: {bad}"
    return out


def merge(parts: list[Path], out_path: Path) -> None:
    ds, cl = {}, {}
    for p in parts:
        d = json.loads(p.read_text())
        if d.get("boot") == "clusters":
            assert d["set"] not in cl and d["panel_every"] == 1, p
            cl[d["set"]] = d | {"part": {"path": str(p), "sha256": hashlib.sha256(
                p.read_bytes()).hexdigest()}}
            continue
        assert d["set"] not in ds, f"set {d['set']} twice"
        assert d["panel_every"] == 1 and d["n_boot"] >= 200, f"{p}: a smoke part"
        ds[d["set"]] = d | {"part": {"path": str(p), "sha256": hashlib.sha256(
            p.read_bytes()).hexdigest()}}
    assert sorted(ds) == sorted(SETS), f"sets present: {sorted(ds)}"
    for s, d in cl.items():  # the logit lens runs on a GPU: float32 rounding may differ by node
        diff = max(abs(d["point"][k] - v) for k, v in ds[s]["point"].items())
        assert set(d["point"]) == set(ds[s]["point"]) and diff < 1e-6, f"{s}: points differ"
        ds[s]["token_clusters"] = {k: d[k] for k in ("ci_token_clusters", "n_boot", "n_clusters",
                                                     "seconds", "git_commit", "command", "part")}
        ds[s]["token_clusters"]["max_abs_point_diff"] = diff
    out = {"prompt": "073", "sets": ds, "checks": checks(ds),
           "prediction": C.prediction(ds["A0"]["ci_prompts"], ds["A"]["ci_prompts"],
                                      ds["A"]["validity"]),
           "merge_command": provenance.command(), "merge_commit": provenance.git_commit()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: v["holds"] for k, v in out["prediction"].items()}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_name", choices=SETS)
    ap.add_argument("--check-lens", default=None, help="model id")
    ap.add_argument("--merge", type=Path, nargs="+", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--max-models", type=int, default=None, help="smoke")
    ap.add_argument("--panel-every", type=int, default=1, help="smoke")
    ap.add_argument("--boot", choices=("prompts", "clusters"), default="prompts")
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (D036)"
    if a.merge:
        merge(a.merge, a.out)
    elif a.check_lens:
        man = json.loads(MANIFEST.read_text())
        check_lens(next(m for m in man["models"] if m["model_id"] == a.check_lens), a.out)
    else:
        run_set(a.set_name, a.out, a.n_boot, a.max_models, a.panel_every, a.boot)


if __name__ == "__main__":
    main()
