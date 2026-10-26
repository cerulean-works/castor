"""Cross-twin ablation transfer (castor/transfer.py).

    # CPU, once: the Procrustes fit set (committed)
    uv run --frozen python scripts/transfer_018.py prep
    # GPU (one dev job): fit-set activations of the 12 twins, every token, own-site layers
    uv run --frozen python scripts/transfer_018.py extract \
        --out-root /work/harrychang08/castor/activations/018 [--model-ids ID ...]
    # CPU: one Procrustes map per (donor, target) pair, matched and mismatched
    uv run --frozen python scripts/transfer_018.py fit \
        --acts-root /work/harrychang08/castor/activations/018 \
        --out /home/harrychang08/castor/runs/018/procrustes.json
    # GPU, per cell (shell/mech/transfer_genjudge_018.sh): one ablated checkpoint
    uv run --frozen python scripts/transfer_018.py ablate --target ID --arm ARM \
        --out /work/harrychang08/castor/models/ablate/018/ID-ARM
    # CPU: transfer ratios, intervals, the rule
    uv run --frozen python scripts/transfer_018.py analyze \
        --out /home/harrychang08/castor/runs/018/transfer.json

Directions: the donor's own-site difference of means (runs/007/selection.json["own"],
candidates.safetensors of 007's extraction), unit-normalised, moved into the target by
castor.transfer.transport (identity for the unrotated arms) and removed from every layer of
the target by weight orthogonalization (castor.directions.orthogonalize_, Arditi Eq. 5), with
007's checks: orthogonalized = hooked logits on 16 panel prompts, no residual component left.
The Procrustes map of a pair is fitted at the donor's own-site layer (resid_pre) in both twins,
on every token of the fit prompts except BOS. Diagnostics beside it (no arm):
the fit with BOS, the fit on content tokens only (the shared template prefix after BOS is one
activation repeated in every prompt under causal attention, 5 of 25 tokens per prompt), and the
standardized fit (Bosco App. A.9), whose map and std vectors are saved for the diff-rot-std arm.
"""

import argparse
import functools
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from castor import directions as Dn
from castor import evalset, provenance, sanityb
from castor import transfer as T
from castor.judge.common import read_jsonl

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "data" / "manifests" / "mech_007.json"
SELECTION = ROOT / "runs" / "007" / "selection.json"
EXTRACTION = ROOT / "data" / "directions" / "extraction_v1.jsonl"
FIT = ROOT / "data" / "directions" / "procrustes_fit_v1.jsonl"
FIT_MANIFEST = ROOT / "data" / "directions" / "procrustes_fit_v1.MANIFEST.json"
W = Path("/work/harrychang08/castor")
ACTS_007 = W / "activations" / "007"
RUNS = W / "runs"
N_FIT = 256
SUBSETS = ("plain_harmful", "adv_harmful", "benign")
SERVED_DTYPE = "bfloat16"  # as scripts/generate.py serves every twin


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def specs() -> dict[str, dict]:
    return {m["model_id"]: m for m in json.loads(MODELS.read_text())["models"]}


def own_sites() -> dict[str, tuple[int, int]]:
    own = json.loads(SELECTION.read_text())["own"]
    return {m: (s["layer"], s["pos_index"]) for m, s in own.items()}


def fit_layers() -> list[int]:
    """Every own-site layer: a pair is fitted at its donor's layer in both twins."""
    return sorted({lay for lay, _ in own_sites().values()})


def own_direction(model_id: str) -> tuple[np.ndarray, dict]:
    """The twin's own-site difference of means (raw) and where it was read."""
    from safetensors import safe_open

    layer, pos = own_sites()[model_id]
    p = ACTS_007 / model_id / "candidates.safetensors"
    with safe_open(str(p), "pt") as f:
        d = f.get_tensor("mean_diff")[layer, pos].numpy().astype(np.float64)
    return d, {"layer": layer, "pos_index": pos, "position": pos - Dn.N_POST,
               "candidates": str(p), "candidates_sha256": sha(p)}


def candidate(model_id: str, layer: int, pos: int) -> np.ndarray:
    from safetensors import safe_open

    with safe_open(str(ACTS_007 / model_id / "candidates.safetensors"), "pt") as f:
        return f.get_tensor("mean_diff")[layer, pos].numpy().astype(np.float64)


# ---- prep: the fit set ----------------------------------------------------------------------


def cmd_prep(a) -> None:
    spec = importlib.util.spec_from_file_location("prep_007", ROOT / "scripts" / "prep_007.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    from castor import decontam as D

    used = {json.loads(x)["prompt"] for x in EXTRACTION.read_text().splitlines()}
    index = D.PanelIndex([r["prompt"] for r in evalset.load_panel()])

    def pool(rows):  # prep_007's dedup + decontamination, unchanged
        seen, dedup = set(), []
        for r in rows:
            if r[2] not in seen:
                seen.add(r[2])
                dedup.append(r)
        why = P.matches(index, [r[2] for r in dedup])
        return [r for r, w in zip(dedup, why) if not w]

    harmful = pool([(f, i, x["instruction"]) for f in P.HARMFUL_FILES
                    for i, x in enumerate(P.load(f"processed/{f}"))])
    harmless = pool([("alpaca/harmless_train", i, x["instruction"])
                     for i, x in enumerate(P.load("splits/harmless_train"))])
    assert len(harmful) == 577, len(harmful)  # 007 MANIFEST pools.harmful.n_pool
    rest_h = [r for r in harmful if r[2] not in used]
    rest_b = [r for r in harmless if r[2] not in used]
    assert len(rest_h) == 577 - 288, len(rest_h)
    rng = np.random.default_rng(T.SEED)
    draw = {"harmful": [rest_h[i] for i in rng.permutation(len(rest_h))[:N_FIT]],
            "harmless": [rest_b[i] for i in rng.choice(len(rest_b), N_FIT, replace=False)]}
    lines = [json.dumps({"id": f"{c}-fit-{j:03d}", "class": c, "split": "fit", "source": s,
                         "source_index": i, "prompt": t}, ensure_ascii=False)
             for c, rows in draw.items() for j, (s, i, t) in enumerate(rows)]
    data = ("\n".join(lines) + "\n").encode()
    assert not used & {json.loads(x)["prompt"] for x in lines}
    FIT.write_bytes(data)
    man = {"file": str(FIT.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest(),
           "counts": {c: len(r) for c, r in draw.items()},
           "pools": {"harmful_remainder": len(rest_h), "harmless_train_unused": len(rest_b)},
           "rule": "harmful: the decontaminated 007 pool (577) minus extraction_v1 (288), "
                   "permuted, first 256; harmless: Arditi's harmless_train pool (decontaminated "
                   "as in 007) minus extraction_v1, choice(256)",
           "draw": {"rng": f"numpy.random.default_rng({T.SEED})",
                    "order": "harmful permutation, then harmless choice, one generator"},
           "source": {"repo": P.ARDITI_REPO, "commit": P.ARDITI_COMMIT},
           "extraction_v1_sha256": sha(EXTRACTION),
           "git_commit": provenance.git_commit(allow_dirty=True),
           "command": provenance.command()}
    FIT_MANIFEST.write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: man[k] for k in ("counts", "pools", "sha256")}))


def load_fit() -> list[dict]:
    data = FIT.read_bytes()
    assert hashlib.sha256(data).hexdigest() == json.loads(FIT_MANIFEST.read_text())["sha256"]
    return [json.loads(x) for x in data.decode().splitlines()]


# ---- extract (GPU) --------------------------------------------------------------------------


def all_token_acts(model, ids, pad_id: int, layers: list[int], max_tokens: int = 65536):
    """resid_pre at `layers` for every real token: (sum of lengths, len(layers), d) float32,
    prompts concatenated in input order."""
    import torch

    store: dict = {}
    bl = Dn.blocks(model)

    def make(layer):
        def pre(module, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            store[layer] = x.detach().float()
        return pre

    per = [None] * len(ids)
    with torch.no_grad(), Dn.hooks([(bl[i], make(i)) for i in layers]):
        for b in Dn.length_batches(ids, max_tokens):
            x = Dn.batch_inputs([ids[i] for i in b], pad_id, model.device)
            model(**x, use_cache=False, logits_to_keep=1)
            a = torch.stack([store[i] for i in layers], 2).cpu()  # (batch, seq, L, d)
            for j, i in enumerate(b):
                per[i] = a[j, a.shape[1] - len(ids[i]):]  # left padded
    return torch.cat(per, 0)


def cmd_extract(a) -> None:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert a.out_root.is_absolute(), "--out-root must be absolute (D036)"
    commit = provenance.git_commit()
    rows = load_fit()
    layers = fit_layers()
    sp = specs()
    for mid in a.model_ids or T.targets():
        out = a.out_root / mid
        s = sp[mid]
        if (out / "fit.json").exists() and acts_complete(
                json.loads((out / "fit.json").read_text()), layers, s["checkpoint_resolved"]):
            print(f"skip {mid} (complete)")
            continue
        t0 = time.time()
        ckpt = s["checkpoint"]
        assert str(Path(ckpt).resolve()) == s["checkpoint_resolved"]
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16).cuda()
        model.eval()
        ids = Dn.encode(tok, [r["prompt"] for r in rows])
        assert all(x[0] == tok.bos_token_id for x in ids), "first token is not BOS"
        acts = all_token_acts(model, ids, tok.pad_token_id, layers, a.max_tokens)
        out.mkdir(parents=True, exist_ok=True)
        save_file({"acts": acts.to(torch.bfloat16).contiguous()}, str(out / "fit.safetensors"))
        lengths = [len(x) for x in ids]
        meta = {"model_id": mid, "checkpoint": ckpt, "checkpoint_resolved":
                s["checkpoint_resolved"], "dtype": "bfloat16", "layers": layers,
                "hook": "resid_pre (block input)", "lengths": lengths,
                "prompt_ids": [r["id"] for r in rows],
                "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "n_tokens": int(sum(lengths)), "fit_sha256": sha(FIT),
                "acts_sha256": sha(out / "fit.safetensors"), "seconds": time.time() - t0,
                "git_commit": commit, "command": provenance.command()}
        (out / "fit.json").write_text(json.dumps(meta) + "\n")
        print(json.dumps({k: meta[k] for k in ("model_id", "n_tokens", "seconds")}))
        del model
        torch.cuda.empty_cache()


# ---- fit (CPU) ------------------------------------------------------------------------------


def acts_complete(meta: dict, layers: list[int], checkpoint_resolved: str) -> bool:
    """An extracted twin is reused only for the same fit set, layers and checkpoint."""
    return (meta.get("fit_sha256") == sha(FIT) and meta.get("layers") == layers
            and meta.get("checkpoint_resolved") == checkpoint_resolved)


@functools.cache
def load_acts(root: Path, mid: str, layer: int) -> tuple[np.ndarray, dict]:
    """Cached: every twin is read once per layer across the pairs it enters."""
    from safetensors import safe_open

    meta = json.loads((root / mid / "fit.json").read_text())
    assert meta["fit_sha256"] == sha(FIT), f"{mid}: activations of another fit set"
    with safe_open(str(root / mid / "fit.safetensors"), "pt") as f:
        x = f.get_tensor("acts")[:, meta["layers"].index(layer)].float().numpy()
    return x, meta


def fit_token_ids(checkpoint: str, token_ids_sha256: str) -> list[list[int]]:
    """The fit prompts' token ids, checked against the extraction's hash."""
    from transformers import AutoTokenizer

    ids = Dn.encode(AutoTokenizer.from_pretrained(checkpoint), [r["prompt"] for r in load_fit()])
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == token_ids_sha256
    return ids


def save_map(path: Path, **arrays: np.ndarray) -> dict:
    """One map on disk: .npy for a bare rotation, .npz for a rotation with its std vectors."""
    if path.suffix == ".npz":
        np.savez(path, **{k: v.astype(np.float64) for k, v in arrays.items()})
    else:
        np.save(path, arrays["r"].astype(np.float64))
    return {"path": str(path), "sha256": sha(path)}


def load_map(m: dict) -> dict[str, np.ndarray]:
    """The arrays of a saved map, after checking that the file is the one recorded."""
    p = Path(m["path"])
    assert sha(p) == m["sha256"], f"map changed: {p}"
    if p.suffix == ".npz":
        with np.load(p) as z:
            return dict(z)
    return {"r": np.load(p)}


def cmd_fit(a) -> None:
    assert a.out.is_absolute() and a.maps.is_absolute()
    commit = provenance.git_commit()
    sites = own_sites()
    a.maps.mkdir(parents=True, exist_ok=True)
    pairs, lengths_seen, span = {}, set(), None
    for tgt in a.targets or T.targets():
        for kind, don in T.donors(tgt).items():
            t0 = time.time()
            layer, pos = sites[don]
            xa, ma = load_acts(a.acts_root, don, layer)
            xb, mb = load_acts(a.acts_root, tgt, layer)
            assert ma["token_ids_sha256"] == mb["token_ids_sha256"], "token ids differ"
            lengths_seen.add(ma["token_ids_sha256"])
            lengths = ma["lengths"]
            if span is None:
                span = T.template_span(fit_token_ids(ma["checkpoint"], ma["token_ids_sha256"]))
            d_a, src_a = own_direction(don)
            d_b, src_b = own_direction(tgt)
            d_b_at_a = candidate(tgt, layer, pos)  # the target's direction at the donor's site
            m_rows = T.matched_rows(lengths)
            am, bm = T.centred(xa[m_rows[0]], xb[m_rows[1]])
            r = T.procrustes(am, bm)
            row = {"target": tgt, "donor": don, "kind": kind,
                   "fit_layer": layer, "donor_site": src_a, "target_site": src_b,
                   "n_rows_matched": len(m_rows[0]),
                   "resid_identity": T.fit_residual(am, bm, np.eye(xa.shape[1])),
                   "resid_matched": T.fit_residual(am, bm, r),
                   "r_minus_identity_fro_over_sqrt_d": float(
                       np.linalg.norm(r - np.eye(len(r))) / np.sqrt(len(r))),
                   "cos_unrot_vs_target_own": T.cos(d_a, d_b),
                   "cos_rot_vs_target_own": T.cos(T.transport(d_a, r), d_b),
                   "cos_unrot_vs_target_at_donor_site": T.cos(d_a, d_b_at_a),
                   "cos_rot_vs_target_at_donor_site": T.cos(T.transport(d_a, r), d_b_at_a)}
            maps = {"matched": {"r": r}}
            if kind == "diff":
                mm = T.mismatched_rows(lengths, T.derangement(len(lengths)))
                rm = T.procrustes(xa[mm[0]], xb[mm[1]])
                maps["mismatched"] = {"r": rm}
                rs, sa, sb = T.standardized_procrustes(am, bm)
                maps["standardized"] = {"r": rs, "sa": sa, "sb": sb}
                ds = T.transport_standardized(d_a, rs, sa, sb)
                row |= {"n_rows_mismatched": len(mm[0]),
                        "resid_mismatched_on_matched": T.fit_residual(am, bm, rm),
                        "cos_mismatch_vs_target_own": T.cos(T.transport(d_a, rm), d_b),
                        "cos_mismatch_vs_target_at_donor_site": T.cos(
                            T.transport(d_a, rm), d_b_at_a),
                        "cos_rot_standardized_vs_target_own": T.cos(ds, d_b),
                        "cos_rot_standardized_vs_target_at_donor_site": T.cos(ds, d_b_at_a)}
                # diagnostics, not arms: BOS rows included; content tokens only
                with_bos = T.matched_rows(lengths, skip_first=0)
                rb = T.procrustes(xa[with_bos[0]], xb[with_bos[1]])
                bos = [x[0] for x in T.token_rows(lengths, 0)]
                nb = np.linalg.norm(xa[bos], axis=1).mean()
                no = np.linalg.norm(xa[m_rows[0]], axis=1).mean()
                cr = T.content_rows(lengths, *span)
                rc = T.procrustes(xa[cr[0]], xb[cr[1]])
                row |= {"diag_bos_norm_over_other": float(nb / no),
                        "diag_cos_rot_with_bos_vs_target_own": T.cos(T.transport(d_a, rb), d_b),
                        "n_rows_content": len(cr[0]),
                        "diag_cos_rot_content_vs_target_own": T.cos(T.transport(d_a, rc), d_b),
                        "diag_cos_rot_content_vs_target_at_donor_site": T.cos(
                            T.transport(d_a, rc), d_b_at_a),
                        "diag_resid_content_on_matched": T.fit_residual(am, bm, rc)}
            for name, arrays in maps.items():
                ext = "npz" if len(arrays) > 1 else "npy"
                row[f"map_{name}"] = save_map(a.maps / f"{don}__{tgt}__{name}.{ext}", **arrays)
            row["seconds"] = time.time() - t0
            pairs[f"{don}->{tgt}"] = row
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in row.items() if k.startswith(("cos", "resid", "diag"))
                              or k in ("target", "donor")}))
    assert len(lengths_seen) == 1
    out = {"prompt": "018", "fit_set": str(FIT.relative_to(ROOT)), "fit_sha256": sha(FIT),
           "selection_sha256": sha(SELECTION), "layers_extracted": fit_layers(),
           "rows": "every token of each fit prompt except BOS; the donor's own-site layer "
                   "(resid_pre) in both twins; centred per model",
           "mismatch": f"prompt derangement from default_rng({T.SEED}); within a pair, "
                       "independent random position permutations over the shorter prompt, "
                       "BOS excluded",
           "standardized": "rows divided by their per-dimension std after centring (Bosco "
                           "App. A.9); the direction moves as unit(((unit(d) / sigma_A) R) "
                           "* sigma_B)",
           "template_span": {"n_prefix_with_bos": span[0], "n_suffix": span[1]},
           "pairs": pairs, "git_commit": commit, "command": provenance.command()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")


# ---- ablate (GPU) ---------------------------------------------------------------------------


def arm_direction(target: str, arm: str, procrustes_json: Path | None) -> tuple[np.ndarray, dict]:
    """The unit direction removed from `target` under `arm`, and where it came from."""
    don = T.donors(target)[T.DONOR_OF_ARM[arm]]
    d_a, src = own_direction(don)
    info = {"arm": arm, "target": target, "donor": don, "donor_site": src, "map": None}
    which = T.MAP_OF_ARM[arm]
    d = T.transport(d_a, None)
    if which:
        pj = json.loads(procrustes_json.read_text())
        assert pj["selection_sha256"] == sha(SELECTION), "maps fitted under another selection"
        assert pj["fit_sha256"] == sha(FIT), "maps fitted on another fit set"
        pair = pj["pairs"][f"{don}->{target}"]
        assert pair["fit_layer"] == src["layer"], "map fitted at another layer than the donor's"
        m = pair[f"map_{which}"]
        arrays = load_map(m)
        d = (T.transport_standardized(d_a, arrays["r"], arrays["sa"], arrays["sb"])
             if which == "standardized" else T.transport(d_a, arrays["r"]))
        info["map"] = {"kind": which, **m, "procrustes_json": str(procrustes_json),
                       "procrustes_json_sha256": sha(procrustes_json)}
    info["direction_sha256"] = Dn.array_sha256(d)
    return d, info


def cmd_direction_sha(a) -> None:
    """Print the sha of the direction an (target, arm) cell would remove (skip-if-done)."""
    print(arm_direction(a.target, a.arm, a.procrustes)[1]["direction_sha256"])


def cmd_ablate(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert a.out.is_absolute(), "--out must be absolute (D036)"
    commit = provenance.git_commit()
    t0 = time.time()
    s = specs()[a.target]
    ckpt = Path(s["checkpoint"])
    assert str(ckpt.resolve()) == s["checkpoint_resolved"]
    r, info = arm_direction(a.target, a.arm, a.procrustes)

    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32).eval()
    source_dtype = str(json.loads((ckpt / "config.json").read_text()).get("torch_dtype"))
    if torch.cuda.is_available():
        model = model.cuda()
    probe = Dn.encode(tok, [x["prompt"] for x in evalset.load_eval()[:16]])
    checks = Dn.orthogonalize_checkpoint(model, tok, r, ckpt, a.out, probe, a.tol)
    prov = {"prompt": "018", "kind": a.arm, **info, "source_model_id": a.target,
            "source_checkpoint": str(ckpt), "source_resolved": s["checkpoint_resolved"],
            "source_dtype": source_dtype, "saved_dtype": "torch.float32",
            "served_dtype": SERVED_DTYPE, **checks,
            "selection_sha256": sha(SELECTION),
            "seconds": time.time() - t0, "git_commit": commit, "command": provenance.command()}
    (a.out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    np.save(a.out / "direction.npy", r)
    print(json.dumps({k: prov[k] for k in ("kind", "source_model_id", "donor",
                                           "check_max_abs_dlogit_vs_hooks",
                                           "check_max_abs_dlogit_hooks_vs_base")}))


# ---- analyze (CPU) --------------------------------------------------------------------------


def labels_path(mid: str, arm: str, prompt_id: str = "018") -> Path:
    if arm == "base":
        return RUNS / mid / "006" / "judge_v1" / "labels.jsonl"
    if arm == "own":
        return RUNS / f"{mid}-ablate-ownsite" / "022" / "judge_v1" / "labels.jsonl"
    if arm == "random":
        return RUNS / f"{mid}-ablate-random" / "007" / "judge_v1" / "labels.jsonl"
    return RUNS / f"{mid}-ablate-{arm}" / prompt_id / "judge_v1" / "labels.jsonl"


def per_prompt(rows: list[dict], ids: list[str], field: str) -> tuple[np.ndarray, dict]:
    from castor import reliability

    y, counts = sanityb.outcome_matrix(rows, ids, field)
    return reliability.per_prompt_p(y), counts | {"share_nan": float(np.isnan(y).mean())}


def cmd_analyze(a) -> None:
    assert a.out.is_absolute()
    commit = provenance.git_commit()
    panel = evalset.load_eval()
    ids = {s: sanityb.subset_ids(panel, s) for s in SUBSETS}
    if a.limit:  # dry runs: the first N eval prompts only
        keep = {r["id"] for r in panel[: a.limit]}
        ids = {s: [i for i in v if i in keep] for s, v in ids.items()}
    assert ids["plain_harmful"], f"no plain-harmful prompt among the first {a.limit} (--limit)"
    tgts = a.targets or T.targets()
    arms = a.arms or list(T.ARMS)
    per, src, pp_all = {}, {}, {}
    for m in tgts:
        per[m], pp_all[m] = {}, {}
        for arm in ("base", "own", "random", *arms):
            p = labels_path(m, arm, a.prompt_id)
            if arm == "random" and not p.exists():
                continue
            rows = list(read_jsonl(p))
            src[f"{m}:{arm}"] = {"path": str(p), "sha256": sha(p)}
            if arm not in ("base",):
                ck = json.loads((p.parent.parent / "ablation_checkpoint.json").read_text())
                assert ck["source_model_id"] == m, p
                want = {"own": "own-site", "random": "random"}.get(arm, arm)
                assert ck["kind"] == want, (p, ck["kind"])
                src[f"{m}:{arm}"] |= {"donor": ck.get("donor", m),
                                      "direction_sha256": ck.get("direction_sha256"),
                                      "map": ck.get("map")}
            per[m][arm], pp_all[m][arm] = {}, {}
            for s, sid in ids.items():
                v, c = per_prompt(rows, sid, "wg_vulnerable")
                h, _ = per_prompt(rows, sid, "wg_harmful")
                if not a.limit:
                    assert c["missing"] == 0, f"{m} {arm} {s}: incomplete labels"
                per[m][arm][s] = {"vulnerability": float(np.nanmean(v)),
                                  "harmful": float(np.nanmean(h)),
                                  "unusable": c["share_nan"], "missing": c["missing"]}
                pp_all[m][arm][s] = v
    ratios, inv = {}, {}
    for m in tgts:
        pp = pp_all[m]
        ratios[m] = {s: T.bootstrap_ratio({k: pp[k][s] for k in ("base", "own", *arms)}, arms,
                                          a.n_boot) for s in SUBSETS if len(ids[s])}
        if "random" in pp:
            ratios[m]["plain_harmful"]["random"] = {"T": T.ratio(
                per[m]["random"]["plain_harmful"]["vulnerability"],
                per[m]["base"]["plain_harmful"]["vulnerability"],
                per[m]["own"]["plain_harmful"]["vulnerability"])}
        if "diff-unrot" in arms:
            v = {k: per[m][k]["plain_harmful"]["vulnerability"] for k in ("base", "own",
                                                                          "diff-unrot")}
            inv[m] = {"diff_unrot_minus_base": v["diff-unrot"] - v["base"],
                      "sign_flip": bool(v["diff-unrot"] < v["base"]),
                      "cross_exceeds_within": bool(abs(v["diff-unrot"] - v["base"])
                                                   > abs(v["own"] - v["base"]))}
    point = {m: {arm: ratios[m]["plain_harmful"][arm]["T"] for arm in arms} for m in tgts}
    secondary = {}  # diff-rot-std beside diff-rot, no rule
    if {"diff-rot", "diff-mismatch", "diff-rot-std"} <= set(arms):
        pairs = json.loads(a.procrustes.read_text())["pairs"]
        for m in tgts:
            pr = pairs[f"{T.donors(m)['diff']}->{m}"]
            t = ratios[m]["plain_harmful"]
            secondary[m] = {
                "diff_rot": t["diff-rot"], "diff_rot_std": t["diff-rot-std"],
                "diff_rot_minus_mismatch": t["diff-rot"]["T"] - t["diff-mismatch"]["T"],
                "diff_rot_std_minus_mismatch": t["diff-rot-std"]["T"] - t["diff-mismatch"]["T"],
                "cos_vs_target_own": {"diff-rot": pr["cos_rot_vs_target_own"],
                                      "diff-rot-std": pr["cos_rot_standardized_vs_target_own"]},
                "cos_vs_target_at_donor_site": {
                    "diff-rot": pr["cos_rot_vs_target_at_donor_site"],
                    "diff-rot-std": pr["cos_rot_standardized_vs_target_at_donor_site"]}}
        src["procrustes_json"] = {"path": str(a.procrustes), "sha256": sha(a.procrustes)}
    summary = {arm: {"mean": float(np.mean([point[m][arm] for m in tgts])),
                     "median": float(np.median([point[m][arm] for m in tgts]))} for arm in arms}
    out = {"prompt": "018", "primary": "T on plain-harmful (wg_vulnerable, K = 20, judge_v1)",
           "transfer": ratios, "summary_plain_harmful": summary, "per_model": per,
           "inversion": inv, "diff_rot_std_secondary": secondary, "n_prompts": {s: len(x) for s, x in ids.items()},
           "limit": a.limit, "prompt_id": a.prompt_id, "sources": src,
           "git_commit": commit, "command": provenance.command(), "time": time.time()}
    if set(T.RULE_ARMS) <= set(arms) and len(tgts) == 12 and not a.limit:
        out["verdict"] = T.verdict(point)  # reads the four registered arms only
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({"summary": summary, "verdict": out.get("verdict", {}).get("reading"),
                      "T": {m[11:]: {k: round(v, 3) for k, v in r.items()}
                            for m, r in point.items()}}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    e = sub.add_parser("extract")
    e.add_argument("--out-root", type=Path, required=True)
    e.add_argument("--model-ids", nargs="*")
    e.add_argument("--max-tokens", type=int, default=65536)
    f = sub.add_parser("fit")
    f.add_argument("--acts-root", type=Path, required=True)
    f.add_argument("--maps", type=Path, default=W / "activations" / "018" / "maps")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--targets", nargs="*", choices=T.targets(), help="smoke tests: a subset")
    for name in ("ablate", "direction-sha"):
        b = sub.add_parser(name)
        b.add_argument("--target", required=True, choices=T.targets())
        b.add_argument("--arm", required=True, choices=T.ARMS)
        b.add_argument("--procrustes", type=Path, default=ROOT / "runs" / "018" /
                       "procrustes.json")
        if name == "ablate":
            b.add_argument("--out", type=Path, required=True)
            b.add_argument("--tol", type=float, default=1e-2)
    n = sub.add_parser("analyze")
    n.add_argument("--out", type=Path, required=True)
    n.add_argument("--targets", nargs="*")
    n.add_argument("--arms", nargs="*", choices=T.ARMS)
    n.add_argument("--prompt-id", default="018")
    n.add_argument("--limit", type=int, default=None, help="dry runs: first N eval prompts")
    n.add_argument("--n-boot", type=int, default=1000)
    n.add_argument("--procrustes", type=Path, default=ROOT / "runs" / "018" / "procrustes.json")
    a = ap.parse_args()
    {"prep": cmd_prep, "extract": cmd_extract, "fit": cmd_fit, "ablate": cmd_ablate,
     "direction-sha": cmd_direction_sha, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
