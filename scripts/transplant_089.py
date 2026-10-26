"""Projection transplant (089): dose gate along one model's refusal direction, then a per-prompt
transplant of a donor's projection scalar (castor.transplant).

    # GPU, 1 H200 (--dev): the site check, then one arm per job
    uv run --frozen python scripts/transplant_089.py check-site --model-id A --out <abs>.json
    uv run --frozen python scripts/transplant_089.py generate --model-id A --dose K
    uv run --frozen python scripts/transplant_089.py generate --model-id A --donor B \
        [--permute] [--site 11,-2]
    # CPU
    uv run --frozen python scripts/transplant_089.py analyze --step dose --out <abs>/dose.json
    uv run --frozen python scripts/transplant_089.py analyze --step transplant \
        --out <abs>/transplant.json --tsv <abs>/closure.tsv
    # 091 (D129): arms under <model>/091/, then the twelve-pair analysis
    uv run --frozen python scripts/transplant_089.py generate --prompt-id 091 --model-id A ...
    uv run --frozen python scripts/transplant_089.py analyze --step replication \
        --out <abs>/transplant.json --tsv <abs>/closure.tsv

Inputs: data/manifests/mech_021.json (checkpoints, labels of record), 021's activations
(/work/harrychang08/castor/activations/021/<model>/{candidates,panel}.safetensors).
Prompts: the adversarial-harmful subset of the evaluated panel (3,212), panel order.
proj_M(p) is read from 021's panel.safetensors (scores.json, which the prompt names, holds no
per-prompt panel projections; note 089, Deviation 1).

Generation = the sampling parameters of record (castor.generate.D017: T 1.0, top_p 1.0, no
top-k, 512 new tokens, K = 20, stop at EOS, bf16, prompt ids from castor.evalset) through HF
`generate` with castor.transplant.Addition on block `layer`'s input; vLLM cannot take a
per-position hook. No greedy pass (P uses the K samples only). Sampling noise: torch's global
generator seeded per batch from (arm id, batch index). Rows follow castor.judge.common with
model = "<model-id>+089/<arm>", written to <out-root>/<model-id>/089/<arm>/samples.jsonl;
judge with shell/nchc/judge_stages.sh into <arm>/judge_v1/ (shell/mech/transplant_089.sh).
"""

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from castor import directions as Dn
from castor import evalset, generate, posttrain, provenance, sanityb
from castor import transplant as T
from castor.judge.common import read_jsonl

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifests" / "mech_021.json"
ACTS = Path("/work/harrychang08/castor/activations/021")
RUNS = Path("/work/harrychang08/castor/runs")
PROMPT_ID = "089"
SUBSET = "adv_harmful"
FIELD = "wg_vulnerable"
A_DOSE = "dd-dclm-baseline-1b-default-dpo-seed1"
PAIRS = (("default", "large-aux-2"), ("large-aux-2", "large-aux-3"), ("large-aux-3", "default"))
# 091 (D129): A_MZ is A's other post-training seed (seed 1 -> 2 as in 089, 3 <-> 4)
MZ_SEED = {1: 2, 3: 4, 4: 3}
REPLICATION_ID = "091"
REPLICATION = {"reversed": tuple((b, a, 1) for a, b in PAIRS),
               "seed3": tuple((a, b, 3) for a, b in PAIRS),
               "seed4": tuple((a, b, 4) for a, b in PAIRS)}
CLOSURE_089 = 0.483  # 089's registered mean closure, the 091 prediction's reference
WITHIN = 0.10


def mid(base: str, seed: int = 1) -> str:
    return f"dd-dclm-baseline-1b-{base}-dpo-seed{seed}"


def pair_name(a_base: str, b_base: str, seed: int) -> str:
    return f"{a_base}<-{b_base}" + ("" if seed == 1 else f"@seed{seed}")


def replication_pairs() -> list[dict]:
    """The twelve directed pairs of 091: 089's three (arms under 089/) and the nine new ones
    (arms under 091/). Identity arms: 089's dose_+0 for every seed-1 A, 091's for seeds 3, 4."""
    out = []
    groups = [("089", "089", tuple((a, b, 1) for a, b in PAIRS))]
    groups += [(g, REPLICATION_ID, ps) for g, ps in REPLICATION.items()]
    for group, pid, ps in groups:
        for a_base, b_base, seed in ps:
            A, B = mid(a_base, seed), mid(b_base, seed)
            ident = "089" if seed == 1 else REPLICATION_ID
            out.append({"name": pair_name(a_base, b_base, seed), "group": group, "seed": seed,
                        "a": A, "b": B, "mz_a": mid(a_base, MZ_SEED[seed]),
                        "arms": {"transplant": f"{A}/{pid}/transplant_{B}",
                                 "identity": f"{A}/{ident}/{T.dose_key(0.0)}"}})
    return out


def parse_site(s: str) -> tuple[int, int]:
    layer, pos = (int(x) for x in s.split(","))
    assert -Dn.N_POST <= pos <= -1, "the site is a post-instruction position"
    return layer, pos


def spec_of(model_id: str) -> dict:
    return next(m for m in json.loads(MANIFEST.read_text())["models"] if m["model_id"] == model_id)


def subset_rows() -> list[dict]:
    return [r for r in evalset.load_eval() if sanityb.SUBSETS[SUBSET](r)]


def subset_ids() -> list[str]:
    return sanityb.subset_ids(evalset.load_eval(), SUBSET)


def site_quantities(model_id: str, site: tuple[int, int], acts_root: Path = ACTS) -> dict:
    """d̂ from 021's candidates (float64 mean_diff), projections of 021's stored panel
    activations on it over the subset, μ and σ."""
    from safetensors import safe_open

    layer, pos = site
    d = acts_root / model_id
    with safe_open(str(d / "candidates.safetensors"), "np") as f:
        md = f.get_tensor("mean_diff")[layer, pos + Dn.N_POST]
    direction = Dn.unit(md)
    j = pos + Dn.N_STORED
    with safe_open(str(d / "panel.safetensors"), "pt") as f:
        ids = json.loads(f.metadata()["ids"])
        x = f.get_slice("acts")[:, layer:layer + 1, j:j + 1, :][:, 0, 0, :].double().numpy()
    want = subset_ids()
    pos_of = {p: i for i, p in enumerate(ids)}
    proj = Dn.project(x[[pos_of[p] for p in want]], direction)
    mu, sigma = T.site_stats(proj)
    return {"direction": direction, "ids": want, "proj": proj, "mu": mu, "sigma": sigma,
            "z": T.zscore(proj, mu, sigma), "layer": layer, "position": pos,
            "candidates_sha256": posttrain.sha256_file(d / "candidates.safetensors")}


# ---- GPU --------------------------------------------------------------------------------------


def load_model(model_id: str, cudnn_sdp: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = spec_of(model_id)
    ckpt = spec["checkpoint"]
    resolved = str(Path(ckpt).resolve())
    assert resolved == spec["checkpoint_resolved"], f"{ckpt} now points to {resolved}"
    # cuDNN's SDPA kernel fails on some large left-padded decode batches (all seven dose arms
    # of the first launch, batch 42 of 88); flash / memory-efficient attention take over
    torch.backends.cuda.enable_cudnn_sdp(cudnn_sdp)
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16).cuda().eval()
    return tok, model, spec


def prompt_batches(lengths: list[int], k: int, max_tokens: int, max_seqs: int) -> list[list[int]]:
    """Prompt indices by length (descending), grouped so that a batch's k copies of each
    prompt fit max_tokens padded (prompt + new) tokens and max_seqs sequences."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    out, cur, cur_len = [], [], 0
    new = generate.D017.max_new_tokens
    for i in order:
        n = max(cur_len, lengths[i])
        if cur and ((len(cur) + 1) * k * (n + new) > max_tokens or (len(cur) + 1) * k > max_seqs):
            out.append(cur)
            cur, n = [], lengths[i]
        cur.append(i)
        cur_len = n
    if cur:
        out.append(cur)
    return out


def read_parts(path: Path | None) -> dict[int, dict]:
    """Completed batches of an interrupted run (a torn last line is dropped)."""
    done: dict[int, dict] = {}
    if path is None or not path.exists():
        return done
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[r["batch"]] = r
    return done


def sample_hooked(model, tok, ids: list[list[int]], coeffs: np.ndarray, hook: "T.Addition",
                  layer: int, arm_id: str, max_tokens: int, max_seqs: int,
                  parts: Path | None = None, stamp: dict | None = None, commit: str = "",
                  log=print):
    """K samples per prompt under the addition hook. Returns per prompt a list of K
    (token ids without EOS, finish_reason), the number of batches resumed from `parts`, and
    the commits that made the batches (one line per completed batch; the batches and their
    seeds are fixed by the prompt list, so a resumed run draws what an uninterrupted one
    would). A stored batch is reused only if its `stamp` (everything that shapes the draw
    except the code commit) equals this run's."""
    import torch
    from transformers import GenerationConfig

    cfg = generate.D017
    eos, pad = tok.eos_token_id, tok.pad_token_id
    gcfg = GenerationConfig(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
                            top_k=0, max_new_tokens=cfg.max_new_tokens, eos_token_id=eos,
                            pad_token_id=pad, bos_token_id=None)
    out: list = [None] * len(ids)
    batches = prompt_batches([len(x) for x in ids], cfg.k, max_tokens, max_seqs)
    done = read_parts(parts)
    for bi, r in done.items():
        assert r["prompts"] == batches[bi], f"{parts}: batch {bi} is not this run's batch"
        assert r.get("stamp") == stamp, f"{parts}: batch {bi} was made under other settings"
        for i, res in zip(r["prompts"], r["res"], strict=True):
            out[i] = [tuple(x) for x in res]
    t0 = time.time()
    with torch.inference_mode(), Dn.hooks([(Dn.blocks(model)[layer], hook)]):
        for bi, b in enumerate(batches):
            if bi in done:
                continue
            rows = [ids[i] for i in b for _ in range(cfg.k)]
            x = Dn.batch_inputs(rows, pad, model.device)
            hook.coeff = torch.tensor(np.repeat(coeffs[b], cfg.k), dtype=torch.float32)
            torch.manual_seed(int.from_bytes(hashlib.sha256(f"089:{arm_id}:{bi}".encode())
                                             .digest()[:4], "big") % 2**31)
            seq = model.generate(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                                 generation_config=gcfg)
            gen = seq[:, x["input_ids"].shape[1]:].cpu().tolist()
            for j, i in enumerate(b):
                res = []
                for g in gen[j * cfg.k:(j + 1) * cfg.k]:
                    if eos in g:
                        res.append((g[:g.index(eos)], "stop"))
                    else:
                        res.append((g, "length"))
                out[i] = res
            if parts is not None:
                with open(parts, "a") as f:
                    f.write(json.dumps({"batch": bi, "prompts": b, "stamp": stamp,
                                        "git_commit": commit,
                                        "res": [out[i] for i in b]}) + "\n")
            if bi % 10 == 0 or bi == len(batches) - 1:
                log(json.dumps({"batch": bi + 1, "of": len(batches), "prompts": len(b),
                                "seconds": round(time.time() - t0, 1)}), flush=True)
    commits = {r.get("git_commit", "") for r in done.values()}
    if len(done) < len(batches):
        commits.add(commit)
    return out, len(done), sorted(commits)


def cmd_generate(a) -> None:
    import torch
    import transformers

    assert a.out_root.is_absolute(), "--out-root must be absolute (D036)"
    commit = provenance.git_commit()
    site = parse_site(a.site)
    qa = site_quantities(a.model_id, site)
    n = len(qa["ids"])
    if a.dose is not None:
        assert a.donor is None and not a.permute
        coeffs = T.dose_coeffs(n, a.dose, qa["sigma"])
        arm = T.dose_key(a.dose)
        donor = None
    else:
        qb = site_quantities(a.donor, site)
        assert qb["ids"] == qa["ids"]
        zb = T.permuted(qb["z"]) if a.permute else qb["z"]
        coeffs = T.transplant_coeffs(qa["z"], zb, qa["sigma"])
        arm = ("permuted_" if a.permute else "transplant_") + a.donor
        donor = {"model_id": a.donor, "mu": qb["mu"], "sigma": qb["sigma"],
                 "candidates_sha256": qb["candidates_sha256"]}
    if site != T.SITE:
        arm = f"site{site[0]}{site[1]}_" + arm
    if a.limit is not None:
        arm += f"_limit{a.limit}"  # a smoke never writes into a full arm
    out = a.out_root / a.model_id / a.prompt_id / arm
    prov_path = out / "provenance.json"
    if prov_path.exists() and a.limit is None:
        p = json.loads(prov_path.read_text())
        if p.get("limit") is None and p.get("complete"):
            print(f"generation exists: {out}")
            return
    rows = subset_rows()
    assert [r["id"] for r in rows] == qa["ids"]
    if a.limit:
        rows, coeffs = rows[: a.limit], coeffs[: a.limit]
    tok, model, spec = load_model(a.model_id)
    cudnn_sdp = bool(torch.backends.cuda.cudnn_sdp_enabled())
    kind = generate.template_kind(tok.chat_template, posttrain.tulu_template(True),
                                  posttrain.tulu_template(False))
    ids = [evalset.prompt_ids(tok, r["prompt"]) for r in rows]
    assert all(len(x) + generate.D017.max_new_tokens <= generate.CONTEXT for x in ids)
    hook = T.Addition(qa["direction"], site[1])
    out.mkdir(parents=True, exist_ok=True)
    parts = None if a.limit else out / "parts.jsonl"
    stamp = {"coeffs_sha256": Dn.array_sha256(np.asarray(coeffs, np.float64)),
             "direction_sha256": Dn.array_sha256(qa["direction"]), "mu": qa["mu"],
             "sigma": qa["sigma"], "cudnn_sdp": cudnn_sdp, "site": list(site),
             "sampling": generate.config_dict(), "transformers": transformers.__version__,
             "torch": torch.__version__, "max_tokens": a.max_tokens, "max_seqs": a.max_seqs}
    t0 = time.time()
    res, n_resumed, batch_commits = sample_hooked(
        model, tok, ids, coeffs, hook, site[0], f"{a.model_id}/{arm}", a.max_tokens, a.max_seqs,
        parts, stamp, commit)
    secs = time.time() - t0
    model_tag = f"{a.model_id}+{a.prompt_id}/{arm}"
    with open(out / "samples.jsonl", "w") as f:
        for r, samples in zip(rows, res, strict=True):
            f.writelines(json.dumps({
                "model": model_tag, "prompt_id": r["id"], "sample": k, "prompt": r["prompt"],
                "response": tok.decode(g, skip_special_tokens=True), "n_tokens": len(g),
                "finish_reason": fr}, ensure_ascii=False) + "\n" for k, (g, fr) in enumerate(samples))
    prov = {"model_id": a.model_id, "model_tag": model_tag, "arm": arm,
            "checkpoint": spec["checkpoint"], "checkpoint_resolved": spec["checkpoint_resolved"],
            "chat_template": kind, "site": {"layer": site[0], "position": site[1]},
            "hook": "forward pre-hook on block layer (resid_pre): + c_p * d_hat at the last "
                    f"{-site[1]} prompt position(s) and at every generated position",
            "direction_sha256": Dn.array_sha256(qa["direction"]),
            "projection_source": "021 panel.safetensors acts[:, layer, pos] . d_hat "
                                 "(scores.json has no per-prompt panel projections; note 089, "
                                 "Deviation 1)",
            "candidates_sha256": qa["candidates_sha256"], "mu": qa["mu"], "sigma": qa["sigma"],
            "dose_k": a.dose, "donor": donor, "permute": a.permute,
            "permute_seed": T.PERMUTE_SEED if a.permute else None,
            "coeffs": {"mean": float(np.mean(coeffs)), "sd": float(np.std(coeffs)),
                       "min": float(np.min(coeffs)), "max": float(np.max(coeffs)),
                       "sha256": Dn.array_sha256(np.asarray(coeffs, np.float64))},
            "hook_calls": hook.calls, "n_batches_resumed": n_resumed,
            "batch_commits": batch_commits, "cudnn_sdp": cudnn_sdp,
            "sampling": generate.config_dict() | {"engine": "transformers.generate",
                                                  "greedy": False, "top_k": 0},
            "transformers": transformers.__version__,
            "torch": torch.__version__, "max_tokens": a.max_tokens, "max_seqs": a.max_seqs,
            "seed_rule": "torch.manual_seed(sha256('089:<model-id>/<arm>:<batch>')[:4] % 2**31)",
            "subset": SUBSET, "n_prompts": len(rows), "limit": a.limit, "seconds": secs,
            "prompt_id": a.prompt_id, "out": str(out),
            "complete": True, "eval_manifest": json.loads(evalset.EVAL_MANIFEST.read_text())["eval"],
            "git_commit": commit, "command": provenance.command()}
    prov_path.write_text(json.dumps(prov, indent=1) + "\n")
    print(json.dumps({k: prov[k] for k in ("arm", "out", "n_prompts", "seconds", "hook_calls")}))


def cmd_check_site(a) -> None:
    """(1) 021's extraction path rerun on the full panel reproduces the stored projections;
    (2) the generation path's prefill (this script's batching, coefficient 0) reproduces them
    to bf16 batching noise; (3) a coefficient c moves the site projection by c and the hook
    fires on prefill and on every decode step."""
    import torch

    assert a.out.is_absolute()
    commit = provenance.git_commit()
    site = parse_site(a.site)
    layer, pos = site
    q = site_quantities(a.model_id, site)
    tok, model, _ = load_model(a.model_id, a.cudnn_sdp)
    pad = tok.pad_token_id
    panel = evalset.load_eval()
    sub = {p: i for i, p in enumerate(q["ids"])}
    j = pos + Dn.N_STORED
    # (1) extraction path, every panel prompt, positions −8 … −1, 021's batching
    pids = Dn.encode(tok, [r["prompt"] for r in panel])
    acts = Dn.activations(model, pids, pad, tuple(range(-Dn.N_STORED, 0)))
    x1 = acts[:, layer, j].to(torch.bfloat16).double().numpy()
    idx = [i for i, r in enumerate(panel) if r["id"] in sub]
    p1 = Dn.project(x1[idx], q["direction"])
    d1 = np.abs(p1 - q["proj"])
    del acts
    # (2) generation path: the prefill inside model.generate (positions from the mask, as in
    # sampling), this script's batches, hook at coefficient 0 and 0.5 σ, capture after it
    rows = subset_rows()
    ids = [evalset.prompt_ids(tok, r["prompt"]) for r in rows]
    hook = T.Addition(q["direction"], pos)
    store: dict = {}

    def cap(module, args, kwargs):
        xx = args[0] if args else kwargs["hidden_states"]
        if int(kwargs["cache_position"][0]) == 0:
            store["x"] = xx[:, pos, :].detach().float()

    blk = Dn.blocks(model)[layer]
    p2 = np.zeros(len(ids))
    p3 = np.zeros(len(ids))
    c3 = 0.5 * q["sigma"]
    batches = prompt_batches([len(x) for x in ids], 1, a.max_tokens, a.max_seqs)
    from transformers import GenerationConfig

    one = GenerationConfig(do_sample=False, max_new_tokens=1, eos_token_id=tok.eos_token_id,
                           pad_token_id=pad)
    with torch.inference_mode(), Dn.hooks([(blk, hook), (blk, cap)]):
        for b in batches:
            x = Dn.batch_inputs([ids[i] for i in b], pad, model.device)
            for c, dest in ((0.0, p2), (c3, p3)):
                hook.coeff = torch.full((len(b),), c, dtype=torch.float32)
                model.generate(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                               generation_config=one)
                dest[b] = Dn.project(store["x"].to(torch.bfloat16).double().cpu().numpy(),
                                     q["direction"])
    d2 = np.abs(p2 - q["proj"])
    shift = p3 - p2
    # (3b) the hook on generate: prefill once, one decode call per new token
    hook.calls = {"prefill": 0, "decode": 0}
    hook.coeff = torch.zeros(4, dtype=torch.float32)
    x = Dn.batch_inputs(ids[:4], pad, model.device)
    with torch.inference_mode(), Dn.hooks([(blk, hook)]):
        seq = model.generate(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                             generation_config=GenerationConfig(
                                 do_sample=False, max_new_tokens=8, min_new_tokens=8,
                                 eos_token_id=tok.eos_token_id, pad_token_id=pad))
    out = {"model_id": a.model_id, "site": {"layer": layer, "position": pos},
           "cudnn_sdp": a.cudnn_sdp,
           "hook_point": f"forward pre-hook on model.model.layers[{layer}] (resid_pre), "
                         f"position {pos} of the left-padded prompt",
           "mu": q["mu"], "sigma": q["sigma"], "n_prompts": len(q["ids"]),
           "extraction_path": {"max_abs_diff": float(d1.max()), "mean_abs_diff": float(d1.mean()),
                               "pass_1e-4": bool(d1.max() < 1e-4)},
           "generation_path_prefill": {"max_abs_diff": float(d2.max()),
                                       "mean_abs_diff": float(d2.mean()),
                                       "max_abs_diff_over_sigma": float(d2.max() / q["sigma"]),
                                       "corr": float(np.corrcoef(p2, q["proj"])[0, 1])},
           "addition": {"coeff": c3, "shift_mean": float(shift.mean()),
                        "shift_max_abs_err": float(np.abs(shift - c3).max())},
           "generate_hook_calls": hook.calls, "generate_new_tokens": int(seq.shape[1]
                                                                        - x["input_ids"].shape[1]),
           "git_commit": commit, "command": provenance.command()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


# ---- CPU: analysis ----------------------------------------------------------------------------


def matrix(path: Path, ids: list[str]) -> tuple[np.ndarray, dict]:
    return sanityb.outcome_matrix(read_jsonl(path), ids, FIELD)


def arm_labels(model_id: str, arm: str, runs: Path = RUNS) -> Path:
    return sanityb.labels_path(model_id, f"{PROMPT_ID}/{arm}", runs)


def record_labels(model_id: str) -> Path:
    return Path(spec_of(model_id)["labels"])


def cmd_analyze_dose(a) -> None:
    commit = provenance.git_commit()
    ids = subset_ids()
    ys, counts, inputs = {}, {}, {}
    for k in T.DOSES:
        p = arm_labels(A_DOSE, T.dose_key(k))
        ys[T.dose_key(k)], counts[T.dose_key(k)] = matrix(p, ids)
        inputs[T.dose_key(k)] = {"path": str(p), "sha256": posttrain.sha256_file(p)}
    rec = record_labels(A_DOSE)
    ys["record"], counts["record"] = matrix(rec, ids)
    inputs["record"] = {"path": str(rec), "sha256": posttrain.sha256_file(rec)}
    for key, c in counts.items():
        assert c["missing"] == 0, f"{key}: {c['missing']} label cells missing"

    dstat = T.dose_stat()

    def stat(y: dict) -> dict:
        out = dstat(y)
        out["identity"] = T.corrected(y[T.dose_key(0.0)], y["record"])
        out["record_mean_p"] = T.mean_p(y["record"])
        out["identity_level_diff"] = T.mean_p(y[T.dose_key(0.0)]) - out["record_mean_p"]
        out["r_c_vs_record"] = {key: T.corrected(y[key], y["record"])["r_corrected"]
                                for key in y if key != "record"}
        out["reliability"] = {key: T.corrected(y[key], y[key])["rel_a"] for key in y}
        return out

    ci = T.bootstrap(ys, stat, a.n_boot)
    mean_by = {k: ci["mean_p"][T.dose_key(k)]["point"] for k in T.DOSES}
    gate = T.gate_verdict(mean_by, ci["gate_diff"])
    ident = ci["identity"]["r_corrected"]["point"]
    res = {"prompt": PROMPT_ID, "step": "dose", "model_id": A_DOSE, "site": list(T.SITE),
           "doses": list(T.DOSES), "subset": SUBSET, "field": FIELD, "n_prompts": len(ids),
           "n_boot": a.n_boot, "boot_seed": T.BOOT_SEED, "gate": gate,
           "gate_rule": f"mean P strictly decreasing in k over {list(T.DOSES)} and "
                        f"P({T.GATE_DOSES[0]:+g}) - P({T.GATE_DOSES[1]:+g}) >= {T.GATE_MIN} "
                        "with the 95% prompt-bootstrap interval excluding 0",
           "identity_control": {"r_corrected": ident, "reading": T.identity_reading(ident),
                                "predicted_min": T.IDENTITY_PREDICTED,
                                "review_line": T.IDENTITY_REVIEW},
           "ci": ci,
           "nan_share": {key: c["nan"] / c["cells"] for key, c in counts.items()},
           "inputs": inputs, "git_commit": commit, "command": provenance.command()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"gate": gate, "identity": res["identity_control"],
                      "mean_p": mean_by, "nan_share": res["nan_share"]}, indent=1))


ARMS = ("transplant", "permuted", "mzdonor", "site11")


def arm_dirs(a_base: str, b_base: str) -> dict[str, str]:
    b = mid(b_base)
    return {"transplant": f"transplant_{b}", "permuted": f"permuted_{b}",
            "mzdonor": f"transplant_{mid(b_base, 2)}",
            "site11": f"site{T.COMPARE_SITE[0]}{T.COMPARE_SITE[1]}_transplant_{b}"}


def cmd_analyze_transplant(a) -> None:
    commit = provenance.git_commit()
    ids = subset_ids()
    ys, inputs, pairs = {}, {}, []

    def add(key: str, path: Path) -> None:
        if key not in ys:
            y, c = matrix(path, ids)
            assert c["missing"] == 0, f"{key}: {c['missing']} label cells missing"
            ys[key], inputs[key] = y, {"path": str(path), "sha256": posttrain.sha256_file(path)}

    for a_base, b_base in PAIRS:
        A, B = mid(a_base), mid(b_base)
        add(A, record_labels(A))
        add(B, record_labels(B))
        add(mid(a_base, 2), record_labels(mid(a_base, 2)))
        add(mid(b_base, 2), record_labels(mid(b_base, 2)))
        arms = {}
        for arm, d in arm_dirs(a_base, b_base).items():
            arms[arm] = f"{A}/{d}"
            add(arms[arm], arm_labels(A, d))
        arms["identity"] = f"{A}/{T.dose_key(0.0)}"
        add(arms["identity"], arm_labels(A, T.dose_key(0.0)))
        pairs.append({"name": f"{a_base}<-{b_base}", "a": A, "b": B, "mz_a": mid(a_base, 2),
                      "mz_b": mid(b_base, 2), "arms": arms})
    arm_names = (*ARMS, "identity")
    cstat = T.closure_stat(pairs, arm_names)
    btwin = T.closure_stat([p | {"b": p["mz_b"]} for p in pairs], arm_names)

    def stat(y: dict) -> dict:
        out = cstat(y)
        out["vs_b_mz_twin"] = btwin(y)["mean_closure"]
        sec = {}
        for p in pairs:
            row = {"mean_p_a": T.mean_p(y[p["a"]]),
                   "rel_a": T.corrected(y[p["a"]], y[p["a"]])["rel_a"]}
            for arm in arm_names:
                e = y[p["arms"][arm]]
                row[f"mean_p_{arm}"] = T.mean_p(e)
                row[f"rel_{arm}"] = T.corrected(e, e)["rel_a"]
                row[f"r_c_edit_vs_a_{arm}"] = T.corrected(e, y[p["a"]])["r_corrected"]
            sec[p["name"]] = row
        out["secondaries"] = sec
        return out

    ci = T.bootstrap(ys, stat, a.n_boot)
    offsets, z = {}, {}
    for p in pairs:
        for m in (p["a"], p["b"]):
            if m not in z:
                z[m] = site_quantities(m, T.SITE)["z"]
        za, zb = z[p["a"]], z[p["b"]]
        dz = zb - za
        offsets[p["name"]] = {"sd_dz": float(dz.std()), "share_abs_dz_gt_0.5": float(
            (np.abs(dz) > 0.5).mean()), "r_proj": float(np.corrcoef(za, zb)[0, 1])}
    res = {"prompt": PROMPT_ID, "step": "transplant", "pairs": pairs, "arms": list(arm_names),
           "offsets": offsets,
           "site": list(T.SITE), "compare_site": list(T.COMPARE_SITE), "subset": SUBSET,
           "field": FIELD, "n_prompts": len(ids), "n_boot": a.n_boot, "boot_seed": T.BOOT_SEED,
           "predictions": T.predictions(ci), "ci": ci, "inputs": inputs,
           "git_commit": commit, "command": provenance.command()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    if a.tsv:
        lines = ["pair\tarm\tclosure\tlo\thi"]
        for p in pairs:
            for arm in arm_names:
                c = ci["per_pair"][p["name"]][f"closure_{arm}"]
                lines.append(f"{p['name']}\t{arm}\t{c['point']:.4f}\t{c['lo']:.4f}\t{c['hi']:.4f}")
        for arm in arm_names:
            c = ci["mean_closure"][arm]
            lines.append(f"mean\t{arm}\t{c['point']:.4f}\t{c['lo']:.4f}\t{c['hi']:.4f}")
        a.tsv.write_text("\n".join(lines) + "\n")
    print(json.dumps({"predictions": res["predictions"],
                      "mean_closure": {k: v["point"] for k, v in ci["mean_closure"].items()}},
                     indent=1))


def group_means(per_pair: dict, pairs: Sequence[dict], arm: str = "transplant") -> dict:
    """Mean closure of `arm` over each 091 group, the nine new pairs, the six other-seed pairs
    and all twelve; plus the nine-pair mean minus 089's three (same draws)."""
    def mean(sel):
        return float(np.mean([per_pair[p["name"]][f"closure_{arm}"] for p in pairs if sel(p)]))

    out = {g: mean(lambda p, g=g: p["group"] == g) for g in ("089", *REPLICATION)}
    out["new9"] = mean(lambda p: p["group"] != "089")
    out["other_seed6"] = mean(lambda p: p["seed"] != 1)
    out["all12"] = mean(lambda p: True)
    out["new9_minus_089"] = out["new9"] - out["089"]
    return out


def replication_verdict(ci: dict, ref: float = CLOSURE_089, within: float = WITHIN) -> dict:
    """091's prediction: the nine-pair mean within `within` of 089's `ref`, interval excluding
    0; the secondaries in words (each group within `within`; identity r_c(A', A) >= 0.95)."""
    g = ci["groups"]
    new9 = g["new9"]
    out = {"primary": bool(abs(new9["point"] - ref) <= within and new9["lo"] > 0),
           "primary_within": bool(abs(new9["point"] - ref) <= within),
           "primary_ci_excludes_0": bool(new9["lo"] > 0)}
    for k in ("reversed", "seed3", "seed4", "other_seed6"):
        out[f"{k}_within"] = bool(abs(g[k]["point"] - ref) <= within)
    out["identity_ge_0.95"] = {m: bool(v["point"] >= T.IDENTITY_PREDICTED)
                               for m, v in ci["identity_vs_record"].items()}
    return out


def cmd_analyze_replication(a) -> None:
    commit = provenance.git_commit()
    ids = subset_ids()
    pairs = replication_pairs()
    ys, inputs = {}, {}

    def add(key: str, path: Path) -> None:
        if key not in ys:
            y, c = matrix(path, ids)
            assert c["missing"] == 0, f"{key}: {c['missing']} label cells missing"
            ys[key], inputs[key] = y, {"path": str(path), "sha256": posttrain.sha256_file(path)}

    for p in pairs:
        for m in (p["a"], p["b"], p["mz_a"]):
            add(m, record_labels(m))
        for key in p["arms"].values():
            model_id, pid, arm = key.split("/")
            add(key, sanityb.labels_path(model_id, f"{pid}/{arm}", RUNS))
    arm_names = ("transplant", "identity")
    cstat = T.closure_stat(pairs, arm_names)
    a_models = sorted({p["a"] for p in pairs})
    ident_of = {p["a"]: p["arms"]["identity"] for p in pairs}

    def stat(y: dict) -> dict:
        out = cstat(y)
        out["groups"] = group_means(out["per_pair"], pairs)
        out["groups_identity"] = group_means(out["per_pair"], pairs, "identity")
        out["identity_vs_record"] = {m: T.corrected(y[ident_of[m]], y[m])["r_corrected"]
                                     for m in a_models}
        sec = {}
        for p in pairs:
            e = y[p["arms"]["transplant"]]
            sec[p["name"]] = {"mean_p_a": T.mean_p(y[p["a"]]), "mean_p_transplant": T.mean_p(e),
                              "rel_a": T.corrected(y[p["a"]], y[p["a"]])["rel_a"],
                              "rel_transplant": T.corrected(e, e)["rel_a"],
                              "r_c_edit_vs_a_transplant": T.corrected(e, y[p["a"]])["r_corrected"]}
        out["secondaries"] = sec
        return out

    ci = T.bootstrap(ys, stat, a.n_boot)
    offsets, z = {}, {}
    for p in pairs:
        for m in (p["a"], p["b"]):
            if m not in z:
                z[m] = site_quantities(m, T.SITE)["z"]
        za, zb = z[p["a"]], z[p["b"]]
        dz = zb - za
        offsets[p["name"]] = {"sd_dz": float(dz.std()), "share_abs_dz_gt_0.5": float(
            (np.abs(dz) > 0.5).mean()), "r_proj": float(np.corrcoef(za, zb)[0, 1])}
    res = {"prompt": REPLICATION_ID, "step": "replication", "pairs": pairs,
           "arms": list(arm_names), "mz_seed": {str(k): v for k, v in MZ_SEED.items()},
           "reference_closure_089": CLOSURE_089, "within": WITHIN,
           "verdict": replication_verdict(ci), "offsets": offsets, "site": list(T.SITE),
           "subset": SUBSET, "field": FIELD, "n_prompts": len(ids), "n_boot": a.n_boot,
           "boot_seed": T.BOOT_SEED, "ci": ci, "inputs": inputs, "git_commit": commit,
           "command": provenance.command()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    if a.tsv:
        lines = ["pair\tgroup\tarm\tclosure\tlo\thi"]
        for p in pairs:
            for arm in arm_names:
                c = ci["per_pair"][p["name"]][f"closure_{arm}"]
                lines.append(f"{p['name']}\t{p['group']}\t{arm}\t{c['point']:.4f}\t"
                             f"{c['lo']:.4f}\t{c['hi']:.4f}")
        for g, c in ci["groups"].items():
            lines.append(f"mean\t{g}\ttransplant\t{c['point']:.4f}\t{c['lo']:.4f}\t{c['hi']:.4f}")
        a.tsv.write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": res["verdict"],
                      "groups": {k: v["point"] for k, v in ci["groups"].items()}}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("generate")
    g.add_argument("--model-id", required=True)
    g.add_argument("--dose", type=float, default=None)
    g.add_argument("--donor", default=None)
    g.add_argument("--permute", action="store_true")
    g.add_argument("--site", default="13,-1")
    g.add_argument("--out-root", type=Path, default=RUNS)
    g.add_argument("--prompt-id", default=PROMPT_ID, choices=(PROMPT_ID, REPLICATION_ID),
                   help="arm directory <model>/<id>/; the instrument (seed rule included) is 089's")
    g.add_argument("--limit", type=int, default=None, help="first N prompts (smoke)")
    g.add_argument("--max-tokens", type=int, default=600_000, help="padded tokens per batch")
    g.add_argument("--max-seqs", type=int, default=1280)
    c = sp.add_parser("check-site")
    c.add_argument("--model-id", default=A_DOSE)
    c.add_argument("--site", default="13,-1")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--max-tokens", type=int, default=65536)
    c.add_argument("--max-seqs", type=int, default=256)
    c.add_argument("--cudnn-sdp", action="store_true",
                   help="021's attention backend (sampling runs with it off)")
    an = sp.add_parser("analyze")
    an.add_argument("--step", choices=("dose", "transplant", "replication"), required=True)
    an.add_argument("--out", type=Path, required=True)
    an.add_argument("--tsv", type=Path, default=None)
    an.add_argument("--n-boot", type=int, default=T.N_BOOT)
    a = ap.parse_args()
    if a.cmd == "analyze":
        assert a.out.is_absolute() and (a.tsv is None or a.tsv.is_absolute()), \
            "--out / --tsv must be absolute (D036)"
    if a.cmd == "generate":
        assert (a.dose is None) != (a.donor is None), "one of --dose / --donor"
        cmd_generate(a)
    elif a.cmd == "check-site":
        cmd_check_site(a)
    elif a.step == "dose":
        cmd_analyze_dose(a)
    elif a.step == "transplant":
        cmd_analyze_transplant(a)
    else:
        cmd_analyze_replication(a)


if __name__ == "__main__":
    main()
