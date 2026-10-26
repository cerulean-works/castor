"""One twin on one GPU: activations, candidate directions, and the selection scores of Arditi et
al. (2024).

    uv run --frozen python scripts/extract_007.py --models data/manifests/mech_007.json \
        --model-id dd-dclm-1b-default-dpo-seed1 \
        --out /work/harrychang08/castor/activations/007/dd-dclm-1b-default-dpo-seed1 \
        [--dtype bfloat16|float32] [--panel-every K] [--no-panel]

Everything here is identical for every model in the manifest: the extraction
prompts (data/directions/extraction_v1.jsonl, sha in its MANIFEST), the refusal-token set
(data/directions/refusal_tokens_v1.json), the chat formatting of the instrument
(castor.evalset.prompt_ids), and the evaluated panel (castor.evalset.load_eval).

Writes to --out (absolute; /work, never git):
  extraction.safetensors  acts_{harmful,harmless}_{train,val}: (n, L, 8, d) resid_pre at
                          positions -8 … -1 (bf16 values; float32 for --dtype float32)
  candidates.safetensors  mean_diff (L, 7, d) float64 over post-instruction positions -7 … -1
  scores.json             bypass / induce / kl (L, 7) means and per-prompt arrays, baselines,
                          baseline refusal scores of every extraction prompt (for the per-model
                          filter sensitivity), tail-token check, provenance
  panel.safetensors       acts (6226, L, 8, d) in panel order + ids (unless --no-panel)
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from castor import directions as Dn
from castor import evalset, provenance

ROOT = Path(__file__).resolve().parent.parent
EXTRACTION = ROOT / "data" / "directions" / "extraction_v1.jsonl"
EXTRACTION_MANIFEST = ROOT / "data" / "directions" / "MANIFEST.json"
TOKENS = ROOT / "data" / "directions" / "refusal_tokens_v1.json"
POSITIONS = tuple(range(-Dn.N_STORED, 0))  # -8 … -1; the last 7 are post-instruction


def load_extraction() -> dict[tuple[str, str], list[dict]]:
    data = EXTRACTION.read_bytes()
    m = json.loads(EXTRACTION_MANIFEST.read_text())
    assert hashlib.sha256(data).hexdigest() == m["sha256"], "extraction prompts changed"
    out: dict = {}
    for line in data.decode().splitlines():
        r = json.loads(line)
        out.setdefault((r["class"], r["split"]), []).append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--panel-every", type=int, default=1, help="every K-th panel prompt (smoke)")
    ap.add_argument("--no-panel", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=65536, help="padded tokens per batch")
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (D036)"
    commit = provenance.git_commit()
    t0 = time.time()

    spec = next(m for m in json.loads(a.models.read_text())["models"]
                if m["model_id"] == a.model_id)
    ckpt = spec["checkpoint"]
    resolved = str(Path(ckpt).resolve())
    assert resolved == spec["checkpoint_resolved"], f"{ckpt} now points to {resolved}"
    dtype = getattr(torch, a.dtype)
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=dtype).cuda().eval()
    pad = tok.pad_token_id
    n_layer = model.config.num_hidden_layers

    toks = json.loads(TOKENS.read_text())
    refusal = toks["tokens"]
    suffix = Dn.post_instruction_ids(tok)
    assert len(suffix) == Dn.N_POST, f"post-instruction tokens {suffix}"

    ex = load_extraction()
    ids = {k: Dn.encode(tok, [r["prompt"] for r in v]) for k, v in ex.items()}
    tail_mismatch = {f"{c}_{s}": sum(x[-Dn.N_POST:] != suffix for x in v)
                     for (c, s), v in ids.items()}

    # activations of the extraction prompts at every layer, positions -8 … -1
    acts = {k: Dn.activations(model, v, pad, POSITIONS, a.max_tokens) for k, v in ids.items()}
    a.out.mkdir(parents=True, exist_ok=True)
    save_dtype = torch.bfloat16 if a.dtype == "bfloat16" else torch.float32
    save_file({f"acts_{c}_{s}": t.to(save_dtype).contiguous() for (c, s), t in acts.items()},
              str(a.out / "extraction.safetensors"))

    # candidates: difference of means over the training prompts, post-instruction positions
    post = slice(Dn.N_STORED - Dn.N_POST, Dn.N_STORED)
    md = Dn.mean_diff(acts[("harmful", "train")][:, :, post].double().numpy(),
                      acts[("harmless", "train")][:, :, post].double().numpy())  # (L, 7, d)
    save_file({"mean_diff": torch.from_numpy(md).contiguous()},
              str(a.out / "candidates.safetensors"))

    # Arditi's three scores for every candidate
    hv, bv = ids[("harmful", "val")], ids[("harmless", "val")]
    base_h = Dn.last_logits(model, hv, pad, max_tokens=a.max_tokens)
    base_b = Dn.last_logits(model, bv, pad, max_tokens=a.max_tokens)
    shape = (n_layer, Dn.N_POST)
    by = np.zeros(shape)
    ind = np.zeros(shape)
    kl = np.zeros(shape)
    by_pp = np.zeros((*shape, len(hv)))
    ind_pp = np.zeros((*shape, len(bv)))
    kl_pp = np.zeros((*shape, len(bv)))
    for layer in range(n_layer):
        for p in range(Dn.N_POST):
            r = md[layer, p]
            if not np.any(r):  # layer 0 at a template position: no direction (Arditi: NaN)
                by[layer, p] = ind[layer, p] = kl[layer, p] = np.nan
                by_pp[layer, p] = ind_pp[layer, p] = kl_pp[layer, p] = np.nan
                continue
            pre, post_h = Dn.ablation_hooks(model, r)
            s = Dn.refusal_score(Dn.last_logits(model, hv, pad, pre, post_h, a.max_tokens),
                                 refusal).cpu().numpy()
            lb = Dn.last_logits(model, bv, pad, pre, post_h, a.max_tokens)
            k = Dn.kl_div(base_b, lb).cpu().numpy()
            pre_a, _ = Dn.actadd_hooks(model, layer, r)
            i = Dn.refusal_score(Dn.last_logits(model, bv, pad, pre_a, (), a.max_tokens),
                                 refusal).cpu().numpy()
            by[layer, p], ind[layer, p], kl[layer, p] = s.mean(), i.mean(), k.mean()
            by_pp[layer, p], ind_pp[layer, p], kl_pp[layer, p] = s, i, k
    scores = {"bypass": by.tolist(), "induce": ind.tolist(), "kl": kl.tolist()}
    own = Dn.select_own(scores)

    # baseline refusal scores of every extraction prompt (per-model filter sensitivity)
    base_scores = {f"{c}_{s}": Dn.refusal_score(Dn.last_logits(model, v, pad,
                                                               max_tokens=a.max_tokens),
                                                refusal).cpu().numpy().tolist()
                   for (c, s), v in ids.items()}

    out = {"model_id": a.model_id, "checkpoint": ckpt, "checkpoint_resolved": resolved,
           "dtype": a.dtype, "n_layer": n_layer, "positions_stored": list(POSITIONS),
           "post_instruction_positions": list(range(-Dn.N_POST, 0)),
           "post_instruction_tokens": [tok.decode([t]) for t in suffix],
           "tail_mismatch": tail_mismatch, "refusal_tokens": refusal,
           "scores": scores, "pass": Dn.pass_matrix(scores).tolist(),
           "own_selection": None if own is None else {"layer": own[0], "pos_index": own[1],
                                                       "position": own[1] - Dn.N_POST},
           "baseline": {"refusal_harmful_val": float(Dn.refusal_score(base_h, refusal).mean()),
                        "refusal_harmless_val": float(Dn.refusal_score(base_b, refusal).mean())},
           "per_prompt": {"bypass": by_pp.tolist(), "induce": ind_pp.tolist(),
                          "kl": kl_pp.tolist()},
           "baseline_refusal_scores": base_scores,
           "extraction_sha256": json.loads(EXTRACTION_MANIFEST.read_text())["sha256"],
           "refusal_tokens_sha256": hashlib.sha256(TOKENS.read_bytes()).hexdigest()}

    if not a.no_panel:
        rows = evalset.load_eval()
        rows = rows[:: a.panel_every]
        pids = Dn.encode(tok, [r["prompt"] for r in rows])
        out["panel_tail_mismatch"] = sum(x[-Dn.N_POST:] != suffix for x in pids)
        pa = Dn.activations(model, pids, pad, POSITIONS, a.max_tokens)
        save_file({"acts": pa.to(save_dtype).contiguous()}, str(a.out / "panel.safetensors"),
                  metadata={"ids": json.dumps([r["id"] for r in rows])})
        out["panel"] = {"n": len(rows), "every": a.panel_every,
                        "eval_sha256": json.loads(evalset.EVAL_MANIFEST.read_text())["eval"]
                        ["sha256"]}
    out |= {"seconds": time.time() - t0, "git_commit": commit,
            "command": provenance.command()}
    (a.out / "scores.json").write_text(json.dumps(out) + "\n")
    print(json.dumps({k: out[k] for k in ("model_id", "own_selection", "baseline",
                                          "tail_mismatch", "seconds")}))


if __name__ == "__main__":
    main()
