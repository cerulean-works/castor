"""One orthogonalized checkpoint (own direction or random control).

    uv run --frozen python scripts/ablate_007.py --models data/manifests/mech_007.json \
        --model-id dd-dclm-1b-default-dpo-seed1 --kind own|random \
        --selection runs/007/selection.json --acts-root /work/harrychang08/castor/activations/007 \
        --out /work/harrychang08/castor/models/ablate/007/dd-dclm-1b-default-dpo-seed1-own

own:    the twin's unit direction at the common (layer, position) of selection.json (the
        difference of means in candidates.safetensors, i.e. the direction the ladder reads).
own-site: the same, at the twin's own site, selection.json["own"][model_id]
        (Arditi's per-model procedure). Nothing is re-selected.
random: a random unit direction seeded by the model id (castor.directions.random_direction).
The weights (stored in bfloat16) are loaded in float32, orthogonalized in float64 (Arditi Eq. 5)
and saved in float32 as a plain HF checkpoint, with every non-weight file of the twin's
checkpoint copied unchanged (tokenizer, chat template, generation config), so the unchanged
instrument (vLLM generation; judge_v1) serves it. generate.py loads it in bfloat16, as every twin. Before saving, the orthogonalized model's last-position logits
on 16 panel prompts are compared with the original model under Arditi's ablation hooks
(float32); the max |Δlogit| is recorded and must be below --tol.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from castor import directions as Dn
from castor import evalset, provenance

SERVED_DTYPE = "bfloat16"  # as scripts/generate.py serves every twin
PROMPT_OF_KIND = {"own": "007", "random": "007", "own-site": "022"}


def site_for(sel: dict, kind: str, model_id: str) -> tuple[int, int]:
    """(layer, pos_index) whose difference of means is ablated: the common site for 007's
    kinds (random ignores it), the twin's own committed site for 022's own-site."""
    s = sel["own"][model_id] if kind == "own-site" else sel["common"]
    return s["layer"], s["pos_index"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--kind", required=True, choices=tuple(PROMPT_OF_KIND))
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--acts-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=1e-2)
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (D036)"
    commit = provenance.git_commit()
    t0 = time.time()
    spec = next(m for m in json.loads(a.models.read_text())["models"]
                if m["model_id"] == a.model_id)
    ckpt = Path(spec["checkpoint"])
    assert str(ckpt.resolve()) == spec["checkpoint_resolved"]
    sel = json.loads(a.selection.read_text())
    layer, pos = site_for(sel, a.kind, a.model_id)

    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32).eval()
    source_dtype = str(json.loads((ckpt / "config.json").read_text()).get("torch_dtype"))
    loaded_dtype = str(next(model.parameters()).dtype)
    d = model.config.hidden_size
    cand_path = a.acts_root / a.model_id / "candidates.safetensors"
    cand_sha = None
    if a.kind != "random":
        cand_sha = hashlib.sha256(cand_path.read_bytes()).hexdigest()
        with safe_open(str(cand_path), "pt") as f:
            r = f.get_tensor("mean_diff")[layer, pos].numpy()
    else:
        r = Dn.random_direction(a.model_id, d)
    r = Dn.unit(r)
    if torch.cuda.is_available():
        model = model.cuda()

    probe = Dn.encode(tok, [x["prompt"] for x in evalset.load_eval()[:16]])
    checks = Dn.orthogonalize_checkpoint(model, tok, r, ckpt, a.out, probe, a.tol)
    prov = {"prompt": PROMPT_OF_KIND[a.kind], "kind": a.kind, "source_model_id": a.model_id,
            "source_checkpoint": str(ckpt), "source_resolved": spec["checkpoint_resolved"],
            "site": {"layer": layer, "pos_index": pos, "position": pos - Dn.N_POST},
            "direction_sha256": Dn.array_sha256(r),
            "candidates": str(cand_path) if cand_sha else None, "candidates_sha256": cand_sha,
            "source_dtype": source_dtype, "saved_dtype": loaded_dtype,
            "served_dtype": SERVED_DTYPE, **checks,
            "selection_sha256": hashlib.sha256(a.selection.read_bytes()).hexdigest(),
            "seconds": time.time() - t0, "git_commit": commit, "command": provenance.command()}
    (a.out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    np.save(a.out / "direction.npy", r)
    print(json.dumps(prov))


if __name__ == "__main__":
    main()
