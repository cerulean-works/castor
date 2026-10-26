"""Which base checkpoint is Ai2's released OLMo-2-0425-1B-SFT descended from? (diagnostic)

    uv run --frozen python scripts/sft_parent_probe_004.py --out /abs/path/sft_parent_probe.json

Cosine similarity and L2 distance (float64) of three tensors (token embedding, layer-0 q_proj,
layer-15 MLP down_proj) between the released -SFT@0d85a3d0 (and our SFT seed 1 as a control) and
candidate bases: OLMo-2-0425-1B `main`@a1847dff (our SFT's base), the last stage-1 branch, the three
stage-2 ingredients' final steps, their uniform average ("soup"), and stage-1 step 0 (the
initialisation). Fine-tuning leaves cosine near 1 with the parent and an L2 distance of about
one SFT update (our SFT vs `main` is the reference); unrelated pretraining runs give cosine ~0.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open

from castor import provenance

BASE = "allenai/OLMo-2-0425-1B"
TENSORS = ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
           "model.layers.15.mlp.down_proj.weight"]
M = "/work/harrychang08/castor/models/posttrain/olmo2-1b"


def load(repo: str, revision: str | None = None, local: str | None = None) -> dict:
    if local:
        d = Path(os.path.realpath(local))
        sd = torch.load(d / "pytorch_model.bin", map_location="cpu", mmap=True, weights_only=True)
        return {t: sd[t].double() for t in TENSORS}
    files = [f for f in HfApi().list_repo_files(repo, revision=revision)
             if f.endswith((".safetensors", ".bin"))]
    out = {}
    for f in files:
        p = hf_hub_download(repo, f, revision=revision)
        if f.endswith(".safetensors"):
            with safe_open(p, "pt") as h:
                out |= {t: h.get_tensor(t).double() for t in TENSORS if t in h.keys()}  # noqa: SIM118
        else:
            sd = torch.load(p, map_location="cpu", mmap=True, weights_only=True)
            out |= {t: sd[t].double() for t in TENSORS if t in sd}
    assert set(out) == set(TENSORS), f"{repo}@{revision}: missing {set(TENSORS) - set(out)}"
    return out


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(a.flatten() @ b.flatten() / (a.norm() * b.norm()))


def dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    commit = provenance.git_commit()
    branches = [b.name for b in HfApi().list_repo_refs(BASE).branches]
    stage1 = max((b for b in branches if b.startswith("stage1-step")),
                 key=lambda b: int(b.split("-")[1][4:]))
    cands = {"main@a1847dff": (BASE, "a1847dff35000b4271fa70afc5db10fd29fedbdf"),
             stage1: (BASE, stage1), "stage1-step0-tokens0B": (BASE, "stage1-step0-tokens0B")}
    for i in (1, 2, 3):
        name = max((b for b in branches if b.startswith(f"stage2-ingredient{i}-step")),
                   key=lambda b: int(b.split("-")[2][4:]))
        cands[name] = (BASE, name)
    probes = {"sft_ai2": load("allenai/OLMo-2-0425-1B-SFT",
                              "0d85a3d037876ce6ac7d4311d994400fc66ac27f"),
              "sft_ours_seed1": load("", local=f"{M}/sft/seed1/model")}
    res = {"tensors": TENSORS, "candidates": {k: f"{r}@{v}" for k, (r, v) in cands.items()},
           "cosine": {}, "l2": {}}
    loaded = {name: load(repo, rev) for name, (repo, rev) in cands.items()}
    ingr = [v for k, v in loaded.items() if k.startswith("stage2-ingredient")]
    loaded["soup(ingredient1-3)"] = {t: sum(c[t] for c in ingr) / len(ingr) for t in TENSORS}
    res["candidates"]["soup(ingredient1-3)"] = "uniform mean of the three stage2 ingredients above"
    for name, c in loaded.items():
        res["cosine"][name] = {p: {t: cos(sd[t], c[t]) for t in TENSORS}
                               for p, sd in probes.items()}
        res["l2"][name] = {p: {t: dist(sd[t], c[t]) for t in TENSORS} for p, sd in probes.items()}
        print(name, json.dumps(res["cosine"][name]), json.dumps(res["l2"][name]), flush=True)
    res |= {"git_commit": commit, "command": provenance.command()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
