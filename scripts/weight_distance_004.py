"""Weight-space distances among the 004 models (diagnostic, not pre-registered).

    uv run --frozen python scripts/weight_distance_004.py --out runs/posttrain/004/weight_distance.json
    # 010: parent = ingredient 3, ours from models/posttrain/olmo2-1b-ingr3
    uv run --frozen python scripts/weight_distance_004.py --out $PWD/runs/posttrain/010/weight_distance.json \
        --base-revision f83cd641d4f00954eb25c781e0ac360debb3239e \
        --local-root /work/harrychang08/castor/models/posttrain/olmo2-1b-ingr3

For every pair of models, the global L2 norm of the parameter difference (fp32, all tensors), and
per model the norm of its update from its parent (SFT: base OLMo-2-0425-1B at --base-revision,
default a1847dff = main; DPO: its own SFT). If Ai2's release were one more post-training seed of our
pipeline, ||SFT_ours1 - SFT_ai2|| would be about ||SFT_ours1 - SFT_ours2|| and the update norms
would match. Before loading, each local SFT's effective open-instruct config must name the base repo
at --base-revision, and each DPO's must name its own SFT (check_parents), so a missing flag cannot
measure the update from the wrong parent. Tensors are memory-mapped and compared one at a time
(CPU).
"""

import argparse
import itertools
import json
import os
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors import safe_open

from castor import provenance

M = "/work/harrychang08/castor/models/posttrain/olmo2-1b"
HF = {
    "base": ("allenai/OLMo-2-0425-1B", "a1847dff35000b4271fa70afc5db10fd29fedbdf"),
    "sft_ai2": ("allenai/OLMo-2-0425-1B-SFT", "0d85a3d037876ce6ac7d4311d994400fc66ac27f"),
    "dpo_ai2": ("allenai/OLMo-2-0425-1B-DPO", "c4b0485961ab24c2433b090f3b922f0913a9290f"),
}
LOCAL = ("sft1", "sft2", "dpo1", "dpo2")  # <local-root>/<stage>/seed<k>/model
PARENT = {"sft1": "base", "sft2": "base", "sft_ai2": "base",
          "dpo1": "sft1", "dpo2": "sft2", "dpo_ai2": "sft_ai2"}


def check_parents(local_root: str, base: tuple[str, str]) -> dict:
    """Assert every local model was trained from the parent PARENT names; return the parents
    as recorded in <local-root>/<stage>/seed<k>/oi_config.yaml (what open-instruct was given)."""
    out = {}
    for k in LOCAL:
        cfg = yaml.safe_load(Path(f"{local_root}/{k[:3]}/seed{k[3]}/oi_config.yaml").read_text())
        got = (cfg["model_name_or_path"], cfg.get("model_revision", "main"))
        if PARENT[k] == "base":
            want = base
        else:
            p = PARENT[k]
            want = (os.path.realpath(f"{local_root}/{p[:3]}/seed{p[3]}/model"), got[1])
            got = (os.path.realpath(got[0]), got[1])
        assert got == want, f"{k} was trained from {got}, expected {want} (--base-revision?)"
        out[k] = got
    return out


def check_extra(spec: str, local_root: str, hf: dict) -> tuple[str, str, str, tuple]:
    """--extra KEY=RUN_DIR:PARENT_KEY: one more post-trained model at RUN_DIR/model whose
    open-instruct config must name PARENT_KEY's checkpoint (an HF repo@revision in HF, or a
    local <stage><k> under --local-root). Returns (key, model path, parent key, parent)."""
    key, _, rest = spec.partition("=")
    run, _, parent = rest.rpartition(":")
    assert key and run and parent, f"bad --extra {spec!r}: KEY=RUN_DIR:PARENT_KEY"
    assert key not in HF and key not in LOCAL, f"--extra key {key} clashes with a built-in model"
    cfg = yaml.safe_load(Path(run, "oi_config.yaml").read_text())
    got = (cfg["model_name_or_path"], cfg.get("model_revision", "main"))
    if parent in hf:
        want = hf[parent]
    else:
        assert parent in LOCAL, f"--extra {key}: unknown parent {parent}"
        want = (os.path.realpath(f"{local_root}/{parent[:3]}/seed{parent[3]}/model"), got[1])
        got = (os.path.realpath(got[0]), got[1])
    assert got == want, f"{key} was trained from {got}, expected {parent} = {want}"
    return key, os.path.realpath(f"{run}/model"), parent, got


def open_state(path: str) -> dict:
    """name -> tensor accessor (lazy) for a bin or safetensors checkpoint dir."""
    d = Path(path)
    st = sorted(d.glob("*.safetensors"))
    if st:
        out = {}
        for f in st:
            h = safe_open(str(f), "pt")
            out |= {k: (lambda h=h, k=k: h.get_tensor(k)) for k in h.keys()}  # noqa: SIM118 (safe_open handle)
        return out
    bins = sorted(d.glob("pytorch_model*.bin"))
    out = {}
    for f in bins:
        sd = torch.load(f, map_location="cpu", mmap=True, weights_only=True)
        out |= {k: (lambda sd=sd, k=k: sd[k]) for k in sd}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-revision", default=HF["base"][1],
                    help="OLMo-2-0425-1B revision our SFTs were trained from (010: f83cd641)")
    ap.add_argument("--local-root", default=M, help="<root>/<stage>/seed<k>/model")
    ap.add_argument("--extra", action="append", default=[], metavar="KEY=RUN_DIR:PARENT_KEY",
                    help="another post-trained model and its parent's key (012)")
    a = ap.parse_args()
    commit = provenance.git_commit()
    hf = HF | {"base": (HF["base"][0], a.base_revision)}
    parents = check_parents(a.local_root, hf["base"])
    extra = [check_extra(e, a.local_root, hf) for e in a.extra]
    parent_of = PARENT | {k: p for k, _, p, _ in extra}
    parents |= {k: g for k, _, _, g in extra}
    paths = {k: snapshot_download(r, revision=v, allow_patterns=["*.json", "*.bin",
                                                                 "*.safetensors"])
             for k, (r, v) in hf.items()}
    paths |= {k: os.path.realpath(f"{a.local_root}/{k[:3]}/seed{k[3]}/model") for k in LOCAL}
    paths |= {k: p for k, p, _, _ in extra}
    states = {k: open_state(p) for k, p in paths.items()}
    names = sorted(states["base"])
    for k, s in states.items():
        assert sorted(s) == names, f"{k}: parameter names differ from base"
    pairs = list(itertools.combinations(states, 2))
    sq = {p: 0.0 for p in pairs}
    norm2 = {k: 0.0 for k in states}
    for n in names:
        t = {k: s[n]().float() for k, s in states.items()}
        for x, y in pairs:
            sq[(x, y)] += float((t[x] - t[y]).pow(2).sum())
        for k in states:
            norm2[k] += float(t[k].pow(2).sum())
    dist = {f"{x}|{y}": v ** 0.5 for (x, y), v in sq.items()}

    def d(x, y):
        return dist.get(f"{x}|{y}", dist.get(f"{y}|{x}"))

    res = {"distance": dist, "param_norm": {k: v ** 0.5 for k, v in norm2.items()},
           "update_from_parent": {k: d(k, p) for k, p in parent_of.items()},
           "paths": paths, "trained_from": parents, "revisions": {k: v for k, (_, v) in hf.items()},
           "git_commit": commit, "command": provenance.command()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("distance", "update_from_parent")}, indent=1))


if __name__ == "__main__":
    main()
