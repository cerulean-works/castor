"""Does a released checkpoint "remember" our data order? (recency test)

A language model has lower loss on the batches it trained on most recently. So if our data
order for seed s equals Ai2's, the released seed-s checkpoint at step T scores lower on our
order-s batches from steps (T − W, T] than on the same-position batches of any other order, and
than on batches no run has reached yet. The matrix model × order answers whether our data
pipeline (mix file list, file contents, global-index shuffle) matches Ai2's, independently of
GPU numerics. Order s = OLMo's global indices for data seed s (numpy PCG64 shuffle of all
instances, olmo/data/iterable_dataset.py), rebuilt here with the same code path; for seed 6198
the rebuilt indices are checked against a training run's own global_indices.npy.

Runs in the OLMo env.
  uv run --frozen --project $CASTOR_WORK/third_party/olmo-env python shell/olmo/recency_test.py \
      --config $CASTOR_WORK/runs/003/eval/config-150M.yaml --step 1250 --window 50 \
      --orders 6198 14 15 1001 --check-indices 6198=<run>/ckpt/train_data/global_indices.npy \
      --model default=allenai/DataDecide-dclm-baseline-150M@step1250-seed-default ... --out r.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from olmo.model import OLMo
from safetensors.torch import load_file


def load(spec: str) -> dict[str, torch.Tensor]:
    if "@" in spec:
        repo, rev = spec.split("@")
        path = Path(snapshot_download(repo, revision=rev))
        sd = {}
        for f in sorted(path.glob("*.safetensors")):
            sd.update({k.removeprefix("model."): v for k, v in load_file(str(f)).items()})
        return sd
    return torch.load(spec, map_location="cpu")


def order(n: int, seed: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.uint32)
    np.random.Generator(np.random.PCG64(seed=seed)).shuffle(idx)
    return idx


@torch.no_grad()
def mean_ce(model: OLMo, ds, ids: np.ndarray, bs: int = 32) -> float:
    tot, n = 0.0, 0
    for i in range(0, len(ids), bs):
        x = torch.stack([ds[int(j)]["input_ids"] for j in ids[i:i + bs]]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).logits
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                               x[:, 1:].reshape(-1), reduction="sum")
        tot += loss.item()
        n += x[:, 1:].numel()
    return tot / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--step", type=int, required=True, help="checkpoint step T")
    ap.add_argument("--window", type=int, default=50, help="W steps before T")
    ap.add_argument("--future", type=int, default=30000, help="start step of an unseen window")
    ap.add_argument("--orders", type=int, nargs="+", required=True)
    ap.add_argument("--check-indices", default=None, help="seed=path to a run's global_indices.npy")
    ap.add_argument("--model", action="append", required=True, help="name=path-or-REPO@REV")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cfg = TrainConfig.load(a.config)
    cfg.model.init_device = "cpu"
    ds = build_memmap_dataset(cfg, cfg.data)
    gbs = cfg.global_train_batch_size
    lo, hi = (a.step - a.window) * gbs, a.step * gbs
    windows, check = {}, None
    for s in a.orders:
        o = order(len(ds), s)
        if a.check_indices and a.check_indices.startswith(f"{s}="):
            ref = np.memmap(a.check_indices.split("=", 1)[1], mode="r", dtype=np.uint32)
            check = {"seed": s, "equal": bool(np.array_equal(o[:len(ref)], ref)),
                     "n_ref": len(ref), "n": len(o)}
            print("index check", check, flush=True)
        windows[f"order{s}"] = o[lo:hi]
        if s == a.orders[0]:
            f0 = a.future * gbs
            windows[f"order{s}-future{a.future}"] = o[f0:f0 + (hi - lo)]
    res = {"step": a.step, "window_steps": a.window, "global_batch": gbs, "n_instances": len(ds),
           "index_check": check, "models": dict(m.split("=", 1) for m in a.model), "ce": {}}
    for m in a.model:
        name, spec = m.split("=", 1)
        model = OLMo(cfg.model)
        model.load_state_dict({k: v.float() for k, v in load(spec).items()}, strict=True)
        model = model.cuda().eval()
        res["ce"][name] = {w: mean_ce(model, ds, ids) for w, ids in windows.items()}
        print(name, json.dumps(res["ce"][name]), flush=True)
        del model
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
