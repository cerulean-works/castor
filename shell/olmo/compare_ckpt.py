"""Compare an OLMo unsharded checkpoint (model.pt) bit for bit with a released DataDecide one.

Runs in the OLMo env.
  uv run --frozen --project $CASTOR_WORK/third_party/olmo-env python shell/olmo/compare_ckpt.py \
      --ours $RUN/step0-unsharded/model.pt --repo allenai/DataDecide-dclm-baseline-150M \
      --revisions step0-seed-default step0-seed-small-aux-2 --out cmp.json
"""
import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


def load_released(repo: str, rev: str) -> dict[str, torch.Tensor]:
    path = Path(snapshot_download(repo, revision=rev, local_files_only=True))
    out = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(str(shard), "pt") as f:
            for k in f.keys():  # noqa: SIM118  (safe_open is not a dict)
                out[k.removeprefix("model.")] = f.get_tensor(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revisions", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ours = {k: v.float() for k, v in torch.load(a.ours, map_location="cpu").items()}
    res = {"ours": str(a.ours), "repo": a.repo, "vs": {}}
    for rev in a.revisions:
        ref = load_released(a.repo, rev)
        assert set(ref) == set(ours), set(ref) ^ set(ours)
        per = {k: {"equal": bool(torch.equal(ref[k], ours[k])),
                   "max_abs_diff": (ref[k] - ours[k]).abs().max().item()} for k in ref}
        n_eq = sum(v["equal"] for v in per.values())
        res["vs"][rev] = {"tensors_equal": n_eq, "tensors": len(per), "exact": n_eq == len(per),
                          "max_abs_diff": max(v["max_abs_diff"] for v in per.values())}
        print(rev, res["vs"][rev], flush=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
