"""Bitwise and relative-L2 comparison of same-seed 1B replicates, step by step.

Runs in the OLMo env, like compare_ckpt.py / weight_distance.py, whose compare() it reuses.
For every saved step, every pair of replicates with the same invocation tag (s10: steps 0-10;
l100: steps 0, 10, .., 100), plus the within-replicate pair s10 vs l100 at the shared steps 0 and 10:
number of tensors with any bit difference, max |a - b|, relative L2 ||a - b|| / ||b||, global cosine.
  uv run --frozen --no-sync --project $CASTOR_WORK/third_party/olmo-env python shell/olmo/pairs_025.py \
      --replicates P1 P2 P3 --out /home/harrychang08/castor/runs/025/probe1/pairs.json
"""
import argparse
import importlib.util
import itertools
import json
import subprocess
from pathlib import Path

import torch

RUNS = Path("/work/harrychang08/castor/runs/003")
_spec = importlib.util.spec_from_file_location("weight_distance", Path(__file__).with_name("weight_distance.py"))
wd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wd)


INT_OF = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def bits(t: torch.Tensor) -> torch.Tensor:
    """The tensor's stored bits as integers (so -0.0 != 0.0 and NaN payloads count)."""
    assert t.dtype.is_floating_point, t.dtype
    return t.contiguous().view(INT_OF[t.element_size()])


def bitwise(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict:
    """Tensors that differ in any bit (on the stored dtype, before any cast) and the largest |a - b|."""
    assert a.keys() == b.keys(), sorted(set(a) ^ set(b))
    assert all(a[k].dtype == b[k].dtype for k in a)
    diff = {k: int((bits(a[k]) != bits(b[k])).sum().item()) for k in sorted(a)}
    unequal = [k for k, n in diff.items() if n]
    mx = max(((a[k].double() - b[k].double()).abs().max().item() for k in unequal), default=0.0)
    return {"n_tensors": len(a), "n_unequal": len(unequal), "max_abs": mx,
            "n_unequal_elements": sum(diff.values()),
            "n_elements": sum(v.numel() for v in a.values()), "unequal_tensors": unequal}


def raw(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")


def ckpt(name: str, step: int) -> Path:
    return RUNS / name / "ckpt" / f"step{step}-unsharded" / "model.pt"


def pair_row(step: int, na: str, nb: str) -> dict:
    a, b = raw(ckpt(na, step)), raw(ckpt(nb, step))
    dtypes = sorted({str(v.dtype) for v in a.values()})
    bw = bitwise(a, b)
    c = wd.compare({k: v.float() for k, v in a.items()}, {k: v.float() for k, v in b.items()})
    return {"step": step, "a": na, "b": nb, "dtypes": dtypes, **bw,
            **{k: c[k] for k in ("rel_l2", "cos", "tensor_cos_mean", "tensor_cos_min")}}


def plan(reps: list[str]) -> list[tuple[int, str, str, str]]:
    """(step, a, b, kind) for every comparison; kind = same-tag pair or within-replicate."""
    out = []
    for tag, steps in (("s10", range(11)), ("l100", [0, *range(10, 101, 10)])):
        for s in steps:
            for x, y in itertools.combinations(reps, 2):
                out.append((s, f"025-{x}-{tag}", f"025-{y}-{tag}", f"{x}~{y}"))
    for x in reps:
        for s in (0, 10):
            out.append((s, f"025-{x}-s10", f"025-{x}-l100", f"{x}:s10~l100"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", nargs="+", default=["P1", "P2", "P3"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for s, na, nb, kind in plan(a.replicates):
        r = pair_row(s, na, nb) | {"pair": kind}
        print(f"step {s:3d} {kind:14s} unequal {r['n_unequal']:3d}/{r['n_tensors']} max|d| {r['max_abs']:.3e} "
              f"relL2 {r['rel_l2']:.3e} cos {r['cos']:.8f}", flush=True)
        rows.append(r)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
                            text=True, check=True).stdout.strip()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"rows": rows, "git_commit": commit, "runs_root": str(RUNS)}, indent=1) + "\n")


if __name__ == "__main__":
    main()
