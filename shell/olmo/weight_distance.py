"""How far apart are two checkpoints, relative to a seed-to-seed distance?

Runs in the OLMo env. Loads OLMo model.pt files and/or released hf_olmo checkpoints (REPO@REV).

Pair mode: the global relative L2 distance ||a − b|| / ||b|| for every pair of --ckpt.
  uv run --frozen --project $CASTOR_WORK/third_party/olmo-env python shell/olmo/weight_distance.py \
      --ckpt ours=<run>/ckpt/step1250-unsharded/model.pt \
      --ckpt def=allenai/DataDecide-dclm-baseline-150M@step1250-seed-default --out d.json

Trajectory mode: at every --steps N, our <run>/ckpt/stepN-unsharded/model.pt against the
released REPO@stepN-seed-<ref> for each --refs, plus the released refs against each other (the
different-seed reference). Per pair: ||a − b|| / ||b|| (b = the later name: released, and the
first --refs for the released pair), global cosine, and per-tensor cosine (all tensors in the
JSON; mean / median / min in the TSV). One JSON per step under --outdir (skipped when it exists
with the same checkpoint specs and resolved HF commits), then <outdir>/../divergence.tsv from all
of them. Steps whose checkpoint does not exist yet are skipped with a message, so reruns fill in.
--peer NAME=RUNDIR adds another of our runs at the same step as pair ours~NAME (b = the peer).
  uv run --frozen --no-sync --project $CASTOR_WORK/third_party/olmo-env python shell/olmo/weight_distance.py \
      --run /work/harrychang08/castor/runs/003/castor-dclm-1B-i6198-d6198 \
      --repo allenai/DataDecide-dclm-baseline-1B --refs default large-aux-2 --steps 0 2500 5000
"""
import argparse
import itertools
import json
import math
import statistics
import subprocess
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file

RELEASED_1B_STEPS = [*range(0, 67501, 2500), 69369]


def load(spec: str) -> dict[str, torch.Tensor]:
    if "@" in spec:
        repo, rev = spec.split("@")
        path = Path(snapshot_download(repo, revision=rev))
        sd = {}
        for f in sorted(path.glob("*.safetensors")):
            sd.update({k.removeprefix("model."): v.float() for k, v in load_file(str(f)).items()})
        return sd
    return {k: v.float() for k, v in torch.load(spec, map_location="cpu").items()}


def compare(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict:
    """Relative L2 ||a − b|| / ||b||, global cosine, per-tensor cosine (fp64 accumulation)."""
    assert a.keys() == b.keys(), sorted(set(a) ^ set(b))
    num = den = dot = na2 = 0.0
    per = {}
    for k in sorted(a):
        x, y = a[k].double(), b[k].double()
        assert x.shape == y.shape, (k, x.shape, y.shape)
        d, xx, yy = (x * y).sum().item(), (x * x).sum().item(), (y * y).sum().item()
        num += ((x - y) ** 2).sum().item()
        den, dot, na2 = den + yy, dot + d, na2 + xx
        per[k] = d / (xx * yy) ** 0.5 if xx > 0 and yy > 0 else float("nan")
    cos = [v for v in per.values() if not math.isnan(v)]
    return {"rel_l2": (num / den) ** 0.5, "cos": dot / (na2 * den) ** 0.5,
            "tensor_cos_mean": statistics.fmean(cos), "tensor_cos_median": statistics.median(cos),
            "tensor_cos_min": min(cos), "tensor_cos": per}


def git_commit() -> str:
    here = Path(__file__).resolve().parent
    run = lambda *c: subprocess.run(["git", "-C", str(here), *c], capture_output=True, text=True,
                                    check=True).stdout.strip()
    if run("status", "--porcelain", "--", str(Path(__file__).resolve())):
        raise SystemExit("weight_distance.py: uncommitted changes; commit before measuring")
    return run("rev-parse", "HEAD")


def trajectory(a: argparse.Namespace) -> None:
    run = Path(a.run)
    outdir = Path(a.outdir) if a.outdir else run / "divergence"
    outdir.mkdir(parents=True, exist_ok=True)
    commit, api = git_commit(), HfApi()
    for step in a.steps:
        ours = run / "ckpt" / f"step{step}-unsharded" / "model.pt"
        if not ours.exists():
            print(f"step {step}: {ours} not there yet, skipped", flush=True)
            continue
        peers = {n: Path(d) / "ckpt" / f"step{step}-unsharded" / "model.pt"
                 for n, d in (p.split("=", 1) for p in a.peer)}
        if missing := [str(p) for p in peers.values() if not p.exists()]:
            raise SystemExit(f"step {step}: peer checkpoint missing: {missing}")
        specs = ({"ours": str(ours)} | {n: str(p) for n, p in peers.items()}
                 | {r: f"{a.repo}@step{step}-seed-{r}" for r in a.refs})
        shas = {r: api.repo_info(a.repo, revision=specs[r].split("@")[1]).sha for r in a.refs}
        f = outdir / f"step{step}.json"
        if f.exists():
            old = json.loads(f.read_text())
            if old["ckpts"] == specs and old["hf_sha"] == shas:
                print(f"step {step}: {f} exists, skipped", flush=True)
                continue
        sds = {n: load(s) for n, s in specs.items()}
        res = {"step": step, "ckpts": specs, "hf_sha": shas, "git_commit": commit, "pairs": {}}
        for n in [*peers, *a.refs]:
            res["pairs"][f"ours~{n}"] = compare(sds["ours"], sds[n])
        for n1, n2 in itertools.combinations(a.refs, 2):
            res["pairs"][f"{n2}~{n1}"] = compare(sds[n2], sds[n1])
        del sds
        f.write_text(json.dumps(res, indent=1) + "\n")
        print(f"step {step}: " + ", ".join(f"{p} L2 {v['rel_l2']:.4f} cos {v['cos']:.4f}"
                                         for p, v in res["pairs"].items()), flush=True)
    write_tsv(outdir, outdir.parent / "divergence.tsv")


def write_tsv(outdir: Path, out: Path) -> None:
    cols = ["rel_l2", "cos", "tensor_cos_mean", "tensor_cos_median", "tensor_cos_min"]
    rows = ["\t".join(["step", "pair", *cols, "git_commit"])]
    for f in sorted(outdir.glob("step*.json"), key=lambda p: int(p.stem[4:])):
        r = json.loads(f.read_text())
        rows += ["\t".join([str(r["step"]), p, *(f"{v[c]:.6f}" for c in cols), r["git_commit"]])
                 for p, v in r["pairs"].items()]
    out.write_text("\n".join(rows) + "\n")
    print(f"wrote {out} ({len(rows) - 1} rows)", flush=True)


def pairs(a: argparse.Namespace) -> None:
    sds = {n: load(s) for n, s in (c.split("=", 1) for c in a.ckpt)}
    res = {"ckpts": dict(c.split("=", 1) for c in a.ckpt), "rel_l2": {}}
    for (na, sa), (nb, sb) in itertools.combinations(sds.items(), 2):
        assert sa.keys() == sb.keys()
        num = sum(((sa[k] - sb[k]) ** 2).sum().item() for k in sa)
        den = sum((sb[k] ** 2).sum().item() for k in sb)
        res["rel_l2"][f"{na}~{nb}"] = (num / den) ** 0.5
        print(f"{na} vs {nb}: {(num / den) ** 0.5:.4f}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", help="pair mode: name=path-or-REPO@REV")
    ap.add_argument("--out", type=Path, help="pair mode: output JSON")
    ap.add_argument("--run", help="trajectory mode: run dir with ckpt/step<N>-unsharded/model.pt")
    ap.add_argument("--repo", default="allenai/DataDecide-dclm-baseline-1B")
    ap.add_argument("--refs", nargs="+", default=["default", "large-aux-2"],
                    help="released seeds (REV = step<N>-seed-<ref>); the first is the L2 denominator")
    ap.add_argument("--steps", nargs="+", type=int, default=RELEASED_1B_STEPS)
    ap.add_argument("--peer", action="append", default=[],
                    help="trajectory mode: NAME=RUNDIR, another of our runs compared at each step")
    ap.add_argument("--outdir", help="trajectory mode: per-step JSON dir (default <run>/divergence)")
    a = ap.parse_args()
    if a.run:
        trajectory(a)
    else:
        assert a.ckpt and a.out, "pair mode needs --ckpt ... --out"
        pairs(a)


if __name__ == "__main__":
    main()
