"""Diagnostic: the global fp32 L2 between two DPO checkpoints computed two ways on the real
files (per-tensor `equiv_024.l2`, mmap, one tensor at a time; flattened `confeff.l2_flat`) and
against a float64 reference, so the implementations are checked on a 1.28 B parameter pair, not
only on the synthetic tensors of tests/test_confeff.py. Dev job only: two flattened vectors
(~5 GB each) at a time.

    uv run --frozen python scripts/check_l2_027.py --out /abs/runs/027/check_l2.json \
        [--a <model dir> --b <model dir>]
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

from castor import confeff, provenance, sanityb

M = sanityb.MODELS_ROOT / "dd-dclm-1B-seed-default"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--a", type=Path, default=M / "dpo" / "seed1" / "model")
    ap.add_argument("--b", type=Path, default=M / "dpo" / "seed2" / "model")
    args = ap.parse_args()
    assert args.out.is_absolute()
    spec = importlib.util.spec_from_file_location(
        "equiv_024", Path(__file__).resolve().parent / "equiv_024.py")
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    t0 = time.time()
    sa, sb = E.load_state(args.a), E.load_state(args.b)
    n_a = sum(v.numel() for v in sa.values())
    dtypes = sorted({str(v.dtype) for v in sa.values()})
    l2_024 = E.l2(sa, sb)
    t1 = time.time()
    fa, fb = confeff.flatten(sa), confeff.flatten(sb)
    l2_027 = confeff.l2_flat(fa, fb)
    t2 = time.time()
    # float64 reference, per tensor: sum of squares of the fp64 difference (the ground truth
    # both fp32 methods are checked against), with each tensor's fp32 per-tensor sum beside it
    import torch

    per_tensor, ss64 = {}, 0.0
    for n in sorted(sa):
        d64 = sa[n].double() - sb[n].double()
        t64 = float(torch.sum(d64 * d64))
        t32 = float(torch.sum((sa[n].float() - sb[n].float()) ** 2))
        per_tensor[n] = {"numel": sa[n].numel(), "ss_fp64": t64, "ss_fp32": t32,
                         "rel_err_fp32": abs(t32 - t64) / t64 if t64 else 0.0}
        ss64 += t64
        del d64
    l2_64 = ss64 ** 0.5
    t3 = time.time()
    # the flat method with fp64 accumulation in chunks
    d = fa[1] - fb[1]
    l2_flat64 = float(sum(float(torch.sum(c.double() * c.double())) for c in torch.split(d, 1 << 26))) ** 0.5
    del d
    t4 = time.time()
    worst = sorted(per_tensor.items(), key=lambda kv: -kv[1]["rel_err_fp32"])[:5]
    res = {"a": str(args.a), "a_resolved": str(args.a.resolve()), "b": str(args.b),
           "b_resolved": str(args.b.resolve()), "n_tensors": len(sa), "n_params": n_a,
           "dtypes": dtypes, "l2_fp64_reference": l2_64, "l2_equiv_024": l2_024,
           "l2_confeff_flat": l2_027, "l2_flat_fp64_chunked": l2_flat64,
           "rel_err_024": abs(l2_024 - l2_64) / l2_64, "rel_err_027": abs(l2_027 - l2_64) / l2_64,
           "rel_err_flat64": abs(l2_flat64 - l2_64) / l2_64,
           "per_tensor": per_tensor, "seconds_024": t1 - t0, "seconds_027": t2 - t1,
           "seconds_fp64": t3 - t2, "seconds_flat64": t4 - t3, "script": "scripts/check_l2_027.py",
           "git_commit": provenance.git_commit(), "command": provenance.command()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"L2 fp64 reference {l2_64:.4f}; 024-style {l2_024:.4f} (rel err {res['rel_err_024']:.2e}); "
          f"027-style {l2_027:.4f} (rel err {res['rel_err_027']:.2e}); flat fp64 chunked {l2_flat64:.4f} "
          f"(rel err {res['rel_err_flat64']:.2e}); {len(sa)} tensors, {n_a:,} params, dtypes {dtypes}")
    for n, v in worst:
        print(f"  worst fp32 per-tensor: {n} numel {v['numel']:,} ss64 {v['ss_fp64']:.6g} ss32 {v['ss_fp32']:.6g} rel {v['rel_err_fp32']:.2e}")


if __name__ == "__main__":
    main()
