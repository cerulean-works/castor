"""Weight-space divergence of the second replicate R2, with 011's figure code.

Reads R2's divergence.tsv (shell/olmo/weight_distance.py trajectory mode with --peer r1=<R>) and
011's copy of R's TSV, and draws relative L2 and global cosine against step for R2 ~ R (same seed,
same stack), R2 ~ D and R ~ D (same seed, different stack) and released aux-2 ~ D (different
seed). Descriptive, not a gate. Writes divergence.svg, both TSVs and summary.json into --out.

    uv run --frozen python scripts/divergence_035.py \
        --tsv /work/harrychang08/castor/runs/003/castor-dclm-1B-i6198-d6198-r2/divergence.tsv \
        --tsv-011 runs/011/divergence/divergence.tsv --out runs/035/divergence
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from castor import provenance

D011 = provenance.load_script("divergence_011")

# pair key in the merged frame -> (source, pair in that source's TSV)
SOURCE = {"r2~r1": ("035", "ours~r1"), "r2~default": ("035", "ours~default"),
          "r1~default": ("011", "ours~default"), "large-aux-2~default": ("035", "large-aux-2~default")}
LABEL = {"r2~r1": "R2 ~ R (same seed, same stack)",
         "r2~default": "R2 ~ released default (same seed)",
         "r1~default": "R ~ released default (same seed)",
         "large-aux-2~default": "released aux-2 ~ default (different seed)"}
COLOR = {"r2~r1": "#2a78d6", "r2~default": "#1baa7d", "r1~default": "#1baa7d",
         "large-aux-2~default": "#eb6834"}
DASH = {"r1~default"}
SAME_SEED = ["r2~r1", "r2~default", "r1~default"]


def merge(d35: pd.DataFrame, d11: pd.DataFrame, col: str) -> pd.DataFrame:
    src = {"035": d35, "011": d11}
    out = {}
    for k, (s, p) in SOURCE.items():
        d = src[s][src[s].pair == p]
        assert len(d), (s, p)
        out[k] = d.set_index("step")[col]
    m = pd.DataFrame(out)[list(LABEL)]
    assert not m.isna().any().any(), m[m.isna().any(axis=1)]
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, required=True)
    ap.add_argument("--tsv-011", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d35, d11 = pd.read_csv(a.tsv, sep="\t"), pd.read_csv(a.tsv_011, sep="\t")
    l2, cos = merge(d35, d11, "rel_l2"), merge(d35, d11, "cos")
    D011.LABEL, D011.COLOR, D011.DASH = LABEL, COLOR, DASH
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.tsv, a.out / "divergence.tsv")
    shutil.copy(a.tsv_011, a.out / "divergence_011.tsv")
    (a.out / "divergence.svg").write_text(D011.svg(l2, cos))
    seed = l2["large-aux-2~default"]
    last = int(l2.index.max())
    summary = {
        "steps": [int(s) for s in l2.index], "tsv": str(a.tsv), "tsv_011": str(a.tsv_011),
        "final_step": last,
        "rel_l2_final": {k: round(float(l2.loc[last, k]), 4) for k in LABEL},
        "cos_final": {k: round(float(cos.loc[last, k]), 4) for k in LABEL},
        "l2_ratio_over_different_seed": {k: {int(s): round(float(v), 4)
                                             for s, v in (l2[k] / seed).items() if s > 0}
                                         for k in SAME_SEED},
        "git_commit": provenance.git_commit(), "command": provenance.command()}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("rel_l2_final", "cos_final")}, indent=1))


if __name__ == "__main__":
    main()
