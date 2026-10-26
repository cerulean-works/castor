"""018 diagnostic (no rule): where in the target does each arm's removed direction align?

For every target and arm, the cosine of the unit direction removed from the target (exactly as
`transfer_018.py ablate` builds it) with the target's own difference of means at every one of
007's 16 x 7 candidate sites (zero-norm sites skipped): the value at the target's own site, at the donor's site, and the
largest |cos| over all sites with its site.
    uv run --frozen python scripts/diag_018_sitecos.py --out /abs/runs/018/site_cosines.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

from castor import provenance
from castor import transfer as T

tr = provenance.load_script("transfer_018")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procrustes", type=Path,
                    default=provenance.ROOT / "runs" / "018" / "procrustes.json")
    a = ap.parse_args()
    assert a.out.is_absolute()
    commit = provenance.git_commit()
    own = tr.own_sites()
    out = {}
    for m in T.targets():
        with safe_open(str(tr.ACTS_007 / m / "candidates.safetensors"), "pt") as f:
            c = f.get_tensor("mean_diff").numpy().astype(np.float64)
        n = np.linalg.norm(c, axis=-1, keepdims=True)
        c = np.where(n > 0, c / np.where(n > 0, n, 1), np.nan)  # layer 0: template rows, zero
        out[m] = {}
        for arm in T.ARMS:
            d, info = tr.arm_direction(m, arm, a.procrustes)
            cos = c @ d
            i = np.unravel_index(np.nanargmax(np.abs(cos)), cos.shape)
            ds = info["donor_site"]
            out[m][arm] = {"donor": info["donor"], "direction_sha256": info["direction_sha256"],
                           "cos_at_target_own": float(cos[own[m]]),
                           "cos_at_donor_site": float(cos[ds["layer"], ds["pos_index"]]),
                           "max_abs_cos": float(abs(cos[i])), "max_site": [int(x) for x in i],
                           "max_abs_cos_by_layer": [None if np.isnan(x) else round(float(x), 4)
                                                    for x in np.nanmax(np.abs(cos), 1)]}
    res = {"prompt": "018", "what": "cos(removed direction, target diff-of-means) per 007 site",
           "sites_shape": list(c.shape[:2]), "targets": out,
           "procrustes_json_sha256": tr.sha(a.procrustes), "git_commit": commit,
           "command": provenance.command(), "time": time.time()}
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    for m, r in out.items():
        print(m[11:], {k: (round(v["cos_at_target_own"], 3), round(v["max_abs_cos"], 3),
                           tuple(v["max_site"])) for k, v in r.items()})


if __name__ == "__main__":
    main()
