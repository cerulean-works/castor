"""The common (layer, position) of the refusal direction for every twin. CPU only.

    uv run --frozen python scripts/select_007.py --models data/manifests/mech_007.json \
        --acts-root /work/harrychang08/castor/activations/007 \
        --out /home/harrychang08/castor/runs/007/selection.json [--filtered] [--allow-subset]

Reads <acts-root>/<model-id>/scores.json (scripts/extract_007.py) for every model in the
manifest. Among candidates that pass Arditi's three filters in every model, the lowest mean
bypass_score; else the candidate passing in the most models, ties by mean bypass
(castor.directions.select_common). Also records each model's own Arditi selection (the
per-twin sensitivity) and the pass counts per candidate.

--filtered: a sensitivity analysis with Arditi's per-model filter (his `filter_data`): validation
prompts are kept per model only if their baseline refusal score is > 0 (harmful) or < 0 (harmless),
and the scores are re-averaged over the kept prompts before selecting. The primary selection never
uses it.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from castor import directions as Dn
from castor import provenance


def filtered_scores(s: dict) -> dict:
    """Re-average the per-prompt scores over the validation prompts Arditi's filter keeps."""
    bh = np.asarray(s["baseline_refusal_scores"]["harmful_val"])
    bb = np.asarray(s["baseline_refusal_scores"]["harmless_val"])
    kh, kb = bh > 0, bb < 0
    pp = {k: np.asarray(v) for k, v in s["per_prompt"].items()}
    return {"bypass": pp["bypass"][..., kh].mean(-1), "induce": pp["induce"][..., kb].mean(-1),
            "kl": pp["kl"][..., kb].mean(-1), "n_kept": {"harmful_val": int(kh.sum()),
                                                         "harmless_val": int(kb.sum())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--acts-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--filtered", action="store_true")
    ap.add_argument("--allow-subset", action="store_true", help="smoke: skip missing models")
    ap.add_argument("--set", dest="set_name", default=None,
                    help="021: one pair set of the manifest (`sets`) instead of every model")
    a = ap.parse_args()
    commit = provenance.git_commit()
    man = json.loads(a.models.read_text())
    ids = man["sets"][a.set_name]["models"] if a.set_name else [m["model_id"] for m in man["models"]]
    per, src, missing = {}, {}, []
    for m in ids:
        p = a.acts_root / m / "scores.json"
        if not p.exists():
            missing.append(m)
            continue
        s = json.loads(p.read_text())
        assert s["dtype"] == "bfloat16", f"{m}: selection uses the generation dtype"
        per[m] = filtered_scores(s) if a.filtered else s["scores"]
        src[m] = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                  "git_commit": s["git_commit"], "tail_mismatch": s["tail_mismatch"],
                  "panel_tail_mismatch": s.get("panel_tail_mismatch")}
    assert not missing or a.allow_subset, f"missing scores: {missing}"
    common = Dn.select_common(per)
    common["position"] = common["pos_index"] - Dn.N_POST
    own = {}
    for m, s in per.items():
        o, dropped = Dn.select_own(s), False
        if o is None:  # sensitivity: KL filter dropped, labelled as such
            o, dropped = Dn.select_own_kl_dropped(s), True
        own[m] = None if o is None else {"layer": o[0], "pos_index": o[1],
                                         "position": o[1] - Dn.N_POST,
                                         "bypass": float(np.asarray(s["bypass"])[o]),
                                         "kl_filter_dropped": dropped}
    n_pass = np.sum([Dn.pass_matrix(s) for s in per.values()], 0)
    mean_by = np.mean([np.asarray(s["bypass"], float) for s in per.values()], 0)
    out = {"prompt": "007", "filtered": a.filtered, "set": a.set_name, "models": sorted(per),
           "missing": missing,
           "common": common, "own": own, "n_pass": n_pass.tolist(),
           "n_sites_passing": {m: int(Dn.pass_matrix(s).sum()) for m, s in per.items()},
           "mean_bypass": mean_by.tolist(),
           "n_kept": {m: s["n_kept"] for m, s in per.items()} if a.filtered else None,
           "rule": "lowest mean bypass among candidates passing in all models; else most "
                   "models passing, ties by mean bypass (D047 b); filters: Arditi filter_fn",
           "sources": src, "git_commit": commit, "command": provenance.command()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"common": common, "own": own}, indent=1))


if __name__ == "__main__":
    main()
