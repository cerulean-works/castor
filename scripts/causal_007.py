"""The own-direction ablation check ("causal at 1B"). CPU.

    uv run --frozen python scripts/causal_007.py --models data/manifests/mech_007.json \
        --out /home/harrychang08/castor/runs/007/causal.json

Per twin, plain-harmful vulnerability = the mean over plain-harmful prompts of per-prompt P
(`wg_vulnerable`, K = 20, greedy excluded; castor.sanityb.model_stats, the quantity 006 gates),
for the twin as trained (006's labels), under own-direction ablation and, for the three seed-1
twins, under the random-direction control (labels in /work/.../runs/<id>-ablate-<kind>/007/).
Verdict: castor.mechladder.ablation_verdict (every twin doubles; random |change| < 1/4 of the
own increase). Also reported, descriptive: the same figures on adversarial-harmful and benign
prompts, the harmful-compliance rate (`wg_harmful`) and the share of unusable (incoherent or
judge-error) samples, since an ablation that breaks the model would raise that share.

Own site: `--own-kind own-site` takes the ablated labels from each twin's own-site cell
(/work/.../runs/<id>-ablate-ownsite/022/), with the random controls above reused unchanged.

    uv run --frozen python scripts/causal_007.py --models data/manifests/mech_007.json \
        --own-kind own-site --out /home/harrychang08/castor/runs/022/causal.json
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from castor import evalset, provenance, sanityb
from castor import mechladder as ML
from castor.judge.common import read_jsonl

RUNS = Path("/work/harrychang08/castor/runs")
SUBSETS = ("plain_harmful", "adv_harmful", "benign")


def labels_path(model_id: str, kind: str | None) -> Path:
    if kind is None:
        return RUNS / model_id / "006" / "judge_v1" / "labels.jsonl"
    if kind == "own-site":  # 022 (scripts/ablate_007.py PROMPT_OF_KIND)
        return RUNS / f"{model_id}-ablate-ownsite" / "022" / "judge_v1" / "labels.jsonl"
    return RUNS / f"{model_id}-ablate-{kind}" / "007" / "judge_v1" / "labels.jsonl"


def summarize(rows: list[dict], ids_by_subset: dict[str, list[str]]) -> dict:
    out = {}
    for s, ids in ids_by_subset.items():
        y, counts = sanityb.outcome_matrix(rows, ids, "wg_vulnerable")
        h, _ = sanityb.outcome_matrix(rows, ids, "wg_harmful")
        st = sanityb.model_stats(y)
        out[s] = {"vulnerability": st["mean_vulnerability"], "share_nan": st["share_nan"],
                  "harmful": sanityb.model_stats(h)["mean_vulnerability"],
                  "missing": counts["missing"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--own-kind", default="own", choices=("own", "own-site"))
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (D036)"
    commit = provenance.git_commit()
    panel = evalset.load_eval()
    ids = {s: sanityb.subset_ids(panel, s) for s in SUBSETS}
    specs = json.loads(a.models.read_text())["models"]
    per, src = {}, {}
    for spec in specs:
        m = spec["model_id"]
        per[m] = {}
        for kind in (None, a.own_kind, "random"):
            p = labels_path(m, kind)
            if kind == "random" and not p.exists():
                continue
            rows = list(read_jsonl(p))
            key = "own" if kind == a.own_kind else kind or "base"
            per[m][key] = summarize(rows, ids)
            src[f"{m}:{key}"] = {"path": str(p), "sha256": hashlib.sha256(
                p.read_bytes()).hexdigest()}
            if kind:
                ck = json.loads((p.parent.parent / "ablation_checkpoint.json").read_text())
                assert ck["source_model_id"] == m and ck["kind"] == kind, p
                src[f"{m}:{key}"]["checkpoint_site"] = ck.get("site")
                src[f"{m}:{key}"]["selection_sha256"] = ck.get("selection_sha256")
    for m in per:
        assert per[m]["own"]["plain_harmful"]["missing"] == 0, f"{m}: incomplete own labels"
    v = {k: {m: per[m][k]["plain_harmful"]["vulnerability"] for m in per if k in per[m]}
         for k in ("base", "own", "random")}
    verdict = ML.ablation_verdict(v["base"], v["own"], v["random"])
    out = {"prompt": "022" if a.own_kind == "own-site" else "007", "own_kind": a.own_kind,
           "verdict": verdict, "per_model": per,
           "n_prompts": {s: len(x) for s, x in ids.items()}, "sources": src,
           "git_commit": commit, "command": provenance.command(), "time": time.time()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({"causal": verdict["causal"],
                      "per_model": {m: {k: round(x, 3) if isinstance(x, float) else x
                                        for k, x in r.items()}
                                    for m, r in verdict["per_model"].items()}}, indent=1))
    print("plain-harmful share_nan base/own:",
          {m[11:]: (round(per[m]["base"]["plain_harmful"]["share_nan"], 3),
                    round(per[m]["own"]["plain_harmful"]["share_nan"], 3)) for m in per})
    print("mean plain-harmful vulnerability (base, own, random):",
          np.mean(list(v["base"].values())), np.mean(list(v["own"].values())),
          np.mean(list(v["random"].values())) if v["random"] else None)


if __name__ == "__main__":
    main()
