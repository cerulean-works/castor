"""Is 006's headline interval optimistic, and does the gap hold within each benchmark? Pure CPU;
reads 006's judge_v1 labels for the 12 twins only.

    uv run --frozen python scripts/cluster_ci_pitch.py --out /abs/path/runs/006/cluster_ci.json \
        [--n-boot 1000]

The primary subset (adv_harmful, wg_vulnerable) is two sources: WildJailbreak's adversarial
eval split and HarmBench HumanJailbreaks (behaviour x template, source_id "<behaviour>::jbNN").
Blocks of the output:
  pooled        006's point estimate (asserted equal to runs/006/signal.json) and four intervals:
                prompts (006's, rows independent), template clusters, behaviour clusters (both
                drawn within source; WildJailbreak prompts carry no template id in the panel and
                stay singletons), and bases (006's ci_bases, copied).
  ratio         (1 - r_U) / (1 - r_MZ), corrected, under the same prompt and template resampling.
  by_source     the contrast within each source, with its prompt interval (and template and
                behaviour intervals for HumanJailbreaks).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from castor import evalset, provenance, reliability, sanityb
from castor.judge.common import read_jsonl

KEYS = ("r_MZ_corrected", "r_U_corrected", "MZ_minus_U_corrected", "MZ_minus_U")


HUMANJB = "harmbench_humanjb"


def cluster_ids(src: list[str], sid: list[str]) -> tuple[list[str], list[str]]:
    """Template and behaviour cluster ids per prompt. HarmBench HumanJailbreaks prompts carry
    source_id "<behaviour>::jbNN": the template cluster is jbNN, the behaviour cluster the
    behaviour. Every other prompt (WildJailbreak carries no template id) is its own singleton
    cluster, named by its row index so it collides with nothing."""
    template = [s.split("::")[1] if so == HUMANJB else f"wj{j}"
                for j, (s, so) in enumerate(zip(sid, src, strict=True))]
    behaviour = [s.split("::")[0] if so == HUMANJB else f"wj{j}"
                 for j, (s, so) in enumerate(zip(sid, src, strict=True))]
    return template, behaviour


def stat(res: dict) -> dict:
    s = reliability._summary(res)
    out = {k: s[k] for k in KEYS}
    out["ratio_unshared_U_over_MZ"] = (1 - s["r_U_corrected"]) / (1 - s["r_MZ_corrected"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert a.out.is_absolute()
    t0 = time.time()
    panel = {r["id"]: r for r in evalset.load_eval()}
    ids = sanityb.subset_ids(list(panel.values()), sanityb.PRIMARY_SUBSET)
    cells = sanityb.cells()
    info = {c.model_id: c.info for c in cells}
    ys = {}
    for c in cells:
        rows = read_jsonl(sanityb.RUNS / c.model_id / sanityb.PROMPT_ID / "judge_v1" / "labels.jsonl")
        ys[c.model_id] = sanityb.outcome_matrix(rows, ids, sanityb.PRIMARY_FIELD)[0]

    src = [panel[i]["source"] for i in ids]
    sid = [panel[i]["source_id"] for i in ids]
    hb = HUMANJB
    template, behaviour = cluster_ids(src, sid)

    def run(sel: np.ndarray | None, with_clusters: bool) -> dict:
        y = ys if sel is None else {m: v[sel] for m, v in ys.items()}
        pick = (lambda xs: xs) if sel is None else (lambda xs: [xs[i] for i in sel])
        b = {"n_prompts": int(next(iter(y.values())).shape[0]),
             "point": stat(reliability.twin_correlations(y, info)),
             "ci_prompts": reliability.bootstrap_prompts(y, info, a.n_boot, a.seed, stat=stat)}
        if with_clusters:
            for name, cl in (("template", template), ("behaviour", behaviour)):
                b[f"ci_{name}_clusters"] = reliability.bootstrap_prompt_clusters(
                    y, info, pick(cl), a.n_boot, a.seed, strata=pick(src), stat=stat)
                b[f"n_{name}_clusters_humanjb"] = len({c for c, s in zip(pick(cl), pick(src))
                                                       if s == hb})
        return b

    res = {"pooled": run(None, True)}
    sig = json.loads(Path(__file__).resolve().parent.parent.joinpath("runs/006/signal.json")
                     .read_text())["adv_harmful"]["primary"]
    ref = sig["ci_prompts"]["MZ_minus_U_corrected"]
    assert abs(res["pooled"]["point"]["MZ_minus_U_corrected"] - ref["estimate"]) < 1e-12
    if a.n_boot == 1000 and a.seed == 0:  # 006's settings: the prompt interval must reproduce
        assert res["pooled"]["ci_prompts"]["MZ_minus_U_corrected"]["lo"] == ref["lo"]
    res["pooled"]["ci_bases_from_signal_json"] = {k: sig["ci_bases"][k] for k in
                                                  ("MZ_minus_U_corrected", "r_MZ_corrected",
                                                   "r_U_corrected")}
    res["pooled"]["jackknife_bases_from_signal_json"] = sig["jackknife_bases"]["MZ_minus_U_corrected"]
    res["by_source"] = {}
    for s in sorted(set(src)):
        sel = np.array([i for i, x in enumerate(src) if x == s])
        res["by_source"][s] = run(sel, s == hb)
    res.update({"script": "scripts/cluster_ci_pitch.py", "git_commit": provenance.git_commit(),
                "command": provenance.command(), "n_boot": a.n_boot, "seed": a.seed,
                "seconds": time.time() - t0})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float))

    def show(label, d):
        return f"{label} {d['estimate']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
    for blk, b in [("pooled", res["pooled"])] + list(res["by_source"].items()):
        print(blk, b["n_prompts"], {k: round(v, 4) for k, v in b["point"].items()})
        for k in [k for k in b if k.startswith("ci_") and "signal" not in k]:
            print("  ", k, show("gap", b[k]["MZ_minus_U_corrected"]),
                  show("ratio", b[k]["ratio_unshared_U_over_MZ"]))


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    main()
