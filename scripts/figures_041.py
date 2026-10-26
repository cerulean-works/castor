"""Data behind four appendix figures, from existing judge_v1 labels only (no generation, no
judging). CPU only; streams the label files.

    uv run --frozen python scripts/figures_041.py --out /home/harrychang08/castor/runs/041

Writes under --out (committed: small, label-free, no prompt or response text):
  scatter.json      per-prompt vulnerability of one MZ-twin pair and one unrelated pair of 006's
                    twins on the adversarial-harmful subset (006's headline subset and outcome,
                    wg_vulnerable); each pair is the lower-median pair of its type by corrected r
                    in runs/006/signal.json (castor.figdata.lower_median_pair; never the best).
  by_source.json    006's twin statistics per prompt source (castor.reliability.by_group: each
                    source its own panel, reliabilities recomputed, prompt bootstrap n_boot / seed
                    as 006 ran it); plus 006's adversarial-harmful subset recomputed the same way,
                    asserted equal to signal.json (the reference line and the reproduction check).
  seed_stable.json  per adversarial-harmful prompt the mean / min / max vulnerability over the 12
                    main-recipe rerun twins (008 stage A0, castor.ladder.anchor_cells), and the
                    shares with min >= 0.9 (every twin breaks) and max <= 0.1 (no twin breaks).
  judge_audit.json  023's 1,200 items: confusion counts of judge_v1's label against the adjudicated
                    human label and against each annotator's own label (023's gold rule, its
                    scripts' readers); refusal precision / recall recomputed from the counts and
                    asserted equal to runs/006/023/result.json.
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from castor import evalset, figdata, ladder, provenance, reliability, sanityb
from castor.judge import labels as L
from castor.judge.common import read_jsonl

ROOT = Path(__file__).resolve().parent.parent
SIGNAL_006 = ROOT / "runs" / "006" / "signal.json"
RESULT_023 = ROOT / "runs" / "006" / "023" / "result.json"
HT_DIR = sanityb.RUNS / "006" / "human_twin"
ANNOTATORS = ("harry", "coauthor")
FIELD = sanityb.PRIMARY_FIELD  # wg_vulnerable
SUBSET = sanityb.PRIMARY_SUBSET  # adv_harmful

# prompt sources (data/panel/panel_v1_eval.jsonl: source, source_split), in figure order
SOURCES = {
    "harmbench_standard": lambda r: r["source"] == "harmbench" and r["source_split"].endswith("standard"),
    "harmbench_contextual": lambda r: r["source"] == "harmbench" and r["source_split"].endswith("contextual"),
    "harmbench_humanjb": lambda r: r["source"] == "harmbench_humanjb",
    "wildjailbreak_adv_harmful": lambda r: r["source_split"] == "eval/adversarial_harmful",
    "sorrybench": lambda r: r["source"] == "sorrybench",
    "strongreject": lambda r: r["source"] == "strongreject",
    "xstest_unsafe": lambda r: r["source_split"] == "unsafe",
    "xstest_safe": lambda r: r["source_split"] == "safe",
    "wildjailbreak_adv_benign": lambda r: r["source_split"] == "eval/adversarial_benign",
    "orbench": lambda r: r["source"] == "orbench",
}
GAP_KEYS = ("r_MZ", "r_U", "r_MZ_corrected", "r_U_corrected", "MZ_minus_U", "MZ_minus_U_corrected",
            "mean_reliability")
HUMAN_ORDER = ("R", "H", "N", "I")
JUDGE_SHORT = {L.REFUSAL: "R", L.HARMFUL: "H", L.NONHARMFUL: "N", L.INCOHERENT: "I",
               L.JUDGE_ERROR: "E"}
JUDGE_ORDER = ("R", "H", "N", "I", "E")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def matrices(models: list[str], prompt_id: str, ids: list[str]) -> dict[str, np.ndarray]:
    """(len(ids), K) wg_vulnerable matrices, streamed from each model's labels.jsonl."""
    out = {}
    for m in models:
        y, c = sanityb.outcome_matrix(read_jsonl(sanityb.labels_path(m, prompt_id)), ids, FIELD)
        assert c["missing"] == 0, f"{m}/{prompt_id}: {c['missing']} missing label cells"
        out[m] = y
    return out


def _ci(ci: dict) -> dict:
    return {k: ci[k] for k in GAP_KEYS}


def scatter_and_sources(panel: list[dict], signal: dict, n_boot: int, seed: int) -> tuple[dict, dict]:
    cells = {c.model_id: c for c in sanityb.cells()}
    assert sorted(cells) == sorted(m for m in signal["models"] if not signal["models"][m].get("reference"))
    info = {m: c.info for m, c in cells.items()}
    all_ids = [r["id"] for r in panel]
    ys = matrices(sorted(cells), sanityb.PROMPT_ID, all_ids)
    pos = {pid: i for i, pid in enumerate(all_ids)}

    sub_ids = sanityb.subset_ids(panel, SUBSET)
    rows = [pos[p] for p in sub_ids]
    prim = signal[SUBSET]["primary"]
    picked = {lvl: figdata.lower_median_pair(prim["pairs"], lvl) for lvl in ("MZ", "U")}
    sub = {m: y[rows] for m, y in ys.items()}
    tw = reliability.twin_correlations(sub, info)
    scat = {"subset": SUBSET, "field": FIELD, "n_prompts": len(sub_ids), "prompt_ids": sub_ids,
            "rule": "lower-median pair of each type by corrected r over 006's pairs "
                    "(sorted ascending, index (n - 1) // 2; 18 MZ -> 9th, 48 U -> 24th)",
            "pairs": {}}
    for lvl, p in picked.items():
        again = next(x for x in tw["pairs"] if (x["a"], x["b"]) == (p["a"], p["b"]))
        assert abs(again["r"] - p["r"]) < 1e-12 and abs(again["r_corrected"] - p["r_corrected"]) < 1e-12
        n_lvl = sum(x["level"] == lvl for x in prim["pairs"])
        rank = sorted(x["r_corrected"] for x in prim["pairs"] if x["level"] == lvl).index(p["r_corrected"])
        pa, pb = (reliability.per_prompt_p(sub[m]) for m in (p["a"], p["b"]))
        scat["pairs"][lvl] = {"a": p["a"], "b": p["b"], "r": p["r"], "r_corrected": p["r_corrected"],
                              "rank_ascending": rank + 1, "n_pairs": n_lvl,
                              "reliability_a": tw["reliability"][p["a"]],
                              "reliability_b": tw["reliability"][p["b"]],
                              "p_a": [round(float(v), 6) for v in pa],
                              "p_b": [round(float(v), 6) for v in pb]}

    # the headline subset is added as the reference group and checked against signal.json
    groups = {name: [pos[r["id"]] for r in panel if pred(r)] for name, pred in SOURCES.items()}
    covered = sorted(i for g in groups.values() for i in g)
    assert covered == list(range(len(panel))), "sources must partition the panel"
    groups[SUBSET] = rows
    res = reliability.by_group(ys, info, groups, n_boot=n_boot, seed=seed)
    ref = res[SUBSET]["ci_prompts"]
    for k in GAP_KEYS:
        for f in ("estimate", "lo", "hi"):
            a, b = ref[k][f], prim["ci_prompts"][k][f]
            assert abs(a - b) < 1e-12, f"{SUBSET} {k} {f}: {a} != signal.json {b}"
    src = {"field": FIELD, "n_boot": n_boot, "seed": seed, "order": list(SOURCES),
           "harmful": {name: bool(panel[groups[name][0]]["is_harmful"]) for name in SOURCES},
           "reference": SUBSET, "reference_reproduces_signal_006": True,
           "groups": {name: {"n_prompts": g["n_prompts"],
                             "n_pairs": {lvl: g["levels"][lvl]["n_pairs"] for lvl in ("MZ", "U")},
                             "n_pairs_defined": {lvl: g["levels"][lvl]["n_pairs_defined"]
                                                 for lvl in ("MZ", "U")},
                             "reliability_min": float(min(g["reliability"].values())),
                             "ci_prompts": _ci(g["ci_prompts"])}
                      for name, g in res.items()}}
    return scat, src


def seed_stable(panel: list[dict]) -> dict:
    cells = ladder.anchor_cells()
    ids = sanityb.subset_ids(panel, SUBSET)
    ys = matrices([c.model_id for c in cells], ladder.ANCHOR_PROMPT_ID, ids)
    ps = {m: reliability.per_prompt_p(y) for m, y in ys.items()}
    rg = figdata.prompt_range(ps)
    kept = [p for p, k in zip(ids, rg["keep"], strict=True) if k]
    order = np.lexsort((np.arange(len(kept)), rg["mean"]))  # ascending mean, ties by panel order
    return {"subset": SUBSET, "field": FIELD, "models": rg["models"], "n_models": len(rg["models"]),
            "n_prompts": len(ids), "n_dropped_undefined": rg["n_dropped"],
            "shares": figdata.stable_shares(rg["min"], rg["max"]),
            "sorted": {"prompt_id": [kept[i] for i in order],
                       "mean": [round(float(rg["mean"][i]), 6) for i in order],
                       "min": [round(float(rg["min"][i]), 6) for i in order],
                       "max": [round(float(rg["max"][i]), 6) for i in order]}}


def judge_audit(result_023: dict) -> dict:
    score = _load_script("audit_score")
    keyrows = {r["item_id"]: r for r in read_jsonl(HT_DIR / "key.jsonl")}
    sheets = [score.read_sheet(str(HT_DIR / f"sheet_{a}.csv")) for a in ANNOTATORS]
    adj = score.read_sheet(str(HT_DIR / "sheet_adjudicate.csv"))
    gold, info = score.gold_labels(sheets, adj)
    assert info["unresolved"] == 0 and set(gold) == set(keyrows)
    short_h = {v: k for k, v in {"R": L.REFUSAL, "H": L.HARMFUL, "N": L.NONHARMFUL,
                                 "I": L.INCOHERENT}.items()}
    sets = {"adjudicated": gold, **dict(zip(ANNOTATORS, sheets, strict=True))}
    out = {"n_items": len(keyrows), "rows": "human label", "cols": "judge_v1 label (wg_label)",
           "human_order": list(HUMAN_ORDER), "judge_order": list(JUDGE_ORDER), "matrices": {}}
    for name, lab in sets.items():
        pairs = [(short_h[lab[i]["label"]], JUDGE_SHORT[keyrows[i]["wg_label"]]) for i in sorted(lab)]
        out["matrices"][name] = figdata.confusion(pairs, HUMAN_ORDER, JUDGE_ORDER)
    # refusal agreement (adjudicated, coherent on both sides), recomputed from the counts
    m = out["matrices"]["adjudicated"]
    coh = ("R", "H", "N")
    n = sum(m[h][j] for h in coh for j in coh)
    judge_nr = sum(m[h][j] for h in coh for j in ("H", "N"))
    tp = sum(m[h][j] for h in ("H", "N") for j in ("H", "N"))
    human_nr = sum(m[h][j] for h in ("H", "N") for j in coh)
    ref = result_023["judge_vs_human_items"]["wg_refusal"]
    check = {"n": n, "precision": tp / judge_nr, "recall": tp / human_nr}
    assert check["n"] == ref["n"], (check, ref)
    for k in ("precision", "recall"):
        assert abs(check[k] - ref[k]) < 1e-12, (k, check[k], ref[k])
    out["refusal_check_vs_023"] = check
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", choices=("sources", "stable", "audit"), nargs="*")
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute"
    commit, t0 = provenance.git_commit(), time.time()
    panel = evalset.load_eval()
    signal = json.loads(SIGNAL_006.read_text())
    meta = {"script": "scripts/figures_041.py", "git_commit": commit, "command": provenance.command()}
    a.out.mkdir(parents=True, exist_ok=True)
    todo = set(a.only or ("sources", "stable", "audit"))

    def write(name: str, obj: dict) -> None:
        (a.out / name).write_text(json.dumps({**obj, **meta, "seconds": time.time() - t0},
                                             indent=1, default=float))
        print(f"wrote {a.out / name} ({time.time() - t0:.0f} s)")

    if "sources" in todo:
        scat, src = scatter_and_sources(panel, signal, signal["n_boot"], signal["seed"])
        write("scatter.json", scat)
        write("by_source.json", src)
    if "stable" in todo:
        write("seed_stable.json", seed_stable(panel))
    if "audit" in todo:
        write("judge_audit.json", judge_audit(json.loads(RESULT_023.read_text())))


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    main()
