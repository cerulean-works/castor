"""Human-labelled twin contrast on the stored generations of runs/006.

    # 1. draw the 1,200 items and write the blinded sheets (once; refuses to overwrite)
    uv run --frozen python scripts/human_twin_023.py draw --out HT_DIR
    # 2. after both sheets are labelled: the adjudication sheet (label disagreements)
    uv run --frozen python scripts/audit_disagree.py --sheets HT_DIR/sheet_harry.csv \
        HT_DIR/sheet_coauthor.csv --names harry coauthor --out-json HT_DIR/disagree.json \
        --out-sheet HT_DIR/sheet_adjudicate.csv
    # 3. after adjudication: the contrast
    uv run --frozen python scripts/human_twin_023.py analyze --dir HT_DIR \
        --sheets HT_DIR/sheet_harry.csv HT_DIR/sheet_coauthor.csv \
        --adjudicated HT_DIR/sheet_adjudicate.csv --out HT_DIR/023/result.json
    # dry run: placeholder human sheets derived from judge labels (*_DRYRUN*)
    uv run --frozen python scripts/human_twin_023.py dryrun --dir HT_DIR --stage sheets
    ... audit_disagree.py on the two DRYRUN sheets -> HT_DIR/dryrun/sheet_adjudicate_DRYRUN.csv
    uv run --frozen python scripts/human_twin_023.py dryrun --dir HT_DIR --stage adjudicate
    uv run --frozen python scripts/human_twin_023.py analyze --dryrun ...

HT_DIR = /work/harrychang08/castor/runs/006/human_twin (sheets carry harmful text: never git).
Design, outcome and statistic: castor.humantwin. Sheets are written by castor.audit in
scripts/audit_sample.py's format (shuffled, item ids random, no model id or judge label);
`key.jsonl` maps item ids back and carries judge_v1's labels (never shown to annotators).

Blind order: `analyze` computes nothing unless both sheets pass
scripts/audit_check_sheets.py's check and every label disagreement is adjudicated; it reads
gold labels with scripts/audit_score.py's own `read_sheet` / `gold_labels`.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from castor import audit, evalset, humantwin, provenance, sanityb
from castor.judge import labels as L
from castor.judge.common import key, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path("/work/harrychang08/castor/runs")
AUDIT_006 = RUNS / "006" / "audit"
GUIDE = ROOT / "docs" / "audit_labelling_guide.md"
REPO_COPY = ROOT / "runs" / "006" / "023" / "result.json"
REPO_COPY_NAMES = ("result.json", "result_ordered.json")  # of record and the 056 rerun
ANNOTATORS = ("harry", "coauthor")
SHORT = humantwin.SHORT  # REFUSAL -> R, ...
JUDGE_SHORT = {L.REFUSAL: "R", L.HARMFUL: "H", L.NONHARMFUL: "N", L.INCOHERENT: "I",
               L.JUDGE_ERROR: "I"}  # dry-run placeholder only
KEY_FIELDS = ("coherent", "coherence_rule", "llm_invalid", "wg_label", "wg_partial",
              "wg_vulnerable", "wg_vulnerable_incl", "q3g_label", "q3g_partial",
              "refusal_agree")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sources_006() -> tuple[list[str], list[str]]:
    """The label and generation files of the four twins, as 006's audit manifest names them."""
    m = json.loads((AUDIT_006 / "manifest.json").read_text())
    labels = [p for p in m["labels"] if any(f"/runs/{t}/006/" in p for t in humantwin.MODELS)]
    gens = [p for p in m["generations"] if any(f"/runs/{t}/006/samples.jsonl" in p
                                                for t in humantwin.MODELS)]
    assert len(labels) == len(gens) == len(humantwin.MODELS), (labels, gens)
    return labels, gens


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- draw -----------------------------------------------------------------------------------


def cmd_draw(args) -> None:
    out = Path(args.out)
    if out.exists() and (any(out.glob("sheet_*.csv")) or (out / "key.jsonl").exists()):
        sys.exit(f"{out} already holds a draw; refusing to overwrite (sheets may be labelled)")
    commit, cmd = provenance.git_commit(), provenance.command()
    panel = evalset.load_eval()
    source = {r["id"]: r["source"] for r in panel}
    adv = sanityb.subset_ids(panel, sanityb.PRIMARY_SUBSET)
    ids_by_source: dict[str, list[str]] = {}
    for pid in adv:
        ids_by_source.setdefault(source[pid], []).append(pid)
    prompts, items = humantwin.draw(ids_by_source)
    want = set(items)

    label_files, gen_files = sources_006()
    lab = {key(r): r for p in label_files for r in read_jsonl(p) if key(r) in want}
    gen = {key(r): r for p in gen_files for r in read_jsonl(p) if key(r) in want}
    missing = [k for k in items if k not in lab or k not in gen]
    assert not missing, f"{len(missing)} drawn items lack a label or generation row: {missing[:3]}"
    in_audit = {key(r) for r in read_jsonl(AUDIT_006 / "key.jsonl")} & want

    rng = random.Random(humantwin.SEED + 1)  # sheet order and item ids, as audit_sample.py
    order = list(items)
    rng.shuffle(order)
    ids = [f"{i:06x}" for i in rng.sample(range(16 ** 6), len(order))]
    keyrows = [{"item_id": i, "model": k[0], "prompt_id": k[1], "sample": k[2],
                "source": source[k[1]], **{f: lab[k][f] for f in KEY_FIELDS},
                "in_d023_audit": k in in_audit} for k, i in zip(order, ids)]
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "key.jsonl", keyrows)
    for a in ANNOTATORS:
        audit.write_sheet(out / f"sheet_{a}.csv",
                          [(i, gen[k]["prompt"], gen[k]["response"]) for k, i in zip(order, ids)])
    manifest = {
        "n": len(items), "seed": humantwin.SEED, "n_per_source": humantwin.N_PER_SOURCE,
        "k_draw": humantwin.K_DRAW, "k_stored": humantwin.K_STORED,
        "models": list(humantwin.MODELS), "prompts": prompts,
        "per_source": dict(Counter(source[p] for p in prompts)),
        "population_per_source": {s: len(v) for s, v in sorted(ids_by_source.items())},
        "subset": sanityb.PRIMARY_SUBSET, "eval_manifest": {  # repo path: snapshots are deleted
            "path": str(evalset.EVAL_MANIFEST.relative_to(ROOT)),
            "sha256": sha256(evalset.EVAL_MANIFEST)},
        "labels": label_files, "generations": gen_files, "annotators": list(ANNOTATORS),
        "overlap_d023_audit": len(in_audit),
        "judge_nan_items": sum(lab[k]["wg_vulnerable"] is None for k in items),
        "guide_sha256": sha256(GUIDE),
        "git_commit": commit, "command": cmd, "script": "scripts/human_twin_023.py draw",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("n", "per_source", "overlap_d023_audit",
                                               "judge_nan_items")}))


# ---- dry run --------------------------------------------------------------------------------


def cmd_dryrun(args) -> None:
    """Placeholder human sheets from judge labels, named *_DRYRUN*. First annotator: judge label
    with each judge non-refusal turned into R with prob 0.4; second: the first's labels with
    10 % of items relabelled uniformly. Adjudication takes the first annotator's labels."""
    d = Path(args.dir)
    dr = d / "dryrun"
    keyrows = list(read_jsonl(d / "key.jsonl"))
    if args.stage == "sheets":
        dr.mkdir(exist_ok=True)
        r1, r2 = random.Random(1), random.Random(2)
        harry = {}
        for r in keyrows:
            lab = JUDGE_SHORT[r["wg_label"]]
            if lab in ("H", "N") and r1.random() < 0.4:
                lab = "R"
            harry[r["item_id"]] = lab
        coauthor = {i: (r2.choice("RHNI") if r2.random() < 0.1 else v) for i, v in harry.items()}
        for name, labs in (("harry", harry), ("coauthor", coauthor)):
            _fill(d / f"sheet_{name}.csv", dr / f"sheet_{name}_DRYRUN.csv", labs)
    else:
        harry = {i: v["label"] for i, v in _load("audit_score").read_sheet(
            str(dr / "sheet_harry_DRYRUN.csv")).items()}
        harry = {i: SHORT[v] for i, v in harry.items()}
        _fill(dr / "sheet_adjudicate_DRYRUN.csv", dr / "sheet_adjudicate_DRYRUN.csv", harry)
    print(f"dry-run {args.stage} written under {dr}")


def _fill(src: Path, dst: Path, labels: dict[str, str]) -> None:
    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())
    for r in rows:
        if not r["item_id"].startswith("#"):
            r["label"] = labels[r["item_id"]]
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---- analysis -------------------------------------------------------------------------------


def _contrast(items: list[dict], prompts: list[str], incl: bool, n_boot: int) -> dict:
    vals = humantwin.values(items, incl)
    ys = humantwin.combine({lab: humantwin.matrices(v, prompts) for lab, v in vals.items()})
    ci = humantwin.contrast(ys, n_boot=n_boot, seed=0)
    nan = {lab: int(sum(np.isnan(v) for v in vals[lab].values())) for lab in vals}
    undefined = {k: int(np.isnan(humantwin.reliability.per_prompt_p(y)).sum())
                 for k, y in ys.items()}  # (prompt, twin) cells whose 3 items are all NaN
    return {"ci": ci, "reading": humantwin.reading(ci), "n_prompts": len(prompts),
            "nan_items": nan, "undefined_prompt_cells": undefined}


def cmd_analyze(args) -> None:
    is_dry = [("DRYRUN" in Path(s).name) for s in args.sheets + [args.adjudicated or ""]]
    if args.dryrun != any(is_dry) or (args.dryrun and not all(is_dry[:2])) \
            or (args.dryrun and "DRYRUN" not in args.out):
        sys.exit("--dryrun goes with *_DRYRUN* sheets and a *DRYRUN* --out, and only with them")
    d = Path(args.dir)
    manifest = json.loads((d / "manifest.json").read_text())
    keyrows = {r["item_id"]: r for r in read_jsonl(d / "key.jsonl")}

    # 1. sheets complete and untouched (label-only check: no judge label read)
    chk = _load("audit_check_sheets")
    want = {key(r) for r in keyrows.values()}
    text = {key(r): (r["prompt"], r["response"]) for g in manifest["generations"]
            for r in read_jsonl(g) if key(r) in want}
    checks = {s: chk.check_sheet(s, keyrows, text) for s in args.sheets}
    bad = {s: c for s, c in checks.items() if not c["ok"]}
    if errs := humantwin.input_errors(len(keyrows), len(text), bad):
        sys.exit("nothing computed: " + "; ".join(errs))
    # 2. every label disagreement adjudicated (no judge-vs-human number before)
    score = _load("audit_score")
    sheets = [score.read_sheet(s) for s in args.sheets]
    adj = score.read_sheet(args.adjudicated) if args.adjudicated else None
    gold, info = score.gold_labels(sheets, adj)
    if errs := humantwin.gold_errors(info["unresolved"], set(gold), set(keyrows)):
        sys.exit("nothing computed: " + "; ".join(errs))
    commit, cmd = provenance.git_commit(), provenance.command()
    # 3. key's judge labels still equal the label files
    lab = {key(r): r for p in manifest["labels"] for r in read_jsonl(p) if key(r) in want}
    drift = [i for i, r in keyrows.items() if any(lab[key(r)][f] != r[f] for f in KEY_FIELDS)]
    assert not drift, f"key.jsonl judge fields differ from the label files: {drift[:3]}"

    items = [{**keyrows[i], "gold": g["label"], "gold_partial": g["partial"], "weight": 1.0}
             for i, g in gold.items()]
    prompts = manifest["prompts"]
    res = {"dryrun": args.dryrun, "n_items": len(items), "n_prompts": len(prompts),
           "primary": _contrast(items, prompts, incl=False, n_boot=args.n_boot),
           "secondary_incoherent_included": _contrast(items, prompts, incl=True,
                                                      n_boot=args.n_boot)}
    by_src = {}
    for src in humantwin.SOURCES:
        in_src = {x["prompt_id"] for x in items if x["source"] == src}
        ps = [p for p in prompts if p in in_src]
        by_src[src] = _contrast([x for x in items if x["source"] == src], ps, incl=False,
                                n_boot=args.n_boot)
    res["secondary_per_source"] = by_src
    a, b = sheets
    both = sorted(set(a) & set(b))
    res["inter_annotator"] = {  # as scripts/audit_score.py
        "n": len(both),
        "kappa_4class": audit.cohen_kappa([a[i]["label"] for i in both],
                                          [b[i]["label"] for i in both]),
        "kappa_refusal": audit.cohen_kappa([a[i]["label"] == L.REFUSAL for i in both],
                                           [b[i]["label"] == L.REFUSAL for i in both]),
        "n_label_disagreements": sum(a[i]["label"] != b[i]["label"] for i in both),
    }
    res["judge_vs_human_items"] = {  # reported, not gated
        "wg_refusal": score.agreement(items, "wg", "refusal"),
        "wg_harm": score.agreement(items, "wg", "harm"),
        "by_model_wg_refusal": {m: score.agreement([x for x in items if x["model"] == m],
                                                   "wg", "refusal") for m in humantwin.MODELS},
        "gold_counts": dict(Counter(SHORT[x["gold"]] for x in items)),
        "judge_counts": dict(Counter(x["wg_label"] for x in items)),
    }
    res["levels"] = {lab: {m: float(np.nanmean(humantwin.reliability.per_prompt_p(y)))
                           for m, y in mats.items()}
                     for lab, mats in ((lab, humantwin.matrices(v, prompts)) for lab, v in
                                       humantwin.values(items).items())}
    # descriptive: gap_human under each annotator's labels alone; no reading rule
    res["per_annotator"] = {name: _contrast(humantwin.with_labels(items, sh), prompts,
                                            incl=False, n_boot=args.n_boot)
                            for name, sh in zip(ANNOTATORS, sheets, strict=True)}
    # descriptive: the adjudication side, recomputed from the sheet itself
    res["adjudication"] = {"gold_info": info}
    if args.adjudicated:
        with open(args.adjudicated, newline="") as f:
            adj_rows = [r for r in csv.DictReader(f) if not r["item_id"].startswith("#")]
        res["adjudication"].update(humantwin.adjudication_summary(adj_rows, ANNOTATORS))
    res["overlap_d023_audit"] = manifest["overlap_d023_audit"]
    res.update({"sheets": args.sheets, "adjudicated": args.adjudicated,
                "sheet_sha256": {s: sha256(Path(s)) for s in args.sheets
                                 + ([args.adjudicated] if args.adjudicated else [])},
                "n_boot": args.n_boot, "bootstrap_seed": 0, "git_commit": commit,
                "command": cmd, "script": "scripts/human_twin_023.py analyze"})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    if not args.dryrun and out.name in REPO_COPY_NAMES:
        # the repo copy carries --out's file name, so a rerun to a new name never overwrites it;
        # only the two names of record are copied, so a scratch run never lands in the tree
        repo_copy = REPO_COPY.with_name(out.name)
        repo_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, repo_copy)
    p = res["primary"]["ci"]
    print(json.dumps({"dryrun": args.dryrun, "reading": res["primary"]["reading"],
                      **{k: p[k] for k in ("gap_human", "gap_judge", "gap_judge_minus_human",
                                            "relgap_judge_minus_human")},
                      "gap_human_per_annotator": {a: res["per_annotator"][a]["ci"]["gap_human"]
                                                  for a in ANNOTATORS},
                      "adjudication": {k: v for k, v in res["adjudication"].items()
                                       if k in ("final_matches", "final_label_counts")},
                      "kappa": res["inter_annotator"]}, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("draw")
    p.add_argument("--out", required=True)
    p = sub.add_parser("dryrun")
    p.add_argument("--dir", required=True)
    p.add_argument("--stage", choices=("sheets", "adjudicate"), required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--dir", required=True)
    p.add_argument("--sheets", nargs=2, required=True)
    p.add_argument("--adjudicated")
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--dryrun", action="store_true")
    args = ap.parse_args()
    {"draw": cmd_draw, "dryrun": cmd_dryrun, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
