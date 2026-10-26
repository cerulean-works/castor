"""Two-annotator pilot: confusion table, PABAK, kappa, disagreement classes, adjudication sheet.

    uv run --frozen python scripts/audit_disagree.py --sheets sheet_a.csv sheet_b.csv \
        --names harry coauthor --out-json AUDIT_DIR/disagree.json \
        --out-sheet AUDIT_DIR/sheet_adjudicate.csv

Kappa uses the scorer's own formula and needs no judge label, so it is safe before adjudication.
Sheets are parsed by `audit_score.read_sheet`. The adjudication sheet lists every item where the
two labels or the two partial flags differ; its `label` / `partial` columns are left empty for
the joint decision and are read by `audit_score.py --adjudicated`. It carries prompt and
response text and must stay on /work.
"""

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

from castor import audit, provenance

_spec = importlib.util.spec_from_file_location(
    "audit_score", Path(__file__).parent / "audit_score.py")
_score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_score)
read_sheet = _score.read_sheet

SHORT = {v: k for k, v in audit.HUMAN.items()}  # REFUSAL -> R, ...


def disagreement_class(a: str, b: str) -> str:
    """Unordered pair of short labels, 'I' pairs pooled as 'I<->*'."""
    x, y = sorted((SHORT[a], SHORT[b]))
    if x == y:
        return "agree"
    if "I" in (x, y):
        return "I<->*"
    return f"{x}<->{y}"


def compare(a: dict, b: dict) -> dict:
    both = sorted(set(a) & set(b))
    la, lb = [a[i]["label"] for i in both], [b[i]["label"] for i in both]
    cats = [audit.HUMAN[k] for k in "RNHI"]
    conf = {SHORT[x]: {SHORT[y]: sum(p == x and q == y for p, q in zip(la, lb)) for y in cats}
            for x in cats}
    classes = Counter(disagreement_class(p, q) for p, q in zip(la, lb))
    partial_only = [i for i in both if a[i]["label"] == b[i]["label"]
                    and bool(a[i]["partial"]) != bool(b[i]["partial"])]
    return {
        "n": len(both),
        "agreement_4class": sum(p == q for p, q in zip(la, lb)) / len(both),
        "pabak_4class": audit.pabak(la, lb, k=4),
        "pabak_refusal": audit.pabak([x == "REFUSAL" for x in la], [x == "REFUSAL" for x in lb]),
        "kappa_4class": audit.cohen_kappa(la, lb),
        "kappa_refusal": audit.cohen_kappa([x == "REFUSAL" for x in la],
                                           [x == "REFUSAL" for x in lb]),
        "confusion_rows_a_cols_b": conf,
        "classes": {k: v for k, v in classes.items() if k != "agree"},
        "partial_mismatch_same_label": len(partial_only),
        "label_disagreements": [i for i in both if a[i]["label"] != b[i]["label"]],
        "partial_only": partial_only,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", nargs=2, required=True)
    ap.add_argument("--names", nargs=2, default=["a", "b"])
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-sheet", required=True)
    args = ap.parse_args()
    a, b = (read_sheet(s) for s in args.sheets)
    res = compare(a, b)
    na, nb = args.names
    with open(args.sheets[0], newline="") as f:
        text = {r["item_id"]: r for r in csv.DictReader(f) if not r["item_id"].startswith("#")}
    rows = res["label_disagreements"] + res["partial_only"]
    with open(args.out_sheet, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "prompt", "response", na, f"{na}_partial", nb, f"{nb}_partial",
                    "class", "label", "partial", "notes"])
        w.writerow(["#", ("label / partial = the jointly adjudicated final label (R/H/N/I) and "
                    "partial flag (1 or empty); read by audit_score.py --adjudicated. Rows with "
                    "class 'partial-only' agree on the label; their final partial is recorded "
                    "but the scorer takes the agreed label from the two sheets."),
                    "", "", "", "", "", "", "", "", ""])
        for i in rows:
            w.writerow([i, text[i]["prompt"], text[i]["response"],
                        SHORT[a[i]["label"]], int(a[i]["partial"]),
                        SHORT[b[i]["label"]], int(b[i]["partial"]),
                        disagreement_class(a[i]["label"], b[i]["label"])
                        if a[i]["label"] != b[i]["label"] else "partial-only", "", "", ""])
    res.update({"names": args.names, "sheets": args.sheets, "adjudication_sheet": args.out_sheet,
                "git_commit": provenance.git_commit(), "command": provenance.command(),
                "script": "scripts/audit_disagree.py"})
    Path(args.out_json).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n", "kappa_refusal", "kappa_4class", "pabak_refusal",
                                          "pabak_4class", "classes",
                                          "partial_mismatch_same_label")}))


if __name__ == "__main__":
    main()
