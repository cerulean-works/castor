"""Check that labelled audit sheets are complete and untouched.

    uv run --frozen python scripts/audit_check_sheets.py --audit AUDIT_DIR \
        --sheets sheet_harry.csv sheet_coauthor.csv --out AUDIT_DIR/check_sheets.json

Label-only check: it never reads a judge label, so it is safe before adjudication.
For each sheet: the item ids equal `key.jsonl`'s; every item has a valid label (R/H/N/I);
the partial cell is 1/y/yes, 0 or blank (blank means not partial; counted); and the prompt and
response text equal the generations the draw read (manifest `generations`, joined on model,
prompt_id, sample), i.e. nobody edited the text while labelling.

008's recipe draw: its key rows carry `model` = the recipe (the stratum key) and `model_id` = the
real model, so the text is joined on (model_id, prompt_id, sample) by `audit.read_generations`
over the files its manifest records (`generations_by_model`). `--grid-manifest
data/manifests/grid_008.json [--runs DIR]` derives the files instead, by the draw's `grid_models`
/ `generation_files`: needed for a draw whose manifest does not record them, and it overrides
recorded paths that no longer resolve. It is refused on a draw whose key rows carry no
`model_id`. A key row with no text is reported (`text_missing`), not raised.
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

from castor import audit, grid, provenance
from castor.judge.common import KEY, key, read_jsonl

PARTIAL_YES, PARTIAL_NO = ("1", "y", "Y", "yes"), ("0", "")
TEXT_KEY_008 = ("model_id", "prompt_id", "sample")


def _sampler_008():
    spec = importlib.util.spec_from_file_location(
        "audit_sample_008", Path(__file__).parent / "audit_sample_008.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check_sheet(path: str, keyrows: dict, text: dict, key=key) -> dict:
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if not r["item_id"].startswith("#")]
    ids = [r["item_id"] for r in rows]
    bad_label = [r["item_id"] for r in rows
                 if r["label"].strip().upper()[:1] not in audit.HUMAN]
    bad_partial = [r["item_id"] for r in rows
                   if r["partial"].strip() not in PARTIAL_YES + PARTIAL_NO]
    missing = [r["item_id"] for r in rows if r["item_id"] in keyrows
               and key(keyrows[r["item_id"]]) not in text]
    changed = [r["item_id"] for r in rows if r["item_id"] in keyrows
               and key(keyrows[r["item_id"]]) in text
               and (r["prompt"], r["response"]) != text[key(keyrows[r["item_id"]])]]
    return {
        "n_rows": len(rows), "n_unique_ids": len(set(ids)),
        "ids_match_key": sorted(ids) == sorted(keyrows),
        "missing_from_sheet": sorted(set(keyrows) - set(ids)),
        "not_in_key": sorted(set(ids) - set(keyrows)),
        "n_labelled": len(rows) - len(bad_label), "unlabelled_or_invalid": bad_label,
        "partial_invalid": bad_partial,
        "partial_blank": sum(r["partial"].strip() == "" for r in rows),
        "partial_yes": sum(r["partial"].strip() in PARTIAL_YES for r in rows),
        "text_changed": changed, "text_missing": missing,
        "ok": (sorted(ids) == sorted(keyrows) and len(set(ids)) == len(ids)
               and not bad_label and not bad_partial and not changed and not missing),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True, help="directory with key.jsonl, manifest.json")
    ap.add_argument("--sheets", nargs="+", required=True)
    ap.add_argument("--grid-manifest", help="008 draw only: derive the generation files from "
                    "the grid manifest instead of the audit manifest's generations_by_model")
    ap.add_argument("--runs", type=Path, help="with --grid-manifest: the draw's --runs "
                    f"(default {grid.RUNS})")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    commit, cmd = provenance.git_commit(), provenance.command()
    d = Path(args.audit)
    keyrows = {r["item_id"]: r for r in read_jsonl(d / "key.jsonl")}
    text = {}
    manifest = json.loads((d / "manifest.json").read_text())
    is_008 = bool(keyrows) and all("model_id" in r for r in keyrows.values())
    if args.runs is not None and not args.grid_manifest:
        raise SystemExit("--runs is read only with --grid-manifest")
    if args.grid_manifest and not is_008:
        raise SystemExit("--grid-manifest is for 008's draw; these key rows carry no model_id")
    files = manifest.get("generations_by_model")
    if args.grid_manifest:  # overrides the recorded files
        if files is not None:
            print("note: --grid-manifest given; the manifest's generations_by_model is ignored")
        s8 = _sampler_008()
        files = s8.generation_files(s8.grid_models(grid.load(args.grid_manifest),
                                                   args.runs or grid.RUNS))
    if files is not None:  # 008: (model_id, prompt_id, sample), the draw's own join
        if not is_008:
            raise SystemExit("manifest records generations_by_model but key rows lack model_id")
        fields = tuple(manifest.get("text_key", TEXT_KEY_008))
        if fields != TEXT_KEY_008:
            raise SystemExit(f"manifest text_key {fields} != {TEXT_KEY_008}")

        def kf(r):
            return tuple(r[k] for k in fields)
        want = {kf(r) for r in keyrows.values()}
        used = {m: files[m] for m in {k[0] for k in want} if m in files}
        gone = sorted(f for fs in used.values() for f in fs if not Path(f).exists())
        if gone:
            raise SystemExit(f"{len(gone)} generation files do not resolve (e.g. {gone[0]}); "
                             "pass --grid-manifest with the --runs where they are now")
        text = {k: (r["prompt"], r["response"])
                for k, r in audit.read_generations(used, want).items()}
    else:
        kf, fields = key, KEY
        want = {key(r) for r in keyrows.values()}
        for g in manifest["generations"]:  # stream; keep only the drawn rows
            for r in read_jsonl(g):
                if key(r) in want:
                    text[key(r)] = (r["prompt"], r["response"])
    res = {"n_key": len(keyrows), "n_text_found": len(text), "key_fields": fields,
           "sheets": {s: check_sheet(s, keyrows, text, kf) for s in args.sheets}}
    res["ok"] = len(text) == len(keyrows) and all(v["ok"] for v in res["sheets"].values())
    res.update({"git_commit": commit, "command": cmd, "script": "scripts/audit_check_sheets.py"})
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"ok": res["ok"], "n_text_found": len(text), **{
        Path(s).name: {k: (v if not isinstance(v, list) else len(v)) for k, v in c.items()}
        for s, c in res["sheets"].items()}}))


if __name__ == "__main__":
    main()
