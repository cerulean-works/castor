"""Export the public castor repository: an allowlist of this repo at a fixed commit plus the
paper's numbers.json (as runs/numbers.json) from a fixed paper commit, copied into a separate git
repo (never this repo's history). The LaTeX source is not exported: arXiv hosts it.

    uv run --frozen python scripts/export_049.py --out /work/harrychang08/castor/release/049/export \
        [--commit HEAD] [--paper-commit HEAD] [--push https://github.com/cerulean-works/castor.git]

--push refuses while the scan finds long strings or missing aggregates (--push-anyway overrides).
"""

import argparse
import csv
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

import yaml

from castor import provenance, release

ROOT = provenance.ROOT
PAPER = ROOT / "paper"

CODE = ["castor", "scripts", "configs", "shell", "tests", "pyproject.toml", "uv.lock",
        ".python-version", "refs/refs.bib", "data/manifests",
        "data/panel/MANIFEST.json", "data/panel/EVAL_MANIFEST_v1.json",
        "data/directions/MANIFEST.json", "data/directions/refusal_tokens_v1.json"]
# the paper's number-to-source map, exported beside the aggregates it points at
NUMBERS = "runs/numbers.json"
# never exported, whatever the allowlist says: prompt text, generations, internal notes, and the
# tools and tests that only read or write the paper tree
DENY = re.compile(r"(^|/)(notes|prompts|docs|logs|\.claude)/|(^|/)(STATE|DECISIONS|CLAUDE)\.md$"
                  r"|^scripts/(paper_\w+|bib_audit|fig_016_trend|raster_05[79])\.py$"
                  r"|^tests/test_(paper_\w+|bib_audit|fig_016_trend)\.py$"
                  r"|panel_v\d+.*\.jsonl$|extraction_v\d+\.jsonl$|decontam_v\d+\.json$"
                  r"|wallet|samples\.jsonl$|greedy\.jsonl$|fallback12_recipes\.json$")
WHERE_PATH = re.compile(r"((?:runs|data)/[^\s:@,;()]+)")
PLACEHOLDER = re.compile(r"<[^<>/]*>")
LONG_STRING = 300
# recorded commands and job logs are provenance, not prompt or response text
COMMAND = re.compile(r"^(uv run |python |bash |node \S+ job \d+)")
# descriptive strings read by hand on 2026-10-23 (049 v1); a new or longer one blocks again
REVIEWED_LONG = {"runs/080/posthoc.json": 1, "runs/080/summary.json": 4}
# a released table carries ids and labels only: no column of annotator notes or other free text
FREE_TEXT = re.compile(r"^(notes?|comments?|text|prompt|response|generation|reason|rationale"
                       r"|explanation|free_?text)$", re.IGNORECASE)
GITIGNORE = ".venv/\n__pycache__/\n*.py[cod]\n.pytest_cache/\n.ruff_cache/\n*.egg-info/\n"


def git(*args, cwd=ROOT, stdin: str | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True, input=stdin).stdout


def tracked(commit: str, paths: list[str], cwd: Path) -> list[str]:
    out = git("ls-tree", "-r", "--name-only", commit, "--", *paths, cwd=cwd).split("\n")
    return [p for p in out if p and not DENY.search(p)]


def where_pattern(p: str) -> re.Pattern:
    """A where path as a full-path regex; each `<...>` placeholder is one path component part."""
    parts = PLACEHOLDER.split(p)
    return re.compile("[^/]+".join(re.escape(x) for x in parts) + r"\Z")


def aggregates(numbers: dict, commit: str, exported: set[str]) -> tuple[list[str], list[str]]:
    """runs/ sources to add to the export, and sources not covered. A data/ source counts only if
    the CODE allowlist already exports it: data/ is never widened from numbers.json."""
    want = set()
    for sec in numbers.values():
        if isinstance(sec, dict):
            for e in sec.values():
                if isinstance(e, dict):
                    want |= set(WHERE_PATH.findall(e.get("where", "")))
    runs = tracked(commit, ["runs"], ROOT)
    found, missing = [], []
    for p in sorted(want):
        pat = where_pattern(p)
        if p.startswith("data/"):
            hits = [h for h in exported if pat.match(h)]
        else:
            hits = [h for h in runs if pat.match(h)]
            found.extend(hits)
        if not hits:
            missing.append(p)
    return sorted(set(found)), missing


def read_blob(commit: str, path: str, cwd: Path) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=cwd, capture_output=True,
                          check=True).stdout


def long_strings(path: str, blob: bytes) -> int:
    """Strings long enough to be prompt or response text: JSON strings, table cells, text lines."""
    if path.endswith((".json", ".jsonl")):
        fields = [f.decode("utf-8", "replace")
                  for f in re.findall(rb'"((?:[^"\\]|\\.)*)"', blob)]
    elif path.endswith(".csv"):
        text = blob.decode("utf-8", "replace")
        fields = [c for row in csv.reader(io.StringIO(text)) for c in row]
    elif path.endswith((".tsv", ".psv", ".txt")):
        sep = {".tsv": "\t", ".psv": "|"}.get(Path(path).suffix)
        lines = blob.decode("utf-8", "replace").splitlines()
        fields = [c for line in lines for c in (line.split(sep) if sep else [line])]
    else:
        return 0
    n = sum(len(s) > LONG_STRING and not COMMAND.match(s) for s in fields)
    return 0 if n <= REVIEWED_LONG.get(path, 0) else n


def free_text_columns(path: str, blob: bytes) -> list[str]:
    """Table columns named like free text (csv, tsv and psv headers, jsonl row keys)."""
    text = blob.decode("utf-8", "replace")
    if path.endswith(".jsonl"):
        cols = {k for line in text.splitlines() if line.strip()
                for k in (json.loads(line) if line.lstrip().startswith("{") else {})}
    elif path.endswith((".csv", ".tsv", ".psv")):
        head = text.splitlines()[0] if text else ""
        sep = {".csv": ",", ".tsv": "\t", ".psv": "|"}[Path(path).suffix]
        cols = {c.strip().strip('"') for c in head.split(sep)}
    else:
        return []
    return sorted(c for c in cols if FREE_TEXT.match(c))


def scan(files: dict[str, bytes]) -> dict:
    rep = {"n_files": len(files), "bytes": sum(len(b) for b in files.values()),
           "em_dash": {}, "decision_refs": {}, "long_json_strings": {}, "large": {},
           "free_text_columns": {}}
    for p, b in files.items():
        t = b.decode("utf-8", "replace")
        if (n := t.count("\u2014")):
            rep["em_dash"][p] = n
        if (n := len(re.findall(r"\bD\d{3}[a-z]?\b", t))):
            rep["decision_refs"][p] = n
        if (n := long_strings(p, b)):
            rep["long_json_strings"][p] = n
        if (cols := free_text_columns(p, b)):
            rep["free_text_columns"][p] = cols
        if len(b) > 5_000_000:
            rep["large"][p] = len(b)
    rep["em_dash_total"] = sum(rep["em_dash"].values())
    rep["decision_refs_total"] = sum(rep["decision_refs"].values())
    return rep


README = """![Cerulean Labs Castor banner](assets/banner.png)

---

# CASTOR: Common Ancestry, Seed-Twinned Outcomes of Refusal

{paper}

Train the same language model twice, on the same data with the same recipe, changing only the
random seed. Do the two copies fail on the same jailbreaks?

CASTOR answers this with seed twins. The base models come from DataDecide: each data recipe and
size was pretrained under several seeds, so two bases that share a recipe differ only in their
pretraining seed. Every base receives the same safety post-training (SFT then DPO, the OLMo 2 1B
Instruct recipe) under several post-training seeds, and every resulting model answers the same
panel of public jailbreak and over-refusal prompts, judged per response. Comparing models that
share a pretraining run, share only a data recipe, or share nothing measures how much of a
model's per-prompt jailbreak vulnerability its pretraining seed fixes.

## What is released

- **Models** (Hugging Face, [`{org}`](https://huggingface.co/{org})): {n_models} post-trained
  models in {n_repos} repos, one repo per data recipe and size (`{org}/castor-<recipe>-<size>`)
  and one revision per model (`s<pretraining seed>-p<post-training seed>`): {n_grid} models at
  1B across {n_recipes} DataDecide data recipes, {n_small} twins at 150M and 300M, {n_rerun}
  models built on our own pretraining reruns of one recipe, and {n_swap} 1B models whose SFT
  safety data were replaced by other public safety data (revisions ending in `-swap080`).
- **Labels** ([`{labels}`](https://huggingface.co/datasets/{labels})): per-response WildGuard and
  Qwen3Guard labels for every model, keyed by benchmark prompt id. No prompt text, no generations.
- **This repository:** the code and the aggregate results behind every number in the paper. The
  paper's LaTeX source is on arXiv.

## Layout

| path | contents |
|---|---|
| `castor/` | library: panel, generation, judging, twin statistics, release |
| `scripts/` | one entry point per experiment step |
| `configs/` | post-training and evaluation configs; `third_party/` pins open-instruct, OLMo and OLMES |
| `shell/` | Slurm submission scripts |
| `runs/` | aggregate artifacts; `runs/numbers.json` names the source of each paper number |
| `tests/` | tests for the code that computes reported numbers |
| `refs/refs.bib` | BibTeX for the prior work the code and the paper cite |

## Setup

```
uv sync
uv run pytest tests    # skips tests whose inputs are not released
```

Evaluation prompts are not redistributed. The panel is rebuilt from the public benchmarks with
`scripts/build_panel.py` (see `data/panel/MANIFEST.json`). Training uses open-instruct and OLMo
unmodified at the commits pinned in `configs/third_party/`.

## Safety

The models were safety-tuned on public data, and their bases are public. They can still produce
harmful text when prompted, as the released labels show. Model outputs on harmful prompts are not
released.

## License

Code: Apache 2.0 (`LICENSE`). Model weights: Apache 2.0, as the DataDecide base models. Labels:
ODC-BY 1.0.

"""


def readme_counts(inv: dict) -> dict:
    """Grid, ladder, swap and rerun counts; every other prompt is a rerun."""
    n = {p: v["n"] for p, v in inv["by_prompt"].items()}
    ms = [m for m in inv["models"] if m["prompt"] == "008"]
    return {"n_models": inv["n_models"], "n_repos": len(inv["repos"]), "n_grid": n.get("008", 0),
            "n_recipes": len({m["recipe"] for m in ms}), "n_small": n.get("016", 0),
            "n_swap": n.get("080", 0),
            "n_rerun": sum(v for p, v in n.items() if p not in ("008", "016", "080"))}


def readme(cfg: dict, inv: dict) -> str:
    return (README.format(paper=release.paper_line(cfg), org=release.ORG,
                          labels=release.LABELS_REPO, **readme_counts(inv))
            + "\n".join(release.citation(cfg)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--paper-commit", default="HEAD")
    ap.add_argument("--push")
    ap.add_argument("--push-anyway", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    assert out.is_absolute(), "--out must be absolute"
    commit = git("rev-parse", args.commit).strip()
    pcommit = git("rev-parse", args.paper_commit, cwd=PAPER).strip()

    files = {p: read_blob(commit, p, ROOT) for p in tracked(commit, CODE, ROOT)}
    files[NUMBERS] = read_blob(pcommit, "numbers.json", PAPER)
    runs, missing = aggregates(json.loads(files[NUMBERS]), commit, set(files))
    files |= {p: read_blob(commit, p, ROOT) for p in runs}
    inv = json.loads(read_blob(commit, "runs/049/inventory.json", ROOT))
    # the README, LICENSE and conftest come from --commit like everything else, never the tree
    files["README.md"] = readme(yaml.safe_load(files["configs/release/card.yaml"]), inv).encode()
    files["LICENSE"] = files["configs/release/LICENSE"]
    files["assets/banner.png"] = files["configs/release/banner.png"]
    files["conftest.py"] = files["configs/release/export_conftest.py"]  # skip unreleased inputs
    files[".gitignore"] = GITIGNORE.encode()
    assert not [p for p in files if DENY.search(p)]

    if not (out / ".git").exists():
        out.mkdir(parents=True, exist_ok=True)
        git("init", "-q", "-b", "main", cwd=out)
        for key in ("user.name", "user.email"):
            git("config", key, git("config", key).strip(), cwd=out)
    for p in git("ls-files", cwd=out).split("\n"):
        if p:
            (out / p).unlink(missing_ok=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for p, b in sorted(files.items()):
            info = tarfile.TarInfo(p)
            info.size = len(b)
            info.mode = 0o755 if p.endswith(".sh") else 0o644
            tar.addfile(info, io.BytesIO(b))
    buf.seek(0)
    with tarfile.open(fileobj=buf) as tar:
        tar.extractall(out, filter="data")
    # stage the export only: a .venv or build output left in the checkout never gets committed
    git("rm", "-q", "-r", "--cached", "--ignore-unmatch", ".", cwd=out)
    git("add", "--pathspec-from-file=-", cwd=out, stdin="\n".join(sorted(files)) + "\n")
    changed = bool(git("status", "--porcelain", "--untracked-files=no", cwd=out).strip())
    if changed:
        git("commit", "-q", "-m", f"Export {commit[:7]} (paper {pcommit[:7]})", cwd=out)

    rep = scan(files) | {"commit": commit, "paper_commit": pcommit, "runs_aggregates": len(runs),
                         "missing_aggregates": missing, "license": "LICENSE" in files,
                         "export_head": git("rev-parse", "HEAD", cwd=out).strip(),
                         "git_commit": provenance.git_commit(), "command": provenance.command()}
    (out.parent / "export_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if not isinstance(v, dict)}, indent=1))
    if args.push:
        if rep["free_text_columns"]:  # never overridden
            raise SystemExit(f"not pushed, free-text columns: {rep['free_text_columns']}")
        blockers = {k: rep[k] for k in ("long_json_strings", "missing_aggregates") if rep[k]}
        if blockers and not args.push_anyway:
            raise SystemExit(f"not pushed, see {out.parent / 'export_report.json'}: "
                             f"{sorted(blockers)}")
        git("push", "-q", args.push, "main", cwd=out)


if __name__ == "__main__":
    main()
