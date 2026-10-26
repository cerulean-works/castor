"""Prompt panel: source loaders and the common schema.

Every loader reads a pinned local copy of the original source (see SOURCES) and returns
records with the panel schema. `build_panel` assigns ids and drops exact cross-source
duplicates. Decontamination lives in `castor.decontam`.
"""

import ast
import csv
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "src"

# Original source for each panel component, pinned to the revision that was downloaded.
SOURCES = {
    "harmbench": {
        "url": "https://github.com/centerforaisafety/HarmBench",
        "revision": "8e1604d1171fe8a48d8febecd22f600e462bdcdd",
        "license": "MIT",
    },
    "sorrybench": {
        "url": "https://huggingface.co/datasets/sorry-bench/sorry-bench-202503",
        "revision": "612a4e1f45db8adf884fa62318ddf9fa1c6e75e9",
        "license": "SORRY-Bench dataset agreement (gated; no redistribution)",
    },
    "strongreject": {
        "url": "https://github.com/alexandrasouly/strongreject",
        "revision": "f7cad6c17e624e21d8df2278e918ae1dddb4cb56",
        "license": "MIT",
    },
    "xstest": {
        "url": "https://github.com/paul-rottger/xstest",
        "revision": "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d",
        "license": "CC-BY-4.0",
    },
    "orbench": {
        "url": "https://huggingface.co/datasets/bench-llm/or-bench",
        "revision": "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272",
        "license": "CC-BY-4.0",
    },
    "wildjailbreak": {
        "url": "https://huggingface.co/datasets/allenai/wildjailbreak",
        "revision": "5ddc12a7894f842b0619b8e1c7ee496b198af009",
        "license": "ODC-BY (gated; AI2 Responsible Use)",
    },
}

# Source ordering decides which copy survives an exact cross-source duplicate.
SOURCE_ORDER = [
    "harmbench", "harmbench_humanjb", "sorrybench", "strongreject", "xstest", "orbench",
    "wildjailbreak",
]


@dataclass
class Record:
    source: str
    source_split: str
    category: str
    is_harmful: bool
    is_adversarial: bool
    prompt: str
    source_url: str
    license: str
    source_id: str = ""  # id inside the source, kept for traceability
    id: str = ""


def _rec(src_key: str, source: str, **kw) -> Record:
    s = SOURCES[src_key]
    return Record(source=source, source_url=f"{s['url']}@{s['revision']}", license=s["license"],
                  **kw)


def _harmbench_rows(raw: Path) -> list[dict]:
    path = raw / "HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    return list(csv.DictReader(path.open()))


def _harmbench_behavior(row: dict) -> str:
    # HarmBench formats contextual behaviors as context, separator, then the request
    # (baselines/direct_request and human_jailbreaks both use this exact string).
    if row["ContextString"]:
        return f"{row['ContextString']}\n\n---\n\n{row['Behavior']}"
    return row["Behavior"]


def _harmbench_split(raw: Path) -> dict[str, str]:
    split = {}
    for name in ("val", "test"):
        path = raw / f"HarmBench/data/behavior_datasets/harmbench_behaviors_text_{name}.csv"
        for r in csv.DictReader(path.open()):
            split[r["BehaviorID"]] = name
    return split


def load_harmbench(raw: Path = RAW) -> list[Record]:
    """Standard + contextual text behaviors, both splits. Copyright behaviors are excluded:
    HarmBench scores them with a hashing classifier for verbatim reproduction, not refusal."""
    split = _harmbench_split(raw)
    out = []
    for r in _harmbench_rows(raw):
        if r["FunctionalCategory"] == "copyright":
            continue
        out.append(_rec("harmbench", "harmbench", source_split=f"{split[r['BehaviorID']]}/"
                        f"{r['FunctionalCategory']}", category=r["SemanticCategory"],
                        is_harmful=True, is_adversarial=False, prompt=_harmbench_behavior(r),
                        source_id=r["BehaviorID"]))
    return out


def harmbench_humanjb_cases(raw: Path = RAW) -> dict[str, list[tuple[int, str]]]:
    """Reproduce HarmBench's HumanJailbreaks test cases exactly as its default pipeline runs
    them (configs/pipeline_configs/run_pipeline.yaml: experiment `random_subset_5`, seed 1):
    seed Python's RNG once, then for each behavior in file order shuffle the template list in
    place and take the first 5 (baselines/human_jailbreaks/human_jailbreaks.py).
    Returns behavior id -> [(template index in the original list, test case)]."""
    templates = humanjb_templates(raw)
    # Shuffle template indices rather than strings: random.shuffle's permutation depends
    # only on the list length, so this reproduces HarmBench's draws while keeping the index
    # (the list has one duplicated template, at indices 1 and 33).
    order = list(range(len(templates)))
    rng_state = random.getstate()
    random.seed(1)
    cases = {}
    try:
        for r in _harmbench_rows(raw):
            random.shuffle(order)
            behavior = _harmbench_behavior(r)
            cases[r["BehaviorID"]] = [(i, f"{templates[i]}\n\n{behavior}") for i in order[:5]]
    finally:
        random.setstate(rng_state)
    return cases


def _join_surrogates(text: str) -> str:
    """jailbreaks.py spells emoji as escaped UTF-16 surrogate pairs ("\\ud83d\\udd13").
    HarmBench's pipeline round-trips test cases through json.dump/json.load, which joins each
    pair into the real character; do the same so the prompt is valid UTF-8."""
    return text.encode("utf-16", "surrogatepass").decode("utf-16")


def load_harmbench_humanjb(raw: Path = RAW) -> list[Record]:
    split = _harmbench_split(raw)
    rows = {r["BehaviorID"]: r for r in _harmbench_rows(raw)}
    out = []
    for bid, cases in harmbench_humanjb_cases(raw).items():
        r = rows[bid]
        if r["FunctionalCategory"] == "copyright":
            continue
        for tidx, text in cases:
            out.append(_rec("harmbench", "harmbench_humanjb",
                            source_split=f"{split[bid]}/{r['FunctionalCategory']}",
                            category=r["SemanticCategory"], is_harmful=True,
                            is_adversarial=True, prompt=text, source_id=f"{bid}::jb{tidx}"))
    return out


def load_sorrybench(raw: Path = RAW) -> list[Record]:
    """Base prompts only (question.jsonl, 440 = 44 categories x 10); the 20 linguistic
    mutation files are not used."""
    d = raw / "sorry-bench__sorry-bench-202503"
    tree = ast.parse((d / "meta_info.py").read_text())
    cats = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and n.targets[0].id == "category_descriptions")
    out = []
    for line in (d / "question.jsonl").open():
        q = json.loads(line)
        assert q["prompt_style"] == "base" and len(q["turns"]) == 1
        c = int(q["category"])
        out.append(_rec("sorrybench", "sorrybench", source_split="202503/base",
                        category=f"{c}: {cats[c - 1]}", is_harmful=True, is_adversarial=False,
                        prompt=q["turns"][0], source_id=str(q["question_id"])))
    return out


def load_strongreject(raw: Path = RAW) -> list[Record]:
    path = raw / "strongreject/strongreject_dataset/strongreject_dataset.csv"
    return [
        _rec("strongreject", "strongreject", source_split=f"full/{r['source']}",
             category=r["category"], is_harmful=True, is_adversarial=False,
             prompt=r["forbidden_prompt"], source_id=str(i))
        for i, r in enumerate(csv.DictReader(path.open()))
    ]


def load_xstest(raw: Path = RAW) -> list[Record]:
    """250 safe prompts (over-refusal probes) and 200 unsafe contrast prompts."""
    path = raw / "xstest/xstest_prompts.csv"
    return [
        _rec("xstest", "xstest", source_split=r["label"], category=r["type"],
             is_harmful=r["label"] == "unsafe", is_adversarial=False, prompt=r["prompt"],
             source_id=r["id"])
        for r in csv.DictReader(path.open())
    ]


def load_orbench(raw: Path = RAW) -> list[Record]:
    """OR-Bench-Hard-1K: seemingly toxic but benign prompts (over-refusal)."""
    df = pd.read_csv(raw / "bench-llm__or-bench/or-bench-hard-1k.csv")
    return [
        _rec("orbench", "orbench", source_split="hard-1k", category=r.category,
             is_harmful=False, is_adversarial=False, prompt=r.prompt, source_id=str(i))
        for i, r in enumerate(df.itertuples())
    ]


def load_wildjailbreak(raw: Path = RAW) -> list[Record]:
    """Eval split: 2000 adversarial harmful + 210 adversarial benign."""
    df = pd.read_csv(raw / "allenai__wildjailbreak/eval/eval.tsv", sep="\t",
                     keep_default_na=False)
    return [
        _rec("wildjailbreak", "wildjailbreak", source_split=f"eval/{r.data_type}",
             category=r.data_type, is_harmful=r.data_type == "adversarial_harmful",
             is_adversarial=True, prompt=r.adversarial, source_id=str(i))
        for i, r in enumerate(df.itertuples())
    ]


LOADERS = {
    "harmbench": load_harmbench,
    "harmbench_humanjb": load_harmbench_humanjb,
    "sorrybench": load_sorrybench,
    "strongreject": load_strongreject,
    "xstest": load_xstest,
    "orbench": load_orbench,
    "wildjailbreak": load_wildjailbreak,
}


def normalize(text: str) -> str:
    """Key for exact-duplicate detection: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def dedupe(records: list[Record]) -> tuple[list[Record], list[dict]]:
    """Keep the first copy of each normalized prompt (records must be in SOURCE_ORDER)."""
    seen: dict[str, Record] = {}
    kept, dropped = [], []
    for r in records:
        k = normalize(r.prompt)
        if k in seen:
            first = seen[k]
            dropped.append({"source": r.source, "source_id": r.source_id,
                            "kept_source": first.source, "kept_source_id": first.source_id})
            continue
        seen[k] = r
        kept.append(r)
    return kept, dropped


def assign_ids(records: list[Record]) -> None:
    counts: dict[str, int] = {}
    for r in records:
        counts[r.source] = counts.get(r.source, 0) + 1
        r.id = f"{r.source}-{counts[r.source]:05d}"


def load_all(raw: Path = RAW) -> list[Record]:
    out = []
    for name in SOURCE_ORDER:
        out.extend(LOADERS[name](raw))
    return out


PANEL_FIELDS = ["id", "source", "source_split", "category", "is_harmful", "is_adversarial",
                "prompt", "source_url", "license", "source_id"]


def to_row(r: Record) -> dict:
    d = asdict(r)
    return {k: d[k] for k in PANEL_FIELDS}


def humanjb_templates(raw: Path = RAW) -> list[str]:
    """HarmBench's HumanJailbreaks template list, in its original order."""
    sys.path.insert(0, str(raw / "HarmBench/baselines/human_jailbreaks"))
    try:
        from jailbreaks import JAILBREAKS
    finally:
        sys.path.pop(0)
    return [_join_surrogates(t) for t in JAILBREAKS]


def humanjb_template_index(r: Record) -> int:
    return int(r.source_id.rsplit("::jb", 1)[1])


def template_level_removals(records: list[Record], removed: set[int],
                            templates: list[str]) -> dict[int, dict]:
    """A HumanJailbreaks template seen in training is not held out, so if any
    instance of a template is decontaminated, every instance of it goes. Templates are keyed
    by text (the list has one duplicated template, at indices 1 and 33).
    Returns record index -> reason, for records not already in `removed`."""
    hit: dict[str, list[str]] = {}
    for i in sorted(removed):
        r = records[i]
        if r.source == "harmbench_humanjb":
            hit.setdefault(templates[humanjb_template_index(r)], []).append(r.source_id)
    out = {}
    for i, r in enumerate(records):
        if i in removed or r.source != "harmbench_humanjb":
            continue
        t = templates[humanjb_template_index(r)]
        if t in hit:
            out[i] = {"template_idx": humanjb_template_index(r),
                      "contaminated_instances": hit[t]}
    return out


def source_counts(raw_counts: dict[str, int], dup_dropped: list[dict], keep: list[Record],
                  removed: list[dict], train_sets: list[str]) -> dict[str, dict]:
    """Per-source panel counts for the decontamination report. `removed` entries carry
    `matches` (training set -> reasons, empty for template-level removals) and optionally
    `template`."""
    out = {}
    for src in SOURCE_ORDER:
        rs = [r for r in keep if r.source == src]
        rm = [x for x in removed if x["source"] == src]
        inst = [x for x in rm if x["matches"]]
        out[src] = {
            "raw": raw_counts.get(src, 0),
            "dup_dropped": sum(1 for d in dup_dropped if d["source"] == src),
            "decontam_removed": len(inst),
            "removed_by_set": {n: sum(1 for x in inst if n in x["matches"]) for n in train_sets},
            "removed_exact_only": sum(1 for x in inst if all(
                "ngram_score" not in m for m in x["matches"].values())),
            "template_removed": sum(1 for x in rm if not x["matches"] and "template" in x),
            "final": len(rs),
            "harmful": sum(r.is_harmful for r in rs),
            "adversarial": sum(r.is_adversarial for r in rs),
        }
    return out
