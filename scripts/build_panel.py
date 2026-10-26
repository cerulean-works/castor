"""Build the frozen, decontaminated prompt panel.

    uv run --frozen python scripts/build_panel.py --version v1 --workers 11 [--out DIR]

Reads pinned source copies under data/raw/src (see castor.panel.SOURCES) and the pinned
post-training mixes under /work/harrychang08/castor/data/tulu3. Writes
data/panel/panel_<version>.jsonl, data/panel/decontam_<version>.json and
data/panel/MANIFEST.json.
"""

import argparse
import glob
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq

from castor import decontam, panel, provenance

ROOT = Path(__file__).resolve().parent.parent
TULU = Path("/work/harrychang08/castor/data/tulu3")
# HF repo, commit sha, message columns
TRAIN_SETS = {
    "sft": ("allenai/tulu-3-sft-olmo-2-mixture-0225", "d91a0785ade02942520280fb484866fce41e448f",
            ["messages"]),
    "dpo": ("allenai/olmo-2-0425-1b-preference-mix", "c6454f3d364622718a8ecd3d307e524febbba569",
            ["chosen", "rejected"]),
    "rlvr1": ("allenai/RLVR-GSM-MATH-IF-Mixed-Constraints",
              "7dbd180f5440c0b90f2944e6efea934b85437a95", ["messages"]),
    "rlvr2": ("allenai/RLVR-MATH", "bd2a93551b503a395fadd1a740d957559cfe6f3c", ["messages"]),
}
LEN_TOKENIZER = ("allenai/DataDecide-dolma1_7-1B", "step69369-seed-default")
LONG = 1536  # prompt tokens; leaves 512 for the response in DataDecide's 2048 context

_INDEX: decontam.PanelIndex | None = None


def _init(prompts: list[str]) -> None:
    global _INDEX
    _INDEX = decontam.PanelIndex(prompts)


def _scan(job: tuple[str, str, list[str]]) -> tuple[str, str, int, list, list]:
    """One parquet file -> docs with panel 8-gram hits, and exact matches."""
    name, path, cols = job
    n_docs, with_hits, exact = 0, [], []
    shard = Path(path).stem
    # stream in batches: a whole SFT shard as Python objects is several GB
    rows = (r for b in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=cols)
            for r in b.to_pylist())
    for row_i, row in enumerate(rows):
        turns = []
        for c in cols:
            turns.extend(m["content"] for m in row[c] if m["role"] == "user")
        # chosen/rejected repeat the same user turns; index each distinct turn once
        for t_i, text in enumerate(dict.fromkeys(turns)):
            n_docs += 1
            doc_id = f"{name}:{shard}:{row_i}:{t_i}"
            grams, ex = _INDEX.hits(text)
            if grams:
                with_hits.append((doc_id, grams))
            exact.extend((pid, doc_id) for pid in ex)
    return name, path, n_docs, with_hits, exact


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "data/panel"))
    args = ap.parse_args()
    commit, command = provenance.git_commit(), provenance.command()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = panel.load_all()
    raw_counts = Counter(r.source for r in records)
    records, dup_dropped = panel.dedupe(records)
    print(f"loaded {sum(raw_counts.values())}, after dedupe {len(records)}", flush=True)
    prompts = [r.prompt for r in records]

    jobs = []
    for name, (repo, sha, cols) in TRAIN_SETS.items():
        d = TULU / f"{repo.split('/')[1]}@{sha[:12]}" / "data"
        files = sorted(glob.glob(str(d / "*.parquet")))
        assert files, f"missing {d}"
        jobs += [(name, f, cols) for f in files]

    doc_counts: Counter = Counter()
    hits_by_set: dict[str, list] = defaultdict(list)
    exact_by_set: dict[str, dict[int, list[str]]] = {n: defaultdict(list) for n in TRAIN_SETS}
    with Pool(args.workers, initializer=_init, initargs=(prompts,)) as pool:
        # imap (ordered), not imap_unordered: the report must not depend on which shard
        # finishes first
        for name, path, n_docs, with_hits, exact in pool.imap(_scan, jobs):
            doc_counts[name] += n_docs
            hits_by_set[name] += with_hits
            for pid, doc_id in exact:
                exact_by_set[name][pid].append(doc_id)
            print(f"scanned {name} {Path(path).name}: {n_docs} user turns, "
                  f"{len(with_hits)} with panel 8-grams", flush=True)

    index = decontam.PanelIndex(prompts)
    per_set = {name: decontam.score(index, hits_by_set[name]) for name in TRAIN_SETS}

    why = {pid: w for pid in range(len(records))
           if (w := decontam.removal_reasons(per_set, exact_by_set, pid))}
    templ = panel.template_level_removals(records, set(why), panel.humanjb_templates())
    removed = []
    for pid, r in enumerate(records):
        if pid in why or pid in templ:
            x = {"source": r.source, "source_id": r.source_id, "matches": why.get(pid, {}),
                 "n_tokens": len(index.tokens[pid])}
            if pid in templ:
                x["template"] = templ[pid]
            removed.append(x)
    keep = [r for pid, r in enumerate(records) if pid not in why and pid not in templ]

    panel.assign_ids(keep)
    panel_path = out / f"panel_{args.version}.jsonl"
    with panel_path.open("w") as f:
        for r in keep:
            f.write(json.dumps(panel.to_row(r), ensure_ascii=False) + "\n")

    # Prompt lengths under the DataDecide tokenizer (reported, not filtered).
    # tokenizer_config.json names hf_olmo's OLMoTokenizer; tokenizer.json is a plain fast
    # tokenizer (castor/convert.py), so load that file directly.
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(
        LEN_TOKENIZER[0], "tokenizer.json", revision=LEN_TOKENIZER[1]))
    lens = defaultdict(list)
    for r in keep:
        lens[r.source].append(len(tok(r.prompt)["input_ids"]))

    by_source = panel.source_counts(raw_counts, dup_dropped, keep, removed, list(TRAIN_SETS))
    for src, c in by_source.items():
        c.update({
            "tokens_median": statistics.median(lens[src]) if lens[src] else None,
            "tokens_max": max(lens[src]) if lens[src] else None,
            f"tokens_gt_{LONG}": sum(1 for n in lens[src] if n > LONG),
        })
    near = {name: sum(1 for s in per_set[name] if 0.3 < s["score"] <= decontam.THRESHOLD)
            for name in TRAIN_SETS}

    report = {
        "version": args.version,
        "method": {"n": decontam.N, "threshold": decontam.THRESHOLD,
                   "tokenizer": decontam.TULU_SPLIT.pattern, "unit": "training user turn",
                   "rule": "remove if best single-document 8-gram token coverage > 0.5, or "
                           "exact normalized match with a training user turn; then remove "
                           "every instance of a HumanJailbreaks template with any removed "
                           "instance (template level, notes/002 Q5)"},
        "train_sets": {n: {"repo": r, "sha": s, "user_turns": doc_counts[n]}
                       for n, (r, s, _) in TRAIN_SETS.items()},
        "by_source": by_source,
        "near_misses_0.3_to_0.5": near,
        "dup_dropped": dup_dropped,
        "removed": removed,
        "length_tokenizer": f"{LEN_TOKENIZER[0]}@{LEN_TOKENIZER[1]}",
        "git_commit": commit,
        "command": command,
    }
    report_path = out / f"decontam_{args.version}.json"
    report_path.write_text(json.dumps(report, indent=1))

    manifest = {
        "version": args.version,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "panel": {"path": provenance.rel(panel_path), "sha256": sha256(panel_path),
                  "n": len(keep)},
        "decontam_report": {"path": provenance.rel(report_path),
                            "sha256": sha256(report_path)},
        "sources": panel.SOURCES,
        "train_sets": report["train_sets"],
        "counts": {s: by_source[s]["final"] for s in panel.SOURCE_ORDER},
        "command": command,
        "git_commit": commit,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(by_source, indent=1))
    print(f"panel: {len(keep)} prompts -> {panel_path}")


if __name__ == "__main__":
    main()
