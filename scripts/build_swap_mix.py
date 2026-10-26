"""Build the 080 swap SFT mixture and its gate report (D125-D128; castor/swapmix.py has the
rules).

    uv run --frozen python scripts/build_swap_mix.py pool --out <out> \
        --report $PWD/runs/080/swap_mix.json
    uv run --frozen python scripts/build_swap_mix.py turns --out <out> \
        --report $PWD/runs/080/swap_mix.json
    uv run --frozen python scripts/judge.py --stage wildguard \
        --generations <out>/judge_in/<shard>.jsonl --out <out>/labels/<shard>   # every shard
    uv run --frozen python scripts/build_swap_mix.py build --workers 11 --out <out> \
        --report $PWD/runs/080/swap_mix.json

`pool` writes every candidate of the five sets (shuffled, exact duplicates dropped) and every
removed-slice row to <out>/pool/, and the same rows in the judge's format as shards under
<out>/judge_in/. `turns` adds the earlier assistant turns of the multi-turn pool and slice
examples as judge_in/turns_<k>.jsonl, and must follow every `pool`. A shard whose text is
unchanged is not rewritten, so its labels stay current. `build` checks that the shards and their
labels describe the current pool, reads WildGuard's labels of all of them, sets the cap (the
slice's harmful-prompt refusals that pass 1 could take), selects the two passes (tier 1 unique
prompts, tier 2 repeats, D128), writes <out>/jsonl/<NAME>@<sha12>.jsonl (the Tulu rows outside
the safety slice byte for byte in their order, then the 110,800 replacements) and its .json
meta, and runs/080/swap_mix.json (selection, composition beside the slice's, prompt multiplicity
beside the slice's, agreement with Nemotron-SFT-Safety-v2's own labels, both-way
decontamination, DPO overlap, the stop rule). If the pools cannot fill 110,800 rows, the report
is written and no mixture.
"""

import argparse
import gzip
import json
import lzma
import os
from multiprocessing import get_context
from pathlib import Path

from castor import decontam, posttrain, provenance, swapmix
from castor.judge.common import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parent.parent
OUT = Path("/work/harrychang08/castor/data/swap")
NAME = "tulu-3-sft-olmo-2-mixture-0225-swap080"
PANEL = ROOT / "data/panel/panel_v1.jsonl"
HF = Path(os.environ.get("HF_HOME", "/work/harrychang08/hf")) / "hub"
SETS = {  # name -> (repo, revision, files under the snapshot); train splits only, in ORDER
    "nemotron_v2": ("nvidia/Nemotron-SFT-Safety-v2", "8a40a63c9a1a340874b874f980953be53bff0a07",
                    ["data/train.jsonl"]),
    "llama_nemotron": ("nvidia/Llama-Nemotron-Post-Training-Dataset",
                       "ab2a40d258a6a4d9d4c277d702aeea445081766c", ["SFT/safety/safety.jsonl"]),
    "pku": ("PKU-Alignment/PKU-SafeRLHF", "9421ffafec3fa40a1f1a7d567b4d525079477ecb",
            ["data/Alpaca-7B/train.jsonl", "data/Alpaca2-7B/train.jsonl",
             "data/Alpaca3-8B/train.jsonl"]),
    "beavertails": ("PKU-Alignment/BeaverTails", "8401fe609d288129cc684a9b3be6a93e41cfe678",
                    ["round0/330k/train.jsonl.xz"]),
    "hh": ("Anthropic/hh-rlhf", "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa",
           ["harmless-base/train.jsonl.gz"]),
}
SHARD = 90_000
DPO_CHUNK = 4000
SLICE_MODEL, POOL_MODEL, TURN_MODEL = "tulu_safety_slice", "swap080_pool", "swap080_turns"

_INDEX: decontam.PanelIndex | None = None  # built before the fork, shared by the workers


def snapshot(repo: str, sha: str) -> Path:
    return HF / f"datasets--{repo.replace('/', '--')}" / "snapshots" / sha


def read_lines(path: Path):
    op = lzma.open if path.suffix == ".xz" else gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        for i, line in enumerate(f):
            yield i, json.loads(line)


def pools() -> tuple[dict, dict]:
    """Every candidate of the five sets as records, each set in its fixed shuffle."""
    out, files = {}, {}
    for name, (repo, sha, rel_files) in SETS.items():
        snap = snapshot(repo, sha)
        paths = [snap / f for f in rel_files]
        files[name] = {"repo": repo, "revision": sha,
                       "files": {f: posttrain.sha256_file(p) for f, p in zip(rel_files, paths)}}
        if name == "nemotron_v2":
            ex = swapmix.nemotron_v2_examples(read_lines(paths[0]))
        elif name == "llama_nemotron":
            ex = ((i, m, None) for i, m in swapmix.llama_nemotron_examples(read_lines(paths[0])))
        elif name == "pku":
            rows = ((Path(f).parent.name, i, r) for f, p in zip(rel_files, paths)
                    for i, r in read_lines(p))
            ex = ((i, m, None) for i, m in swapmix.pku_examples(rows))
        elif name == "beavertails":
            ex = ((i, m, None) for i, m in swapmix.beavertails_examples(read_lines(paths[0])))
        else:
            ex = ((i, m, None) for i, m in swapmix.hh_examples(read_lines(paths[0])))
        recs = [{"id": i, "source": swapmix.SWAP_SOURCE[name], "messages": m}
                | ({"meta": meta} if meta else {}) for i, m, meta in ex]
        out[name] = swapmix.shuffled(recs)
        print(f"{name}: {len(out[name])} candidates", flush=True)
    return out, files


def _dpo_chunk(rows: list[dict]) -> dict:
    return swapmix.dpo_overlap(_INDEX, rows)


def _chunks(path: Path):
    cur = []
    for r in read_jsonl(path):
        cur.append({"chosen": r["chosen"]})
        if len(cur) == DPO_CHUNK:
            yield cur
            cur = []
    if cur:
        yield cur


def dpo_overlap(slice_turns: list[str], workers: int) -> dict:
    global _INDEX
    _INDEX = decontam.PanelIndex(slice_turns)
    path = posttrain.jsonl_path(posttrain.DPO_REPO, posttrain.DPO_SHA)
    with get_context("fork").Pool(workers) as pool:
        tot = swapmix.dpo_shares(pool.imap_unordered(_dpo_chunk, _chunks(path)))
    _INDEX = None
    return {**tot, "dpo_jsonl": str(path),
            "unit": "DPO row matches if any distinct user turn of `chosen` matches one slice "
                    "user turn: exact normalized string, or > 0.5 of its tokens in 8-grams of "
                    "that slice turn"}


def tulu_rows():
    """(line, row) of the Tulu jsonl of record."""
    yield from enumerate(read_jsonl(posttrain.jsonl_path(posttrain.SFT_REPO, posttrain.SFT_SHA)))


def slice_records() -> list[dict]:
    out = [{"id": f"slice/{i}", "source": r["source"], "messages": r["messages"]}
           for i, r in tulu_rows() if r["source"] in swapmix.SAFETY_SOURCES]
    assert len(out) == swapmix.SLICE_N, len(out)
    return out


def write_shards(out: Path, tag: str, rows: list[dict]) -> dict[str, str]:
    """judge_in/<tag>_<k>.jsonl of SHARD rows each; an unchanged shard keeps its file and mtime."""
    shards = {}
    for k in range(0, len(rows), SHARD):
        path = out / "judge_in" / f"{tag}_{k // SHARD}.jsonl"
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[k:k + SHARD])
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
        shards[path.stem] = str(path)
    return shards


def write_report(path: Path, key: str, value: dict, fresh: bool = False) -> None:
    report = json.loads(path.read_text()) if path.exists() and not fresh else {}
    report.update({"prompt": "080", "decision": "D125, D126, D127, D128", key: value})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1) + "\n")


def pool(args) -> None:
    commit, command = provenance.git_commit(), provenance.command()
    out = Path(args.out)
    for d in ("pool", "judge_in"):
        (out / d).mkdir(parents=True, exist_ok=True)
    raw, files = pools()
    cands, dups = swapmix.dedupe_pool(raw)
    slice_ = slice_records()
    flat = [r for n in swapmix.ORDER for r in cands[n]]
    write_jsonl(out / "pool" / "pool.jsonl", flat)
    write_jsonl(out / "pool" / "slice.jsonl", slice_)
    shards = {}
    for tag, model, rows in (("pool", POOL_MODEL, flat), ("slice", SLICE_MODEL, slice_)):
        shards |= write_shards(out, tag, [swapmix.judge_input(r, model) for r in rows])
    write_report(Path(args.report), "pool", {
        "sets": files, "order": list(swapmix.ORDER), "shuffle_seed": swapmix.SEED,
        "candidates": {n: len(raw[n]) for n in swapmix.ORDER},
        "exact_duplicates_dropped": dups,
        "after_dedupe": {n: len(cands[n]) for n in swapmix.ORDER},
        "slice_rows": len(slice_), "pool_jsonl": str(out / "pool" / "pool.jsonl"),
        "slice_jsonl": str(out / "pool" / "slice.jsonl"), "judge_shards": shards,
        "git_commit": commit, "command": command}, fresh=True)  # drops `turns`: run it again
    print(json.dumps({"after_dedupe": {n: len(cands[n]) for n in swapmix.ORDER},
                      "shards": list(shards)}))


def current_shards(out: Path, report: dict) -> list[str]:
    """The judge_in shards of the report's last pool and turns steps, after checking that turns
    read the current pool and slice and that judge_in holds no other shard."""
    if "pool" not in report:
        raise ValueError("the report has no pool step: run pool, then turns")
    turns_ = report.get("turns", {})
    if "inputs" not in turns_:
        raise ValueError("no turns step since the last pool step: run turns")
    for name, f in turns_["inputs"].items():
        if posttrain.sha256_file(out / "pool" / f"{name}.jsonl") != f["sha256"]:
            raise ValueError(f"pool/{name}.jsonl changed after turns read it: run turns")
    want = report["pool"]["judge_shards"].keys() | turns_["judge_shards"].keys()
    have = {p.stem for p in (out / "judge_in").glob("*.jsonl")}
    if have != want:
        raise ValueError(f"judge_in shards {sorted(have ^ want)} are not those of the last pool "
                         "and turns steps")
    return sorted(want)


def label_files(out: Path, stems: list[str]) -> list[Path]:
    """labels/<shard>/wildguard.jsonl of every shard in `stems`: one directory per shard, judged
    from that shard's path after the shard was last written (older labels may describe other
    text under the same ids). judge.py records no hash of its input, so mtime is the evidence."""
    dirs = {p.name for p in (out / "labels").iterdir() if p.is_dir()}
    if dirs != set(stems):
        raise ValueError(f"label dirs {sorted(dirs)} do not match judge_in shards {stems}")
    files = []
    for stem in stems:
        shard = out / "judge_in" / f"{stem}.jsonl"
        path = out / "labels" / stem / "wildguard.jsonl"
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        judged = [Path(g).resolve() for g in meta["generations"]]
        if judged != [shard.resolve()] or path.stat().st_mtime < shard.stat().st_mtime:
            raise ValueError(f"{path} was not judged from the current {shard}")
        files.append(path)
    return files


def turns(args) -> None:
    commit, command = provenance.git_commit(), provenance.command()
    out = Path(args.out)
    inputs = {name: out / "pool" / f"{name}.jsonl" for name in ("pool", "slice")}
    rows = [t for path in inputs.values() for r in read_jsonl(path)
            for t in swapmix.turn_inputs(r, TURN_MODEL)]
    shards = write_shards(out, "turns", rows)
    write_report(Path(args.report), "turns", {
        "unit": "every assistant turn but the last of a pool or slice example, judged as (the "
                "user turn before it, the turn)", "turns": len(rows), "judge_shards": shards,
        "inputs": {name: {"path": str(path), "sha256": posttrain.sha256_file(path)}
                   for name, path in inputs.items()},
        "git_commit": commit, "command": command})
    print(json.dumps({"turns": len(rows), "shards": list(shards)}))


def read_labels(files: list[Path], ids: set[str]) -> dict[str, dict]:
    labels = {}
    for path in files:
        for r in read_jsonl(path):
            if r["prompt_id"] in labels:
                raise ValueError(f"{r['prompt_id']} labelled twice")
            labels[r["prompt_id"]] = {**{k: r[k] for k in swapmix.LABELS},
                                      "truncated": r.get("truncated_to_fit", False)}
    missing = ids - labels.keys()
    if missing:
        raise ValueError(f"{len(missing)} rows have no WildGuard label, e.g. {sorted(missing)[:3]}")
    return labels


def build(args) -> None:
    commit, command = provenance.git_commit(), provenance.command()
    out = Path(args.out)
    (out / "jsonl").mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    tulu = posttrain.jsonl_path(posttrain.SFT_REPO, posttrain.SFT_SHA)

    recs = list(read_jsonl(out / "pool" / "pool.jsonl"))
    slice_ = list(read_jsonl(out / "pool" / "slice.jsonl"))
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    files = label_files(out, current_shards(out, report))
    labels = read_labels(files, {r["id"] for r in recs} | {r["id"] for r in slice_})
    earlier = swapmix.fold_earlier(labels, [*recs, *slice_])
    by_name = {v: k for k, v in swapmix.SWAP_SOURCE.items()}
    cands = {n: [] for n in swapmix.ORDER}
    for r in recs:
        cands[by_name[r["source"]]].append(r)
    cap = sum(swapmix.usable(labels[r["id"]], swapmix.harmful_refusal) for r in slice_)

    panel = [r["prompt"] for r in read_jsonl(PANEL)]
    index = decontam.PanelIndex(panel)
    drop, kept = set(), set()  # Tulu line numbers of the slice; norm_prompts of the kept rows
    with open(tulu) as f:
        for n_tulu, line in enumerate(f, 1):
            r = json.loads(line)
            if r["source"] in swapmix.SAFETY_SOURCES:
                drop.add(n_tulu - 1)
            else:
                kept.add(swapmix.norm_prompt(r["messages"]))
    if len(drop) != swapmix.SLICE_N:
        raise SystemExit(f"{len(drop)} Tulu rows in the safety slice, expected {swapmix.SLICE_N}")
    new, sel = swapmix.select(cands, labels, cap, swapmix.SLICE_N, index, frozenset(kept))
    del kept
    print(f"cap {cap}; selected {len(new)}, short {sel['short']}: " + json.dumps(
        {p: {n: [sel[p][n]["taken_tier1"], sel[p][n]["taken_tier2"]] for n in swapmix.ORDER}
         for p, _ in swapmix.PASSES}), flush=True)

    def rid(r):
        return r["id"].removeprefix("castor_swap/")
    comp = {"replacement": swapmix.composition(new, labels, rid) if new else None,
            "removed_slice": swapmix.composition(slice_, labels)}
    stop = swapmix.stop_rule(comp["replacement"]["refusal_rate"],
                             comp["removed_slice"]["refusal_rate"]) if new else None
    removed_ids = {p: {n: sel[p][n].pop("removed_ids") for n in swapmix.ORDER}
                   for p, _ in swapmix.PASSES}
    gate = {
        "labels": {"judge": "WildGuard (judge_v1 primary), raw prompt_harmful / refusal / "
                            "response_harmful on the (last user turn, final response) pair and on "
                            "each earlier (user turn, assistant turn) pair, no coherence stage; "
                            "a label on a response cut to fit the judge "
                            "makes the row ineligible",
                   "files": [str(p) for p in files],
                   "pool_rows": len(recs), "slice_rows": len(slice_),
                   "truncated": {"pool": sum(labels[r["id"]]["truncated"] for r in recs),
                                 "slice": sum(labels[r["id"]]["truncated"] for r in slice_),
                                 "turns": sum(c["turns_truncated"] for c in earlier.values())},
                   "earlier_turns": {"rule": "a multi-turn pool or slice row is eligible (pool) "
                                             "or counts toward the cap (slice) only if every "
                                             "earlier assistant turn is labelled, non-empty, "
                                             "uncut and not a harmful response",
                                     "by_source": earlier}},
        "cap": {"value": cap, "rule": "removed-slice rows pass 1 could take: prompt_harmful, "
                                      "refusal and not response_harmful on an uncut label, every "
                                      "earlier turn clean (castor.swapmix.usable), of all 110,800"},
        "selection": sel, "composition": comp,
        "prompt_multiplicity": {"key": "castor.swapmix.norm_prompt (decontam-normalized user turns)",
                                "replacement": swapmix.prompt_multiplicity(new),
                                "removed_slice": swapmix.prompt_multiplicity(slice_)},
        "agreement_nemotron_v2": {"pool": swapmix.agreement(cands["nemotron_v2"], labels)},
        "decontamination": {
            "panel": {"path": provenance.rel(PANEL), "sha256": posttrain.sha256_file(PANEL),
                      "n": len(panel)},
            "method": {"n": decontam.N, "threshold": decontam.THRESHOLD,
                       "tokenizer": decontam.TULU_SPLIT.pattern, "unit": "user turn",
                       "directions": ["panel prompt > 0.5 covered by one candidate turn",
                                      "candidate turn > 0.5 covered by one panel prompt",
                                      "exact normalized match"]},
            "removed_ids": removed_ids},
        "stop_rule": {"rule": "replacement refusal rate < half the slice's -> stop, ask /direct",
                      "replacement": comp["replacement"]["refusal_rate"] if new else None,
                      "slice": comp["removed_slice"]["refusal_rate"], "stop": stop},
        "git_commit": commit, "command": command, "script": "scripts/build_swap_mix.py"}
    if sel["short"]:
        write_report(report_path, "build", {**gate, "mixture": None})
        raise SystemExit(f"pools fill {len(new)} of {swapmix.SLICE_N}; report written, no mixture")

    tmp = out / "jsonl" / f"{NAME}.tmp.jsonl"
    n_kept = 0
    with open(tulu) as f, open(tmp, "w") as g:
        for i, line in enumerate(f):
            if i not in drop:
                g.write(line if line.endswith("\n") else line + "\n")
                n_kept += 1
        g.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in new)
    if n_kept != n_tulu - swapmix.SLICE_N:
        tmp.unlink()
        raise SystemExit(f"kept {n_kept} Tulu rows of {n_tulu}, expected {n_tulu - swapmix.SLICE_N}")
    sha = posttrain.sha256_file(tmp)
    final = out / "jsonl" / f"{NAME}@{sha[:12]}.jsonl"
    tmp.rename(final)

    slice_turns = list(dict.fromkeys(t for r in slice_ for t in swapmix.user_turns(r["messages"])))
    dpo = dpo_overlap(slice_turns, args.workers)
    mixture = {"path": str(final), "sha256": sha, "rows": n_kept + len(new),
               "kept_tulu_rows": n_kept, "replacement_rows": len(new),
               "source_tulu": {"repo": posttrain.SFT_REPO, "revision": posttrain.SFT_SHA,
                               "jsonl": str(tulu)}}
    write_report(report_path, "build", {**gate, "mixture": mixture, "dpo_overlap": dpo})
    # the flat fields scripts/posttrain_run.py render checks and records (as the Tulu export's)
    meta = {"repo": f"{posttrain.SFT_REPO} (080 swap, castor)", "revision": posttrain.SFT_SHA,
            "jsonl": str(final), "rows": n_kept + len(new), "sha256": sha,
            "report": str(args.report), "git_commit": commit, "command": command}
    final.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"mixture": str(final), "sha256": sha, "stop": stop,
                      "dpo_share_either": dpo["share_either"]}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pool", "turns", "build"])
    ap.add_argument("--out", default=str(OUT), help="absolute (D036)")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--report", required=True, help="absolute: runs/080/swap_mix.json in the repo")
    args = ap.parse_args()
    for name in ("out", "report"):
        if not Path(getattr(args, name)).is_absolute():
            ap.error(f"--{name} must be absolute (a job's snapshot worktree is deleted, D036)")
    {"pool": pool, "turns": turns, "build": build}[args.step](args)


if __name__ == "__main__":
    main()
