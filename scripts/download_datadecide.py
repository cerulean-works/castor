"""Download final DataDecide checkpoints into the HF cache and write a manifest.

Each checkpoint is resolved by HF repo + branch revision (e.g.
allenai/DataDecide-dolma1_7-1B@step69369-seed-default); the manifest records the commit sha
the branch pointed to at download time, so later loads can pin it.

    uv run --frozen python scripts/download_datadecide.py --size 1B \
        --seeds default large-aux-2 large-aux-3 --out data/manifests/datadecide_1B.json
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ORG = "allenai"
IGNORE = ["training/*"]


def recipe_repos(api: HfApi, size: str) -> list[str]:
    ids = [m.id for m in api.list_models(author=ORG, search="DataDecide", limit=1000)]
    return sorted(i for i in ids if i.endswith(f"-{size}"))


def final_revision(api: HfApi, repo: str, seed: str, step: int | None) -> str:
    """Branch name of the requested (or highest) step for one seed."""
    steps = []
    for b in api.list_repo_refs(repo).branches:
        m = re.fullmatch(r"step(\d+)-seed-(.+)", b.name)
        if m and m.group(2) == seed:
            steps.append(int(m.group(1)))
    if not steps:
        raise ValueError(f"{repo}: no branches for seed {seed}")
    s = step if step is not None else max(steps)
    if s not in steps:
        raise ValueError(f"{repo}: seed {seed} has no step {s}")
    return f"step{s}-seed-{seed}"


def fetch(api: HfApi, repo: str, revision: str) -> dict:
    info = api.model_info(repo, revision=revision, files_metadata=True)
    # training/optim.pt (optimizer state, ~10 GB per 1B checkpoint) is not needed for
    # post-training or evaluation. Download by the resolved commit sha, not the branch, so
    # the files on disk are exactly the commit the manifest records.
    path = snapshot_download(repo, revision=info.sha, ignore_patterns=IGNORE)
    assert Path(path).name == info.sha, f"{repo}: snapshot {path} is not commit {info.sha}"
    files = {s.rfilename: s.size for s in info.siblings if not s.rfilename.startswith("training/")}
    return {
        "repo": repo,
        "revision": revision,
        "sha": info.sha,
        "bytes": sum(v or 0 for v in files.values()),
        "files": files,
        "local_path": path,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, help="DataDecide size suffix, e.g. 1B, 150M")
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--step", type=int, default=None, help="default: highest step per seed")
    ap.add_argument("--recipes", nargs="*", default=None, help="repo ids; default: all 25")
    # >1 concurrent snapshot_download calls race on tqdm's class lock inside thread_map
    # (AttributeError: 'tqdm' has no attribute '_lock'); each call already fetches its
    # files in parallel, so checkpoints go one at a time by default.
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    api = HfApi()
    repos = args.recipes or recipe_repos(api, args.size)
    jobs = [(r, final_revision(api, r, s, args.step)) for r in repos for s in args.seeds]
    print(f"{len(jobs)} checkpoints", flush=True)

    out = Path(args.out)
    done = {}
    if out.exists():
        done = {(e["repo"], e["revision"]): e for e in json.loads(out.read_text())["entries"]}

    entries = list(done.values())
    todo = [j for j in jobs if j not in done]
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, api, r, rev): (r, rev) for r, rev in todo}
        for f in as_completed(futs):
            e = f.result()
            entries.append(e)
            print(f"ok {e['repo']}@{e['revision']} {e['sha'][:10]} {e['bytes'] / 1e9:.2f} GB",
                  flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "size": args.size,
                "seeds": args.seeds,
                "n": len(entries),
                "total_bytes": sum(x["bytes"] for x in entries),
                "entries": sorted(entries, key=lambda x: (x["repo"], x["revision"])),
            }, indent=1))
    print(f"done: {len(entries)}/{len(jobs)}")


if __name__ == "__main__":
    main()
