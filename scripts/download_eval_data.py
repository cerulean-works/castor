"""Download the 11 held-out perplexity sets of OLMo's `all-small-ppl-validation` evaluator.

The paths are the ones the ladder config lists (`v3_small_gptneox20b/<set>/val/part-0-00000.npy`,
read from a config written by shell/olmo/make_config.py). There is no upstream sha256 (the files
are not in allenai/DataDecide-data-recipes), so a file is kept only if its size equals the
mirror's Content-Length and its MD5 equals the mirror's ETag (single-part S3-style ETags are the
content MD5).

    uv run --frozen python scripts/download_eval_data.py \
        --config /work/harrychang08/castor/runs/003/eval/config-150M.yaml \
        --root /work/harrychang08/castor/data/datadecide-recipes
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import requests
import yaml

from castor import provenance

MIRROR = "https://olmo-data.org"
LOCAL_ROOT = "/work/harrychang08/castor/data/datadecide-recipes"


def eval_paths(config: Path, local_root: str) -> list[str]:
    cfg = yaml.safe_load(config.read_text())
    ev = [e for e in cfg["evaluators"] if e["label"] == "all-small-ppl-validation"]
    assert len(ev) == 1
    paths = [p for ps in ev[0]["data"]["datasets"].values() for p in ps]
    assert all(p.startswith(local_root + "/") for p in paths), paths[:2]
    return sorted(p[len(local_root) + 1:] for p in paths)


def md5(p: Path) -> str:
    h = hashlib.md5()  # compared with the server's ETag, not used for security
    h.update(p.read_bytes())
    return h.hexdigest()


def fetch(rel: str, root: Path) -> dict:
    url = f"{MIRROR}/{rel}"
    head = requests.head(url, timeout=60)
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    etag = head.headers["ETag"].strip('"')
    assert re.fullmatch(r"[0-9a-f]{32}", etag), f"multipart ETag, not an MD5: {etag}"
    dst = root / rel
    if dst.exists() and dst.stat().st_size == size and md5(dst) == etag:
        status = "skip"
    else:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_bytes(r.content)
        if tmp.stat().st_size != size or md5(tmp) != etag:
            raise RuntimeError(f"{rel}: size/md5 mismatch after download")
        tmp.rename(dst)
        status = "ok"
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    return {"path": rel, "bytes": size, "md5_etag": etag, "sha256": sha, "status": status}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path(LOCAL_ROOT))
    args = ap.parse_args()
    commit = provenance.git_commit()
    rows = [fetch(rel, args.root) for rel in eval_paths(args.config, str(args.root))]
    for r in rows:
        print(r["status"], r["bytes"], r["path"], flush=True)
    (args.root / "eval-data" / "SHA256SUMS").write_text(
        "".join(f"{r['sha256']}  {r['path']}\n" for r in rows))
    manifest = {"mirror": MIRROR, "files": rows, "git_commit": commit,
                "command": provenance.command()}
    (args.root / "eval-data" / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
