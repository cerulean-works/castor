"""Download one DataDecide pretraining-data mix (pre-tokenised .npy shards), verified by sha256.

The file list is OLMo's own mix definition (dumped by shell/olmo/dump_mix.py), so the local
copy has exactly the files, in exactly the order, that OLMo's ladder config reads. Expected
size and sha256 come from the HF LFS metadata of allenai/DataDecide-data-recipes at a pinned
revision; bytes come from https://olmo-data.org (same paths, ~5x faster from NCHC). A file is
kept only if both match; a `<file>.sha256` sidecar marks it verified so reruns skip it.

    uv run --frozen python scripts/download_recipe_data.py \
        --paths configs/data/dclm-baseline-paths.json \
        --root /work/harrychang08/castor/data/datadecide-recipes --workers 24
"""

import argparse
import hashlib
import json
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import HfApi

from castor import provenance

REPO = "allenai/DataDecide-data-recipes"
REVISION = "3baf34ba"
MIRROR = "https://olmo-data.org"
CHUNK = 16 << 20


def hf_metadata(paths: list[str], revision: str) -> dict[str, dict]:
    api = HfApi()
    sha = api.dataset_info(REPO, revision=revision).sha
    meta = {}
    for i in range(0, len(paths), 100):
        for f in api.get_paths_info(REPO, paths[i:i + 100], repo_type="dataset", revision=sha,
                                    expand=True):
            meta[f.path] = {"size": f.size, "sha256": f.lfs.sha256 if f.lfs else None}
    missing = [p for p in paths if p not in meta or meta[p]["sha256"] is None]
    if missing:
        raise RuntimeError(f"{len(missing)} paths lack LFS metadata at {REPO}@{sha}: {missing[:3]}")
    return {"repo_sha": sha, "files": meta}


class QuotaGate:
    """Start a file only if it fits under a disk budget shared with other jobs.

    The per-user /work quota (10 TB on NCHC) is shared with other jobs, so the downloader must
    stop early rather than fill it. Usage is `du` of `usage_dir`, refreshed
    every `refresh_s`; every in-flight file reserves its full size (conservative)."""

    def __init__(self, usage_dir: Path, limit_bytes: int, refresh_s: float = 300):
        self.usage_dir, self.limit, self.refresh_s = usage_dir, limit_bytes, refresh_s
        self.lock, self.reserved, self.usage, self.t = threading.Lock(), 0, 0, 0.0

    def _du(self) -> int:
        out = subprocess.run(["du", "-s", "--block-size=1", str(self.usage_dir)],
                             capture_output=True, text=True, check=True)
        return int(out.stdout.split()[0])

    def acquire(self, size: int) -> bool:
        with self.lock:
            if time.time() - self.t > self.refresh_s:
                self.usage, self.t = self._du(), time.time()
            if self.usage + self.reserved + size > self.limit:
                return False
            self.reserved += size
            return True

    def release(self, size: int, written: bool) -> None:
        with self.lock:
            self.reserved -= size
            if written:
                self.usage += size  # counted until the next du refresh sees it


def fetch(rel: str, want: dict, root: Path, gate: QuotaGate | None = None,
          retries: int = 30) -> str:
    dst = root / rel
    side = dst.with_name(dst.name + ".sha256")
    if dst.exists() and side.exists() and side.read_text().strip() == want["sha256"] \
            and dst.stat().st_size == want["size"]:
        return "skip"
    if gate is not None and not gate.acquire(want["size"]):
        return "deferred"
    status = "failed"
    try:
        status = _download(rel, want, dst, side, retries)
        return status
    finally:
        if gate is not None:
            gate.release(want["size"], written=status == "ok")


def _download(rel: str, want: dict, dst: Path, side: Path, retries: int) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    for attempt in range(retries):
        try:
            # Resume a partial file: re-hash what is already on disk, then continue.
            h = hashlib.sha256()
            done = part.stat().st_size if part.exists() else 0
            if done > want["size"]:
                part.unlink()
                done = 0
            if done:
                with part.open("rb") as f:
                    while b := f.read(CHUNK):
                        h.update(b)
            headers = {"Range": f"bytes={done}-"} if done else {}
            if done < want["size"]:
                with requests.get(f"{MIRROR}/{rel}", headers=headers, stream=True,
                                  timeout=60) as r:
                    r.raise_for_status()
                    if done and r.status_code != 206:
                        # A full-body answer to a Range request: restart from byte 0 next try.
                        part.unlink()
                        raise RuntimeError(f"mirror ignored Range for {rel}; restarting")
                    with part.open("ab") as f:
                        for b in r.iter_content(CHUNK):
                            f.write(b)
                            h.update(b)
            size = part.stat().st_size
            if size != want["size"] or h.hexdigest() != want["sha256"]:
                part.unlink()
                raise RuntimeError(f"{rel}: size {size} sha {h.hexdigest()[:12]} != expected")
            part.rename(dst)
            side.write_text(want["sha256"] + "\n")
            return "ok"
        except Exception as e:  # retry any network or verification failure
            if attempt == retries - 1:
                raise
            print(f"retry {attempt + 1} {rel}: {e}", flush=True)
            # olmo-data.org answers bursts with 503s: back off up to 5 min, jittered so the
            # workers do not retry in lockstep.
            time.sleep(min(300, 5 * 2 ** attempt) * random.uniform(0.5, 1.5))
    raise AssertionError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--quota-gb", type=float, default=None,
                    help="per-user quota in GB (1e9 B); omit for no disk budget")
    ap.add_argument("--headroom-gb", type=float, default=1500,
                    help="left free under the quota for other sessions' jobs")
    ap.add_argument("--usage-dir", type=Path, default=Path("/work/harrychang08"))
    args = ap.parse_args()

    mix = json.loads(args.paths.read_text())
    commit = provenance.git_commit()
    meta = hf_metadata(mix["paths"], args.revision)
    total = sum(meta["files"][p]["size"] for p in mix["paths"])
    print(f"{mix['mix']}: {len(mix['paths'])} files, {total / 1e12:.3f} TB", flush=True)

    gate = None
    if args.quota_gb is not None:
        gate = QuotaGate(args.usage_dir, int((args.quota_gb - args.headroom_gb) * 1e9))
        print(f"disk budget: {args.usage_dir} stays under {gate.limit / 1e12:.2f} TB", flush=True)
    t0, done_bytes, status = time.time(), 0, {}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, p, meta["files"][p], args.root, gate): p for p in mix["paths"]}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                status[p] = fut.result()
            except Exception as e:  # noqa: BLE001 out of retries: report, keep the other workers going
                status[p] = "failed"
                print(f"FAILED {p}: {e}", flush=True)
            done_bytes += meta["files"][p]["size"] if status[p] in ("ok", "skip") else 0
            el = time.time() - t0
            print(f"[{len(status)}/{len(futs)}] {status[p]} {p} "
                  f"{done_bytes / 1e12:.2f} TB {done_bytes / el / 1e6:.0f} MB/s", flush=True)

    failed = [p for p, v in status.items() if v == "failed"]
    if failed:
        print(f"FAILED: {len(failed)} files out of retries; rerun to resume them", flush=True)
        raise SystemExit(4)
    deferred = [p for p, v in status.items() if v == "deferred"]
    if deferred:
        print(f"STOPPED AT DISK BUDGET: {len(status) - len(deferred)}/{len(status)} files verified, "
              f"{len(deferred)} deferred ({sum(meta['files'][p]['size'] for p in deferred) / 1e12:.2f} TB);"
              " rerun with a larger --quota-gb to finish", flush=True)
        raise SystemExit(3)
    manifest = {
        "mix": mix["mix"], "olmo_commit": mix["olmo_commit"], "hf_repo": REPO,
        "hf_repo_sha": meta["repo_sha"], "mirror": MIRROR, "root": str(args.root),
        "n_files": len(mix["paths"]), "bytes": total, "files": meta["files"],
        "git_commit": commit, "command": provenance.command(),
    }
    out = args.root / f"{mix['mix']}.manifest.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"all {len(status)} files verified; manifest {out}", flush=True)


if __name__ == "__main__":
    main()
