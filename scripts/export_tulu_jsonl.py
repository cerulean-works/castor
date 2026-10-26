"""Export the pinned Tulu 3 SFT and DPO mixes to local jsonl for open-instruct.

    uv run --frozen python scripts/export_tulu_jsonl.py

open-instruct @ fc7748ea4 loads hub datasets at `main` only; a local .jsonl is loaded as is,
so exporting the pinned snapshot is how the revision is enforced. Rows keep the
snapshot's order (load_dataset over the sorted parquet shards, the order a hub load gives).
Writes <name>@<sha12>.jsonl and <name>@<sha12>.json (rows, sha256, ordered-content hash).
"""

import json

from datasets import load_dataset

from castor import posttrain, provenance


def main() -> None:
    commit = provenance.git_commit()
    for repo, sha in [(posttrain.SFT_REPO, posttrain.SFT_SHA), (posttrain.DPO_REPO, posttrain.DPO_SHA)]:
        src = posttrain.snapshot_dir(repo, sha)
        ds = load_dataset(str(src), split="train")
        out = posttrain.jsonl_path(repo, sha)
        out.parent.mkdir(parents=True, exist_ok=True)
        ds.to_json(str(out), lines=True, force_ascii=False)
        back = load_dataset("json", data_files=str(out), split="train")
        assert len(back) == len(ds), (len(back), len(ds))
        key = "messages" if "messages" in ds.column_names else "chosen"
        for i in (0, len(ds) // 2, len(ds) - 1):
            assert back[i][key] == ds[i][key], f"row {i} differs after the jsonl round trip"
        meta = {"repo": repo, "revision": sha, "snapshot": str(src), "jsonl": str(out),
                "rows": len(ds), "columns": ds.column_names,
                "sha256": posttrain.sha256_file(out),
                "ordered_hash": posttrain.ordered_hash(ds),
                "git_commit": commit, "command": provenance.command()}
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
        print(json.dumps({k: meta[k] for k in ("repo", "rows", "sha256")}))


if __name__ == "__main__":
    main()
