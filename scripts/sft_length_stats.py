"""Fraction of Tulu 3 SFT rows truncated / dropped at a max length, per tokenizer.

    uv run --frozen python scripts/sft_length_stats.py --workers 12 \
        --out runs/posttrain/004/sft_length_stats.json

Arms: DataDecide tokenizer at 2048 without BOS and with EOS as BOS, and
OLMo-2's tokenizer at 4096 with BOS (Ai2's own run, for reference). The label mask follows
open-instruct's (castor.posttrain.sft_labels); "dropped" = no trainable token after truncation,
which open-instruct's sft_tulu_filter_v1 removes.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from transformers import AutoTokenizer

from castor import posttrain, provenance

DD_TOK = ("/work/harrychang08/castor/models/datadecide-llama/DataDecide-dclm-baseline-1B/"
          "step69369-seed-default")
ARMS = {
    "datadecide_nobos_2048": (DD_TOK, None, False, 2048),
    "datadecide_eosbos_2048": (DD_TOK, None, True, 2048),
    "olmo2_bos_4096": ("allenai/OLMo-2-0425-1B", "a1847dff35000b4271fa70afc5db10fd29fedbdf", True, 4096),
    "olmo2_bos_2048": ("allenai/OLMo-2-0425-1B", "a1847dff35000b4271fa70afc5db10fd29fedbdf", True, 2048),
}
_TOK = None


def _init(path, rev, bos):
    global _TOK
    _TOK = AutoTokenizer.from_pretrained(path, revision=rev)
    if bos and _TOK.bos_token is None:
        _TOK.bos_token = _TOK.eos_token  # EOS as BOS
    _TOK.chat_template = posttrain.tulu_template(bos)


def _chunk(args):
    rows, max_len = args
    return posttrain.length_stats(_TOK, rows, max_len)


def merge(parts: list[dict]) -> dict:
    n = sum(p["rows"] for p in parts)
    t = sum(p["truncated"] for p in parts)
    d = sum(p["dropped"] for p in parts)
    return {"max_len": parts[0]["max_len"], "rows": n, "truncated": t, "dropped": d,
            "frac_truncated": t / n, "frac_dropped": d / n,
            "len_max": max(p["len_max"] for p in parts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (smoke)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    commit = provenance.git_commit()
    src = posttrain.jsonl_path(posttrain.SFT_REPO, posttrain.SFT_SHA)
    rows = []
    with open(src) as f:
        for i, line in enumerate(f):
            if a.limit and i >= a.limit:
                break
            rows.append({"messages": json.loads(line)["messages"]})
    chunks = [rows[i:i + 2000] for i in range(0, len(rows), 2000)]
    result = {"sft_jsonl": str(src), "sft_sha256": posttrain.sha256_file(src), "arms": {}}
    for arm, (path, rev, bos, max_len) in ARMS.items():
        with Pool(a.workers, initializer=_init, initargs=(path, rev, bos)) as pool:
            parts = pool.map(_chunk, [(c, max_len) for c in chunks])
        result["arms"][arm] = {"tokenizer": path, "revision": rev, "bos": bos} | merge(parts)
        print(arm, json.dumps(result["arms"][arm]))
    result |= {"git_commit": commit, "command": provenance.command()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
