"""Recount the over-length prompts with each ladder base's own tokenizer and context, by the
rule that built the evaluated panel (scripts/build_eval_subset.py, castor.evalset), and compare
the kept set with panel_v1_eval (6,226 prompts).

    uv run --frozen python scripts/overlength_016.py --out /abs/path/runs/016/overlength.json

CPU only. Exits non-zero if any size's kept set differs from the evaluated panel.
"""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from castor import evalset, ladder, posttrain, provenance


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute"
    rows = evalset.load_panel()
    eval_ids = [r["id"] for r in evalset.load_eval()]
    res, same = {}, True
    for size in ladder.SIZES:
        for seed in ladder.PRETRAIN_SEEDS:
            cell = ladder.Cell(size, seed, 1)
            d = cell.base_dir
            ctx = AutoConfig.from_pretrained(d).max_position_embeddings
            tok = AutoTokenizer.from_pretrained(d)
            tok.bos_token = tok.eos_token  # EOS as BOS, as build_eval_subset.py
            tok.chat_template = posttrain.tulu_template(bos=True)
            n = [len(evalset.prompt_ids(tok, r["prompt"])) for r in rows]
            kept, dropped = evalset.select(rows, n, context=ctx)
            ids = [r["id"] for r in kept]
            ok = ids == eval_ids
            same &= ok
            res[cell.base_ref] = {
                "context": ctx, "tokenizer_json_sha256": sha(d / "tokenizer.json"),
                "n_kept": len(kept), "n_dropped": len(dropped),
                "max_prompt_tokens": max(r["n_prompt_tokens"] for r in kept),
                "kept_equals_eval_panel": ok,
                "only_in_kept": sorted(set(ids) - set(eval_ids)),
                "only_in_eval": sorted(set(eval_ids) - set(ids))}
            print(cell.base_ref, ctx, len(kept), len(dropped), ok)
    out = {"eval_panel": {"n": len(eval_ids),
                          "sha256": json.loads(evalset.EVAL_MANIFEST.read_text())["eval"]["sha256"]},
           "rule": f"keep iff chat-formatted prompt tokens (EOS as BOS, tulu template) + "
                   f"{evalset.MAX_NEW_TOKENS} <= the base's max_position_embeddings (D021 b)",
           "bases": res, "all_equal_eval_panel": same,
           "script": "scripts/overlength_016.py", "git_commit": provenance.git_commit(),
           "command": provenance.command()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    raise SystemExit(0 if same else 1)


if __name__ == "__main__":
    main()
