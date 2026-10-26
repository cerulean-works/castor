"""Write the evaluated subset of panel_v1 (prompts that fit the context) and its manifest.

    uv run --frozen python scripts/build_eval_subset.py

Outputs data/panel/panel_v1_eval.jsonl (panel rows + n_prompt_tokens, panel order) and
data/panel/EVAL_MANIFEST_v1.json (rule, tokenizer, sha256, per-source counts, dropped ids).
"""

import hashlib
import json
from collections import Counter

from transformers import AutoTokenizer

from castor import evalset, posttrain, provenance

DD_TOK = ("/work/harrychang08/castor/models/datadecide-llama/DataDecide-dclm-baseline-1B/"
          "step69369-seed-default")
OLMO2 = ("allenai/OLMo-2-0425-1B-SFT", posttrain.TEMPLATE_SHA)


def main() -> None:
    commit = provenance.git_commit()
    rows = evalset.load_panel()
    tok = AutoTokenizer.from_pretrained(DD_TOK)
    tok.bos_token = tok.eos_token  # variant B, the longer of the two twin variants
    tok.chat_template = posttrain.tulu_template(bos=True)
    n = [len(evalset.prompt_ids(tok, r["prompt"])) for r in rows]
    kept, dropped = evalset.select(rows, n)

    # OLMo-2 references see the same prompts under their own tokenizer; they must fit too.
    otok = AutoTokenizer.from_pretrained(OLMO2[0], revision=OLMO2[1])
    n_olmo2 = max(len(evalset.prompt_ids(otok, r["prompt"])) for r in kept)
    assert n_olmo2 + evalset.MAX_NEW_TOKENS <= evalset.CONTEXT, n_olmo2

    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept).encode()
    evalset.EVAL.write_bytes(body)
    by_src = Counter(r["source"] for r in rows)
    kept_src = Counter(r["source"] for r in kept)
    manifest = {
        "version": "v1",
        "panel": {"path": str(evalset.PANEL.relative_to(provenance.ROOT)),
                  "sha256": evalset.PANEL_SHA256, "n": len(rows)},
        "rule": f"keep iff chat-formatted prompt tokens + {evalset.MAX_NEW_TOKENS} <= "
                f"{evalset.CONTEXT} (D021 b, D017)",
        "tokenizer": f"{DD_TOK} with bos_token=eos_token, tulu template from "
                     f"{posttrain.TEMPLATE_REPO}@{posttrain.TEMPLATE_SHA}, generation prompt",
        "eval": {"path": str(evalset.EVAL.relative_to(provenance.ROOT)),
                 "sha256": hashlib.sha256(body).hexdigest(), "n": len(kept),
                 "n_harmful": sum(r["is_harmful"] for r in kept)},
        "per_source": {s: {"panel": by_src[s], "eval": kept_src[s],
                           "dropped": by_src[s] - kept_src[s]} for s in by_src},
        "dropped_ids": [r["id"] for r in dropped],
        "max_prompt_tokens": {"datadecide_eosbos": max(r["n_prompt_tokens"] for r in kept),
                              "olmo2": n_olmo2},
        "git_commit": commit, "command": provenance.command(),
    }
    evalset.EVAL_MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("eval", "per_source", "max_prompt_tokens")},
                     indent=1))


if __name__ == "__main__":
    main()
