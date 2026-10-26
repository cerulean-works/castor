"""Provisional WildGuard refusal judge, superseded by the instrument in castor/judge/; nothing
downstream reads this output.

    uv run --frozen python scripts/judge_wildguard_provisional.py \
        --gen /work/harrychang08/castor/runs/<model-id>/004/samples.jsonl [--limit N]

allenai/wildguard @ cbba4823, the official classifier prompt verbatim (model card README
L65-78; identical to the wildguard library's `wildguard/utils.py:11-24` @ 495b9dc, which
omits the literal "<s>" and tokenizes with special tokens, as done here), greedy, 128 new
tokens (library: wildguard.py:201-203), and the library's parser (wildguard.py:126-166):
three "label: value" lines; a value other than yes/no is a parse failure (None), never a
non-refusal. Writes <gen dir>/<gen stem>.wildguard_provisional.jsonl.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from castor import provenance
from castor.judge_provisional import REPO, SHA, TEMPLATE, parse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    commit = provenance.git_commit()
    gens = [json.loads(line) for line in a.gen.read_text().splitlines()]
    if a.limit:
        gens = gens[: a.limit]
    tok = AutoTokenizer.from_pretrained(REPO, revision=SHA)
    ids = [tok(TEMPLATE.format(prompt=g["prompt"], response=g["response"]),
               add_special_tokens=True)["input_ids"] for g in gens]
    llm = LLM(model=REPO, revision=SHA, tokenizer_revision=SHA, dtype="bfloat16",
              max_model_len=8192, seed=0, gpu_memory_utilization=0.85)
    t0 = time.time()
    res = llm.generate([{"prompt_token_ids": x} for x in ids],
                       SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128))
    secs = time.time() - t0
    out = a.gen.with_name(a.gen.stem + ".wildguard_provisional.jsonl")
    with open(out, "w") as f:
        f.writelines(json.dumps({"id": g["prompt_id"], "k": g["sample"], **parse(o.outputs[0].text),
                                 "raw": o.outputs[0].text}) + "\n" for g, o in zip(gens, res, strict=True))
    prov = {"judge": f"{REPO}@{SHA}", "template_sha256": hashlib.sha256(TEMPLATE.encode()).hexdigest(),
            "gen": str(a.gen), "n": len(gens), "seconds": secs, "out": str(out),
            "git_commit": commit, "command": provenance.command()}
    out.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
