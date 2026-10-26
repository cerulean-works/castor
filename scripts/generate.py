"""Generate samples (generate.D017 config) for one model on the evaluated panel.

    uv run --frozen python scripts/generate.py --model allenai/OLMo-2-0425-1B-DPO \
        --revision c4b0485961ab24c2433b090f3b922f0913a9290f --model-id olmo2-1b-dpo-ai2 \
        --out-root /work/harrychang08/castor/runs --prompt-id 004 [--limit N]

Writes <out-root>/<model-id>/<prompt-id>/{samples,greedy}.jsonl and provenance.json. Rows are
in the instrument's schema (`castor.judge.common`): {"model": <model-id>, "prompt_id",
"sample" (0..K-1, or -1 for greedy), "prompt", "response", "n_tokens" (EOS excluded),
"finish_reason"}, so `scripts/judge.py` reads them as they are.
"""

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from castor import evalset, generate, posttrain, provenance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo or local checkpoint dir")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out-root", type=Path, default=Path("/work/harrychang08/castor/runs"))
    ap.add_argument("--prompt-id", default="004")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts (smoke)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    a = ap.parse_args()
    commit = provenance.git_commit()
    cfg = generate.D017

    rows = evalset.load_eval()
    if a.limit:
        rows = rows[: a.limit]
    # a local model is often a symlink (<run>/model -> checkpoint): record what it pointed to now
    resolved = str(Path(a.model).resolve()) if Path(a.model).exists() else None
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    kind = generate.template_kind(tok.chat_template, posttrain.tulu_template(True),
                                  posttrain.tulu_template(False))
    ids = [evalset.prompt_ids(tok, r["prompt"]) for r in rows]
    too_long = [r["id"] for r, x in zip(rows, ids) if len(x) + cfg.max_new_tokens > generate.CONTEXT]
    assert not too_long, f"prompts over the context under this tokenizer: {too_long[:5]}"

    llm = LLM(model=a.model, revision=a.revision, tokenizer_revision=a.revision,
              dtype="bfloat16", max_model_len=generate.CONTEXT, seed=0,
              gpu_memory_utilization=a.gpu_memory_utilization)
    stop = [tok.eos_token_id]
    passes = {"samples": [SamplingParams(n=cfg.k, temperature=cfg.temperature, top_p=cfg.top_p,
                                         max_tokens=cfg.max_new_tokens, stop_token_ids=stop,
                                         seed=generate.prompt_seed(r["id"], a.model_id))
                          for r in rows]}
    if cfg.greedy:
        passes["greedy"] = SamplingParams(n=1, temperature=0.0, max_tokens=cfg.max_new_tokens,
                                          stop_token_ids=stop)
    out = a.out_root / a.model_id / a.prompt_id
    out.mkdir(parents=True, exist_ok=True)
    timing = {}
    for name, params in passes.items():
        t0 = time.time()
        res = llm.generate([{"prompt_token_ids": x} for x in ids], params)
        timing[name] = time.time() - t0
        with open(out / f"{name}.jsonl", "w") as f:
            for r, o in zip(rows, res, strict=True):
                f.writelines(json.dumps({
                    "model": a.model_id, "prompt_id": r["id"],
                    "sample": -1 if name == "greedy" else k, "prompt": r["prompt"],
                    "response": c.text,
                    "n_tokens": generate.completion_tokens(list(c.token_ids), c.finish_reason,
                                                           set(stop)),
                    "finish_reason": c.finish_reason}, ensure_ascii=False) + "\n"
                    for k, c in enumerate(o.outputs))
    prov = {"model": a.model, "model_resolved": resolved, "revision": a.revision,
            "model_id": a.model_id,
            "chat_template": kind, "sampling": generate.config_dict(cfg),
            "seed_rule": "sha256(salt:model_id:prompt_id)[:4] mod (2**31 - K); vLLM child seeds +i",
            "eval_manifest": json.loads(evalset.EVAL_MANIFEST.read_text())["eval"],
            "n_prompts": len(rows), "limit": a.limit, "seconds": timing,
            "git_commit": commit, "command": provenance.command()}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
