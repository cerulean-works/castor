"""Check converted DataDecide checkpoints against Ai2's reference OLMo code, then smoke-test
HF and vLLM generation on the converted copy.

    PYTHONPATH=/work/harrychang08/castor/third_party/OLMo \
    uv run --frozen python scripts/validate_conversion.py \
        --model allenai/DataDecide-dclm-baseline-1B --revision step69369-seed-default

1. Reference: OLMo `DataDecide` branch @ 7094aab (olmo.model.OLMo) with the released weights,
   float32. Converted: transformers LlamaForCausalLM, float32. Same token ids (a short text,
   and a 2048-token sequence). Reports max |logit diff| and top-1 agreement, and FAILS (exit
   1) unless every input has max |diff| <= MAX_ABS_DIFF and top-1 agreement >= MIN_TOP1.
2. Greedy generation, bfloat16, HF vs vLLM on the converted copy: shared-prefix length
   (reported, not gated: bf16 kernels differ between the engines).

Run once per distinct source config (d_model, n_layers, heads, MLP size); the conversion is
the same code for every recipe and seed of one size.
"""

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import torch
import transformers
import vllm
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from castor import convert, localckpt, provenance

# Fixed before the ladder conversions were validated. fp32 re-indexing is exact up to
# summation order; the 1B dclm-baseline check gave 5e-5 on logits of scale ~30.
MAX_ABS_DIFF = 1e-3
MIN_TOP1 = 0.999
README_REPO, README_REV = "allenai/DataDecide-dclm-baseline-1B", "step69369-seed-default"
OLMO_DIR = Path("/work/harrychang08/castor/third_party/OLMo")
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Photosynthesis is the process by which",
    "Once upon a time, in a small village,",
]


def is_local(src: Path) -> bool:
    """Our own run's hf_olmo copy (castor.localckpt) rather than a released checkpoint."""
    return (src / "pytorch_model.bin").exists()


def source_config(src: Path) -> dict:
    return localckpt.load_config(src)[0] if is_local(src) else json.loads((src / "config.json").read_text())


def source_state(src: Path) -> dict[str, torch.Tensor]:
    return localckpt.load_state(src) if is_local(src) else convert.load_olmo_state(src)


def reference_logits(src: Path, texts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Logits of Ai2's reference OLMo implementation, one model build for all inputs."""
    from olmo.config import ModelConfig
    from olmo.model import OLMo

    # The checkout is at ccabb54e (used for pretraining); its olmo/ equals 7094aab's.
    same = subprocess.run(["git", "diff", "--quiet", "7094aab", "HEAD", "--", "olmo/"],
                          cwd=OLMO_DIR, check=False).returncode == 0
    assert same, "reference OLMo code (olmo/) differs from 7094aab"
    raw = source_config(src)
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in raw.items() if k in fields})
    cfg.init_device = "cpu"
    cfg.flash_attention = False
    model = OLMo(cfg)
    state = {f"transformer.{k}": v for k, v in source_state(src).items()}
    model.load_state_dict(state)
    model = model.cuda().eval()
    with torch.no_grad():
        out = {k: model(ids.cuda()).logits.float().cpu() for k, ids in texts.items()}
    del model
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--converted", default=None)
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument("--source-dir", default=None,
                    help="our run's hf_olmo copy (castor.localckpt); --model/--revision then name "
                         "its castor/ id, and the natural-text probe is the README of "
                         "allenai/DataDecide-dclm-baseline-1B@step69369-seed-default")
    args = ap.parse_args()
    commit, command = provenance.git_commit(), provenance.command()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    conv = Path(args.converted or convert.converted_dir(args.model, args.revision))
    conv_prov = json.loads((conv / "provenance.json").read_text())
    assert conv_prov["source"] == f"{args.model}@{args.revision}", conv_prov
    # the exact commit that was converted, not whatever the branch points to now
    if args.source_dir:
        src = Path(args.source_dir)
        assert json.loads((src / "provenance.json").read_text())["model_pt_sha256"] \
            == conv_prov["source_sha"], "source dir is not the converted checkpoint"
        readme = Path(hf_hub_download(README_REPO, "README.md", revision=README_REV)).read_text()
    else:
        src = Path(snapshot_download(args.model, revision=conv_prov["source_sha"],
                                     ignore_patterns=["training/*"]))
        readme = (src / "README.md").read_text()
    tok = AutoTokenizer.from_pretrained(conv)
    vocab = tok.vocab_size

    g = torch.Generator().manual_seed(0)
    texts = {"short": tok(PROMPTS[0], return_tensors="pt")["input_ids"],
             "random_2048": torch.randint(0, vocab, (1, 2048), generator=g)}
    texts["readme_2048"] = tok(readme * 4, return_tensors="pt")["input_ids"][:, :2048]

    llama = AutoModelForCausalLM.from_pretrained(conv, dtype=torch.float32,
                                                 attn_implementation="eager").cuda().eval()
    ours = {}
    with torch.no_grad():
        for k, ids in texts.items():
            ours[k] = llama(ids.cuda()).logits.float().cpu()
    del llama
    torch.cuda.empty_cache()

    refs = reference_logits(src, texts)
    logit_check = {}
    for k, ids in texts.items():
        ref = refs[k]
        diff = (ref - ours[k]).abs()
        logit_check[k] = {
            "n_tokens": ids.shape[1],
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "ref_logit_absmax": ref.abs().max().item(),
            "top1_agreement": (ref.argmax(-1) == ours[k].argmax(-1)).float().mean().item(),
        }
    passed = all(c["max_abs_diff"] <= MAX_ABS_DIFF and c["top1_agreement"] >= MIN_TOP1
                 for c in logit_check.values())

    print(json.dumps({"logits_fp32_reference_vs_converted": logit_check}, indent=1), flush=True)

    # generation smoke test, bf16, greedy
    hf = AutoModelForCausalLM.from_pretrained(conv, dtype=torch.bfloat16).cuda().eval()
    hf_out = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").to("cuda")
        o = hf.generate(**ids, max_new_tokens=48, do_sample=False)
        hf_out.append(o[0, ids["input_ids"].shape[1]:].tolist())
    del hf
    torch.cuda.empty_cache()
    llm = vllm.LLM(str(conv), dtype="bfloat16", gpu_memory_utilization=0.3, seed=0)
    res = llm.generate(PROMPTS, vllm.SamplingParams(temperature=0.0, max_tokens=48))
    gen = []
    for p, h, r in zip(PROMPTS, hf_out, res, strict=True):
        v = list(r.outputs[0].token_ids)
        agree = next((i for i, (a, b) in enumerate(zip(h, v, strict=False)) if a != b),
                     min(len(h), len(v)))
        gen.append({"prompt": p, "hf": tok.decode(h), "vllm": r.outputs[0].text,
                    "shared_prefix_tokens": agree, "n": len(h)})

    report = {
        "passed": passed, "thresholds": {"max_abs_diff": MAX_ABS_DIFF, "min_top1": MIN_TOP1},
        "source": f"{args.model}@{args.revision}", "converted": str(conv),
        "converted_provenance": conv_prov,
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "transformers": transformers.__version__, "vllm": vllm.__version__},
        "gpu": torch.cuda.get_device_name(0),
        "logits_fp32_reference_vs_converted": logit_check,
        "generation_bf16_hf_vs_vllm": gen,
        "git_commit": commit, "command": command,
    }
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
    print("PASS" if passed else "FAIL", flush=True)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
