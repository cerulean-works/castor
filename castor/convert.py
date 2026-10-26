"""Convert DataDecide checkpoints (legacy `hf_olmo` format) to transformers' LlamaForCausalLM.

DataDecide's released weights load only through Ai2's old `hf_olmo` package, which neither
current transformers nor vLLM supports. Their architecture (OLMo `DataDecide` branch, commit
7094aab, olmo/model.py) is a Llama block exactly: pre-norm RMSNorm with a weight and no bias
(RMSLayerNorm.forward, fp32 normalization), SwiGLU MLP, full-width RoPE with the
half-split rotate_half, no linear biases, untied output embedding. The conversion is a pure
re-indexing of tensors:

  att_proj  [3d, d]  -> q_proj, k_proj, v_proj (split in that order, OLMoBlock.forward)
  ff_proj   [2h, d]  -> up_proj = rows [:h], gate_proj = rows [h:]
                        (SwiGLU.forward: `x, gate = x.chunk(2, -1); silu(gate) * x`)
  attn_out, ff_out   -> o_proj, down_proj
  attn_norm, ff_norm -> input_layernorm, post_attention_layernorm
  wte, ln_f, ff_out  -> embed_tokens, norm, lm_head

`scripts/validate_conversion.py` checks logits against the reference OLMo code.
"""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import LlamaConfig

OUT = Path("/work/harrychang08/castor/models/datadecide-llama")


def converted_dir(repo: str, revision: str) -> Path:
    """Where the converted copy of `repo@revision` lives (named by HF repo + branch)."""
    return OUT / repo.split("/")[1] / revision


def converter_sha256() -> str:
    """Hash of this module: an output converted by different code must not be reused."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# Config fields whose value the conversion assumes; anything else is refused.
REQUIRED = {
    "activation_type": "swiglu",
    "alibi": False,
    "attention_layer_norm": False,
    "bias_for_layer_norm": False,
    "block_type": "sequential",
    "clip_qkv": None,
    "include_bias": False,
    "layer_norm_type": "rms",
    "layer_norm_with_affine": True,
    "multi_query_attention": None,
    "n_kv_heads": None,
    "norm_after": False,
    "rope": True,
    "scale_logits": False,
    "weight_tying": False,
    "block_group_size": 1,
}


def llama_config(src: dict) -> LlamaConfig:
    for k, v in REQUIRED.items():
        if src.get(k) != v:
            raise ValueError(f"unsupported DataDecide config: {k}={src.get(k)!r} (need {v!r})")
    d = src["d_model"]
    hidden = src["mlp_hidden_size"] or src["mlp_ratio"] * d
    return LlamaConfig(
        vocab_size=src["embedding_size"] or src["vocab_size"],
        hidden_size=d,
        intermediate_size=hidden // 2,  # SwiGLU halves ff_proj's output
        num_hidden_layers=src["n_layers"],
        num_attention_heads=src["n_heads"],
        num_key_value_heads=src["n_heads"],
        head_dim=d // src["n_heads"],
        hidden_act="silu",
        max_position_embeddings=src["max_sequence_length"],
        rms_norm_eps=src["layer_norm_eps"],
        rope_theta=src["rope_theta"],
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        bos_token_id=None,
        eos_token_id=src["eos_token_id"],
        pad_token_id=src["pad_token_id"],
        torch_dtype="float32",
        architectures=["LlamaForCausalLM"],
    )


def load_olmo_state(src_dir: Path) -> dict[str, torch.Tensor]:
    """Sharded (1B: model.safetensors.index.json) or single-file (<= 300M) checkpoints."""
    index = src_dir / "model.safetensors.index.json"
    shards = (sorted(set(json.loads(index.read_text())["weight_map"].values()))
              if index.exists() else ["model.safetensors"])
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(str(src_dir / shard)))
    prefix = "model.transformer."
    assert all(k.startswith(prefix) for k in state), "unexpected key prefix"
    return {k[len(prefix):]: v for k, v in state.items()}


def convert_state(olmo: dict[str, torch.Tensor], cfg: LlamaConfig) -> dict[str, torch.Tensor]:
    d, h = cfg.hidden_size, cfg.intermediate_size
    out = {
        "model.embed_tokens.weight": olmo.pop("wte.weight"),
        "model.norm.weight": olmo.pop("ln_f.weight"),
        "lm_head.weight": olmo.pop("ff_out.weight"),
    }
    for i in range(cfg.num_hidden_layers):
        b, L = f"blocks.{i}.", f"model.layers.{i}."
        q, k, v = olmo.pop(b + "att_proj.weight").split(d, dim=0)
        up, gate = olmo.pop(b + "ff_proj.weight").split(h, dim=0)
        out.update({
            L + "self_attn.q_proj.weight": q,
            L + "self_attn.k_proj.weight": k,
            L + "self_attn.v_proj.weight": v,
            L + "self_attn.o_proj.weight": olmo.pop(b + "attn_out.weight"),
            L + "mlp.up_proj.weight": up,
            L + "mlp.gate_proj.weight": gate,
            L + "mlp.down_proj.weight": olmo.pop(b + "ff_out.weight"),
            L + "input_layernorm.weight": olmo.pop(b + "attn_norm.weight"),
            L + "post_attention_layernorm.weight": olmo.pop(b + "ff_norm.weight"),
        })
    if olmo:
        raise ValueError(f"unconverted tensors: {sorted(olmo)}")
    return {k: v.contiguous() for k, v in out.items()}


TOKENIZER_CONFIG = {
    # the released tokenizer_config names `OLMoTokenizer` (hf_olmo); tokenizer.json is a
    # standard fast tokenizer, so load it as such. No BOS token exists in this vocabulary.
    "tokenizer_class": "PreTrainedTokenizerFast",
    "eos_token": "<|endoftext|>",
    "pad_token": "<|padding|>",
    "clean_up_tokenization_spaces": False,
    "model_max_length": 2048,
    "model_input_names": ["input_ids", "attention_mask"],  # no token_type_ids
}
