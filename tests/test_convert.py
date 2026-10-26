import dataclasses
import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaForCausalLM

from castor import convert

OLMO_DIR = Path("/work/harrychang08/castor/third_party/OLMo")
pytestmark = pytest.mark.skipif(not OLMO_DIR.exists(), reason="reference OLMo code not cloned")

TINY = {  # a DataDecide-shaped config, scaled down
    "activation_type": "swiglu", "alibi": False, "attention_layer_norm": False,
    "bias_for_layer_norm": False, "block_type": "sequential", "block_group_size": 1,
    "clip_qkv": None, "d_model": 64, "embedding_size": 128, "vocab_size": 120,
    "eos_token_id": 0, "pad_token_id": 1, "include_bias": False, "layer_norm_eps": 1e-6,
    "layer_norm_type": "rms", "layer_norm_with_affine": True, "max_sequence_length": 64,
    "mlp_hidden_size": None, "mlp_ratio": 8, "multi_query_attention": None, "n_heads": 4,
    "n_kv_heads": None, "n_layers": 2, "norm_after": False, "rope": True,
    "rope_full_precision": True, "rope_theta": 10000, "scale_logits": False,
    "weight_tying": False, "attention_dropout": 0.0, "embedding_dropout": 0.0,
    "residual_dropout": 0.0, "flash_attention": False, "init_device": "cpu",
}


def test_tiny_random_olmo_matches_llama():
    sys.path.insert(0, str(OLMO_DIR))
    from olmo.config import ModelConfig
    from olmo.model import OLMo

    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    torch.manual_seed(0)
    ref = OLMo(ModelConfig(**{k: v for k, v in TINY.items() if k in fields})).eval()
    ref.reset_parameters()
    # norms start at ones; perturb so the test sees whether they are mapped
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if "norm" in n or "ln_f" in n:
                p.add_(torch.randn_like(p) * 0.1)
    olmo_state = {k.removeprefix("transformer."): v.clone() for k, v in ref.state_dict().items()}
    cfg = convert.llama_config(TINY)
    llama = LlamaForCausalLM(cfg).eval()
    llama.load_state_dict(convert.convert_state(olmo_state, cfg))
    ids = torch.randint(0, 120, (2, 40))
    with torch.no_grad():
        a = ref(ids).logits
        b = llama(ids).logits
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_refuses_unsupported_config():
    with pytest.raises(ValueError):
        convert.llama_config({**TINY, "norm_after": True})


def test_converted_dir_and_hash():
    p = convert.converted_dir("allenai/DataDecide-dclm-baseline-1B", "step69369-seed-default")
    assert p == convert.OUT / "DataDecide-dclm-baseline-1B" / "step69369-seed-default"
    assert len(convert.converter_sha256()) == 64


def test_skip_only_if_same_source_and_converter(tmp_path):
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location(
        "convert_datadecide", Path(__file__).parent.parent / "scripts/convert_datadecide.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert not mod.is_current(tmp_path, "abc", "h")
    (tmp_path / "provenance.json").write_text(json.dumps(
        {"source_sha": "abc", "converter_sha256": "h"}))
    assert mod.is_current(tmp_path, "abc", "h")
    assert not mod.is_current(tmp_path, "abc", "h2")  # converter changed
    assert not mod.is_current(tmp_path, "def", "h")   # source commit changed


def test_load_olmo_state_single_and_sharded(tmp_path):
    import json

    from safetensors.torch import save_file
    a = {"model.transformer.wte.weight": torch.ones(2, 2)}
    b = {"model.transformer.ln_f.weight": torch.zeros(2)}
    single, sharded = tmp_path / "single", tmp_path / "sharded"
    single.mkdir()
    sharded.mkdir()
    save_file({**a, **b}, str(single / "model.safetensors"))
    save_file(a, str(sharded / "s1.safetensors"))
    save_file(b, str(sharded / "s2.safetensors"))
    (sharded / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {**dict.fromkeys(a, "s1.safetensors"), **dict.fromkeys(b, "s2.safetensors")}}))
    for d in (single, sharded):
        assert sorted(convert.load_olmo_state(d)) == ["ln_f.weight", "wte.weight"]
