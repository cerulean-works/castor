import dataclasses
import json
import sys

import pytest
import torch
from transformers import LlamaForCausalLM

from castor import convert, localckpt
from tests.test_convert import OLMO_DIR, TINY


def _hf_olmo_dir(tmp_path, state: dict, cfg: dict):
    """An hf_olmo copy as shell/olmo/to_hf.sh leaves it: pytorch_model.bin + config.json
    without the FILL fields."""
    torch.save({f"model.transformer.{k}": v for k, v in state.items()},
               tmp_path / "pytorch_model.bin")
    (tmp_path / "config.json").write_text(json.dumps(
        {k: v for k, v in cfg.items() if k not in localckpt.FILL}))
    return tmp_path


@pytest.mark.skipif(not OLMO_DIR.exists(), reason="reference OLMo code not cloned")
def test_local_copy_converts_to_matching_llama(tmp_path):
    sys.path.insert(0, str(OLMO_DIR))
    from olmo.config import ModelConfig
    from olmo.model import OLMo

    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    torch.manual_seed(0)
    ref = OLMo(ModelConfig(**{k: v for k, v in TINY.items() if k in fields})).eval()
    ref.reset_parameters()
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if "norm" in n or "ln_f" in n:
                p.add_(torch.randn_like(p) * 0.1)
    state = {k.removeprefix("transformer."): v.clone() for k, v in ref.state_dict().items()}
    src = _hf_olmo_dir(tmp_path, state, TINY)

    cfg_in, filled = localckpt.load_config(src)
    assert sorted(filled) == sorted(localckpt.FILL)
    cfg = convert.llama_config(cfg_in)
    assert cfg.rope_theta == 10000
    llama = LlamaForCausalLM(cfg).eval()
    llama.load_state_dict(convert.convert_state(localckpt.load_state(src), cfg))
    ids = torch.randint(0, 120, (2, 40))
    with torch.no_grad():
        a, b = ref(ids).logits, llama(ids).logits
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_fill_refuses_other_values(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"rope_theta": 500000}))
    with pytest.raises(AssertionError):
        localckpt.load_config(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({**localckpt.FILL, "d_model": 8}))
    cfg, filled = localckpt.load_config(tmp_path)
    assert filled == [] and cfg["d_model"] == 8


def test_data_seed_from_changes():
    base = {"ladder_args": {"seed": 6198}, "changes": ["run_name a -> b"]}
    assert localckpt.data_seed(base) == 6198
    assert localckpt.data_seed({**base, "changes": [
        "data.seed None (falls back to seed=6198) -> 4"]}) == 4


def test_parse_ref():
    assert localckpt.parse_ref("castor/DataDecide-dclm-baseline-1B@init-default-data-aux2") == (
        "castor/DataDecide-dclm-baseline-1B", "init-default-data-aux2")
    with pytest.raises(AssertionError):
        localckpt.parse_ref("allenai/DataDecide-dclm-baseline-1B@step69369-seed-default")


def test_sha256_file(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    assert localckpt.sha256_file(p, chunk=2) == \
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
