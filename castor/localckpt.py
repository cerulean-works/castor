"""Our own OLMo pretraining runs as twins: the hf_olmo copy that
`shell/olmo/to_hf.sh` writes (OLMo's hf_olmo/convert_olmo_to_hf.py @ ccabb54e, the same copy the
OLMES instrument evaluates) is read here and handed to `castor.convert`, the converter of every
released DataDecide twin. castor/convert.py itself is not edited: its sha256 gates every
converted base (castor.grid.check_base).

Two differences from a released hf_olmo checkpoint, both handled here and nowhere else:
  - weights are one `pytorch_model.bin` (fp32), not safetensors shards;
  - config.json lacks five fields the released files carry: the ccabb54e
    converter predates them. OLMo @ ccabb54e and @ 7094aab have none of them as options (RoPE
    base 10000 is hard-coded in RotaryEmbedding, olmo/model.py:280 in both), so each is filled
    with the value every released 1B/150M DataDecide config.json carries.

Model ids: `castor/DataDecide-<recipe>-<size>@<cell>`, e.g.
`castor/DataDecide-dclm-baseline-1B@init-default-data-aux2`; registered in
`data/manifests/castor_twins_<size>.json` with the source run, step and sha256 of model.pt.
"""

import hashlib
import json
from pathlib import Path

import torch

# absent from our hf_olmo config.json; values as in the released DataDecide config.json
FILL = {"emb_init_std": None, "embedding_layer_norm": False, "norm_after": False,
        "rope_theta": 10000, "scale_emb_init": False}
PREFIX = "model.transformer."


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_config(src_dir: Path) -> tuple[dict, list[str]]:
    """config.json with exactly the FILL fields added; returns (config, fields filled).
    A field outside FILL that is missing, or a FILL field present with another value, is an
    error: that would be a checkpoint this module was not written for."""
    cfg = json.loads((Path(src_dir) / "config.json").read_text())
    filled = []
    for k, v in FILL.items():
        if k in cfg:
            assert cfg[k] == v, f"{src_dir}: {k}={cfg[k]!r}, expected {v!r}"
        else:
            cfg[k] = v
            filled.append(k)
    return cfg, filled


def load_state(src_dir: Path) -> dict[str, torch.Tensor]:
    """pytorch_model.bin of an hf_olmo copy -> OLMo parameter names (prefix stripped), the
    form castor.convert.convert_state takes."""
    state = torch.load(Path(src_dir) / "pytorch_model.bin", map_location="cpu", weights_only=True)
    assert all(k.startswith(PREFIX) for k in state), "unexpected key prefix"
    return {k[len(PREFIX):]: v for k, v in state.items()}


def data_seed(changes: dict) -> int:
    """Data-order seed of a run from its config.yaml.changes.json (shell/olmo/make_config.py):
    the `data.seed ... -> N` change when pretrain.sh set one, else the init seed (OLMo's
    fallback)."""
    hits = [c for c in changes["changes"] if c.startswith("data.seed ")]
    assert len(hits) <= 1, hits
    return int(hits[0].rsplit("->", 1)[1]) if hits else int(changes["ladder_args"]["seed"])


def parse_ref(ref: str) -> tuple[str, str]:
    """castor/DataDecide-<recipe>-<size>@<cell> -> (repo, revision)."""
    repo, rev = ref.split("@")
    assert repo.startswith("castor/DataDecide-") and rev and "/" not in rev, ref
    return repo, rev
