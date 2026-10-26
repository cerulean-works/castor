"""Build the DataDecide tokenizer variant whose BOS is EOS.

    uv run --frozen python scripts/make_eosbos_tokenizer.py

The OLMo-1 SFT/Instruct models with this tokenizer family prepend <|endoftext|> through their
chat template ("{{ eos_token }}<|user|>...", e.g. allenai/OLMo-7B-0724-Instruct-hf), and
open-instruct's GPTNeoX branch sets bos_token = eos_token and requires --add_bos
(dataset_transformation.py:476-481 @ fc7748ea4). The converted DataDecide tokenizer is a
PreTrainedTokenizerFast with a pad token, so that branch is skipped; this copy sets bos_token =
<|endoftext|> so that `--add_bos` gives the same leading EOS. Weights are not copied.
open-instruct's tokenizer loader also reads config.json (AutoConfig, :454), so the Llama
config of the source checkpoint is copied alongside; only model_type is read from it.
"""

import json
import shutil
from pathlib import Path

from transformers import AutoTokenizer

from castor import posttrain, provenance

SRC = Path("/work/harrychang08/castor/models/datadecide-llama/DataDecide-dclm-baseline-1B/"
           "step69369-seed-default")
OUT = Path("/work/harrychang08/castor/models/tokenizers/datadecide-neox-eosbos")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(SRC)
    assert tok.bos_token is None and tok.eos_token == "<|endoftext|>" and tok.pad_token_id == 1
    tok.bos_token = tok.eos_token
    OUT.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(OUT)
    shutil.copy(SRC / "config.json", OUT / "config.json")
    back = AutoTokenizer.from_pretrained(OUT)
    assert back.bos_token_id == back.eos_token_id == 50279 and back.pad_token_id == 1
    assert back("Hello world")["input_ids"] == tok("Hello world")["input_ids"]
    meta = {"source": str(SRC), "change": "bos_token = eos_token (<|endoftext|>, 50279)",
            "files": {p.name: posttrain.sha256_file(p) for p in sorted(OUT.iterdir())
                      if p.name != "provenance.json"},
            "git_commit": provenance.git_commit(), "command": provenance.command()}
    (OUT / "provenance.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
