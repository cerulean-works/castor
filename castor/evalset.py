"""The evaluated subset of the frozen panel: prompts that fit a 2048 context with room for the
generation budget.

A prompt is kept iff (chat-formatted prompt tokens) + MAX_NEW_TOKENS <= CONTEXT, counted with
the DataDecide tokenizer (the binding 2048 context) under open-instruct's tulu template with
the generation prompt, and with EOS as BOS (one token longer than the no-BOS variant, so the
subset holds under either). The same subset is used for every model, OLMo-2 references
included.
"""

import hashlib
import json
from pathlib import Path

CONTEXT = 2048
MAX_NEW_TOKENS = 512
PANEL = Path(__file__).resolve().parent.parent / "data" / "panel" / "panel_v1.jsonl"
PANEL_SHA256 = "7b7c4fac5e285e5dfc3aa3b4bdd22397bfd3e645ebb6e7ee476c9e2a951578d7"
EVAL = PANEL.with_name("panel_v1_eval.jsonl")
EVAL_MANIFEST = PANEL.with_name("EVAL_MANIFEST_v1.json")


def load_panel(path: Path = PANEL, sha256: str = PANEL_SHA256) -> list[dict]:
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha256, f"{path} is not the frozen panel"
    return [json.loads(line) for line in data.decode().splitlines()]


def prompt_ids(tok, prompt: str) -> list[int]:
    """Token ids a model sees for one user turn, generation prompt included."""
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                   add_generation_prompt=True)


def select(rows: list[dict], n_tokens: list[int], context: int = CONTEXT,
           max_new: int = MAX_NEW_TOKENS) -> tuple[list[dict], list[dict]]:
    """(kept rows with n_prompt_tokens added, dropped rows); order preserved."""
    kept, dropped = [], []
    for r, n in zip(rows, n_tokens, strict=True):
        (kept if n + max_new <= context else dropped).append(r | {"n_prompt_tokens": n})
    return kept, dropped


def load_eval(path: Path = EVAL, manifest: Path = EVAL_MANIFEST) -> list[dict]:
    m = json.loads(manifest.read_text())
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == m["eval"]["sha256"], f"{path} changed"
    return [json.loads(line) for line in data.decode().splitlines()]
