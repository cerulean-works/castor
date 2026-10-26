"""The sampling instrument: one vLLM config for every twin and reference model.

vLLM 0.23.0, bf16, max_model_len 2048, T = 1.0, top_p = 1.0, K = 20 samples,
max_new_tokens = 512, plus one greedy pass. The prompt is the model's own chat template with
the generation prompt, passed as token ids (so a template BOS is never doubled). Each request's
sampling seed is derived from the panel id *and* the model id: two models never share
their sampling noise, so the K samples of different models are independent draws, which is
what the split-half ceiling and the disattenuation in `castor.reliability` assume. Nothing
here depends on the model beyond its weights, tokenizer and id.

Rows written by the generation step follow `castor.judge.common` (model, prompt_id, sample,
prompt, response, n_tokens, finish_reason); `n_tokens` excludes the EOS / stop token.
"""

import hashlib
from dataclasses import asdict, dataclass

CONTEXT = 2048


@dataclass(frozen=True)
class SamplingConfig:
    k: int = 20
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    greedy: bool = True
    seed_salt: str = "castor-d017"


D017 = SamplingConfig()


def prompt_seed(prompt_id: str, model_id: str, salt: str = D017.seed_salt) -> int:
    """vLLM request seed for one (model, prompt): a hash of both ids, so no two models share
    random draws. vLLM gives the n samples of a request seeds seed, seed+1, ...,
    seed+n-1 (v1 parallel sampling), so the seed is kept below 2**31 - K."""
    h = hashlib.sha256(f"{salt}:{model_id}:{prompt_id}".encode()).digest()
    return int.from_bytes(h[:4], "big") % (2**31 - D017.k)


def completion_tokens(token_ids: list[int], finish_reason: str | None,
                      stop_ids: set[int]) -> int:
    """Length of a completion in the generating model's tokens, EOS / stop token excluded
    (vLLM keeps the token that stopped the sequence in `token_ids`)."""
    n = len(token_ids)
    if n and finish_reason == "stop" and token_ids[-1] in stop_ids:
        n -= 1
    return n


def template_kind(chat_template: str | None, tulu_bos: str, tulu_nobos: str) -> str:
    """Which tulu variant a model's saved template is; anything else is refused."""
    if chat_template == tulu_bos:
        return "tulu+bos"
    if chat_template == tulu_nobos:
        return "tulu"
    raise ValueError("model chat template is not open-instruct's tulu template")


def config_dict(cfg: SamplingConfig = D017) -> dict:
    return asdict(cfg) | {"engine": "vllm==0.23.0", "dtype": "bfloat16", "max_model_len": CONTEXT}
