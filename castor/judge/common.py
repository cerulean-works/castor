"""Shared pieces of the judges: the generation schema, JSONL IO, a greedy vLLM classifier
runner that keeps top-k log-probs, and soft scores for `Key: value` answer lines.

Generation rows (written by the generation step) carry at least:
    model      str   model id, `<hf repo>@<revision>` (or a castor post-train id)
    prompt_id  str   panel id, e.g. `harmbench-00001`
    sample     int   0..K-1 for the T=1 samples, -1 for the greedy pass
    prompt     str   the panel prompt as given to the model (user turn)
    response   str   the decoded completion, special tokens stripped
    n_tokens   int   completion length in the generating model's tokens (EOS excluded)
Every judge writes one row per generation row with the same three key fields.

Soft scores (2609.04482: parse failures silently become non-refusal). For each answer line
`<key>: <value>` we read the distribution of the first token of the value from the top-k
log-probs vLLM returns at that position, sum probability mass per class (tokens whose
decoded text, stripped and lower-cased, equals the class name), and report
p = P(class_pos) / (P(class_pos) + P(class_neg)) plus the covered mass. A class that is not
in the top-k contributes 0.
"""

import hashlib
import json
import math
from collections.abc import Iterable, Iterator
from pathlib import Path

from numpy import logaddexp

KEY = ("model", "prompt_id", "sample")
TOP_LOGPROBS = 20  # vLLM default max_logprobs


def key(row: dict) -> tuple:
    return tuple(row[k] for k in KEY)


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _hit(tok: str, names: list[str], prefix: bool) -> bool:
    t = tok.strip().lower()
    return bool(t) and any(n.lower().startswith(t) if prefix else n.lower() == t for n in names)


def value_distribution(tokens: list[str], top: list[dict[str, float]], field: str,
                       classes: dict[str, list[str]], prefix: bool = False) -> dict | None:
    """Distribution over `classes` of the first value token after `<field>:`.

    tokens: decoded text of each generated token (in order); top: per position, a map
    decoded-token-text -> logprob (the top-k alternatives, including the chosen token).
    classes: class name -> accepted token strings (compared stripped, lower-cased). With
    prefix=True a token counts for a class if it is a non-empty prefix of one of its names,
    for values the tokenizer splits (Qwen3Guard's "Controversial" is Cont|rovers|ial); the
    names must then not share a prefix.
    Returns {class: prob, ..., "mass": covered prob} or None if the field was not generated.
    """
    text = ""
    target = field.lower() + ":"
    start = None
    for i, t in enumerate(tokens):
        text += t
        if start is None and text.lower().rstrip().endswith(target):
            start = i + 1
            continue
        if start is not None and i >= start and t.strip():
            dist = {c: sum(math.exp(lp) for tok, lp in top[i].items() if _hit(tok, names, prefix))
                    for c, names in classes.items()}
            dist["mass"] = sum(dist[c] for c in classes)
            return dist
    return None


def first_token_distribution(top: list[dict[str, float]], classes: dict[str, list[str]],
                             prefix: bool = False) -> dict | None:
    """Class distribution at the first generated position (one-word answers). With
    prefix=True a token counts for a class if it is a non-empty prefix of one of its names
    (for answers the tokenizer may split, e.g. IN|VALID); names must not share prefixes."""
    if not top:
        return None
    dist = {c: sum(math.exp(lp) for tok, lp in top[0].items() if _hit(tok, names, prefix))
            for c, names in classes.items()}
    dist["mass"] = sum(dist[c] for c in classes)
    return dist


def binary_p(dist: dict | None, pos: str, neg: str) -> float | None:
    """P(pos) / (P(pos) + P(neg)); None if the field was missing or both are out of top-k."""
    if dist is None or dist[pos] + dist[neg] == 0:
        return None
    return dist[pos] / (dist[pos] + dist[neg])


# Over-long judge inputs. A generation can tokenize far longer under a judge's tokenizer than
# under the generating model's (a 512-token OLMo run of `0` is 12,594 Qwen tokens), and an input
# over max_model_len - max_new_tokens cannot be judged. Such a response is cut to fit, from the
# response only (template and task intact): its first `budget` judge tokens, decoded back to
# text, budget = limit - len(input with an empty response) - FIT_MARGIN. Inputs within the limit
# take the unchanged path. Truncating a classifier's input to its context is the standard
# handling (vLLM `truncate_prompt_tokens`, HF `truncation=True`); the validity
# prompt is itself truncation-aware (2605.22984 App. B.5).
FIT_MARGIN = 8  # slack below the limit for tokens that merge across the cut seam
FIT_STEP = 8  # tokens the budget shrinks by when the rebuilt input still overshoots
INPUT_TRUNCATION = "response cut to fit max_model_len - max_new_tokens"
JUDGE_V1_PATCH = "040"


def cut_text(tokenizer, ids: list[int], budget: int) -> str:
    """The first `budget` tokens of `ids` as text. A byte-level (Qwen) or byte-fallback
    (Llama/Mistral) tokenizer can split one character over several tokens, and decoding a
    partial character yields U+FFFD, a glyph the generation never contained; that tail is
    dropped so the judge sees only text the model emitted."""
    return tokenizer.decode(ids[:budget]).rstrip("�")


def fit_response(tokenizer, build, response: str, limit: int) -> tuple[list[int], bool]:
    """build(response) -> judge token ids. Returns (ids, truncated): build(response) itself
    when it fits within `limit`, else build(cut) with `cut` the longest leading part of the
    response (in judge tokens, at most limit - len(build("")) - FIT_MARGIN) that fits."""
    ids = build(response)
    if len(ids) <= limit:
        return ids, False
    resp = tokenizer(response, add_special_tokens=False)["input_ids"]
    n_frame = len(build(""))
    budget = min(len(resp), limit - n_frame - FIT_MARGIN)
    while budget > 0:
        # re-tokenizing the cut inside the template can merge across the cut; shrink and retry
        ids = build(cut_text(tokenizer, resp, budget))
        if len(ids) <= limit:
            return ids, True
        budget -= FIT_STEP
    raise ValueError(f"judge template plus the panel prompt leave no room for a response: "
                     f"{n_frame} tokens with an empty response, limit {limit} "
                     f"(an over-long prompt, not a template bug: only the response is cut)")


def truncation_fields(ids: list[int], truncated: bool) -> dict:
    """Row fields recording a cut; empty when none, so uncut rows are unchanged."""
    return {"truncated_to_fit": True, "n_tokens_judge_input": len(ids)} if truncated else {}


class VLLMClassifier:
    """Greedy vLLM generation over pre-tokenized prompts, keeping top-k log-probs."""

    def __init__(self, model: str, revision: str, max_model_len: int, max_tokens: int,
                 gpu_memory_utilization: float = 0.9, seed: int = 0):
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model, revision=revision, tokenizer_revision=revision,
                       dtype="bfloat16", max_model_len=max_model_len, seed=seed,
                       gpu_memory_utilization=gpu_memory_utilization,
                       max_logprobs=TOP_LOGPROBS)
        self.tokenizer = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                                     logprobs=TOP_LOGPROBS, seed=seed)
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens

    @property
    def limit(self) -> int:
        """Longest prompt that leaves room for max_tokens new tokens."""
        return self.max_model_len - self.max_tokens

    def run(self, token_ids: list[list[int]]) -> list[dict]:
        """-> per prompt: text, tokens (decoded per position), top (text -> logprob)."""
        from vllm.inputs import TokensPrompt

        limit = self.limit
        too_long = [i for i, t in enumerate(token_ids) if len(t) > limit]
        if too_long:  # unreachable: every stage fits its inputs (fit_response)
            raise ValueError(f"{len(too_long)} judge inputs exceed {limit} tokens "
                             f"(first index {too_long[0]})")
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=t) for t in token_ids],
                                 self.params, use_tqdm=True)
        res = []
        for o in outs:
            c = o.outputs[0]
            ids = list(c.token_ids)
            # Per-position text as the difference of prefix decodes: sentencepiece drops the
            # leading space when a single token is decoded alone.
            prefix = [self.tokenizer.decode(ids[:i]) for i in range(len(ids) + 1)]
            tokens = [prefix[i + 1][len(prefix[i]):] for i in range(len(ids))]
            top = []
            for lp in c.logprobs or []:
                alt: dict[str, float] = {}
                for aid, a in lp.items():
                    s = self.tokenizer.decode([aid])
                    alt[s] = float(logaddexp(alt.get(s, -math.inf), a.logprob))
                top.append(alt)
            res.append({"text": c.text, "tokens": tokens, "top": top,
                        "n_prompt_tokens": len(o.prompt_token_ids)})
        return res
