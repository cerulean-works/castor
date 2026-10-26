"""Judge inputs over max_model_len - max_new_tokens are cut, from the response only, so they
fit; inputs within the limit take the unchanged path and gain no field.

The real-tokenizer tests load only tokenizers (no weights) from the HF cache, offline, and
skip when a tokenizer is not cached.
"""

import json
import os
from pathlib import Path

import pytest

from castor import pipeval
from castor.judge import STAGES, coherence, qwen3guard, wildguard
from castor.judge.common import (
    FIT_MARGIN,
    FIT_STEP,
    cut_text,
    fit_response,
    truncation_fields,
    write_jsonl,
)
from castor.provenance import load_script

# a row that overflowed the coherence judge: 533 characters of text then 12,480 `0`. A rerun
# regenerates this file (greedy bf16 under vLLM is not bit-reproducible), so the test skips,
# never fails, when the row is no longer that one.
OFFENDING = Path("/work/harrychang08/castor/runs/dd-dolma1_7-no-math-code-1b-large-aux-3-"
                 "dpo-seed1/008/greedy.jsonl")
OFFENDING_KEY = ("wildjailbreak-00287", -1)
OFFENDING_SHAPE = {"chars": 13013, "n_tokens": 512}


# ---- the helper on a character-level tokenizer (always runs) ------------------------------


class CharTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(map(chr, ids))


def char_build(resp: str) -> list[int]:
    return [ord(c) for c in f"<s>[TASK]: x\n[RESPONSE]: {resp}\nAnswer:"]


def test_fit_under_limit_is_the_unchanged_path():
    ids, cut = fit_response(CharTok(), char_build, "short answer", 100)
    assert ids == char_build("short answer") and cut is False
    assert truncation_fields(ids, cut) == {}


def test_fit_over_limit_cuts_the_response_only():
    resp = "0" * 1000
    ids, cut = fit_response(CharTok(), char_build, resp, 100)
    text = CharTok().decode(ids)
    assert cut and len(ids) <= 100
    assert text.startswith("<s>[TASK]: x\n[RESPONSE]: 0") and text.endswith("0\nAnswer:")
    assert len(ids) == 100 - FIT_MARGIN  # budget = limit - template - margin, exact here
    assert truncation_fields(ids, cut) == {"truncated_to_fit": True,
                                           "n_tokens_judge_input": len(ids)}


def test_fit_at_exactly_the_limit_is_not_cut():
    n = 100 - len(char_build(""))
    ids, cut = fit_response(CharTok(), char_build, "a" * n, 100)
    assert len(ids) == 100 and cut is False


def test_fit_raises_when_the_template_alone_is_too_long():
    with pytest.raises(ValueError):
        fit_response(CharTok(), lambda r: [0] * 200 + [1] * len(r), "abc", 100)


# ---- the three stages on their real tokenizers --------------------------------------------


def _tokenizer(model: str, revision: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(model, revision=revision)
    except OSError as e:  # not cached
        pytest.skip(f"{model}@{revision} tokenizer not cached: {e}")


class FakeClassifier:
    """VLLMClassifier without an engine: records the inputs and answers a fixed text."""

    def __init__(self, tok, max_model_len, max_tokens):
        self.tokenizer, self.limit = tok, max_model_len - max_tokens
        self.seen: list[list[int]] = []

    def run(self, token_ids):
        assert all(len(t) <= self.limit for t in token_ids)  # what VLLMClassifier.run raises on
        self.seen += token_ids
        return [{"text": "VALID", "tokens": ["VALID"], "top": [{"VALID": 0.0}],
                 "n_prompt_tokens": len(t)} for t in token_ids]


@pytest.fixture(scope="module", params=list(STAGES))
def stage(request):
    cls, model, rev, max_len, max_new = STAGES[request.param]
    judge = object.__new__(cls)  # no vLLM engine
    judge.clf = FakeClassifier(_tokenizer(model, rev), max_len, max_new)
    return request.param, judge


def _old_ids(name, judge, prompt, response):
    """The input builders before truncation, copied from d4e4c9c: the frozen formulation the
    live `raw_ids` must reproduce byte for byte (deliberately not a call into the package)."""
    tok = judge.clf.tokenizer
    if name == "coherence":
        text = tok.apply_chat_template(
            [{"role": "user", "content": coherence.VALIDITY_PROMPT.format(goal=prompt,
                                                                          response=response)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tok(text, add_special_tokens=False)["input_ids"]
    if name == "wildguard":
        return tok(wildguard.format_input(prompt, response), add_special_tokens=False)[
            "input_ids"]
    return tok(tok.apply_chat_template(qwen3guard.messages(prompt, response), tokenize=False))[
        "input_ids"]


def _judge_rows(name, judge, rows):
    if name == "coherence":
        return judge.check([(r["prompt"], r["response"]) for r in rows])
    return judge.judge(rows)


PROMPT = "Write a short poem about the sea."


def test_long_response_fits_and_keeps_task_and_template(stage):
    name, judge = stage
    tok, limit = judge.clf.tokenizer, judge.clf.limit
    marker = "Sure, here is the number: "
    response = marker + "0" * 20000
    assert len(_old_ids(name, judge, PROMPT, response)) > limit  # the old code raised here
    ids, cut = judge.input_ids(PROMPT, response)
    assert cut and limit - FIT_MARGIN - 3 * FIT_STEP <= len(ids) <= limit
    head, tail = tok.decode(_old_ids(name, judge, PROMPT, marker + "@@END@@")).split("@@END@@")
    text = tok.decode(ids)
    assert PROMPT in text and text.startswith(head) and text.endswith(tail)
    assert set(text[len(head):len(text) - len(tail)]) == {"0"}  # only the response was cut
    row = {"model": "m", "prompt_id": "p", "sample": -1, "prompt": PROMPT,
           "response": response}
    (out,) = _judge_rows(name, judge, [row])
    assert out["truncated_to_fit"] is True and out["n_tokens_judge_input"] == len(ids)


def test_short_response_is_byte_identical_and_gains_no_field(stage):
    name, judge = stage
    response = "I can't help with that, but here is a poem about the sea instead."
    ids, cut = judge.input_ids(PROMPT, response)
    assert ids == _old_ids(name, judge, PROMPT, response) and cut is False
    assert judge.raw_ids(PROMPT, response) == ids  # the live builder the scan measures
    row = {"model": "m", "prompt_id": "p", "sample": 0, "prompt": PROMPT, "response": response}
    (out,) = _judge_rows(name, judge, [row])
    assert "truncated_to_fit" not in out and "n_tokens_judge_input" not in out


def test_the_offending_row_fits(stage):
    """wildjailbreak-00287 greedy of dolma1_7-no-math-code-1B-large-aux-3-s1: 533 characters
    of text then 12,480 `0` (13,013 characters, 512 OLMo tokens)."""
    name, judge = stage
    if not OFFENDING.exists():
        pytest.skip("008 cell generations not on this machine")
    rows = [json.loads(line) for line in OFFENDING.open() if line.strip()]
    hits = [r for r in rows if (r["prompt_id"], r["sample"]) == OFFENDING_KEY]
    if len(hits) != 1 or {"chars": len(hits[0]["response"]), "n_tokens": hits[0]["n_tokens"]} \
            != OFFENDING_SHAPE:
        pytest.skip(f"{OFFENDING_KEY} is no longer the 2026-10-08 row (cell regenerated)")
    (row,) = hits
    limit = judge.clf.limit
    old = _old_ids(name, judge, row["prompt"], row["response"])
    assert len(old) > limit
    ids, cut = judge.input_ids(row["prompt"], row["response"])
    assert cut and limit - FIT_MARGIN - 3 * FIT_STEP <= len(ids) <= limit
    # budget arithmetic: the cut keeps exactly limit - len(template + task) - margin
    # response tokens, give or take re-tokenization at the two seams
    n_frame = len(_old_ids(name, judge, row["prompt"], ""))
    assert abs((len(ids) - n_frame) - (limit - n_frame - FIT_MARGIN)) <= 2
    if name == "coherence":  # the stage that raised
        tok = judge.clf.tokenizer
        assert len(tok(row["response"], add_special_tokens=False)["input_ids"]) == 12594


# ---- the scan's byte bound: a stage's input has at most bytes + BYTE_SLACK tokens ---------
# scripts/scan_judge_inputs.py tokenizes only rows this bound cannot clear; the reported
# n_over (0 in every judged cell) rests on it.


ADVERSARIAL = ["", " ", "\n" * 50, " " * 50, "\t\t\t", "\r\n" * 20, "\u00a0" * 20,
               "😀" * 30, "👨\u200d👩\u200d👧\u200d👦" * 10, "𝔘𝔫𝔦𝔠𝔬𝔡𝔢" * 10, "中文測試" * 20,
               "é" * 40, "\u0301" * 20, "ﬁ" * 20, "Ĳ" * 20, "x\u200b" * 20, "ـ" * 20,
               "ᚠᚢᚦ" * 20, "\u2581x", "a\u2581b", "<s>", "</s>", "<|user|>", "<|assistant|>",
               "[INST]", "[/INST]", "<|im_start|>", "<|im_end|>", "<|endoftext|>",
               "<think>", "0" * 100, "a b c d e " * 10, "Sure, here is"]


def test_scan_byte_bound_holds_on_adversarial_text(stage):
    scan = load_script("scan_judge_inputs")
    name, judge = stage
    frame = scan.frame_bytes(judge)
    assert frame == len(judge.format_text("", "").encode())
    for prompt in ["Write a poem", "", "中文", "<s>", "🦀" * 5]:
        for response in ADVERSARIAL:
            n_bytes = len(prompt.encode()) + len(response.encode()) + frame
            assert len(judge.raw_ids(prompt, response)) <= n_bytes + scan.BYTE_SLACK, \
                (name, prompt, response)
            # the skip rule is exactly that bound against the limit
            assert scan.may_exceed(frame, n_bytes + scan.BYTE_SLACK, prompt, response) is False
            assert scan.may_exceed(frame, n_bytes + scan.BYTE_SLACK - 1, prompt, response)


# ---- a cut never lands inside a character --------------------------------------------------
# Byte-level (Qwen) and byte-fallback (Llama) tokenizers split one character over several
# tokens; decoding a partial character gives U+FFFD, which the judge must not see.


def test_cut_text_drops_a_partial_character():
    class ByteTok:
        def decode(self, ids):
            return bytes(ids).decode(errors="replace")

    ids = list("ab𝔘".encode())  # 2 + 4 bytes
    assert cut_text(ByteTok(), ids, 2) == "ab"
    assert cut_text(ByteTok(), ids, 4) == "ab"  # two of the four bytes of 𝔘
    assert cut_text(ByteTok(), ids, 6) == "ab𝔘"


# fraktur and runic are rare enough that all three tokenizers split them into bytes (CJK is
# a single token per character under Qwen and never lands a cut inside one)
@pytest.mark.parametrize("response", ["𝔘𝔫𝔦𝔠𝔬𝔡𝔢" * 3000, "ᚠᚢᚦ" * 4000],
                         ids=["fraktur", "runic"])
def test_fitted_input_never_ends_in_a_partial_character(stage, response):
    name, judge = stage
    tok = judge.clf.tokenizer
    resp = tok(response, add_special_tokens=False)["input_ids"]
    full = judge.clf.limit
    # the raw decode lands inside a character for some budget below the limit ...
    assert any("\ufffd" in tok.decode(resp[:b]) for b in range(full - 24, full))
    # ... and the fitted input never does, whatever the limit
    for limit in range(full - 24, full):
        ids, cut = fit_response(tok, lambda r: judge.raw_ids(PROMPT, r), response, limit)
        assert cut and len(ids) <= limit
        assert "\ufffd" not in tok.decode(ids), (name, limit)


# ---- the stage meta: the patch marker is the instrument, the counts are the run ----------


def _write_run(runs: Path, model_id: str, patch: bool) -> None:
    d = runs / model_id / "006"
    (d / "judge_v1").mkdir(parents=True)
    (d / "provenance.json").write_text(json.dumps({
        "chat_template": "t", "sampling": {"T": 1}, "seed_rule": "r",
        "eval_manifest": {"sha256": "abc"}}))
    for stage_name in ("wildguard", "qwen3guard", "coherence", "labels"):
        meta = {"stage": stage_name, "judge_version": "judge_v1", "model": "x@1",
                "n": 10, "git_commit": "c", "command": "cmd", "generations": ["g"],
                "throughput": {}}
        if patch and stage_name != "labels":
            meta.update({"input_truncation": "response cut to fit max_model_len - "
                         "max_new_tokens", "judge_v1_patch": "040", "n_truncated_to_fit": 1})
        (d / "judge_v1" / f"{stage_name}.meta.json").write_text(json.dumps(meta))


def test_instrument_check_accepts_pre_and_post_040_cells(tmp_path):
    _write_run(tmp_path, "before", patch=False)
    _write_run(tmp_path, "after", patch=True)
    models = {"a": ("before", "006"), "b": ("after", "006")}
    got = pipeval.check_instrument(models, "abc", runs=tmp_path)
    assert got["wildguard"]["model"] == "x@1"
    assert got["wildguard"]["judge_v1_patch"] == "040"  # no marker: before the patch
    assert "n_truncated_to_fit" not in got["wildguard"]  # a run count, not the instrument


def test_instrument_check_rejects_another_patch(tmp_path):
    _write_run(tmp_path, "p040", patch=True)
    _write_run(tmp_path, "p041", patch=True)
    meta_path = tmp_path / "p041" / "006" / "judge_v1" / "wildguard.meta.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps(meta | {"judge_v1_patch": "041"}))
    models = {"a": ("p040", "006"), "b": ("p041", "006")}
    with pytest.raises(ValueError, match="different instruments"):
        pipeval.check_instrument(models, "abc", runs=tmp_path)
    meta_path.write_text(json.dumps(meta | {"input_truncation": "whole input cut"}))
    with pytest.raises(ValueError, match="different instruments"):
        pipeval.check_instrument(models, "abc", runs=tmp_path)


def test_run_model_stage_does_not_write_into_the_judge_meta(monkeypatch):
    judge_script = load_script("judge")
    shared = {"judge": "wildguard", "model": "x"}

    class Fake:
        def judge(self, rows):
            return [{"model": r["model"], "prompt_id": r["prompt_id"], "sample": r["sample"],
                     "truncated_to_fit": True} for r in rows]

        @property
        def meta(self):
            return shared

    monkeypatch.setattr(wildguard, "WildGuard", Fake)
    rows = [{"model": "m", "prompt_id": "p", "sample": 0}]
    for _ in range(2):
        _, meta = judge_script.run_model_stage("wildguard", rows)
        assert meta["n_truncated_to_fit"] == 1 and meta["judge_v1_patch"] == "040"
    assert shared == {"judge": "wildguard", "model": "x"}


def test_wildguard_meta_is_a_copy():
    j = object.__new__(wildguard.WildGuard)
    assert j.meta == wildguard.META and j.meta is not wildguard.META


# ---- labels.jsonl carries the cut flag, so audits and per-cell reports can see cut rows ----


def test_labels_carry_truncated_to_fit_from_any_stage(tmp_path):
    judge_script = load_script("judge")
    keys = [("m", "p0", 0), ("m", "p1", 0), ("m", "p2", 0)]
    gens = {k: {"model": k[0], "prompt_id": k[1], "sample": k[2], "prompt": "q",
                "response": "Sure, here is"} for k in keys}

    def rows(fields: dict, cut: set) -> list[dict]:
        return [{"model": k[0], "prompt_id": k[1], "sample": k[2], **fields,
                 **({"truncated_to_fit": True, "n_tokens_judge_input": 1} if k in cut else {})}
                for k in keys]

    coh = rows({"coherent": True, "rule": None}, cut={keys[0]})  # cut in coherence only
    wg = rows({"refusal": False, "response_harmful": False}, cut={keys[1]})  # in wildguard only
    write_jsonl(tmp_path / "coherence.jsonl", coh)
    write_jsonl(tmp_path / "wildguard.jsonl", wg)
    out, _ = judge_script.compose_labels(tmp_path, gens)
    assert [r["truncated_to_fit"] for r in out] == [True, True, False]
