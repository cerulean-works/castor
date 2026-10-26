"""The CASTOR judging instrument.

Generations in, per-response labels out:

    generations.jsonl ──coherence──> coherence.jsonl ─┐
                      ──wildguard──> wildguard.jsonl ─┼─ labels.compose ─> labels.jsonl
                      ──qwen3guard─> qwen3guard.jsonl ┘

Every stage keys its rows by (model, prompt_id, sample); see `castor.judge.common`.
"""

from castor.judge import coherence, qwen3guard, wildguard

JUDGE_VERSION = "judge_v1"

# The LLM stages' classes and pinned engines, for tools that need a stage's input builder on
# its tokenizer alone (scripts/scan_judge_inputs.py, tests): name -> (class, model, revision,
# max_model_len, max_new_tokens). One table, so no copy drifts from the modules.
STAGES = {
    "coherence": (coherence.ValidityLLM, coherence.VALIDITY_MODEL, coherence.VALIDITY_REVISION,
                  coherence.VALIDITY_MAX_MODEL_LEN, coherence.VALIDITY_MAX_NEW_TOKENS),
    "wildguard": (wildguard.WildGuard, wildguard.MODEL, wildguard.REVISION,
                  wildguard.MAX_MODEL_LEN, wildguard.MAX_NEW_TOKENS),
    "qwen3guard": (qwen3guard.Qwen3Guard, qwen3guard.MODEL, qwen3guard.REVISION,
                   qwen3guard.MAX_MODEL_LEN, qwen3guard.MAX_NEW_TOKENS),
}
