from castor import evalset, generate, posttrain


class CharTok:
    """One token per character of a tulu-like rendering (for label-mask tests)."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        s = "B"
        for m in messages:
            s += {"user": "U:", "assistant": "A:", "system": "S:"}[m["role"]] + m["content"]
            s += "E" if m["role"] == "assistant" else "\n"
        if add_generation_prompt:
            s += "A:"
        return list(range(len(s)))


MSGS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "ans"}]
# rendering: B U:hi\n A:yoE U:q\n A:ansE -> lengths: prefix "BU:hi\n"=6, +"A:"=8,
# after asst1 "A:yoE" -> 11, user2 "U:q\n" -> 15, +"A:" -> 17, asst2 "ansE" -> 21


def test_sft_labels_mask_matches_open_instruct_rule():
    n, train = posttrain.sft_labels(CharTok(), MSGS, None)
    assert n == 21
    trainable = [i for i, t in enumerate(train) if t]
    assert trainable == [8, 9, 10] + [17, 18, 19, 20]  # "yoE" and "ansE"


def test_sft_labels_truncation_and_drop():
    n, train = posttrain.sft_labels(CharTok(), MSGS, 12)
    assert n == 21 and len(train) == 12 and [i for i, t in enumerate(train) if t] == [8, 9, 10]
    _, train = posttrain.sft_labels(CharTok(), MSGS, 8)  # prompt fills the window
    assert not any(train)
    s = posttrain.length_stats(CharTok(), [{"messages": MSGS}] * 3, 8)
    assert s["truncated"] == 3 and s["dropped"] == 3 and s["len_max"] == 21


def test_eval_select_rule_is_inclusive():
    rows = [{"id": str(i)} for i in range(3)]
    kept, dropped = evalset.select(rows, [1536, 1537, 10])
    assert [r["id"] for r in kept] == ["0", "2"] and [r["id"] for r in dropped] == ["1"]
    assert kept[0]["n_prompt_tokens"] == 1536


def test_prompt_seed_depends_on_prompt_and_model():
    a = generate.prompt_seed("harmbench-00001", "m1")
    assert a == generate.prompt_seed("harmbench-00001", "m1")
    assert a != generate.prompt_seed("harmbench-00002", "m1")
    assert a != generate.prompt_seed("harmbench-00001", "m2")  # no common random numbers
    assert 0 <= generate.prompt_seed("x", "m") < 2**31 - generate.D017.k  # room for child seeds


def test_completion_tokens_excludes_stop_token():
    assert generate.completion_tokens([5, 6, 7], "stop", {7}) == 2
    assert generate.completion_tokens([5, 6, 7], "length", {7}) == 3
    assert generate.completion_tokens([5, 6, 8], "stop", {7}) == 3  # stopped on a stop string
    assert generate.completion_tokens([], "stop", {7}) == 0


def test_template_kind_refuses_unknown():
    assert generate.template_kind("{{ bos_token }}T", "{{ bos_token }}T", "T") == "tulu+bos"
    assert generate.template_kind("T", "{{ bos_token }}T", "T") == "tulu"
    try:
        generate.template_kind("other", "{{ bos_token }}T", "T")
    except ValueError:
        return
    raise AssertionError("unknown template accepted")


def test_to_flags_matches_hfargparser_conventions():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "posttrain_run", Path(__file__).parent.parent / "scripts" / "posttrain_run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    flags = m.to_flags({"add_bos": True, "push_to_hub": False, "learning_rate": 3e-05,
                        "dataset_mixer_list": ["/x.jsonl", "1.0"], "seed": 1})
    assert flags == ["--add_bos", "true", "--push_to_hub", "false", "--learning_rate", "3e-05",
                     "--dataset_mixer_list", "/x.jsonl", "1.0", "--seed", "1"]
