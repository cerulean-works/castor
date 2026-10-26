import importlib.util
import json
import os
from pathlib import Path

import pytest

from castor import decontam, swapmix

_spec = importlib.util.spec_from_file_location(
    "build_swap_mix", Path(__file__).parents[1] / "scripts/build_swap_mix.py")
bsm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsm)

LONG = " ".join(f"w{i}" for i in range(20))


HR = {"prompt_harmful": True, "refusal": True, "response_harmful": False}
BC = {"prompt_harmful": False, "refusal": False, "response_harmful": False}
HC = {"prompt_harmful": True, "refusal": False, "response_harmful": True}


def _pool(name, items):
    """items: (prompt, response, labels) -> (records, labels by id)."""
    recs = [{"id": f"{name}/{i}", "source": swapmix.SWAP_SOURCE[name],
             "messages": swapmix.pair(p, r)} for i, (p, r, _) in enumerate(items)]
    return recs, {r["id"]: lab for r, (_, _, lab) in zip(recs, items)}


def _pools(**sets):
    pools, labels = {}, {}
    for name, items in sets.items():
        pools[name], lab = _pool(name, items)
        labels.update(lab)
    return pools, labels


def test_nemotron_v2_english_only_drops_reasoning():
    rows = [(0, {"messages": [{"role": "user", "content": "p"},
                              {"role": "assistant", "content": "no", "reasoning_content": "t"}],
                 "translation_languages": None, "response_policy": "refusal_no_redirection",
                 "metadata": {"prompt_label": "unsafe"}}),
            (1, {"messages": [{"role": "user", "content": "q"},
                              {"role": "assistant", "content": "non"}],
                 "translation_languages": "en_fr", "response_policy": "x",
                 "metadata": {"prompt_label": "unsafe"}})]
    got = list(swapmix.nemotron_v2_examples(rows))
    assert got == [("nemotron_v2/train/0", swapmix.pair("p", "no"),
                    {"prompt_label": "unsafe", "response_policy": "refusal_no_redirection"})]


def test_llama_nemotron_first_off_copy_per_prompt():
    def row(sp, p, out):
        return {"system_prompt": sp, "input": [{"role": "user", "content": p}], "output": out}
    rows = list(enumerate([row("detailed thinking on", "a", "<think>x</think>no"),
                           row("detailed thinking off", "a", "<think>\n</think>\nI can't."),
                           row("detailed thinking off", "a ", "second copy"),
                           row("detailed thinking off", "b", "sure")]))
    got = list(swapmix.llama_nemotron_examples(rows))
    assert [i for i, _ in got] == ["llama_nemotron/safety/1", "llama_nemotron/safety/3"]
    assert got[0][1] == swapmix.pair("a", "I can't.")


def test_llama_nemotron_skips_malformed_input():
    def row(turns):
        return {"system_prompt": "detailed thinking off", "output": "x",
                "input": [{"role": r, "content": c} for r, c in turns]}
    rows = list(enumerate([row([]), row([("assistant", "a")]),
                           row([("user", "a"), ("assistant", "b")]), row([("user", "c")])]))
    assert [i for i, _ in swapmix.llama_nemotron_examples(rows)] == ["llama_nemotron/safety/3"]


def test_pku_takes_only_safe_responses():
    rows = [("A", 0, {"prompt": "p", "response_0": "a", "response_1": "b",
                      "is_response_0_safe": "True", "is_response_1_safe": "False"}),
            ("A", 1, {"prompt": "q", "response_0": "c", "response_1": "d",
                      "is_response_0_safe": False, "is_response_1_safe": True})]
    got = list(swapmix.pku_examples(rows))
    assert [i for i, _ in got] == ["pku/A/0/0", "pku/A/1/1"]
    assert got[1][1] == [{"content": "q", "role": "user"}, {"content": "d", "role": "assistant"}]


def test_beavertails_filters_is_safe():
    rows = [(0, {"prompt": "p", "response": "a", "is_safe": True}),
            (1, {"prompt": "p", "response": "b", "is_safe": False})]
    assert [i for i, _ in swapmix.beavertails_examples(rows)] == ["beavertails/330k_train/0"]


def test_parse_hh_multi_turn():
    t = "\n\nHuman: hi\n\nAssistant: hello\n\nHuman: how to x?\n\nAssistant: I won't help."
    m = swapmix.parse_hh(t)
    assert [x["role"] for x in m] == ["user", "assistant", "user", "assistant"]
    assert m[-1]["content"] == "I won't help." and m[2]["content"] == "how to x?"
    assert swapmix.last_turns(m) == ("how to x?", "I won't help.")


def test_parse_hh_rejects_malformed():
    assert swapmix.parse_hh("\n\nHuman: a\n\nHuman: b\n\nAssistant: c") is None
    assert swapmix.parse_hh("\n\nHuman: a\n\nAssistant: ") is None
    assert swapmix.parse_hh("\n\nAssistant: a") is None
    assert swapmix.parse_hh("junk\n\nHuman: a\n\nAssistant: b") is None


def test_shuffle_is_fixed():
    xs = list(range(50))
    assert swapmix.shuffled(xs) == swapmix.shuffled(xs) != xs
    assert sorted(swapmix.shuffled(xs)) == xs


def test_both_directions_and_exact():
    idx = decontam.PanelIndex([LONG, "short prompt"])
    # candidate contains the whole panel prompt: panel prompt covered, candidate not (long)
    big = LONG + " " + " ".join(f"x{i}" for i in range(40))
    assert set(swapmix.contamination(idx, [big])) == {"panel_in_candidate"}
    # candidate is a short piece of the panel prompt: candidate covered, panel prompt not
    piece = " ".join(f"w{i}" for i in range(9))
    assert set(swapmix.contamination(idx, [piece])) == {"candidate_in_panel"}
    assert "exact" in swapmix.contamination(idx, ["Short, prompt!"])
    assert swapmix.contamination(idx, ["nothing in common at all here"]) == {}


def test_dedupe_pool_exact_pairs_across_sets_in_order():
    pools, _ = _pools(nemotron_v2=[("a", "1", HR), ("a", "1", HR), ("a", "2", HR)],
                      pku=[(" a ", "1", HR), ("b", "1", HR)])
    out, dups = swapmix.dedupe_pool(pools)
    assert [r["id"] for r in out["nemotron_v2"]] == ["nemotron_v2/0", "nemotron_v2/2"]
    assert [r["id"] for r in out["pku"]] == ["pku/1"]
    assert dups == {"nemotron_v2": 1, "llama_nemotron": 0, "pku": 1, "beavertails": 0, "hh": 0}


def test_select_two_passes_cap_order_and_fill():
    idx = decontam.PanelIndex([LONG])
    pools, labels = _pools(
        nemotron_v2=[("h1", "no", HR), ("b1", "ok", BC), ("h2", "sure", HC)],
        llama_nemotron=[("h3", "no", HR), ("h4", "no", HR)],
        pku=[("b2", "ok", BC), ("b3", "ok", BC), ("b4", "ok", BC)])
    rows, rep = swapmix.select(pools, labels, cap=2, n=4, index=idx)
    assert [r["id"] for r in rows] == ["castor_swap/nemotron_v2/0", "castor_swap/llama_nemotron/0",
                                       "castor_swap/nemotron_v2/1", "castor_swap/pku/0"]
    p1, p2 = rep["pass1_harmful_refusal"], rep["pass2_benign_compliance"]
    assert (p1["target"], p1["taken"], p2["target"], p2["taken"]) == (2, 2, 2, 2)
    assert p1["nemotron_v2"]["eligible"] == 1 and p1["llama_nemotron"]["scanned"] == 1
    assert p1["pku"]["scanned"] == 0 and rep["short"] == 0


def test_select_prompt_dedupe_kept_decontam_and_short():
    idx = decontam.PanelIndex([LONG])
    pools, labels = _pools(nemotron_v2=[("Same prompt!", "no", HR), ("same prompt", "no 2", HR),
                                        (LONG, "no", HR), ("kept one", "no", HR)],
                           pku=[("same, prompt", "ok", BC), ("fresh", "ok", BC)])
    kept = frozenset({swapmix.norm_prompt(swapmix.pair("Kept one.", "x"))})
    rows, rep = swapmix.select(pools, labels, cap=4, n=6, index=idx, kept=kept)
    # tier 2 adds the second response to "Same prompt!" in pass 1; pku/0's prompt stays pass 1's
    assert [r["id"] for r in rows] == ["castor_swap/nemotron_v2/0", "castor_swap/nemotron_v2/1",
                                       "castor_swap/pku/1"]
    p1 = rep["pass1_harmful_refusal"]["nemotron_v2"]
    assert (p1["duplicate_prompt"], p1["duplicate_of_kept"], p1["removed"]) == (2, 1, 1)
    assert (p1["taken_tier1"], p1["taken_tier2"]) == (1, 1)
    assert rep["pass2_benign_compliance"]["pku"]["duplicate_prompt"] == 1
    assert rep["pass2_benign_compliance"]["pku"]["tier2_candidates"] == 0
    assert rep["short"] == 3


def test_select_tier2_rounds_order_and_ownership():
    idx = decontam.PanelIndex([LONG])
    pools, labels = _pools(
        nemotron_v2=[("a", "1", HR), ("a", "2", HR), ("a", "3", HR), ("b", "1", HR),
                     ("a", "4", HC), ("c", "1", BC)],
        pku=[("b", "2", HR), ("c", "2", BC), ("a", "5", BC), (LONG + " a", "x", HR)])
    # pass 1 tier 1 takes a, b; round 2 = a's copy 2 (nemotron/1), b's copy 2 (pku/0);
    # round 3 = a's copy 3 (nemotron/2). Cap 4 takes rounds in order: nemotron/1, pku/0.
    rows, rep = swapmix.select(pools, labels, cap=4, n=7, index=idx)
    ids = [r["id"].removeprefix("castor_swap/") for r in rows]
    assert ids[:4] == ["nemotron_v2/0", "nemotron_v2/3", "nemotron_v2/1", "pku/0"]
    t2 = rep["pass1_harmful_refusal"]["tier2"]
    assert t2["candidates_by_round"] == {2: 2, 3: 1} and t2["taken_by_round"] == {2: 2}
    assert t2["cross_source"] == 1
    # pass 2: c taken in tier 1, its pku copy in tier 2; pku/2 ("a", benign) is pass 1's prompt
    assert ids[4:] == ["nemotron_v2/5", "pku/1"] and rep["short"] == 1
    p2 = rep["pass2_benign_compliance"]
    assert (p2["taken_tier1"], p2["taken_tier2"], p2["pku"]["tier2_candidates"]) == (1, 1, 1)
    assert p2["pku"]["duplicate_prompt"] == 2


def test_select_tier2_not_used_when_tier1_fills():
    pools, labels = _pools(pku=[("a", "1", HR), ("a", "2", HR), ("b", "1", BC)])
    rows, rep = swapmix.select(pools, labels, cap=1, n=2, index=decontam.PanelIndex(["zzz"]))
    assert [r["id"] for r in rows] == ["castor_swap/pku/0", "castor_swap/pku/2"]
    assert rep["pass1_harmful_refusal"]["tier2"]["candidates_by_round"] == {}


def test_select_cap_above_n_fills_n():
    pools, labels = _pools(pku=[("a", "1", HR), ("b", "1", HR), ("c", "1", HR), ("d", "1", BC)])
    rows, rep = swapmix.select(pools, labels, cap=5, n=2, index=decontam.PanelIndex(["zzz"]))
    assert len(rows) == 2 and rep["pass1_harmful_refusal"]["target"] == 2
    assert rep["pass2_benign_compliance"]["taken"] == 0 and rep["short"] == 0


def test_select_skips_truncated_labels_in_both_tiers():
    pools, labels = _pools(pku=[("a", "1", {**HR, "truncated": True}), ("b", "1", HR),
                                ("b", "2", {**HR, "truncated": True}), ("b", "3", HR)])
    rows, rep = swapmix.select(pools, labels, cap=3, n=3, index=decontam.PanelIndex(["zzz"]))
    assert [r["id"] for r in rows] == ["castor_swap/pku/1", "castor_swap/pku/3"]
    assert rep["pass1_harmful_refusal"]["pku"]["eligible"] == 2


def test_earlier_turns_and_inputs():
    m = swapmix.parse_hh("\n\nHuman: a\n\nAssistant: b\n\nHuman: c\n\nAssistant: d"
                         "\n\nHuman: e\n\nAssistant: f")
    assert swapmix.earlier_turns(m) == [(1, "a", "b"), (3, "c", "d")]
    assert swapmix.earlier_turns(swapmix.pair("p", "r")) == []
    rec = {"id": "hh/harmless-base/7", "source": "S", "messages": m}
    got = swapmix.turn_inputs(rec, "M")
    assert [(t["prompt_id"], t["prompt"], t["response"]) for t in got] == [
        ("hh/harmless-base/7#1", "a", "b"), ("hh/harmless-base/7#3", "c", "d")]


def test_select_drops_rows_with_a_harmful_earlier_turn():
    m = swapmix.parse_hh("\n\nHuman: a\n\nAssistant: b\n\nHuman: c\n\nAssistant: no")
    recs = [{"id": f"hh/{i}", "source": swapmix.SWAP_SOURCE["hh"], "messages": m}
            for i in range(3)]
    for i, r in enumerate(recs):
        r["messages"] = [*m[:-1], {"content": f"no {i}", "role": "assistant"}]
    labels = {r["id"]: dict(HR) for r in recs}
    labels |= {"hh/0#1": HC, "hh/1#1": {**BC, "truncated": True}, "hh/2#1": BC}
    counts = swapmix.fold_earlier(labels, recs)
    assert counts == {swapmix.SWAP_SOURCE["hh"]: {"rows": 3, "turns": 3, "turns_empty": 0,
                                                  "turns_truncated": 1, "rows_failing": 2}}
    rows, _ = swapmix.select({"hh": recs}, labels, cap=3, n=3, index=decontam.PanelIndex(["zzz"]))
    assert [r["id"] for r in rows] == ["castor_swap/hh/2"]


def test_prompt_multiplicity():
    rows = [{"messages": swapmix.pair(p, "x")} for p in ("A", "a!", "a", "b", "c", "C")]
    assert swapmix.prompt_multiplicity(rows) == {"rows": 6, "unique_norm_prompts": 3,
                                                 "repeat_rows": 3,
                                                 "histogram": {"1": 1, "2": 1, "3": 1}}


def test_select_skips_unparsed_labels():
    pools, labels = _pools(pku=[("a", "x", {**HR, "refusal": None}), ("b", "x", HR)])
    rows, _ = swapmix.select(pools, labels, cap=1, n=1, index=decontam.PanelIndex(["zzz"]))
    assert [r["id"] for r in rows] == ["castor_swap/pku/1"]


def test_composition_and_agreement():
    rows = [{"id": "a", "source": "S", "messages": swapmix.pair("p", "x y")},
            {"id": "b", "source": "S", "messages": swapmix.pair("q", "x")},
            {"id": "c", "source": "T", "messages": swapmix.pair("p", "x y z")},
            {"id": "d", "source": "T", "messages": swapmix.pair("r", "x")}]
    labels = {"a": HR, "b": BC, "c": HC,
              "d": {"prompt_harmful": False, "refusal": True, "response_harmful": None}}
    c = swapmix.composition(rows, labels)
    assert (c["rows"], c["prompt_harmful_share"], c["refusal_rate"]) == (4, 0.5, 0.5)
    assert c["refusal_rate_harmful_prompts"] == 0.5 and c["harmful_prompt_refusals"] == 1
    assert (c["benign_prompt_refusals"], c["harmful_responses"]) == (1, 1)
    assert c["unparsed"] == {"prompt_harmful": 0, "refusal": 0, "response_harmful": 1}
    assert c["unique_prompts_total"] == 3 and c["by_source"]["T"]["harmful_responses"] == 1
    recs = [{"id": "a", "meta": {"prompt_label": "unsafe", "response_policy": "refusal_x"}},
            {"id": "b", "meta": {"prompt_label": "safe", "response_policy": "compliance"}},
            {"id": "c", "meta": {"prompt_label": "unsafe", "response_policy": "refusal_x"}}]
    a = swapmix.agreement(recs, labels)
    assert a["prompt_label_x_prompt_harmful"] == {"safe": {"False": 1}, "unsafe": {"True": 2}}
    assert a["response_policy_x_refusal"] == {"compliance": {"False": 1},
                                              "refusal_x": {"False": 1, "True": 1}}


def test_pass_predicates():
    assert swapmix.harmful_refusal(HR) and not swapmix.harmful_refusal(HC)
    assert not swapmix.harmful_refusal({**HR, "response_harmful": None})
    assert swapmix.benign_compliance(BC) and not swapmix.benign_compliance(HR)


def test_dpo_overlap_exact_and_ngram():
    idx = decontam.PanelIndex([LONG, "tell me a joke"])
    rows = [{"chosen": [{"role": "user", "content": "Tell me a joke!"},
                        {"role": "assistant", "content": "x"}]},
            {"chosen": [{"role": "user", "content": " ".join(f"w{i}" for i in range(12))},
                        {"role": "assistant", "content": "x"}]},
            {"chosen": [{"role": "user", "content": "unrelated question entirely"},
                        {"role": "assistant", "content": "x"}]}]
    d = swapmix.dpo_overlap(idx, rows)
    assert d == {"dpo_rows": 3, "exact": 1, "ngram": 1, "either": 2}


def test_dpo_shares_sums_chunks():
    parts = [{"dpo_rows": 3, "exact": 1, "ngram": 1, "either": 2},
             {"dpo_rows": 1, "exact": 0, "ngram": 1, "either": 1}]
    d = swapmix.dpo_shares(iter(parts))
    assert (d["dpo_rows"], d["either"]) == (4, 3)
    assert (d["share_exact"], d["share_ngram"], d["share_either"]) == (0.25, 0.5, 0.75)
    try:
        swapmix.dpo_shares([])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_stop_rule():
    assert swapmix.stop_rule(0.220, 0.550) and not swapmix.stop_rule(0.3, 0.55)
    assert not swapmix.stop_rule(0.275, 0.55)  # exactly half does not stop
    try:
        swapmix.stop_rule(None, 0.5)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_unique_prompts_strip_consistently():
    rows = [{"source": "A", "messages": swapmix.pair("x", "1")},
            {"source": "A", "messages": swapmix.pair(" x ", "2")},
            {"source": "B", "messages": swapmix.pair("x\n", "3")},
            {"source": "B", "messages": swapmix.pair("y", "4")}]
    assert swapmix.unique_prompts(rows) == {"unique_prompts": {"A": 1, "B": 2},
                                            "unique_prompts_total": 2}


def test_lengths():
    rows = [{"messages": swapmix.pair("p", "a b c")}, {"messages": swapmix.pair("p", "a")},
            {"messages": swapmix.pair("p", "a b")}]
    assert swapmix.lengths(rows) == {"response_tokens_median": 2, "response_tokens_mean": 2.0}


def test_fold_earlier_empty_turn_fails_and_missing_label_raises():
    m = swapmix.parse_hh("\n\nHuman: a\n\nAssistant: \n\nHuman: c\n\nAssistant: d")
    rec = {"id": "hh/0", "source": "S", "messages": m}
    labels = {"hh/0": dict(HR), "hh/0#1": BC}
    assert swapmix.fold_earlier(labels, [rec])["S"]["turns_empty"] == 1
    assert labels["hh/0"]["earlier_clean"] is False
    with pytest.raises(ValueError, match="no WildGuard label"):
        swapmix.fold_earlier({"hh/0": dict(HR)}, [rec])


def _shard(out, stem, ids, judged_from=None, stale=False):
    shard = out / "judge_in" / f"{stem}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text("".join(json.dumps({"prompt_id": i}) + "\n" for i in ids))
    d = out / "labels" / stem
    d.mkdir(parents=True)
    (d / "wildguard.jsonl").write_text("".join(
        json.dumps({"prompt_id": i, **HR, **({"truncated_to_fit": True} if i == "t" else {})})
        + "\n" for i in ids))
    (d / "wildguard.meta.json").write_text(json.dumps({"generations": [str(judged_from or shard)]}))
    if stale:
        t = shard.stat().st_mtime
        os.utime(d / "wildguard.jsonl", (t - 10, t - 10))


def test_read_labels_keeps_truncation_and_needs_every_id(tmp_path):
    _shard(tmp_path, "pool_0", ["a", "t"])
    labels = bsm.read_labels(bsm.label_files(tmp_path, ["pool_0"]), {"a", "t"})
    assert labels["a"] == {**HR, "truncated": False} and labels["t"]["truncated"] is True
    with pytest.raises(ValueError, match="no WildGuard label"):
        bsm.read_labels(bsm.label_files(tmp_path, ["pool_0"]), {"a", "b"})


def test_label_files_rejects_stale_or_foreign_labels(tmp_path):
    _shard(tmp_path, "pool_0", ["a"], stale=True)
    with pytest.raises(ValueError, match="not judged from the current"):
        bsm.label_files(tmp_path, ["pool_0"])
    other = tmp_path / "x"
    _shard(other, "pool_0", ["a"], judged_from="/elsewhere/pool_0.jsonl")
    with pytest.raises(ValueError, match="not judged from the current"):
        bsm.label_files(other, ["pool_0"])
    extra = tmp_path / "y"
    _shard(extra, "pool_0", ["a"])
    (extra / "labels" / "pool_9").mkdir()
    with pytest.raises(ValueError, match="do not match judge_in shards"):
        bsm.label_files(extra, ["pool_0"])


def test_label_files_accepts_other_spellings_and_ignores_stray_files(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    _shard(tmp_path / "real", "pool_0", ["a"],
           judged_from=f"{tmp_path}/link//judge_in/./pool_0.jsonl")
    (tmp_path / "real" / "labels" / "job.log").write_text("x")
    assert bsm.label_files(tmp_path / "real", ["pool_0"]) == [
        tmp_path / "real" / "labels" / "pool_0" / "wildguard.jsonl"]


def test_write_shards_keeps_unchanged_files(tmp_path):
    (tmp_path / "judge_in").mkdir()
    rows = [{"prompt_id": str(i)} for i in range(3)]
    path = tmp_path / "judge_in" / "turns_0.jsonl"
    assert bsm.write_shards(tmp_path, "turns", rows) == {"turns_0": str(path)}
    os.utime(path, (1, 1))
    bsm.write_shards(tmp_path, "turns", rows)
    assert path.stat().st_mtime == 1
    bsm.write_shards(tmp_path, "turns", rows[:2])
    assert path.stat().st_mtime > 1 and len(path.read_text().splitlines()) == 2


def test_current_shards_needs_turns_of_the_current_pool(tmp_path):
    for name in ("pool", "slice"):
        (tmp_path / "pool").mkdir(exist_ok=True)
        (tmp_path / "pool" / f"{name}.jsonl").write_text(f"{name}\n")
    (tmp_path / "judge_in").mkdir()
    for stem in ("pool_0", "turns_0"):
        (tmp_path / "judge_in" / f"{stem}.jsonl").write_text("")
    inputs = {n: {"sha256": bsm.posttrain.sha256_file(tmp_path / "pool" / f"{n}.jsonl")}
              for n in ("pool", "slice")}
    report = {"pool": {"judge_shards": {"pool_0": ""}},
              "turns": {"judge_shards": {"turns_0": ""}, "inputs": inputs}}
    assert bsm.current_shards(tmp_path, report) == ["pool_0", "turns_0"]
    with pytest.raises(ValueError, match="run turns"):
        bsm.current_shards(tmp_path, {"pool": report["pool"]})
    (tmp_path / "judge_in" / "pool_1.jsonl").write_text("")
    with pytest.raises(ValueError, match="pool_1"):
        bsm.current_shards(tmp_path, report)
    (tmp_path / "pool" / "pool.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="changed after turns read it"):
        bsm.current_shards(tmp_path, report)
