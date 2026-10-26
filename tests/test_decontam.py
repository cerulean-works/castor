from castor import decontam


def _score(panel, docs):
    idx = decontam.PanelIndex(panel)
    hits = []
    for i, d in enumerate(docs):
        grams, _ = idx.hits(d)
        if grams:
            hits.append((f"d{i}", grams))
    return decontam.score(idx, hits)


def test_tokenize_matches_tulu_analyzer():
    # split on space , . ? ! : ; ( ) " - newline backslash; lowercase; drop empties
    assert decontam.tokenize('Hi, there! (A "b"-c)\nd\\e') == ["hi", "there", "a", "b", "c", "d", "e"]


def test_full_copy_scores_one():
    p = "one two three four five six seven eight nine ten"
    assert _score([p], ["prefix " + p + " suffix"])[0]["score"] == 1.0


def test_coverage_is_per_single_document():
    toks = [f"w{i}" for i in range(16)]
    p = " ".join(toks)
    first, second = " ".join(toks[:8]), " ".join(toks[8:])
    # each doc covers exactly half; the rule takes the best single document, not the union
    s = _score([p], [first, second])[0]
    assert s["score"] == 0.5
    assert s["score"] <= decontam.THRESHOLD  # 0.5 is not > 0.5
    s = _score([p], [" ".join(toks[:9])])[0]
    assert s["score"] == 9 / 16 and s["score"] > decontam.THRESHOLD


def test_short_prompt_needs_exact_match():
    idx = decontam.PanelIndex(["How do I kill a python process?"])
    grams, exact = idx.hits("how do I kill a Python process")
    assert not grams  # 7 tokens: no 8-gram exists
    assert exact == [0]


def test_no_overlap():
    assert _score(["a b c d e f g h i"], ["z y x w v u t s r"])[0]["score"] == 0.0


def test_any_whitespace_splits():
    # CRLF, tabs and NBSP must not glue tokens ('word\r' must match 'word')
    assert decontam.tokenize("a\tb\r\nc d") == ["a", "b", "c", "d"]
    p = "one two three four five six seven eight nine ten"
    crlf = p.replace(" ", "\r\n")
    assert _score([crlf], [p])[0]["score"] == 1.0
    assert decontam.PanelIndex([crlf]).hits(p)[1] == [0]


def test_tie_break_independent_of_doc_order():
    p = " ".join(f"w{i}" for i in range(10))
    idx = decontam.PanelIndex([p])
    g, _ = idx.hits(p)
    a = decontam.score(idx, [("b", g), ("a", g)])[0]
    b = decontam.score(idx, [("a", g), ("b", g)])[0]
    assert a == b and a["doc"] == "a"


def test_removal_reasons():
    per_set = {"sft": [{"score": 0.6, "doc": "s1"}, {"score": 0.5, "doc": "s2"},
                       {"score": 0.0, "doc": None}],
               "dpo": [{"score": 0.0, "doc": None}] * 3}
    exact = {"sft": {}, "dpo": {2: ["d9", "d3", "d1", "d7", "d5", "d2"]}}
    assert decontam.removal_reasons(per_set, exact, 0) == {
        "sft": {"ngram_score": 0.6, "doc": "s1"}}
    assert decontam.removal_reasons(per_set, exact, 1) == {}  # 0.5 is not > 0.5
    assert decontam.removal_reasons(per_set, exact, 2) == {
        "dpo": {"exact": ["d1", "d2", "d3", "d5", "d7"], "n_exact": 6}}
