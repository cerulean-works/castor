import json
from collections import Counter

import pytest

from castor import panel

pytestmark = pytest.mark.skipif(not (panel.RAW / "HarmBench").exists(),
                                reason="raw panel sources not downloaded")


def test_humanjb_reproduces_harmbench_default():
    cases = panel.harmbench_humanjb_cases()
    assert len(cases) == 400 and all(len(v) == 5 for v in cases.values())
    # deterministic, and does not disturb the global RNG
    import random
    random.seed(123)
    a = random.random()
    random.seed(123)
    again = panel.harmbench_humanjb_cases()
    assert random.random() == a
    assert again == cases
    # 5 distinct template indices per behavior
    assert all(len({t for t, _ in v}) == 5 for v in cases.values())


def test_humanjb_matches_harmbench_code():
    """Run HarmBench's own HumanJailbreaks.generate_test_cases logic on a string copy."""
    import random
    import sys
    sys.path.insert(0, str(panel.RAW / "HarmBench/baselines/human_jailbreaks"))
    from jailbreaks import JAILBREAKS
    sys.path.pop(0)
    templates = list(JAILBREAKS)
    random.seed(1)
    ref = {}
    for r in panel._harmbench_rows(panel.RAW):
        random.shuffle(templates)
        b = panel._harmbench_behavior(r)
        # HarmBench saves test cases with json.dump and reloads them with json.load
        ref[r["BehaviorID"]] = json.loads(json.dumps([f"{jb}\n\n{b}" for jb in templates[:5]]))
    ours = {k: [t for _, t in v] for k, v in panel.harmbench_humanjb_cases().items()}
    assert ours == ref


def test_source_sizes_match_papers():
    n = Counter(r.source for r in panel.load_all())
    assert n == {"harmbench": 300, "harmbench_humanjb": 1500, "sorrybench": 440,
                 "strongreject": 313, "xstest": 450, "orbench": 1319, "wildjailbreak": 2210}


def test_labels():
    recs = panel.load_all()
    xs = Counter((r.source_split, r.is_harmful) for r in recs if r.source == "xstest")
    assert xs == {("safe", False): 250, ("unsafe", True): 200}
    wj = Counter(r.is_harmful for r in recs if r.source == "wildjailbreak")
    assert wj == {True: 2000, False: 210}
    assert all(not r.is_harmful for r in recs if r.source == "orbench")
    assert all(r.is_adversarial == (r.source in {"harmbench_humanjb", "wildjailbreak"})
               for r in recs)


def test_dedupe_keeps_first():
    a = panel.Record("x", "", "", True, False, "Same  prompt", "", "", "1")
    b = panel.Record("y", "", "", True, False, "same prompt", "", "", "2")
    kept, dropped = panel.dedupe([a, b])
    assert kept == [a] and dropped[0]["kept_source"] == "x"


def test_all_prompts_valid_utf8():
    for r in panel.load_all():
        r.prompt.encode("utf-8")


def _jb(i, tidx):
    return panel.Record("harmbench_humanjb", "", "", True, True, f"p{i}", "", "", f"b{i}::jb{tidx}")


def test_template_level_removals():
    templates = ["T0", "T1", "T2", "T1"]  # 1 and 3 are the same text, as in HarmBench
    recs = [_jb(0, 1), _jb(1, 1), _jb(2, 3), _jb(3, 2),
            panel.Record("xstest", "", "", False, False, "q", "", "", "7")]
    out = panel.template_level_removals(recs, {0, 4}, templates)
    assert set(out) == {1, 2}  # same template (by text), not already removed; xstest ignored
    assert out[2] == {"template_idx": 3, "contaminated_instances": ["b0::jb1"]}


def test_source_counts():
    keep = [panel.Record("xstest", "", "", True, False, "a", "", "", "1"),
            panel.Record("xstest", "", "", False, False, "b", "", "", "2"),
            panel.Record("harmbench_humanjb", "", "", True, True, "c", "", "", "x::jb0")]
    removed = [
        {"source": "xstest", "matches": {"sft": {"ngram_score": 0.9}, "dpo": {"exact": ["d"]}}},
        {"source": "xstest", "matches": {"dpo": {"exact": ["d"]}}},
        {"source": "harmbench_humanjb", "matches": {}, "template": {"template_idx": 0}},
    ]
    c = panel.source_counts({"xstest": 5, "harmbench_humanjb": 2},
                            [{"source": "xstest"}], keep, removed, ["sft", "dpo"])
    assert c["xstest"] == {"raw": 5, "dup_dropped": 1, "decontam_removed": 2,
                           "removed_by_set": {"sft": 1, "dpo": 2}, "removed_exact_only": 1,
                           "template_removed": 0, "final": 2, "harmful": 1, "adversarial": 0}
    hj = c["harmbench_humanjb"]
    assert (hj["decontam_removed"], hj["template_removed"], hj["final"]) == (0, 1, 1)
    assert c["orbench"]["raw"] == 0
