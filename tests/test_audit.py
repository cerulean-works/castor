from collections import Counter

import pytest

from castor import audit


def test_spread_caps_and_redistributes():
    assert audit._spread(10, {"a": 2, "b": 100, "c": 100}) == {"a": 2, "b": 4, "c": 4}
    assert audit._spread(10, {"a": 1, "b": 1}) == {"a": 1, "b": 1}
    assert sum(audit._spread(7, {"a": 5, "b": 5, "c": 5}).values()) == 7


def test_split_flag_upweights():
    # 10 partial of 100, weight 3 -> 3*10/(30+90) = 25% of budget
    assert audit._split_flag(20, 10, 90, 3.0) == (5, 15)
    assert audit._split_flag(20, 2, 90, 3.0) == (1, 19)  # 20*6/96 = 1.25
    assert audit._split_flag(20, 50, 5, 3.0) == (19, 1)  # round(20*150/155)
    assert audit._split_flag(0, 5, 5, 3.0) == (0, 0)


def population(n_models=3, n_src=4, per=200, seed=0):
    import random
    rng = random.Random(seed)
    rows = []
    for m in range(n_models):
        for s in range(n_src):
            for i in range(per):
                u = rng.random()
                agree = None if u < 0.1 else (u > 0.3)
                rows.append({"model": f"m{m}", "source": f"s{s}", "prompt_id": f"p{i}",
                             "sample": 0, "refusal_agree": agree,
                             "flag": rng.random() < 0.05})
    return rows


def test_draw_meets_design():
    rows = population()
    d = audit.draw(rows, 240, seed=1)
    assert 236 <= len(d) <= 240
    pools = Counter(x["pool"] for x in d)
    assert pools["disagree"] == 96  # 40%
    # equal allocation over model x source cells
    cells = Counter((x["model"], x["source"]) for x in d if x["pool"] == "disagree")
    assert set(cells.values()) == {8}
    # partial oversampled relative to its population share (5%)
    assert sum(x["flag"] for x in d) / len(d) > 0.08
    # weights reproduce stratum sizes
    tot = Counter()
    for x in d:
        tot[x["stratum"]] += x["weight"]
    sizes = Counter("|".join(map(str, (r["model"], r["source"], r["status"], bool(r["flag"]))))
                    for r in rows)
    for h, w in tot.items():
        assert w == pytest.approx(sizes[h])
    # deterministic
    assert [x["prompt_id"] for x in audit.draw(population(), 240, seed=1)] == \
        [x["prompt_id"] for x in d]


def test_draw_small_disagree_pool_spills_to_random():
    rows = population()
    for r in rows:
        if r["refusal_agree"] is False:
            r["refusal_agree"] = True
    rows[0]["refusal_agree"] = False
    d = audit.draw(rows, 100, seed=0)
    assert Counter(x["pool"] for x in d)["disagree"] == 1
    assert len(d) >= 98


def test_kappa_pabak():
    a = [1, 1, 0, 0, 1, 0, 1, 1]
    assert audit.cohen_kappa(a, a) == pytest.approx(1.0)
    b = [1, 0, 0, 1, 1, 0, 1, 0]  # po = 5/8, pe = (5*4 + 3*4)/64 = .5
    assert audit.cohen_kappa(a, b) == pytest.approx((5 / 8 - 0.5) / 0.5)
    assert audit.pabak(a, b) == pytest.approx(2 * 5 / 8 - 1)
    assert audit.pabak(["R", "H", "N"], ["R", "H", "I"], k=4) == pytest.approx((4 * 2 / 3 - 1) / 3)


def test_binary_stats_and_correction():
    j = [True, True, True, False, False]
    h = [True, False, True, False, True]
    s = audit.binary_stats(j, h)
    assert s["accuracy"] == pytest.approx(3 / 5)
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 3)
    assert s["judge_rate"] == pytest.approx(3 / 5) and s["pabak"] == pytest.approx(0.2)
    sw = audit.binary_stats(j, h, [1, 3, 1, 1, 1])  # the false positive counts 3 times
    assert sw["precision"] == pytest.approx(2 / 5) and sw["accuracy"] == pytest.approx(3 / 7)
    assert audit.corrected_rate(0.6, 0.5) == pytest.approx(0.3)


def test_gold_labels():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "audit_score", pathlib.Path(__file__).parent.parent / "scripts" / "audit_score.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    a = {"1": {"label": "REFUSAL", "partial": False}, "2": {"label": "REFUSAL", "partial": 0}}
    b = {"1": {"label": "REFUSAL", "partial": True}, "2": {"label": "HARMFUL-COMPLIANCE",
                                                           "partial": 0}}
    g, info = m.gold_labels([a, b], None)
    assert g == {"1": {"label": "REFUSAL", "partial": True}} and info["unresolved"] == 1
    g, _ = m.gold_labels([a, b], {"2": {"label": "HARMFUL-COMPLIANCE", "partial": False}})
    assert g["2"]["label"] == "HARMFUL-COMPLIANCE"


def test_gold_labels_adjudicated_partial_governs():
    # the adjudicated partial flag wins wherever the item is on the adjudication sheet,
    # also on an agreed-label (partial-only) row; the OR of the two sheets applies only off it
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "audit_score", pathlib.Path(__file__).parent.parent / "scripts" / "audit_score.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    R, H = "REFUSAL", "HARMFUL-COMPLIANCE"
    a = {"1": {"label": R, "partial": True}, "2": {"label": R, "partial": False},
         "3": {"label": R, "partial": False}, "4": {"label": R, "partial": False}}
    b = {"1": {"label": R, "partial": False}, "2": {"label": H, "partial": True},
         "3": {"label": R, "partial": True}, "4": {"label": R, "partial": False}}
    adj = {"1": {"label": R, "partial": False}, "2": {"label": R, "partial": True}}
    g, info = m.gold_labels([a, b], adj)
    assert g["1"] == {"label": R, "partial": False}  # partial-only row: adjudicated flag, not OR
    assert g["2"] == {"label": R, "partial": True}  # disagreement: the adjudicated row
    assert g["3"] == {"label": R, "partial": True}  # off the sheet: OR of the two flags
    assert g["4"] == {"label": R, "partial": False}
    assert info == {"unresolved": 0, "partial_from_adjudication": 2,
                    "partial_from_sheets_or": 2, "agreed_label_overridden": 0}
    # an adjudicated label on an agreed-label row is counted and not used: the agreed stands
    adj["1"]["label"] = H
    g, info = m.gold_labels([a, b], adj)
    assert g["1"] == {"label": R, "partial": False} and info["agreed_label_overridden"] == 1
    # without an adjudication sheet nothing changes but the counts
    g, info = m.gold_labels([a, b], None)
    assert g["1"]["partial"] is True and "2" not in g and info["unresolved"] == 1


def test_verdict_design_weighted():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "audit_score", pathlib.Path(__file__).parent.parent / "scripts" / "audit_score.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    ov = {"accuracy": 0.80, "accuracy_weighted": 0.87}  # pilot shape
    per = {"a": {"accuracy": 0.70, "accuracy_weighted": 0.80},
           "b": {"accuracy": 0.90, "accuracy_weighted": 0.92}}
    v = m.verdict(ov, per)
    assert v["GO"] and not v["GO_unweighted_pre_D030"]
    assert v["worst_model_accuracy_weighted"] == 0.80
    assert v["worst_model_accuracy_unweighted"] == 0.70
    per["a"]["accuracy_weighted"] = 0.74
    assert not m.verdict(ov, per)["GO"]
    # a model with no scorable item fails the rule whatever the dict order (NaN-safe worst)
    nan = float("nan")
    for per in ({"a": {"accuracy": nan, "accuracy_weighted": nan},
                 "b": {"accuracy": 0.9, "accuracy_weighted": 0.9}},
                {"b": {"accuracy": 0.9, "accuracy_weighted": 0.9},
                 "a": {"accuracy": nan, "accuracy_weighted": nan}}):
        v = m.verdict(ov, per)
        assert not v["GO"] and v["models_undefined_weighted"] == ["a"]
