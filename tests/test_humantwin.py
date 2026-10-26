import math

import numpy as np
import pytest

from castor import humantwin as H
from castor.judge import labels as L

M = H.MODELS  # default/s1, default/s2, large-aux-2/s1, large-aux-2/s2


def _ids(n=120):
    return {"harmbench_humanjb": [f"hb-{i:04d}" for i in range(n)],
            "wildjailbreak": [f"wj-{i:04d}" for i in range(n)]}


def test_draw_design():
    prompts, items = H.draw(_ids())
    assert len(prompts) == 100 and len(set(prompts)) == 100
    assert sum(p.startswith("hb-") for p in prompts) == 50
    assert len(items) == 1200 and len(set(items)) == 1200
    cells = {}
    for m, p, s in items:
        assert 0 <= s < 20 and m in M and p in prompts
        cells.setdefault((m, p), []).append(s)
    assert len(cells) == 400 and all(len(set(v)) == 3 for v in cells.values())


def test_draw_reproducible_and_reads_ids_only():
    a = H.draw(_ids())
    assert a == H.draw(_ids())
    shuffled = {k: list(reversed(v)) for k, v in _ids().items()}  # input order is irrelevant
    assert a == H.draw(shuffled)
    assert a != H.draw(_ids(), seed=H.SEED + 1)


def test_human_value():
    assert H.human_value("R") == 0 and H.human_value("H") == 1 and H.human_value("N") == 1
    assert math.isnan(H.human_value("I"))
    assert H.human_value("I", include_incoherent=True) == 1
    assert math.isnan(H.judge_value(None)) and H.judge_value(0) == 0.0


def _values(per_model_p: dict[str, np.ndarray], k=3):
    """values with every item of (model, prompt) equal to that cell's rate (0/1 here)."""
    prompts = [f"p{i:03d}" for i in range(len(next(iter(per_model_p.values()))))]
    vals = {(m, p, s): float(per_model_p[m][i]) for m in M for i, p in enumerate(prompts)
            for s in range(k)}
    return vals, prompts


def test_matrices_shape_and_order():
    vals, prompts = _values({m: np.arange(10) % 2 for m in M})
    vals[(M[0], prompts[3], 1)] = math.nan
    ys = H.matrices(vals, prompts)
    assert ys[M[0]].shape == (10, 3) and math.isnan(ys[M[0]][3, 1])
    del vals[(M[1], prompts[0], 2)]
    with pytest.raises(AssertionError):
        H.matrices(vals, prompts)


def test_gap_known_answer():
    # base A shares x, base B shares y; x ⟂ y exactly (centred, orthogonal) -> r_MZ 1, r_U 0
    x = np.array([1, 1, 0, 0] * 10, dtype=float)
    y = np.array([1, 0, 1, 0] * 10, dtype=float)
    vals, prompts = _values({M[0]: x, M[1]: x, M[2]: y, M[3]: y})
    s = H.summary(H.analyze(H.combine({"human": H.matrices(vals, prompts)})))
    assert s["r_MZ_human"] == pytest.approx(1.0)
    assert s["r_U_human"] == pytest.approx(0.0, abs=1e-12)
    assert s["gap_human"] == pytest.approx(1.0)
    # second known value: base B = x with 10 of 40 prompts flipped -> r_U = corr(x, y2)
    y2 = x.copy()
    y2[:10] = 1 - y2[:10]
    vals, prompts = _values({M[0]: x, M[1]: x, M[2]: y2, M[3]: y2})
    s = H.summary(H.analyze(H.combine({"human": H.matrices(vals, prompts)})))
    assert s["r_U_human"] == pytest.approx(np.corrcoef(x, y2)[0, 1])
    assert s["gap_human"] == pytest.approx(1 - np.corrcoef(x, y2)[0, 1])


def _noisy_labels(seed, n=100, k=3):
    rng = np.random.default_rng(seed)
    a, b = rng.random(n), rng.random(n)
    p = {M[0]: a, M[1]: a, M[2]: b, M[3]: 0.7 * b + 0.3 * a}
    prompts = [f"p{i:03d}" for i in range(n)]
    return {(m, pr, s): float(rng.random() < p[m][i]) for m in M for i, pr in enumerate(prompts)
            for s in range(k)}, prompts


def test_paired_bootstrap_identical_labellers_gives_zero_difference():
    vals, prompts = _noisy_labels(1)
    y = H.matrices(vals, prompts)
    ci = H.contrast(H.combine({"human": y, "judge": {m: v.copy() for m, v in y.items()}}),
                    n_boot=200)
    d = ci["gap_judge_minus_human"]
    # paired: the same prompt rows in every draw, so the difference is exactly 0 throughout;
    # an unpaired bootstrap would spread it by about the gap's own SE
    assert d["estimate"] == 0 and d["lo"] == 0 and d["hi"] == 0
    assert ci["gap_human"]["hi"] - ci["gap_human"]["lo"] > 0.05


def test_paired_bootstrap_detects_base_specific_judge_error():
    # the judge calls every item of base `default` non-refusal on the first 40 prompts and
    # every item of the other base refusal there: identical within a base, so it inflates
    # r_MZ relative to r_U; the paired difference must sit above 0
    vals, prompts = _noisy_labels(2)
    judge = dict(vals)
    for (m, p, s) in vals:
        if int(p[1:]) < 40:
            judge[(m, p, s)] = 1.0 if m in M[:2] else 0.0
    ys = H.combine({"human": H.matrices(vals, prompts), "judge": H.matrices(judge, prompts)})
    ci = H.contrast(ys, n_boot=300)
    assert ci["gap_judge_minus_human"]["lo"] > 0
    assert ci["gap_judge_minus_human"]["estimate"] == pytest.approx(
        ci["gap_judge"]["estimate"] - ci["gap_human"]["estimate"])
    assert H.reading(ci) == "judge inflates the gap"


def test_reading_rule():
    def ci(h, d):
        return {"gap_human": dict(zip(("lo", "hi"), h)),
                "gap_judge_minus_human": dict(zip(("lo", "hi"), d))}
    assert H.reading(ci((0.05, 0.3), (-0.1, 0.1))) == "not a judge artifact"
    assert H.reading(ci((-0.05, 0.3), (-0.1, 0.1))) == "inconclusive at this size"
    assert H.reading(ci((0.05, 0.3), (0.01, 0.2))) == "judge inflates the gap"
    assert H.reading(ci((0.05, 0.3), (-0.3, -0.01))) == "other"


def test_twin_levels_and_pairs():
    vals, prompts = _noisy_labels(3)
    s = H.summary(H.analyze(H.combine({"judge": H.matrices(vals, prompts)})))
    pair_keys = [k for k in s if k.startswith("r_judge|")]
    assert len(pair_keys) == 6  # 2 MZ + 4 U
    mz = [s[f"r_judge|{M[0]}|{M[1]}"], s[f"r_judge|{M[2]}|{M[3]}"]]
    assert s["r_MZ_judge"] == pytest.approx(np.mean(mz))


def test_relgap_invariant_to_common_attenuation():
    def res(r_mz, r_u):
        return {"human": {"levels": {"MZ": {"r": r_mz, "n_pairs": 2}, "U": {"r": r_u, "n_pairs": 4}},
                          "pairs": []}}
    a, b = H.summary(res(0.8, 0.6)), H.summary(res(0.4, 0.3))  # same rs, attenuated by 0.5
    assert a["gap_human"] == pytest.approx(2 * b["gap_human"])
    assert a["relgap_human"] == pytest.approx(b["relgap_human"]) == pytest.approx(0.25)
    assert H.summary(res(0.9, 0.6))["relgap_human"] > a["relgap_human"]  # r_MZ alone moves it


def test_input_errors_messages():
    assert H.input_errors(1200, 1200, {}) == []
    (e,) = H.input_errors(1200, 1198, {})  # text mismatch alone: a message, never `{}`
    assert "2 of 1200" in e and "generation text" in e
    chk = {"ok": False, "n_rows": 1200, "n_unique_ids": 1200, "ids_match_key": True,
           "missing_from_sheet": [], "not_in_key": [], "unlabelled_or_invalid": ["a", "b"],
           "partial_invalid": [], "text_changed": ["c"]}
    assert H.input_errors(1200, 1200, {"a.csv": chk}) == [
        "a.csv: sheet check failed (2 unlabelled_or_invalid, 1 text_changed)"]


def test_gold_errors_messages():
    key = {"a", "b", "c"}
    assert H.gold_errors(0, {"a", "b", "c"}, key) == []
    assert H.gold_errors(1, {"a", "b"}, key) == ["1 label disagreements not adjudicated"]
    (e,) = H.gold_errors(0, {"a", "b"}, key)  # length mismatch with N = 0: not "0 ... adjudicated"
    assert "not adjudicated" not in e and e.startswith("1 key.jsonl items have no gold")
    assert H.gold_errors(0, {"a", "b", "c", "z"}, key) == [
        "1 gold labels have item ids not in key.jsonl"]


# ---- descriptive additions: per-annotator gold and the adjudication side ----------------------


def _items(per_model_value: dict[str, np.ndarray], k=3) -> tuple[list[dict], list[str]]:
    """key.jsonl-like items whose gold is the adjudicated label; the judge column is constant."""
    vals, prompts = _values(per_model_value, k)
    items = [{"item_id": f"{m}|{p}|{s}", "model": m, "prompt_id": p, "sample": s,
              "gold": L.NONHARMFUL if v else L.REFUSAL, "gold_partial": False,
              "wg_vulnerable": 0, "wg_vulnerable_incl": 0} for (m, p, s), v in vals.items()]
    return items, prompts


def _sheet(per_model_value: dict[str, np.ndarray], k=3) -> dict[str, dict]:
    vals, _ = _values(per_model_value, k)
    return {f"{m}|{p}|{s}": {"label": L.NONHARMFUL if v else L.REFUSAL, "partial": 0}
            for (m, p, s), v in vals.items()}


def test_with_labels_swaps_gold_only_and_leaves_items_alone():
    items, _ = _items({m: np.zeros(4) for m in M})
    sheet = {x["item_id"]: {"label": L.INCOHERENT, "partial": 1} for x in items}
    out = H.with_labels(items, sheet)
    assert all(x["gold"] == L.INCOHERENT and x["gold_partial"] is True for x in out)
    assert all(x["gold"] == L.REFUSAL and x["gold_partial"] is False for x in items)
    assert [x["item_id"] for x in out] == [x["item_id"] for x in items]
    with pytest.raises(KeyError):  # every item must be on the annotator's sheet
        H.with_labels(items, dict(list(sheet.items())[:-1]))


def test_per_annotator_gap_known_answer():
    # adjudicated gold: base A shares x, base B shares y ⟂ x -> gap 1; annotator B's own labels
    # give base B = x with 10 prompts flipped -> gap 1 − corr(x, y2); annotator A = adjudicated
    x = np.array([1, 1, 0, 0] * 10, dtype=float)
    y = np.array([1, 0, 1, 0] * 10, dtype=float)
    y2 = x.copy()
    y2[:10] = 1 - y2[:10]
    items, prompts = _items({M[0]: x, M[1]: x, M[2]: y, M[3]: y})
    sheet_a = _sheet({M[0]: x, M[1]: x, M[2]: y, M[3]: y})
    sheet_b = _sheet({M[0]: x, M[1]: x, M[2]: y2, M[3]: y2})

    def gap(its):
        v = H.values(its)
        return H.summary(H.analyze(H.combine({"human": H.matrices(v["human"], prompts)})))

    assert gap(items)["gap_human"] == pytest.approx(1.0)
    assert gap(H.with_labels(items, sheet_a))["gap_human"] == pytest.approx(1.0)
    assert gap(H.with_labels(items, sheet_b))["gap_human"] == pytest.approx(
        1 - np.corrcoef(x, y2)[0, 1])
    assert gap(items)["gap_human"] == pytest.approx(1.0)  # the adjudicated items are untouched
    # values(): the judge column comes from the item's wg field and I is NaN unless included
    items[0]["gold"] = L.INCOHERENT
    v = H.values(items)
    assert math.isnan(v["human"][(M[0], prompts[0], 0)]) and v["judge"][(M[0], prompts[0], 0)] == 0
    assert H.values(items, include_incoherent=True)["human"][(M[0], prompts[0], 0)] == 1.0


def test_adjudication_summary_counts():
    rows = [
        {"item_id": "1", "harry": "R", "coauthor": "N", "class": "N<->R", "label": "R", "partial": ""},
        {"item_id": "2", "harry": "R", "coauthor": "N", "class": "N<->R", "label": "n", "partial": "1"},
        {"item_id": "3", "harry": "H", "coauthor": "N", "class": "H<->N", "label": "I", "partial": ""},
        {"item_id": "4", "harry": "R", "coauthor": "R", "class": "partial-only", "label": "R",
         "partial": "1"},
    ]
    s = H.adjudication_summary(rows, ("harry", "coauthor"))
    assert s["final_matches"] == {"harry": 1, "coauthor": 1, "neither": 1}  # partial-only excluded
    assert s["n_rows"] == 4 and s["n_label_disagreements"] == 3 and s["n_partial_only"] == 1
    assert s["final_label_counts"] == {"R": 2, "N": 1, "I": 1} and s["final_partial_flags"] == 2
    assert s["class_counts"] == {"N<->R": 2, "H<->N": 1, "partial-only": 1}
    rows[0]["label"] = ""
    with pytest.raises(AssertionError, match="1 rows with an empty or invalid"):
        H.adjudication_summary(rows, ("harry", "coauthor"))
