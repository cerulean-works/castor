import math

import numpy as np
import pytest

from castor import reliability, sanityb


def _row(pid, s, v, agree=True, coherent=True, label="NON-HARMFUL-COMPLIANCE", invalid=False):
    return {"prompt_id": pid, "sample": s, "wg_vulnerable": v, "refusal_agree": agree,
            "coherent": coherent, "wg_label": label, "llm_invalid": invalid}


def test_cells_and_naming():
    cs = sanityb.cells()
    assert len(cs) == 12 and len({c.model_id for c in cs}) == 12
    c = sanityb.Cell("large-aux-2", 3)
    assert c.base_ref == "allenai/DataDecide-dclm-baseline-1B@step69369-seed-large-aux-2"
    assert c.out("dpo").parts[-3:] == ("dd-dclm-1B-seed-large-aux-2", "dpo", "seed3")
    assert c.info.base == ("dclm-baseline", "large-aux-2") and c.info.posttrain_seed == "3"
    # 18 MZ pairs and 48 U pairs among the 12 twins
    lv = [reliability.pair_level(a.info, b.info) for i, a in enumerate(cs) for b in cs[i + 1:]]
    assert lv.count("MZ") == 18 and lv.count("U") == 48 and lv.count("R") == 0


def test_subsets():
    panel = [{"id": "a", "is_harmful": True, "is_adversarial": True},
             {"id": "b", "is_harmful": True, "is_adversarial": False},
             {"id": "c", "is_harmful": False, "is_adversarial": True}]
    assert sanityb.subset_ids(panel, "adv_harmful") == ["a"]
    assert sanityb.subset_ids(panel, "plain_harmful") == ["b"]
    assert sanityb.subset_ids(panel, "harmful") == ["a", "b"]
    assert sanityb.subset_ids(panel, "benign") == ["c"]


def test_outcome_matrix():
    rows = [_row("p1", -1, 1), _row("p1", 0, 1), _row("p1", 1, None), _row("p2", 0, 0),
            _row("p2", 1, 1, agree=False), _row("other", 0, 1)]
    y, c = sanityb.outcome_matrix(rows, ["p1", "p2"], "wg_vulnerable", k=2)
    assert np.array_equal(np.nan_to_num(y, nan=-1), [[1, -1], [0, 1]])  # greedy ignored
    assert c == {"cells": 4, "missing": 0, "nan": 1}
    y, _ = sanityb.outcome_matrix(rows, ["p1", "p2"], "wg_vulnerable", k=2, agree_only=True)
    assert np.isnan(y[1, 1]) and y[1, 0] == 0
    y, c = sanityb.outcome_matrix(rows[:2], ["p1", "p2"], "wg_vulnerable", k=2)
    assert c["missing"] == 3
    with pytest.raises(AssertionError):
        sanityb.outcome_matrix([_row("p1", 0, 1)] * 2, ["p1"], "wg_vulnerable", k=2)


def test_model_stats_mean_is_prompt_mean():
    y = np.array([[1, 1, 1, 1], [0, np.nan, np.nan, np.nan]], dtype=float)
    s = sanityb.model_stats(y)
    assert s["mean_vulnerability"] == pytest.approx(0.5)  # prompt mean, not pooled
    assert s["pooled_rate"] == pytest.approx(4 / 5)
    assert s["n_prompts_defined"] == 2


def test_gate():
    ok = {"mean_vulnerability": 0.5, "reliability": 0.8}
    names = ["a", "b"]
    assert sanityb.gate({"a": ok, "b": ok}, names)["verdict"] == "GO"
    assert sanityb.gate({"a": ok, "b": ok | {"mean_vulnerability": 0.95}},
                        names)["verdict"] == "KILL-AT-THIS-SCALE"
    assert sanityb.gate({"a": ok, "b": ok | {"reliability": 0.69}},
                        names)["verdict"] == "KILL-AT-THIS-SCALE"
    assert sanityb.gate({"a": ok, "b": ok | {"mean_vulnerability": 0.1,
                                             "reliability": 0.7}}, names)["verdict"] == "GO"
    g = sanityb.gate({"a": ok}, names)
    assert g["verdict"] == "INCOMPLETE" and g["missing"] == ["b"]
    nan = ok | {"reliability": math.nan}
    assert sanityb.gate({"a": ok, "b": nan}, names)["verdict"] == "KILL-AT-THIS-SCALE"


def test_d035_line():
    tw = {"reliability": {"a": 0.95, "b": 0.93, "c": 0.9},
          "pairs": [{"a": "a", "b": "b", "level": "MZ", "r": 0.921},
                    {"a": "a", "b": "c", "level": "U", "r": 0.5}]}
    d = sanityb.d035_line(tw, n_expected=1)
    assert d["pairs"][0]["mean_ceiling"] == pytest.approx(0.94)
    assert d["pairs"][0]["holds"] and d["fires"]  # 0.921 >= 0.94 - 0.02
    tw["pairs"][0]["r"] = 0.919
    assert not sanityb.d035_line(tw, n_expected=1)["fires"]
    tw["pairs"][0]["r"] = 0.99
    d = sanityb.d035_line(tw, n_expected=18)  # incomplete never fires
    assert not d["fires"] and not d["complete"]


def test_d035_on_simulated_twins():
    """Twins that share one true per-prompt rate: every MZ pair sits near its ceiling."""
    rng = np.random.default_rng(0)
    true = rng.beta(0.5, 0.5, 2000)
    ys = {c.model_id: (rng.random((2000, 20)) < true[:, None]).astype(float)
          for c in sanityb.cells()}
    info = {c.model_id: c.info for c in sanityb.cells()}
    d = sanityb.d035_line(reliability.twin_correlations(ys, info))
    assert d["n_pairs"] == 18 and d["fires"]


def test_response_counts():
    rows = [_row("p", -1, 1, coherent=False), _row("p", 0, 0, label="REFUSAL", invalid=True),
            _row("p", 1, 0, label="REFUSAL"), _row("p", 2, None, coherent=False,
                                                   label="INCOHERENT-OR-OFF-TOPIC"),
            _row("p", 3, None, label="JUDGE-ERROR"), _row("q", 0, 1)]
    c = sanityb.response_counts(rows, {"p"})
    assert c["n"] == 4 and c["incoherent_rate"] == pytest.approx(0.25)
    assert c["judge_error_rate"] == pytest.approx(0.25)
    assert c["refusal_llm_invalid_share"] == pytest.approx(0.5)


def test_gate_incomplete_labels():
    ok = {"mean_vulnerability": 0.5, "reliability": 0.8}
    g = sanityb.gate({"a": ok, "b": ok}, ["a", "b"], incomplete=["b"])
    assert g["verdict"] == "INCOMPLETE" and g["missing"] == ["b"]


def test_d035_reports_geometric_gap():
    tw = {"reliability": {"a": 0.95, "b": 0.75},
          "pairs": [{"a": "a", "b": "b", "level": "MZ", "r": 0.84}]}
    row = sanityb.d035_line(tw, n_expected=1)["pairs"][0]
    assert row["gap"] == pytest.approx(-0.01) and row["holds"]  # decided by the frozen mean
    assert row["gap_geometric"] == pytest.approx(0.84 - math.sqrt(0.95 * 0.75))
