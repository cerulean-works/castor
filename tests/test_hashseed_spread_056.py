"""056: the spread table's arithmetic (flatten, equality, per-key summary) and runs/056/spread.json
against its parts and the `_ordered` artifacts."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from castor import provenance

ROOT = Path(__file__).resolve().parent.parent
SPREAD = ROOT / "runs" / "056" / "spread.json"
H = provenance.load_script("hashseed_spread_056")


def test_flatten_reads_point_or_estimate():
    d = {"a": {"x": {"point": 1.0, "lo": 0.5, "hi": 1.5, "frac_undefined": 0.0}},
         "b": {"ci": {"gap": {"estimate": 2.0, "lo": 1.0, "hi": 3.0, "n_draws": 10}}, "n": 3}}
    assert H.flatten(d) == {"a/x": {"point": 1.0, "lo": 0.5, "hi": 1.5},
                            "b/ci/gap": {"point": 2.0, "lo": 1.0, "hi": 3.0}}


def test_eq_is_exact_with_nan_equal():
    assert H.eq(float("nan"), float("nan"))
    assert not H.eq(0.1 + 0.2, 0.3)
    assert H.eq(None, None) and not H.eq(None, 1)


def test_summarise_ranges_and_distances():
    banked = {"k": {"point": 0.5, "lo": 0.40, "hi": 0.70}, "p": {"point": 0.3}}
    runs = {0: {"k": {"point": 0.5, "lo": 0.38, "hi": 0.71}, "p": {"point": 0.31}},
            1: {"k": {"point": 0.5, "lo": 0.43, "hi": 0.69}, "p": {"point": 0.29}}}
    ordered = {"k": {"point": 0.5, "lo": 0.37, "hi": 0.70}, "p": {"point": 0.30}}
    r = H.summarise(banked, runs, ordered)
    k = r["k"]
    assert k["points_equal_banked"] and k["sorted_point_equal_banked"]
    assert k["lo"]["min"] == 0.38 and k["lo"]["max"] == 0.43
    assert math.isclose(k["lo"]["width"], 0.05) and math.isclose(k["lo"]["max_abs_from_banked"], 0.03)
    assert math.isclose(k["hi"]["max_abs_from_banked"], 0.01)
    assert k["lo"]["banked_within_range"] and not k["lo"]["sorted_within_range"]
    assert math.isclose(k["lo"]["sorted_minus_banked"], -0.03)
    assert r["p"]["values"] == {"min": 0.29, "max": 0.31, "n_distinct": 2}
    assert not r["p"]["points_equal_banked"]
    assert k["lo"]["n_nan"] == 0 and k["hi"]["n_nan"] == 0
    # the headline flag reads intervals only: the moving scalar p never enters it
    agg = H.aggregate(r)
    assert agg["all_interval_points_equal_banked"] and agg["n_intervals"] == 1
    assert agg["interval_points_not_equal"] == [] and k["point_max_abs_from_banked"] == 0.0
    runs[1]["k"]["point"] = 0.5 + 1e-16
    agg = H.aggregate(H.summarise(banked, runs, ordered))
    assert not agg["all_interval_points_equal_banked"]
    (d,) = agg["interval_points_not_equal"]
    assert d["key"] == "k" and 0 < d["max_abs_diff"] < 1e-15  # one ulp of 0.5
    assert agg["n_keys"] == 2 and agg["n_nan_endpoints"] == 0
    assert math.isclose(agg["max_abs_endpoint_from_banked"], 0.03)
    assert math.isclose(agg["max_endpoint_range_width"], 0.05)


def test_nan_endpoint_is_counted_not_dropped():
    banked = {"k": {"point": 0.5, "lo": 0.40, "hi": 0.70},
              "m": {"point": 0.5, "lo": 0.40, "hi": 0.70}}
    runs = {0: {"k": {"point": 0.5, "lo": float("nan"), "hi": 0.71},
                "m": {"point": 0.5, "lo": 0.39, "hi": 0.72}},
            1: {"k": {"point": 0.5, "lo": 0.43, "hi": 0.69},
                "m": {"point": 0.5, "lo": 0.41, "hi": 0.68}}}
    ordered = {"k": {"point": 0.5, "lo": 0.37, "hi": 0.70},
               "m": {"point": 0.5, "lo": None, "hi": 0.70}}
    r = H.summarise(banked, runs, ordered)
    assert r["k"]["lo"]["n_nan"] == 1 and r["k"]["lo"]["min"] == 0.43 == r["k"]["lo"]["max"]
    assert r["m"]["lo"]["sorted_minus_banked"] is None and r["m"]["lo"]["sorted_within_range"] is False
    agg = H.aggregate(r)
    assert agg["n_nan_endpoints"] == 1
    assert math.isclose(agg["max_abs_endpoint_from_banked"], 0.03)  # k lo 0.43, m hi 0.72 (0.02)
    assert math.isclose(agg["max_endpoint_range_width"], 0.04)  # m hi 0.68 to 0.72, not NaN
    all_nan = H.summarise({"k": banked["k"]}, {0: {"k": {"point": 0.5, "lo": float("nan"),
                                                        "hi": 0.7}}}, ordered)
    assert all_nan["k"]["lo"]["max_abs_from_banked"] is None and all_nan["k"]["lo"]["n_nan"] == 1
    assert H.aggregate(all_nan)["max_abs_endpoint_from_banked"] == 0.0  # from hi only


def test_aggregate_reads_pre_fix_rows():
    # a part written before `n_nan` existed: NaN propagated into min/max/width
    rows = {"k": {"points_equal_banked": True, "sorted_point_equal_banked": True,
                  "lo": {"min": float("nan"), "max": float("nan"), "width": float("nan"),
                         "max_abs_from_banked": float("nan")},
                  "hi": {"min": 0.7, "max": 0.7, "width": 0.0, "max_abs_from_banked": 0.01}}}
    agg = H.aggregate(rows)
    assert agg["n_nan_endpoints"] == 1 and agg["max_abs_endpoint_from_banked"] == 0.01
    assert agg["max_endpoint_range_width"] == 0.0


def test_absdiff_tolerates_none():
    assert H.absdiff(None, 17) is None and H.absdiff(3, None) is None
    assert H.absdiff(14, 13) == 1 and math.isclose(H.absdiff(0.5, 0.25), 0.25)


@pytest.mark.skipif(not SPREAD.exists(), reason="runs/056/spread.json not built")
def test_spread_json_against_parts_and_ordered_artifacts():
    s = json.loads(SPREAD.read_text())
    assert set(s["targets"]) == {"015", "023", "051"}
    for t, v in s["targets"].items():
        part = json.loads((ROOT / v["source"]).read_text())
        assert v["headline"] == part["headline"]
        # the headline distances recomputed from the per-seed values
        for k, row in v["headline"].items():
            if "lo" not in row:
                continue
            for e in ("lo", "hi"):
                x = np.array([part["per_seed"][str(sd)][k][e] for sd in s["hash_seeds"]])
                assert row[e]["max_abs_from_banked"] == float(np.max(np.abs(x - row["banked"][e])))
        # the real script's ordered artifact equals the sorted mirror; points never moved
        oc = v["ordered_check"]
        assert oc is not None and oc["n_compared"] == part["n_keys"]
        # last-bit differences only (051: BLAS thread count in the contrast, float sums in item
        # order in the permutation test's T); none at all for 015 and 023
        assert all(m["max_abs_diff"] is not None and m["max_abs_diff"] < 1e-12
                   for m in oc["mirror_mismatches"])
        assert all(d["abs_diff"] <= 1e-12 * max(1.0, abs(d["banked"]))
                   for d in oc["points_not_bit_identical"])
        if t in ("015", "023"):
            assert oc["mirror_mismatches"] == [] and oc["points_not_bit_identical"] == []
        # every interval point is bit-identical in all three targets; the aggregates are those
        # of the part's rows (merge recomputes them, so a stale flag in a part cannot leak)
        assert v["n_nan_endpoints"] == 0
        if t in ("015", "023"):
            assert v["all_interval_points_equal_banked"] and v["interval_points_not_equal"] == []
        else:  # 051's spread children ran under one BLAS thread: ci_audit points off by 1e-16
            assert not v["all_interval_points_equal_banked"]
            assert {d["key"] for d in v["interval_points_not_equal"]} == {
                "contrast/per_source/ci_audit", "contrast/per_group_source/ci_audit"}
            assert all(d["max_abs_diff"] < 1e-12 for d in v["interval_points_not_equal"])
        assert v["n_intervals"] == sum("lo" in r for r in part["per_key"].values())
        assert all(d["abs_diff"] is not None for d in oc["points_not_bit_identical"])
    assert s["targets"]["023"]["max_endpoint_range_width"] == 0.0  # order-free statistic
