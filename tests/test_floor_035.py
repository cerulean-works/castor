"""035: the second same-seed replicate's grouping, three-pair mean, difference, prediction and
branch rule (castor.floor, section 035) and the read-back check of scripts/floor_035.py."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from castor import floor, reliability, sanityb
from castor.provenance import load_script

SEEDS = ("default", "large-aux-2", "large-aux-3", "replicate", "replicate2")


@pytest.fixture
def models(synthetic_twins):
    """20 synthetic models on 5 bases (test_floor's generator plus R2): R and R2 share part of
    default's base latent (partial ancestry), R2 with weight `share_r2`."""
    return lambda seed=0, n=200, k=20, share_r2=0.5: synthetic_twins(
        SEEDS, {"replicate": 0.5, "replicate2": share_r2}, seed, n, k)


def _floor_035():
    return load_script("floor_035")


@pytest.mark.parametrize("a, b, g", [
    ("default", "default", "MZ_006"), ("large-aux-3", "large-aux-3", "MZ_006"),
    ("default", "large-aux-2", "U_006"), ("large-aux-2", "large-aux-3", "U_006"),
    ("default", "replicate", "rung2_RD"), ("replicate", "default", "rung2_RD"),
    ("replicate", "replicate2", "rung2_RR2"), ("replicate2", "replicate", "rung2_RR2"),
    ("default", "replicate2", "rung2_R2D"), ("replicate2", "default", "rung2_R2D"),
    ("replicate", "replicate", "MZ_rep"), ("replicate2", "replicate2", "MZ_R2"),
    ("large-aux-2", "replicate", "rep_other"), ("replicate", "large-aux-3", "rep_other"),
    ("large-aux-2", "replicate2", "R2_other"), ("replicate2", "large-aux-3", "R2_other"),
])
def test_group_of_035(a, b, g):
    assert floor.group_of_035(a, b) == g


def test_011_groups_unchanged_without_r2():
    """On the 16 models of 011/027, group_of_035 names the same groups as group_of."""
    for a in SEEDS[:4]:
        for b in SEEDS[:4]:
            want = floor.group_of(a, b)
            assert floor.group_of_035(a, b) == ("rung2_RD" if want == "rung2" else want)


def test_seeds_035_are_the_closed_set_of_bases():
    assert floor.SEEDS_035 == SEEDS
    assert set(floor.SAME_SEED) < set(floor.SEEDS_035)
    assert set(sanityb.PRETRAIN_SEEDS) < set(floor.SEEDS_035)


@pytest.mark.parametrize("a, b", [
    ("replicate2-i6198-d6198", "default"),  # the manifest's revision, not the base label
    ("default", "replicate2-i6198-d6198"),
    ("replicate3", "replicate3"), ("large-aux-1", "default"),
])
def test_group_of_035_raises_on_an_unknown_label(a, b, models):
    """The 035 groups are closed: an unknown label must not fall silently into U_006 / MZ_006
    (which would leave the rung-2 groups empty and mean3 NaN)."""
    with pytest.raises(ValueError, match="unknown pretraining-seed label"):
        floor.group_of_035(a, b)
    ys, info = models(n=60)
    label = next(x for x in (a, b) if x not in floor.SEEDS_035)
    bad = {m: reliability.ModelInfo(i.recipe, label if i.pretrain_seed == "replicate2" else
                                    i.pretrain_seed, i.posttrain_seed) for m, i in info.items()}
    with pytest.raises(ValueError):
        floor.summary_035(reliability.twin_correlations(ys, bad), bad)


def test_pair_counts_and_027_values_reproduced(models):
    ys, info = models()
    tw = reliability.twin_correlations(ys, info)
    s = floor.summary_035(tw, info)
    assert [s[f"n_{g}"] for g in floor.GROUPS_035] == [18, 48, 16, 16, 16, 6, 6, 32, 32]
    assert s["n_other"] == 64
    # 027's quantities (floor.summary on the 16 models without R2) are reproduced exactly:
    # a pair's r does not depend on which other models are in the panel
    sixteen = {m: y for m, y in ys.items() if info[m].pretrain_seed != "replicate2"}
    s16 = floor.summary(reliability.twin_correlations(sixteen, info), info)
    for g16, g35 in (("MZ_006", "MZ_006"), ("U_006", "U_006"), ("rung2", "rung2_RD"),
                     ("MZ_rep", "MZ_rep"), ("rep_other", "rep_other")):
        for sfx in ("", "_corrected"):
            assert s[f"r_{g35}{sfx}"] == pytest.approx(s16[f"r_{g16}{sfx}"], abs=1e-12)
    assert s["RD_minus_rep_other_corrected"] == pytest.approx(s16["rung2_minus_rep_other_corrected"])
    # the synthetic replicates share part of default's latent: both between U and MZ
    for g in floor.RUNG2_PAIRS_035:
        assert s["r_U_006_corrected"] < s[f"r_{g}_corrected"] < s["r_MZ_006_corrected"]


def test_mean3_is_the_mean_of_the_three_pair_means_and_the_48_pair_mean(models):
    ys, info = models(n=150)
    tw = reliability.twin_correlations(ys, info)
    s = floor.summary_035(tw, info)
    for sfx in ("", "_corrected"):
        three = [s[f"r_{g}{sfx}"] for g in floor.RUNG2_PAIRS_035]
        assert s[f"rung2_mean3{sfx}"] == pytest.approx(np.mean(three), abs=1e-12)
        key = "r_corrected" if sfx else "r"
        pooled = [x[key] for x in tw["pairs"]
                  if floor.group_of_035(info[x["a"]].pretrain_seed, info[x["b"]].pretrain_seed)
                  in floor.RUNG2_PAIRS_035]
        assert len(pooled) == 48
        assert s[f"rung2_mean3{sfx}"] == pytest.approx(np.mean(pooled), abs=1e-12)
    bp = floor.by_base_pair(tw["pairs"], info, floor.group_of_035)
    assert s["rung2_mean3_corrected"] == pytest.approx(np.mean(
        [bp[k]["r_corrected"] for k in ("default|replicate", "replicate|replicate2",
                                        "default|replicate2")]), abs=1e-12)


def test_contrasts_are_differences_of_the_group_means(models):
    ys, info = models(n=150, seed=3)
    s = floor.summary_035(reliability.twin_correlations(ys, info), info)
    for sfx in ("", "_corrected"):
        rd, rr2, r2d = (s[f"r_{g}{sfx}"] for g in floor.RUNG2_PAIRS_035)
        assert s[f"RR2_minus_RD{sfx}"] == pytest.approx(rr2 - rd)
        assert s[f"R2D_minus_RD{sfx}"] == pytest.approx(r2d - rd)
        # row 2b of the branch is a summary key of its own (so it carries a CI and an SE)
        assert s[f"rung2_diffstack{sfx}"] == pytest.approx((rd + r2d) / 2, abs=1e-12)
        assert s[f"RR2_minus_diffstack{sfx}"] == pytest.approx(rr2 - s[f"rung2_diffstack{sfx}"])
        m3, u, mz = s[f"rung2_mean3{sfx}"], s[f"r_U_006{sfx}"], s[f"r_MZ_006{sfx}"]
        assert s[f"mean3_minus_U{sfx}"] == pytest.approx(m3 - u)
        assert s[f"MZ_minus_mean3{sfx}"] == pytest.approx(mz - m3)
        assert s[f"mean3_gap_fraction{sfx}"] == pytest.approx((m3 - u) / (mz - u))
        assert s[f"mean3_minus_other{sfx}"] == pytest.approx(m3 - s[f"r_other{sfx}"])
        assert s[f"RR2_minus_other{sfx}"] == pytest.approx(rr2 - s[f"r_other{sfx}"])
    # `other` pools rep_other (32) and R2_other (32): equal weights, so the plain mean
    assert s["r_other_corrected"] == pytest.approx(
        (s["r_rep_other_corrected"] + s["r_R2_other_corrected"]) / 2, abs=1e-12)


def test_stand_in_r2_has_known_answers(models):
    """R2 := copies of R's matrices (the --stand-in-r2 dry run): r(R2-D) = r(R-D) exactly and
    r(R2-other) = r(R-other); the R-R2 group pairs each model with its copy (r = 1 raw)."""
    ys, info = models(n=120)
    ys = {m: y for m, y in ys.items() if info[m].pretrain_seed != "replicate2"}
    for kk in range(1, 5):
        ys[f"replicate2-s{kk}"] = ys[f"replicate-s{kk}"].copy()
    s = floor.summary_035(reliability.twin_correlations(ys, info), info)
    assert s["r_rung2_R2D_corrected"] == pytest.approx(s["r_rung2_RD_corrected"], abs=1e-12)
    assert s["r_R2_other_corrected"] == pytest.approx(s["r_rep_other_corrected"], abs=1e-12)
    assert s["r_MZ_R2_corrected"] == pytest.approx(s["r_MZ_rep_corrected"], abs=1e-12)
    assert s["R2D_minus_RD_corrected"] == 0.0
    assert s["r_rung2_RR2"] > s["r_MZ_rep"]  # 4 of its 16 pairs are a model against itself


def test_bootstrap_and_jackknife_carry_the_035_quantities(models):
    ys, info = models(n=120)
    stat = floor.stat_for_035(info)
    ci = reliability.bootstrap_prompts(ys, info, 15, 0, stat=stat)
    d = ci["RR2_minus_RD_corrected"]
    assert d["n_draws"] == 15 and d["lo"] <= d["estimate"] <= d["hi"]
    for row in ("r_rung2_RR2_corrected", "rung2_diffstack_corrected"):  # branch rows 2a, 2b
        assert ci[row]["n_draws"] == 15 and ci[row]["lo"] <= ci[row]["estimate"] <= ci[row]["hi"]
    same = sorted(m for m in ys if info[m].pretrain_seed in floor.SAME_SEED)
    assert len(same) == 12
    jk = floor.jackknife_models(ys, info, same, stat=stat)
    r = jk["rung2_mean3_corrected"]
    assert r["n_reps"] == 12 and r["se"] > 0 and r["lo"] < r["estimate"] < r["hi"]
    assert jk["RR2_minus_RD_corrected"]["n_reps"] == 12
    assert jk["rung2_diffstack_corrected"]["n_reps"] == 12
    # 011's jackknife (default stat) is unchanged on the 16 models
    sixteen = {m: y for m, y in ys.items() if info[m].pretrain_seed != "replicate2"}
    eight = sorted(m for m in sixteen if info[m].pretrain_seed in ("default", "replicate"))
    assert floor.jackknife_models(sixteen, info, eight)["r_rung2_corrected"]["n_reps"] == 8


@pytest.mark.parametrize("rr2, diff, lo, hi, holds, branch", [
    (0.845, 0.0055, -0.006, 0.017, True, "one_row_mean3"),    # within 0.02, CI covers 0
    (0.870, 0.0305, 0.019, 0.042, False, "two_rows_2a_2b"),   # > 0.02 above, CI excludes 0
    (0.855, 0.0155, 0.004, 0.027, False, "two_rows_2a_2b"),   # within 0.02 but CI excludes 0
    (0.815, -0.0245, -0.036, -0.013, False, "one_row_mean3"),  # below by > 0.02, CI excludes 0
    (0.8595, 0.02, -0.001, 0.041, True, "one_row_mean3"),      # exactly at the tolerance edge
    (0.8596, 0.0201, -0.001, 0.041, False, "one_row_mean3"),   # just outside it
])
def test_prediction_035(rr2, diff, lo, hi, holds, branch):
    p = floor.prediction_035(rr2, 0.8395, diff, lo, hi)
    assert p["holds"] is holds and p["branch"] == branch
    assert p["two_rows"] is (branch == "two_rows_2a_2b")
    # the flag is a function of the reported numbers alone (no hidden rounding)
    assert p["abs_difference_from_027"] == abs(rr2 - 0.8395)
    assert p["within_tolerance"] is (p["abs_difference_from_027"]
                                     <= p["tolerance"] + p["tolerance_fp_slack"])


def test_tolerance_edge_is_only_floating_point_slack():
    assert 0.8595 - 0.8395 > 0.02  # the edge row of test_prediction_035 lands 2e-17 above 0.02
    assert floor.prediction_035(0.8595, 0.8395, 0.0, -0.01, 0.01)["within_tolerance"] is True
    assert floor.prediction_035(0.8395 + 0.02 + 2e-9, 0.8395, 0.0, -0.01, 0.01)["within_tolerance"] is False
    assert floor.FP_SLACK_035 < 1e-6


def test_branch_flips_at_the_ci_edge():
    base = {"r_rr2": 0.85, "r_rd_027": 0.8395, "diff": 0.0105}
    assert floor.prediction_035(**base, diff_lo=0.0, diff_hi=0.02)["branch"] == "one_row_mean3"
    assert floor.prediction_035(**base, diff_lo=1e-9, diff_hi=0.02)["branch"] == "two_rows_2a_2b"
    assert floor.prediction_035(**base, diff_lo=-1e-9, diff_hi=0.02)["ci_covers_zero"] is True
    neg = floor.prediction_035(0.82, 0.8395, -0.0195, -0.03, -0.009)
    assert neg["ci_excludes_zero_negative"] is True and neg["branch"] == "one_row_mean3"
    assert floor.prediction_035(0.82, 0.8395, -0.0195, -0.03, 0.0)["ci_excludes_zero_negative"] is False


def test_by_base_pair_035(models):
    ys, info = models(n=120)
    tw = reliability.twin_correlations(ys, info)
    bp = floor.by_base_pair(tw["pairs"], info, floor.group_of_035)
    assert len(bp) == 15  # 5 within-base + 10 across-base keys
    assert {k for k, v in bp.items() if v["n_pairs"] == 6} == {f"{s}|{s}" for s in SEEDS}
    assert bp["replicate|replicate2"]["group"] == "rung2_RR2"
    assert bp["default|replicate2"]["group"] == "rung2_R2D"
    assert bp["replicate2|replicate2"]["group"] == "MZ_R2"
    assert bp["large-aux-3|replicate2"]["group"] == "R2_other"
    s = floor.summary_035(tw, info)
    assert bp["replicate|replicate2"]["r_corrected"] == pytest.approx(s["r_rung2_RR2_corrected"])


def test_r2_cells_manifest():
    f35 = _floor_035()
    cells = f35.r2_cells()
    assert [c.k for c in cells] == [1, 2, 3, 4]
    assert all(c.base.pretrain_seed == "replicate2-i6198-d6198" for c in cells)
    assert cells[0].gen_dir().name == "035"
    assert tuple(c.model_id for c in cells) == f35.R2_MODEL_IDS


@pytest.mark.parametrize("bad, msg", [
    ({"prompt_id": "011"}, "manifest prompt_id 011 != 035"),
    ({"posttrain_seeds": [1, 2]}, r"posttrain_seeds \(1, 2\) != 006's"),
    # the variant lets grid.load pass (its guard, b7b7fc2), so the launcher's own check fires
    ({"sft_config": "configs/posttrain/other.yaml", "variant": "other"}, "006's post-training configs"),
    ({"revision": "replicate-i6198-d6198"}, r"bases \[.*@replicate-i6198-d6198'\] != "),
    # per-base overrides grid.Grid.cells honours: refused by their own guard, not silently
    # applied (nor caught only downstream by the R2_MODEL_IDS count)
    ({"posttrain_seeds": [1, 2], "per_base": True}, "base-level posttrain_seeds"),
    ({"base_dir": "/elsewhere", "per_base": True}, "base_dir override"),
])
def test_r2_cells_refuse_another_manifest(tmp_path, bad, msg):
    f35 = _floor_035()
    m = json.loads(f35.MANIFEST_035.read_text())
    bad = dict(bad)
    if bad.pop("per_base", False) or "revision" in bad:
        bad = {"bases": [m["bases"][0] | bad]}
    p = tmp_path / "grid.json"
    p.write_text(json.dumps(m | bad))
    with pytest.raises(ValueError, match=msg):
        f35.r2_cells(p)


def test_record_run_paths_are_011s_with_the_revision_substituted():
    """The record run (no --stand-in-r2) reads the R2 labels and SFT provenance from paths the
    dry run never exercised: they must be 011's paths for R with `replicate` -> `replicate2`,
    i.e. what launch_006.train_cmd (via grid.Cell.out) and gen_judge.sh (grid.Cell.gen_dir)
    produce; launch_035 must build its cells from r2_cells()."""
    f35 = _floor_035()
    rep = {c.k: c for c in load_script("launch_011").cells()}
    for c in f35.r2_cells():
        r = rep[c.k]
        assert str(c.out("sft")) == str(r.out("sft")).replace("-replicate-", "-replicate2-")
        assert c.out("sft") == sanityb.MODELS_ROOT / "dd-dclm-baseline-1B-seed-replicate2-i6198-d6198" / "sft" / f"seed{c.k}"
        labels = sanityb.labels_path(c.model_id, f35.PROMPT_ID)
        assert labels == c.gen_dir() / "judge_v1" / "labels.jsonl"
        assert labels == Path("/work/harrychang08/castor/runs") / c.model_id / "035" / "judge_v1" / "labels.jsonl"
        assert str(labels) == str(sanityb.labels_path(r.model_id, "011")).replace(
            "-replicate-", "-replicate2-").replace("/011/", "/035/")


def _fake_run(f027: dict, rd_shift: float = 0.0) -> dict:
    """A 035 result whose R-D, U, MZ, MZ_rep and rep_other entries copy 027's (shifted by
    `rd_shift` on R-D), for every subset x outcome, plus one R-D prompt CI."""
    out = {}
    for s, block in f027.items():
        if not isinstance(block, dict) or "primary" not in block:
            continue
        out[s] = {}
        for o, v in block.items():
            sm = v["summary"]
            new = {k.replace("rung2", "rung2_RD"): x for k, x in sm.items()
                   if k.startswith(("r_", "n_"))}
            new["r_rung2_RD_corrected"] += rd_shift
            out[s][o] = {"summary": new,
                         "ci_prompts": {"r_rung2_RD_corrected": v["ci_prompts"]["r_rung2_corrected"]}}
    return out


def test_check_against_027_passes_on_identical_values_and_fails_on_a_shift():
    f35 = _floor_035()
    f027 = json.loads(f35.FLOOR_027.read_text())
    outcomes = load_script("floor_011").OUTCOMES
    same = (f027["n_boot"], f027["seed"])
    assert same == (2000, 0)  # 027's pre-registered draws (--n-boot 2000 --seed 0)
    ref = f35.check_against_027(f027, _fake_run(f027), outcomes, *same)
    assert ref["point_values_identical"] is True and ref["ci_identical"] is True
    assert ref["ci_identity_asserted"] is True
    assert ref["prediction"]["r_rung2"] == f027["prediction"]["r_rung2"]
    with pytest.raises(AssertionError, match="not reproduced"):
        f35.check_against_027(f027, _fake_run(f027, rd_shift=1e-6), outcomes, *same)


def test_check_against_027_asserts_the_ci_identity_only_on_027s_draws():
    f35 = _floor_035()
    f027 = json.loads(f35.FLOOR_027.read_text())
    outcomes = load_script("floor_011").OUTCOMES
    run = _fake_run(f027)
    ci = run[sanityb.PRIMARY_SUBSET]["primary"]["ci_prompts"]["r_rung2_RD_corrected"] = copy.deepcopy(
        run[sanityb.PRIMARY_SUBSET]["primary"]["ci_prompts"]["r_rung2_RD_corrected"])
    ci["lo"] -= 1e-4  # a different CI: a changed draw order, or (legitimately) another n_boot
    ref = f35.check_against_027(f027, run, outcomes, 20, 0)  # the dry run's n_boot: recorded
    assert ref["point_values_identical"] is True and ref["ci_identical"] is False
    assert ref["ci_identity_asserted"] is False
    with pytest.raises(AssertionError, match="prompt CI differs"):  # the record run: asserted
        f35.check_against_027(f027, run, outcomes, f027["n_boot"], f027["seed"])
