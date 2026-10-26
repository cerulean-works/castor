"""Prompt 073: castor.confound073 against the 007 / 046 instrument it extends (synthetic data)."""

import numpy as np
import pytest

from castor import confound073 as C
from castor import harmcontrol as HC
from castor import mechladder as ML
from castor import reliability as R

S, N, K = 6, 300, 8


def synth(seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 25, N)
    tok_effect = rng.normal(0, 2, 25)[g]
    base = {s: rng.normal(size=N) for s in ("p1", "p2", "p3")}
    ys, info = {}, {}
    for s, b in base.items():
        for t in ("1", "2"):
            m = f"{s}-{t}"
            latent = b + 0.3 * rng.normal(size=N)
            full = latent + tok_effect + 0.1 * rng.normal(size=N)
            h1 = full[:, None] + 0.05 * rng.normal(size=(N, S))
            h2 = full[:, None] + 0.05 * rng.normal(size=(N, S))
            logit = 0.7 * latent + 0.5 * tok_effect + 0.5 * rng.normal(size=N)
            p = 1 / (1 + np.exp(latent + 0.5 * tok_effect))
            y = (rng.random((N, K)) < p[:, None]).astype(float)
            y[rng.random((N, K)) < 0.05] = np.nan
            ys[m] = C.pack((full, h1, h2), logit, g, y)
            info[m] = R.ModelInfo("rc", s, t)
    return ys, info, g


def test_fast_demean_equals_046():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 5))
    x[rng.random((200, 5)) < 0.1] = np.nan
    g = rng.integers(0, 30, 200)
    assert np.allclose(C.demean_within(x, g), HC.demean_within(x, g), equal_nan=True,
                       atol=1e-12)


def test_raw_variant_is_the_007_link_and_ladder():
    ys, info, _ = synth()
    res = C.analyze(S, gap=True)(ys, info)
    out = C.summary(res)
    ml = {m: C.as_ml(a, S, centre=False) for m, a in ys.items()}
    lk = ML.link_summary(ML.link(S)(ml, info))
    assert out["n_U"] == lk["n_U"] == 12
    assert out["raw_link_corrected"] == pytest.approx(lk["link_U_corrected"], abs=1e-12)
    assert out["raw_link"] == pytest.approx(lk["link_U"], abs=1e-12)
    lad = ML.ladder_summary(ML.ladder(S)(ml, info))
    assert out["ladder_raw_MZ_minus_U_corrected"] == pytest.approx(
        lad["MZ_minus_U_corrected"], abs=1e-12)


def test_centred_ladder_is_046_within_level():
    ys, info, g = synth()
    out = C.summary(C.analyze(S, gap=True)(ys, info))
    within = {}
    for m, a in ys.items():
        proj, _, _, y = C.unpack(a, S)
        within[m] = np.concatenate([HC.demean_within(proj, g), y], axis=1)
    lad = ML.ladder_summary(ML.ladder(S)(within, info))
    assert out["ladder_centred_MZ_minus_U_corrected"] == pytest.approx(
        lad["MZ_minus_U_corrected"], abs=1e-9)


def test_centring_removes_the_token_component():
    ys, info, g = synth()
    out = C.summary(C.analyze(S, gap=True)(ys, info))
    # the token effect is shared by every model, so it inflates r_U; centring removes it
    assert out["ladder_centred_r_U_corrected"] < out["ladder_raw_r_U_corrected"]
    f = C.features(next(iter(ys.values())), S)
    for k in ("z", "zl", "p"):  # z-scoring after centring keeps the group means at zero
        x = f["cen"][k]
        assert np.allclose([np.nanmean(x[g == v]) for v in np.unique(g)], 0, atol=1e-9)


def test_partial_formula_and_correction():
    rng = np.random.default_rng(3)
    w = rng.normal(size=5000)
    x = w + rng.normal(size=5000)
    y = w + x + rng.normal(size=5000)
    rxy, rxw, ryw = (R.corr(a, b) for a, b in ((x, y), (x, w), (y, w)))
    # partial = correlation of residuals after regressing out w
    rx = x - np.polyval(np.polyfit(w, x, 1), w)
    ry = y - np.polyval(np.polyfit(w, y, 1), w)
    assert C.partial(rxy, rxw, ryw) == pytest.approx(R.corr(rx, ry), abs=1e-9)
    assert np.isnan(C.partial(0.5, 1.0, 0.5))


def test_pair_stats_signs_and_corrected_partial_reduces_to_raw():
    ys, _, _ = synth()
    f = {m: C.features(a, S) for m, a in ys.items()}
    a, b = "p1-1", "p2-1"
    st = C.pair_stats(f[a], f[b], "raw", "raw")
    # higher projection and higher logit both mean lower P in the synthetic data
    assert st["link"] > 0 and st["logit_link"] > 0
    assert st["logit_link_corrected"] == pytest.approx(
        st["logit_link"] / np.sqrt(st["rel_dP"]))
    fa = {k: dict(v) for k, v in f[a].items()}
    fb = {k: dict(v) for k, v in f[b].items()}
    for ff in (fa, fb):  # perfectly reliable halves -> rel 1 -> corrected = raw partial
        ff["raw"]["p_even"] = ff["raw"]["p_odd"] = ff["raw"]["p"]
        ff["raw"]["z1"] = ff["raw"]["z2"] = np.repeat(ff["raw"]["z"][:, None], S, 1)
    st1 = C.pair_stats(fa, fb, "raw", "raw")
    assert st1["partial_link_corrected"] == pytest.approx(st1["partial_link"], abs=1e-9)


def test_per_group_keys_and_bootstrap_runs():
    ys, info, _ = synth()
    groups = {m: "rc" for m in ys}
    an = C.analyze(S, gap=False, groups_of=groups)
    ci = R.bootstrap_prompts(ys, info, 5, stat=C.summary, analyze=an)
    assert "group:rc:centred_link_corrected" in ci
    assert ci["centred_link_corrected"]["lo"] <= ci["centred_link_corrected"]["hi"]


def test_prediction_rule():
    def e(v, lo, hi):
        return {"estimate": v, "lo": lo, "hi": hi}
    a0 = {"ladder_centred_MZ_minus_U_corrected": e(0.2, 0.1, 0.3),
          "centred_link_corrected": e(0.35, 0.3, 0.4)}
    a = {"centred_link_corrected": e(0.49, 0.45, 0.52),
         "raw_partial_link_corrected": e(0.4, 0.35, 0.45)}
    val = {"m1": {"raw": {"r_logit_P": -0.5}}, "m2": {"raw": {"r_logit_P": 0.1}}}
    p = C.prediction(a0, a, val)
    assert p["gap_centred_A0"]["holds"] and p["link_centred_A0"]["holds"]
    assert not p["link_centred_A"]["holds"]
    assert not p["logit_validity_A"]["holds"] and p["logit_validity_A"]["n_negative"] == 1
    assert p["partial_link_A"]["holds"]


def test_jackknife_over_groups_runs_on_the_pooled_summary():
    ys, info, _ = synth()
    groups = {m: m.split("-")[0] for m in ys}  # three groups of two models
    info = {m: R.ModelInfo("rc" if groups[m] != "p3" else "rc2", groups[m], m[-1]) for m in ys}
    an = C.analyze(S, gap=False, groups_of={m: info[m].recipe for m in ys})
    jk = ML.jackknife_groups(ys, info, {m: info[m].recipe for m in ys}, C.summary_pooled, an)
    assert "centred_link_corrected" in jk and not any(k.startswith("group:") for k in jk)


def test_token_cluster_bootstrap_runs():
    ys, info, g = synth()
    an = C.analyze(S, gap=True)
    ci = R.bootstrap_prompt_clusters(ys, info, g.tolist(), 4, stat=C.summary, analyze=an)
    assert ci["centred_link_corrected"]["lo"] <= ci["centred_link_corrected"]["hi"]
