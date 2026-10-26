"""Cross-twin transfer: donors, the Procrustes fit and its mismatched control, the transfer ratio, the
pre-registered rule, and the committed fit set."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from castor import transfer as T


def _rotation(d: int, seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return q


def test_donors_cycle():
    tg = T.targets()
    assert len(tg) == 12
    for kind in ("same", "diff"):
        assert sorted(T.donors(m)[kind] for m in tg) == sorted(tg)  # each a donor once
    for m in tg:
        b, k = T.parse_id(m)
        sb, sk = T.parse_id(T.donors(m)["same"])
        db, dk = T.parse_id(T.donors(m)["diff"])
        assert sb == b and sk != k and dk == k and db != b
    assert T.donors("dd-dclm-1b-default-dpo-seed4")["same"] == "dd-dclm-1b-default-dpo-seed1"
    assert T.donors("dd-dclm-1b-large-aux-3-dpo-seed2")["diff"] == "dd-dclm-1b-default-dpo-seed2"
    assert T.donors("dd-dclm-1b-default-dpo-seed1")["diff"] == \
        "dd-dclm-1b-large-aux-2-dpo-seed1"


def test_procrustes_recovers_known_rotation():
    rng = np.random.default_rng(0)
    d = 32
    q = _rotation(d, 1)
    xa = rng.standard_normal((2000, d)) * np.linspace(0.5, 3, d) + 5.0  # offset: centring
    xb = xa @ q + 0.01 * rng.standard_normal((2000, d)) - 2.0
    r = T.procrustes(xa, xb)
    assert np.allclose(r.T @ r, np.eye(d), atol=1e-10)
    assert np.abs(r - q).max() < 1e-2
    v = rng.standard_normal(d)
    assert T.cos(T.transport(v, r), v @ q) > 0.9999
    assert T.fit_residual(xa, xb, r) < 0.01
    assert T.fit_residual(xa, xb, np.eye(d)) > 0.5


def test_identity_when_twins_share_a_basis():
    rng = np.random.default_rng(2)
    xa = rng.standard_normal((1000, 16))
    r = T.procrustes(xa, xa + 1e-3 * rng.standard_normal(xa.shape))
    assert np.abs(r - np.eye(16)).max() < 1e-2


def _prompts(n_prompts: int, d: int, seed: int, n_prefix: int = 0, n_suffix: int = 0):
    """Tokens = a prompt-specific vector plus small token noise; BOS = one huge shared row;
    optionally a chat-template prefix (identical rows in every prompt, as under causal
    attention) and suffix (shared rows plus the prompt's vector, as template tokens that
    attend to the content)."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(4 + n_prefix + n_suffix, 12 + n_prefix + n_suffix, size=n_prompts)
    pre = rng.standard_normal((n_prefix, d)) * 3
    suf = rng.standard_normal((n_suffix, d)) * 3
    rows = []
    for n in lengths:
        c = rng.standard_normal(d)
        bos = np.full((1, d), 50.0)
        content = c + 0.1 * rng.standard_normal((n - 1 - n_prefix - n_suffix, d))
        rows.append(np.vstack([bos, pre, content, suf + c]))
    return np.vstack(rows), [int(x) for x in lengths]


def test_mismatched_control_collapses():
    d = 24
    xa, lengths = _prompts(400, d, 3)
    q = _rotation(d, 4)
    xb = xa @ q
    v = np.random.default_rng(5).standard_normal(d)
    ra, rb = T.matched_rows(lengths)
    r = T.procrustes(xa[ra], xb[rb])
    assert T.cos(T.transport(v, r), v @ q) > 0.999
    ma, mb = T.mismatched_rows(lengths, T.derangement(len(lengths)))
    rm = T.procrustes(xa[ma], xb[mb])
    assert abs(T.cos(T.transport(v, rm), v @ q)) < 0.5
    assert T.fit_residual(xa[ra], xb[rb], rm) > 1.0  # a random rotation of centred rows: ~sqrt 2


def test_mismatched_control_collapses_with_a_shared_template():
    """Prompts share 5 prefix and 7 suffix template rows (the tulu template). Aligning
    partner prompts from the end would pair the suffix id for id and keep a real component
    of the map; the position permutation does not."""
    d = 24
    xa, lengths = _prompts(400, d, 7, n_prefix=5, n_suffix=7)
    q = _rotation(d, 8)
    xb = xa @ q
    partner = T.derangement(len(lengths))
    rows = T.token_rows(lengths)
    end_a = np.concatenate([rows[i][-min(len(rows[i]), len(rows[j])):]
                            for i, j in enumerate(partner)])
    end_b = np.concatenate([rows[j][-min(len(rows[i]), len(rows[j])):]
                            for i, j in enumerate(partner)])
    r_end = T.procrustes(xa[end_a], xb[end_b])
    ma, mb = T.mismatched_rows(lengths, partner)
    rm = T.procrustes(xa[ma], xb[mb])
    ra, rb = T.matched_rows(lengths)

    def dist(r):  # ||R - Q||_F / sqrt(d): sqrt 2 for an unrelated rotation
        return np.linalg.norm(r - q) / np.sqrt(d)

    assert dist(r_end) < 1.2 and T.fit_residual(xa[ra], xb[rb], r_end) < 0.8  # the defect
    assert dist(rm) > 1.3 and T.fit_residual(xa[ra], xb[rb], rm) > 1.3  # a true null


def test_rows_skip_bos_and_pair_within_partner_prompts():
    ra, rb = T.matched_rows([3, 5])
    assert ra.tolist() == [1, 2, 4, 5, 6, 7] and rb.tolist() == ra.tolist()
    assert T.matched_rows([3, 5], skip_first=0)[0].tolist() == list(range(8))
    lengths = [3, 5, 4]
    partner = np.array([1, 2, 0])
    ma, mb = T.mismatched_rows(lengths, partner)
    rows = T.token_rows(lengths)
    assert len(ma) == len(mb) == 2 + 3 + 2
    start = 0
    for i, j in enumerate(partner):
        m = min(len(rows[i]), len(rows[j]))
        a, b = ma[start:start + m], mb[start:start + m]
        assert set(a) <= set(rows[i]) and set(b) <= set(rows[j])  # rows of the right prompts
        assert len(set(a)) == len(set(b)) == m  # no row twice
        start += m
    assert np.array_equal(ma, T.mismatched_rows(lengths, partner)[0])  # deterministic
    assert not np.array_equal(ma, T.mismatched_rows(lengths, partner, seed=1)[0])


def test_template_span_and_content_rows():
    ids = [[0, 7, 8, 1, 2, 3, 9, 9], [0, 7, 8, 4, 9, 9], [0, 7, 8, 5, 6, 9, 9]]
    assert T.template_span(ids) == (3, 2)
    ra, rb = T.content_rows([8, 6, 7], 3, 2)
    assert ra.tolist() == [3, 4, 5, 11, 17, 18] and rb.tolist() == ra.tolist()
    assert T.template_span([[0, 1], [0, 1]]) == (2, 0)  # nothing but template
    assert T.template_span([[0, 1, 2], [0, 3, 2]]) == (1, 1)


def test_standardized_procrustes_recovers_rotation_under_scaling():
    rng = np.random.default_rng(10)
    d, n = 32, 4000
    q = _rotation(d, 11)
    sa = np.exp(rng.uniform(-1.5, 1.5, d))
    sb = np.exp(rng.uniform(-1.5, 1.5, d))
    z = rng.standard_normal((n, d))
    xa = z * sa + 3.0
    xb = (z @ q) * sb - 1.0
    r, sa_hat, sb_hat = T.standardized_procrustes(xa, xb)
    assert np.abs(r - q).max() < 0.05
    assert np.allclose(sa_hat, sa, rtol=0.1) and np.allclose(sb_hat, sb, rtol=0.1)
    v = rng.standard_normal(d)
    want = T.unit(((v / sa) @ q) * sb)
    assert T.cos(T.transport_standardized(v, r, sa_hat, sb_hat), want) > 0.999
    assert T.cos(T.transport(v, T.procrustes(xa, xb)), want) < 0.99  # the raw fit does not
    with pytest.raises(AssertionError):
        T.standardized_procrustes(np.hstack([xa, np.ones((n, 1))]),
                                  np.hstack([xb, np.ones((n, 1))]))


def test_extract_skip_needs_same_fit_set_layers_and_checkpoint():
    import importlib.util

    p = Path(__file__).resolve().parents[1] / "scripts" / "transfer_018.py"
    spec = importlib.util.spec_from_file_location("transfer_018", p)
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    meta = {"fit_sha256": S.sha(S.FIT), "layers": [9, 10, 11], "checkpoint_resolved": "/x"}
    assert S.acts_complete(meta, [9, 10, 11], "/x")
    assert not S.acts_complete(meta, [9, 10], "/x")
    assert not S.acts_complete(meta, [9, 10, 11], "/y")
    assert not S.acts_complete(meta | {"fit_sha256": "0"}, [9, 10, 11], "/x")


def test_derangement_has_no_fixed_point():
    for n in (2, 3, 512):
        p = T.derangement(n)
        assert sorted(p.tolist()) == list(range(n)) and not np.any(p == np.arange(n))
    assert np.array_equal(T.derangement(512), T.derangement(512))  # deterministic


def test_ratio_and_joint_bootstrap():
    rng = np.random.default_rng(6)
    n = 400
    base = rng.uniform(0, 0.4, n)
    own = np.clip(base + 0.6, 0, 1)
    arm = base + 0.5 * (own - base)
    pp = {"base": base, "own": own, "half": arm, "none": base.copy()}
    pp["half"][3] = np.nan  # an unusable prompt is skipped, as in sanityb.model_stats
    out = T.bootstrap_ratio(pp, ["half", "none"], n_boot=200)
    assert out["half"]["T"] == pytest.approx(0.5, abs=0.01)
    lo, hi = out["half"]["ci"]
    assert lo <= out["half"]["T"] <= hi and hi - lo < 0.05
    assert out["none"]["T"] == pytest.approx(0.0) and out["none"]["ci"] == [0.0, 0.0]
    assert T.ratio(0.5, 0.2, 0.8) == pytest.approx(0.5)
    assert np.isnan(T.ratio(0.5, 0.2, 0.2))


def _t(same, unrot, rot, mis):
    return {"same-unrot": same, "diff-unrot": unrot, "diff-rot": rot, "diff-mismatch": mis}


def test_rule():
    tg = [f"m{i:02d}" for i in range(12)]
    go = {m: _t(0.9, 0.1, 0.6, 0.1) for m in tg}
    for m in tg[:4]:
        go[m] = _t(0.9, 0.1, 0.6, 0.4)  # margin 0.2 < 0.3 in 4: 8 pass
    v = T.verdict(go)
    assert v["go"] and len(v["diff_rot_pass"]) == 8 and v["reading"].startswith("GO")
    go[tg[4]] = _t(0.9, 0.1, 0.49, 0.0)  # 7 pass
    v = T.verdict(go)
    assert not v["go"] and v["reading"] == "not transportable by a rotation at this size"
    assert v["predictions"]["diff_rot_ge_0.5_majority"]  # 11 of 12 >= 0.5
    bad = {m: _t(0.9 if i >= 3 else 0.7, 0.1, 0.9, 0.0) for i, m in enumerate(tg)}
    v = T.verdict(bad)  # same-unrot in 9 of 12: inconclusive whatever diff-rot does
    assert not v["go"] and v["reading"].startswith("inconclusive") and not v["sanity_same"]
    bad = {m: _t(0.9, 0.25 if i < 3 else 0.2, 0.9, 0.0) for i, m in enumerate(tg)}
    v = T.verdict(bad)  # diff-unrot <= 0.2 (inclusive) in 9 of 12
    assert not v["sanity_diff_unrot"] and len(v["diff_unrot_pass"]) == 9


def test_rule_ignores_the_secondary_arm():
    tg = [f"m{i:02d}" for i in range(12)]
    t = {m: _t(0.9, 0.1, 0.6, 0.1) for m in tg}
    v = T.verdict(t)
    for x in (-5.0, 5.0):
        assert T.verdict({m: r | {"diff-rot-std": x} for m, r in t.items()}) == v
    assert T.RULE_ARMS == ("same-unrot", "diff-unrot", "diff-rot", "diff-mismatch")
    assert T.MAP_OF_ARM["diff-rot-std"] == "standardized"
    assert T.DONOR_OF_ARM["diff-rot-std"] == T.DONOR_OF_ARM["diff-rot"] == "diff"


def test_committed_fit_set():
    root = Path(__file__).resolve().parents[1] / "data" / "directions"
    data = (root / "procrustes_fit_v1.jsonl").read_bytes()
    man = json.loads((root / "procrustes_fit_v1.MANIFEST.json").read_text())
    assert hashlib.sha256(data).hexdigest() == man["sha256"]
    rows = [json.loads(x) for x in data.decode().splitlines()]
    assert sum(r["class"] == "harmful" for r in rows) == 256
    assert sum(r["class"] == "harmless" for r in rows) == 256
    used = {json.loads(x)["prompt"] for x in
            (root / "extraction_v1.jsonl").read_text().splitlines()}
    texts = [r["prompt"] for r in rows]
    assert len(set(texts)) == 512 and not used & set(texts)
