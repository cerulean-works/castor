import numpy as np
import pytest

from castor import transfer
from castor import transplant as T
from castor.provenance import load_script

TP = load_script("transplant_089")
N, K = 3000, 20


def sim(rate: np.ndarray, rng) -> np.ndarray:
    return (rng.random((len(rate), K)) < rate[:, None]).astype(float)


def logistic(x):
    return 1 / (1 + np.exp(-x))


@pytest.fixture(scope="module")
def planted():
    """Shared ancestry s, base-specific u_A, u_B. The projection z_M = u_M (standardised) is a
    full mediator of the base-specific part: the edited A′ behaves as logistic(s + u_A + Δz)."""
    rng = np.random.default_rng(1)
    s, ua, ub = rng.standard_normal(N), rng.standard_normal(N), rng.standard_normal(N)
    za, zb = T.zscore(ua, *T.site_stats(ua)), T.zscore(ub, *T.site_stats(ub))
    sig = T.site_stats(ua)[1]
    coef_t = T.transplant_coeffs(za, zb, sig)
    coef_p = T.transplant_coeffs(za, T.permuted(zb), sig)

    def edited(c):
        return logistic(s + ua + c)

    ys = {"A": sim(logistic(s + ua), rng), "B": sim(logistic(s + ub), rng),
          "MZ_A": sim(logistic(s + ua), rng),
          "A_t": sim(edited(coef_t), rng), "A_p": sim(edited(coef_p), rng),
          "A_i": sim(edited(np.zeros(N)), rng)}
    return ys


def test_closure_recovered(planted):
    pairs = [{"name": "p", "a": "A", "b": "B", "mz_a": "MZ_A",
              "arms": {"transplant": "A_t", "permuted": "A_p", "identity": "A_i"}}]
    out = T.closure_stat(pairs, ("transplant", "permuted", "identity"))(planted)
    m = out["mean_closure"]
    assert m["transplant"] == pytest.approx(1.0, abs=0.08)
    assert abs(m["permuted"]) < 0.08
    assert abs(m["identity"]) < 0.08
    # identity reproduces A up to sampling: its corrected correlation with A is at the ceiling
    assert T.corrected(planted["A_i"], planted["A"])["r_corrected"] == pytest.approx(1, abs=0.03)


def test_bootstrap_closure_ci(planted):
    pairs = [{"name": "p", "a": "A", "b": "B", "mz_a": "MZ_A",
              "arms": {"transplant": "A_t", "permuted": "A_p", "mzdonor": "A_t",
                       "site11": "A_p"}}]
    st = T.closure_stat(pairs, ("transplant", "permuted", "mzdonor", "site11"))
    ci = T.bootstrap(planted, st, n_boot=50)
    assert ci["mean_closure"]["transplant"]["lo"] > 0.5
    pr = T.predictions(ci)
    assert pr["i_transplant_gt_0"] and pr["ii_permuted_le_0"]
    assert pr["iii_mzdonor_within_0.10"] and pr["iv_site11_below"]


def test_offsets_move_a_to_b():
    rng = np.random.default_rng(0)
    pa, pb = rng.normal(3, 2, 100), rng.normal(-1, 5, 100)
    mu_a, sa = T.site_stats(pa)
    za, zb = T.zscore(pa, mu_a, sa), T.zscore(pb, *T.site_stats(pb))
    new = pa + T.transplant_coeffs(za, zb, sa)
    np.testing.assert_allclose(T.zscore(new, mu_a, sa), zb)
    assert abs(T.transplant_coeffs(za, zb, sa).mean()) < 1e-9


def test_permuted_is_018_derangement():
    z = np.arange(500.0)
    p = T.permuted(z)
    assert np.all(p != z)
    np.testing.assert_array_equal(p, z[transfer.derangement(500, 2028)])
    assert sorted(p) == sorted(z)


def test_gate_verdict():
    ci = {"point": 0.06, "lo": 0.01, "hi": 0.1}
    mono = dict(zip(T.DOSES, [0.9, 0.8, 0.7, 0.65, 0.6, 0.5, 0.3], strict=True))
    assert T.gate_verdict(mono, ci)["pass"]
    bump = mono | {1.0: 0.61}
    assert not T.gate_verdict(bump, ci)["pass"]
    assert not T.gate_verdict(mono, ci | {"point": 0.04})["pass"]
    assert not T.gate_verdict(mono, ci | {"lo": -0.01})["pass"]


def test_dose_stat():
    rng = np.random.default_rng(3)
    ys = {T.dose_key(k): sim(np.full(200, 0.5 - 0.1 * k), rng) for k in T.DOSES}
    out = T.dose_stat()(ys)
    assert out["gate_diff"] == pytest.approx(0.1, abs=0.03)
    assert T.dose_key(0.0) == "dose_+0" and T.dose_key(-0.5) == "dose_-0.5"


def test_prompt_batches_cover_and_fit():
    lengths = [int(x) for x in np.random.default_rng(0).integers(20, 1500, 400)]
    b = TP.prompt_batches(lengths, 20, 600_000, 1280)
    assert sorted(i for g in b for i in g) == list(range(400))
    for g in b:
        assert len(g) * 20 <= 1280
        assert len(g) == 1 or len(g) * 20 * (max(lengths[i] for i in g) + 512) <= 600_000


@pytest.fixture(scope="module")
def tiny():
    torch = pytest.importorskip("torch")
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
                      pad_token_id=0, eos_token_id=63)
    return LlamaForCausalLM(cfg).eval()


def test_hook_positions_and_identity(tiny):
    import torch

    from castor import directions as Dn

    layer, d = 2, np.random.default_rng(0).standard_normal(32)
    d = Dn.unit(d)
    ids = [[5, 6, 7, 8, 9, 10], [11, 12, 13, 14]]
    x = Dn.batch_inputs(ids, 0, "cpu")
    blk = Dn.blocks(tiny)[layer]
    seen = {}

    def cap(module, args, kwargs):
        seen["x"] = (args[0] if args else kwargs["hidden_states"]).clone()

    def run(c, pos):
        h = T.Addition(d, pos)
        h.coeff = torch.tensor(c, dtype=torch.float32)
        with torch.no_grad(), Dn.hooks([(blk, h), (blk, cap)]):
            tiny(**x, use_cache=False)
        return seen["x"].double().numpy(), h

    base, _ = run([0.0, 0.0], -1)
    for pos in (-1, -2):
        edit, h = run([2.0, -3.0], pos)
        diff = edit - base
        np.testing.assert_allclose(diff[:, :pos], 0, atol=1e-6)  # nothing before the site
        np.testing.assert_allclose(diff[:, pos:] @ d, [[2.0] * -pos, [-3.0] * -pos], atol=1e-5)
        assert h.calls == {"prefill": 1, "decode": 0}


def test_hook_on_generate(tiny):
    import torch
    from transformers import GenerationConfig

    from castor import directions as Dn

    d = Dn.unit(np.random.default_rng(1).standard_normal(32))
    ids = [[5, 6, 7, 8, 9, 10], [11, 12, 13, 14]]
    x = Dn.batch_inputs(ids, 0, "cpu")
    g = GenerationConfig(do_sample=False, max_new_tokens=6, min_new_tokens=6, eos_token_id=63,
                         pad_token_id=0)

    def gen(c):
        h = T.Addition(d, -1)
        h.coeff = torch.tensor(c, dtype=torch.float32)
        with torch.no_grad(), Dn.hooks([(Dn.blocks(tiny)[1], h)]):
            out = tiny.generate(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                                generation_config=g)
        return out, h

    with torch.no_grad():
        plain = tiny.generate(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                              generation_config=g)
    zero, h = gen([0.0, 0.0])
    assert torch.equal(plain, zero)  # the hook at coefficient 0 changes nothing
    assert h.calls == {"prefill": 1, "decode": 5}
    big, _ = gen([50.0, 50.0])
    assert not torch.equal(plain, big)


def test_read_parts_drops_torn_line(tmp_path):
    p = tmp_path / "parts.jsonl"
    p.write_text('{"batch": 0, "prompts": [1, 0], "res": [[[[5], "stop"]], [[[6], "length"]]]}\n'
                 '{"batch": 1, "prom')
    done = TP.read_parts(p)
    assert list(done) == [0] and done[0]["prompts"] == [1, 0]
    assert TP.read_parts(tmp_path / "missing.jsonl") == {}


def test_hook_reads_cache_position(tiny):
    import torch
    from transformers import DynamicCache

    from castor import directions as Dn

    h = T.Addition(Dn.unit(np.random.default_rng(2).standard_normal(32)), -1)
    h.coeff = torch.tensor([1.0], dtype=torch.float32)
    ids = torch.tensor([[5, 6, 7, 8]])
    with torch.no_grad(), Dn.hooks([(Dn.blocks(tiny)[1], h)]):
        cache = DynamicCache()
        tiny(input_ids=ids[:, :3], past_key_values=cache, use_cache=True)
        tiny(input_ids=ids[:, 3:], past_key_values=cache, use_cache=True)
        assert h.calls == {"prefill": 1, "decode": 1}
        with pytest.raises(AssertionError, match="decode step with 2 positions"):
            tiny(input_ids=torch.tensor([[9, 10]]), past_key_values=cache, use_cache=True)


def test_replication_pairs_091():
    ps = TP.replication_pairs()
    assert len(ps) == 12 and len({p["name"] for p in ps}) == 12
    by = {p["name"]: p for p in ps}
    assert sum(p["group"] == "089" for p in ps) == 3
    assert [p["group"] for p in ps].count("reversed") == 3
    # 089's pairs read 089's arms; the reversed pairs swap A and B at seed 1
    p = by["default<-large-aux-2"]
    assert p["arms"]["transplant"] == (f"{TP.mid('default')}/089/transplant_"
                                       f"{TP.mid('large-aux-2')}")
    r = by["large-aux-2<-default"]
    assert (r["a"], r["b"], r["mz_a"]) == (TP.mid("large-aux-2"), TP.mid("default"),
                                           TP.mid("large-aux-2", 2))
    assert r["arms"]["transplant"].split("/")[1] == "091"
    assert r["arms"]["identity"] == f"{TP.mid('large-aux-2')}/089/dose_+0"
    # seeds 3 and 4: A and B at the same seed, A_MZ the other one, identity under 091
    for s, mz in ((3, 4), (4, 3)):
        q = by[f"large-aux-3<-default@seed{s}"]
        assert (q["a"], q["b"], q["mz_a"]) == (TP.mid("large-aux-3", s), TP.mid("default", s),
                                               TP.mid("large-aux-3", mz))
        assert q["arms"]["identity"] == f"{TP.mid('large-aux-3', s)}/091/dose_+0"
    assert {(p["a"], p["b"]) for p in ps if p["group"] == "reversed"} == {
        (TP.mid(b), TP.mid(a)) for a, b in TP.PAIRS}


def test_group_means_and_verdict():
    ps = TP.replication_pairs()
    val = {"089": 0.48, "reversed": 0.40, "seed3": 0.50, "seed4": 0.60}
    per = {p["name"]: {"closure_transplant": val[p["group"]]} for p in ps}
    g = TP.group_means(per, ps)
    assert g["new9"] == pytest.approx(0.5) and g["other_seed6"] == pytest.approx(0.55)
    assert g["all12"] == pytest.approx(0.495) and g["new9_minus_089"] == pytest.approx(0.02)
    ci = {"groups": {k: {"point": v, "lo": v - 0.05} for k, v in g.items()},
          "identity_vs_record": {"m": {"point": 0.97}}}
    v = TP.replication_verdict(ci)
    assert v["primary"] and v["reversed_within"] and not v["seed4_within"]
    assert v["identity_ge_0.95"] == {"m": True}
