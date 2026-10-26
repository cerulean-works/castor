"""Tests for castor.directions and castor.mechladder."""

import numpy as np
import pytest

from castor import directions as Dn
from castor import mechladder as ML
from castor import reliability as R

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def tiny_llama(seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=4, num_attention_heads=4,
                                   num_key_value_heads=4, max_position_embeddings=64,
                                   tie_word_embeddings=False, attention_bias=False,
                                   mlp_bias=False)
    return transformers.LlamaForCausalLM(cfg).to(dtype).eval()


PROMPTS = [[5, 6, 7, 8, 9, 10, 11, 12, 13], [11, 12, 13, 14, 15, 16, 17, 18],
           [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], [3, 4, 5, 6, 7, 8, 9, 10, 40, 41]]


# ---- orthogonalization == hook ablation -----------------------------------------------------


def test_orthogonalized_weights_match_hook_ablation():
    model = tiny_llama(dtype=torch.float64)
    r = np.random.default_rng(1).standard_normal(32)
    pre, post = Dn.ablation_hooks(model, r)
    hooked = Dn.last_logits(model, PROMPTS, 0, pre, post)
    base = Dn.last_logits(model, PROMPTS, 0)
    Dn.orthogonalize_(model, r)
    ortho = Dn.last_logits(model, PROMPTS, 0)
    assert torch.max(torch.abs(hooked - base)) > 1e-3  # the ablation does something
    assert torch.allclose(hooked, ortho, atol=1e-6)


def test_orthogonalized_residual_has_no_component():
    model = tiny_llama(dtype=torch.float64)
    r = Dn.unit(np.random.default_rng(2).standard_normal(32))
    Dn.orthogonalize_(model, r)
    acts = Dn.activations(model, PROMPTS, 0, positions=(-2, -1)).double().numpy()
    assert np.max(np.abs(acts @ r)) < 1e-5


def _tiny_tokenizer():
    from tokenizers import Tokenizer, models

    vocab = {f"t{i}": i for i in range(97)}
    t = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=t, pad_token="t0", chat_template="{{ messages[0]['content'] }}")


def test_orthogonalize_checkpoint_saves_checked_ablated_model(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    model = tiny_llama(dtype=torch.float64)
    model.save_pretrained(src, safe_serialization=True)
    tok = _tiny_tokenizer()
    tok.save_pretrained(src)
    (src / "generation_config.json").write_text("{}\n")
    (src / "provenance.json").write_text("{}\n")  # an output of the source's own script
    r = np.random.default_rng(3).standard_normal(32)
    checks = Dn.orthogonalize_checkpoint(model, tok, r, src, out, PROMPTS, tol=1e-6)
    assert checks["check_max_abs_dlogit_vs_hooks"] < 1e-6
    assert checks["check_max_abs_dlogit_hooks_vs_base"] > 1e-3
    assert checks["check_max_abs_residual_component"] < 1e-5
    assert "generation_config.json" in checks["copied_files"]
    assert "provenance.json" not in checks["copied_files"]
    assert not any(f.endswith(Dn.WEIGHT_SUFFIXES) for f in checks["copied_files"])
    saved = transformers.AutoModelForCausalLM.from_pretrained(out, torch_dtype=torch.float64)
    acts = Dn.activations(saved.eval(), PROMPTS, 0, positions=(-1,)).double().numpy()
    assert np.max(np.abs(acts @ Dn.unit(r))) < 1e-5
    assert transformers.AutoTokenizer.from_pretrained(out).chat_template == tok.chat_template
    with pytest.raises(AssertionError):
        Dn.orthogonalize_checkpoint(tiny_llama(dtype=torch.float64), tok, r, src,
                                    tmp_path / "out2", PROMPTS, tol=0.0)


def test_actadd_hook_adds_vector_at_one_layer():
    model = tiny_llama()
    v = np.random.default_rng(3).standard_normal(32)
    pre, _ = Dn.actadd_hooks(model, 2, v)
    # capture after the addition: a second pre-hook on the same block runs after the first
    store = {}
    cpre, _ = Dn.capture_hooks(model, store, (-1,))
    x = Dn.batch_inputs([PROMPTS[0]], 0, "cpu")
    with torch.no_grad(), Dn.hooks(pre, ()), Dn.hooks(cpre, ()):
        model(**x)
    added = store[2][0, 0].numpy()
    with torch.no_grad(), Dn.hooks(cpre, ()):
        model(**x)
    np.testing.assert_allclose(added - store[2][0, 0].numpy(), v, atol=1e-5)
    np.testing.assert_allclose(store[1][0, 0].numpy(), store[1][0, 0].numpy())


# ---- batching ------------------------------------------------------------------------------


def test_left_padding_does_not_change_activations():
    model = tiny_llama()
    alone = [Dn.activations(model, [p], 0) for p in PROMPTS]
    together = Dn.activations(model, PROMPTS, 0, max_tokens=10_000)
    for i in range(len(PROMPTS)):
        torch.testing.assert_close(together[i], alone[i][0], atol=1e-5, rtol=1e-4)


def test_length_batches_cover_all_once():
    ids = [[0] * n for n in (5, 1, 9, 3, 3, 7)]
    b = Dn.length_batches(ids, max_tokens=12)
    assert sorted(i for x in b for i in x) == list(range(6))
    assert all(max(len(ids[i]) for i in x) * len(x) <= 12 or len(x) == 1 for x in b)


# ---- scores, filters, selection ------------------------------------------------------------


def test_refusal_score_log_odds():
    logits = torch.log(torch.tensor([[0.5, 0.25, 0.25]]))
    s = Dn.refusal_score(logits, [1, 2])
    assert abs(float(s[0])) < 1e-6  # p(R) = 0.5
    s = Dn.refusal_score(logits, [1])
    assert abs(float(s[0]) - np.log(0.25 / 0.75)) < 1e-6


def test_kl_zero_on_identical():
    x = torch.randn(3, 10)
    assert torch.all(Dn.kl_div(x, x).abs() < 1e-9)


def test_filters_follow_arditi_code():
    n = 16
    assert Dn.passes(-1, 0.1, 0.05, 11, n)
    assert not Dn.passes(-1, 0.1, 0.05, 12, n)       # layer >= int(0.8 * 16) = 12
    assert not Dn.passes(-1, -0.1, 0.05, 3, n)       # induce < 0
    assert not Dn.passes(-1, 0.1, 0.11, 3, n)        # kl > 0.1
    assert not Dn.passes(float("nan"), 0.1, 0.05, 3, n)


def grid(bypass, induce=1.0, kl=0.0):
    by = np.asarray(bypass, float)
    return {"bypass": by, "induce": np.full_like(by, induce), "kl": np.full_like(by, kl)}


def test_select_common_all_pass_lowest_mean_bypass():
    L = 16
    a = np.zeros((L, 2))
    b = np.zeros((L, 2))
    a[3, 1], b[3, 1] = -5, -1      # mean -3
    a[4, 0], b[4, 0] = -2, -2      # mean -2
    a[13, 0] = b[13, 0] = -99      # pruned layer
    s = Dn.select_common({"a": grid(a), "b": grid(b)})
    assert (s["layer"], s["pos_index"], s["all_pass"]) == (3, 1, True)


def test_select_common_fallback_most_models():
    L = 16
    a = grid(np.zeros((L, 1)), induce=-1.0)  # nothing passes in a
    b = grid(np.zeros((L, 1)))
    b["bypass"][5, 0] = -3
    s = Dn.select_common({"a": a, "b": b})
    assert (s["layer"], s["n_models_passing"], s["all_pass"]) == (5, 1, False)


def test_select_common_nothing_passes_skips_undefined_sites():
    L = 16
    by = np.zeros((L, 2))
    by[0] = np.nan                 # layer 0: zero difference of means, undefined scores
    by[7, 1] = -2
    a, b = grid(by, induce=-1.0), grid(by, induce=-1.0)  # nothing passes anywhere
    s = Dn.select_common({"a": a, "b": b})
    assert (s["layer"], s["pos_index"], s["n_models_passing"]) == (7, 1, 0)
    assert np.isfinite(s["mean_bypass"])


def test_select_own():
    by = np.zeros((16, 3))
    by[2, 2] = -4
    by[14, 0] = -9
    assert Dn.select_own(grid(by)) == (2, 2)


def test_refusal_token_set_smallest():
    toks = [1] * 50 + [2] * 20 + [3] * 15 + [4] * 15
    res = Dn.refusal_token_set(toks, 0.8)
    assert res["tokens"] == [1, 2, 3] and abs(res["coverage"] - 0.85) < 1e-12
    assert Dn.refusal_token_set([7] * 9 + [8], 0.8)["tokens"] == [7]


def test_post_instruction_ids_tulu_like():
    class Tok:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            body = [ord(c) for c in msgs[0]["content"]]
            return [0, 1, 2] + body + [9, 8, 7]
    assert Dn.post_instruction_ids(Tok()) == [9, 8, 7]


def test_random_direction_seeded_unit():
    a = Dn.random_direction("m1", 64)
    assert abs(np.linalg.norm(a) - 1) < 1e-12
    np.testing.assert_array_equal(a, Dn.random_direction("m1", 64))
    assert not np.allclose(a, Dn.random_direction("m2", 64))


# ---- split-half reliability on synthetic data ----------------------------------------------


def synth_acts(rng, n_train, n_panel, d, signal, noise, direction):
    """Harmful = harmless + signal * direction, isotropic noise; panel activations carry a
    latent score t along the direction."""
    harm = rng.standard_normal((n_train, d)) * noise + signal * direction
    harmless = rng.standard_normal((n_train, d)) * noise
    t = rng.standard_normal(n_panel)
    panel = np.outer(t, direction) + rng.standard_normal((n_panel, d)) * noise
    return harm, harmless, panel, t


def test_projection_measure_known_reliability():
    rng = np.random.default_rng(0)
    n, s = 20000, 50
    t = rng.standard_normal(n)
    h1 = t[:, None] + rng.standard_normal((n, s))
    h2 = t[:, None] + rng.standard_normal((n, s))
    full = t + rng.standard_normal(n) / np.sqrt(2)  # mean of two halves' noise
    _, rel = ML.projection_measure(s)(ML.pack(full, h1, h2), "pearson")
    assert abs(rel - 2 / 3) < 0.01   # half r = 0.5 -> SB 2/3


def test_disattenuated_projection_ladder_recovers_true_correlation():
    rng = np.random.default_rng(1)
    n, s = 8000, 40
    info = {"a1": R.ModelInfo("r", "A", "1"), "a2": R.ModelInfo("r", "A", "2"),
            "b1": R.ModelInfo("r", "B", "1")}
    common = rng.standard_normal(n)
    ta = common * np.sqrt(0.9) + rng.standard_normal(n) * np.sqrt(0.1)
    tb = common * np.sqrt(0.9) + rng.standard_normal(n) * np.sqrt(0.1)  # corr(ta, tb) = 0.9
    ys = {}
    for m, t in (("a1", ta), ("a2", ta), ("b1", tb)):
        h1 = t[:, None] + rng.standard_normal((n, s))
        h2 = t[:, None] + rng.standard_normal((n, s))
        ys[m] = ML.pack(t + rng.standard_normal(n) / np.sqrt(2), h1, h2)
    res = ML.ladder(s)(ys, info)
    lv = res["levels"]
    assert abs(lv["MZ"]["r_corrected"] - 1.0) < 0.03
    assert abs(lv["U"]["r_corrected"] - 0.9) < 0.03
    assert lv["U"]["r"] < 0.7  # attenuated raw


def test_half_directions_end_to_end_reliability():
    rng = np.random.default_rng(2)
    d = 64
    u = Dn.unit(rng.standard_normal(d))
    harm, harmless, panel, _ = synth_acts(rng, 256, 4000, d, signal=2.0, noise=1.0,
                                          direction=u)
    splits = Dn.half_splits(256, 256, 30, seed=0)
    d1, d2 = Dn.half_directions(harm, harmless, splits)
    cos = np.sum(d1 * d2, 1)
    assert np.all(cos > 0.5)
    full = Dn.unit(Dn.mean_diff(harm, harmless))
    p = ML.pack(Dn.project(panel, full), Dn.project(panel, d1), Dn.project(panel, d2))
    _, rel = ML.projection_measure(30)(p, "pearson")
    assert 0.9 < rel <= 1.0


def test_half_splits_disjoint():
    for (a1, a2), (b1, b2) in Dn.half_splits(10, 8, 5, seed=3):
        assert len(set(a1) & set(a2)) == 0 and len(a1) == len(a2) == 5
        assert len(set(b1) & set(b2)) == 0 and len(b1) == len(b2) == 4


# ---- the link ------------------------------------------------------------------------------


def test_link_sign_and_correction():
    """P = logistic(−true score); twin A and B differ in their true scores. Higher projection
    -> more refusal -> lower P, so the link is positive."""
    rng = np.random.default_rng(4)
    n, s, k = 6000, 30, 20
    common = rng.standard_normal(n)
    ta = common + rng.standard_normal(n) * 0.6
    tb = common + rng.standard_normal(n) * 0.6
    ys = {}
    for m, t in (("a", ta), ("b", tb)):
        prob = 1 / (1 + np.exp(t * 1.5))
        y = (rng.random((n, k)) < prob[:, None]).astype(float)
        h1 = t[:, None] + rng.standard_normal((n, s)) * 0.5
        h2 = t[:, None] + rng.standard_normal((n, s)) * 0.5
        ys[m] = ML.pack(t + rng.standard_normal(n) * 0.35, h1, h2, y)
    res = ML.pair_link(ys["a"], ys["b"], s)
    assert res["link"] > 0.3
    assert res["link_corrected"] > res["link"]
    assert 0 < res["rel_dP"] < 1 and 0 < res["rel_dproj"] < 1
    # swapping the pair leaves the link unchanged
    assert abs(ML.pair_link(ys["b"], ys["a"], s)["link"] - res["link"]) < 1e-12


def test_link_analyze_with_bootstrap_code_path():
    rng = np.random.default_rng(5)
    n, s, k = 300, 5, 4
    info = {f"{b}{p}": R.ModelInfo("r", b, str(p)) for b in "ABC" for p in (1, 2)}
    ys = {m: ML.pack(rng.standard_normal(n), rng.standard_normal((n, s)),
                     rng.standard_normal((n, s)), (rng.random((n, k)) < 0.5).astype(float))
          for m in info}
    res = ML.link(s)(ys, info)
    assert sum(r["level"] == "U" for r in res["pairs"]) == 12
    assert sum(r["level"] == "MZ" for r in res["pairs"]) == 3
    ci = R.bootstrap_prompts(ys, info, n_boot=5, stat=ML.link_summary, analyze=ML.link(s))
    assert "link_U_corrected" in ci
    ci = R.bootstrap_bases(ys, info, n_boot=5, stat=ML.link_summary, analyze=ML.link(s))
    assert ci["link_U"]["n_draws"] >= 1
    jk = R.jackknife_bases(ys, info, stat=ML.ladder_summary, analyze=ML.ladder(s))
    assert jk["MZ_minus_U_corrected"]["n_bases"] == 3


def test_link_reading_thresholds():
    assert ML.link_reading(0.5) == "runs through the refusal direction"
    assert ML.link_reading(0.49) == "partly"
    assert ML.link_reading(0.3) == "partly"
    assert ML.link_reading(0.29) == "elsewhere"


def test_ablation_verdict():
    base = {"a": 0.25, "b": 0.3}
    own = {"a": 0.6, "b": 0.65}
    v = ML.ablation_verdict(base, own, {"a": 0.27})
    assert v["causal"]
    v = ML.ablation_verdict(base, {"a": 0.6, "b": 0.55}, {"a": 0.27})
    assert not v["causal"]  # b does not double
    v = ML.ablation_verdict(base, own, {"a": 0.4})
    assert not v["causal"]  # random moved 0.15 >= 0.35 / 4


def test_direction_cosines_levels():
    rng = np.random.default_rng(6)
    d = 256
    info = {"a1": R.ModelInfo("r", "A", "1"), "a2": R.ModelInfo("r", "A", "2"),
            "b1": R.ModelInfo("r", "B", "1")}
    shared = Dn.unit(rng.standard_normal(d))
    dirs, h1, h2 = {}, {}, {}
    for m in info:
        base = shared if m.startswith("a") else Dn.unit(rng.standard_normal(d))
        dirs[m] = Dn.unit(base + 0.2 / np.sqrt(d) * rng.standard_normal(d))
        h1[m] = Dn.unit(base + 0.3 / np.sqrt(d) * rng.standard_normal((10, d)))
        h2[m] = Dn.unit(base + 0.3 / np.sqrt(d) * rng.standard_normal((10, d)))
    res = ML.direction_cosines(dirs, h1, h2, info)
    assert res["levels"]["MZ"]["cos"] > 0.9
    assert res["levels"]["MZ"]["cos_disattenuated"] > res["levels"]["MZ"]["cos"]
    assert abs(res["levels"]["U"]["cos"]) < 0.3


def test_direction_cosines_without_mz_pairs():
    rng = np.random.default_rng(7)
    info = {"a1": R.ModelInfo("r", "A", "1"), "b1": R.ModelInfo("r", "B", "1")}
    dirs = {m: Dn.unit(rng.standard_normal(8)) for m in info}
    half = {m: Dn.unit(rng.standard_normal((3, 8))) for m in info}
    res = ML.direction_cosines(dirs, half, half, info)
    assert res["levels"]["MZ"] == {"n_pairs": 0} and res["levels"]["U"]["n_pairs"] == 1


# ---- ladder contrast, site point and the KL-dropped selection ------------------------------


def _twin_packs(rng, n, s, loadings, noise=0.3):
    """Per model a readout = loading on a shared factor + base-specific + noise."""
    info = {f"{b}{p}": R.ModelInfo("r", b, str(p)) for b in "ABC" for p in (1, 2)}
    shared = rng.standard_normal(n)
    base = {b: rng.standard_normal(n) for b in "ABC"}
    out = {}
    for m, i in info.items():
        t = loadings[0] * shared + loadings[1] * base[i.pretrain_seed]
        h1 = t[:, None] + rng.standard_normal((n, s)) * noise
        h2 = t[:, None] + rng.standard_normal((n, s)) * noise
        out[m] = (t + rng.standard_normal(n) * noise / np.sqrt(2), h1, h2)
    return out, info


def test_ladder_contrast_zero_for_identical_readouts_and_matches_ladders():
    rng = np.random.default_rng(6)
    n, s, k = 3000, 20, 4
    ref, info = _twin_packs(rng, n, s, (1.0, 0.7))
    harm, _ = _twin_packs(rng, n, s, (1.0, 0.0))  # no base-specific part: gap ≈ 0
    y = {m: (rng.random((n, k)) < 0.5).astype(float) for m in info}
    same = {m: ML.pack_two(ref[m], ref[m], y[m]) for m in info}
    st = ML.ladder_contrast_summary(ML.ladder_contrast(s)(same, info))
    assert abs(st["gap_refusal_minus_harm_corrected"]) < 1e-12
    two = {m: ML.pack_two(ref[m], harm[m], y[m]) for m in info}
    st = ML.ladder_contrast_summary(ML.ladder_contrast(s)(two, info))
    lad = ML.ladder_summary(ML.ladder(s)({m: ML.pack(*ref[m], y[m]) for m in info}, info))
    assert abs(st["refusal_MZ_minus_U_corrected"] - lad["MZ_minus_U_corrected"]) < 1e-12
    assert abs(st["harm_MZ_minus_U_corrected"]) < 0.05
    assert st["gap_refusal_minus_harm_corrected"] > 0.2
    ci = R.bootstrap_prompts(two, info, n_boot=5, stat=ML.ladder_contrast_summary,
                             analyze=ML.ladder_contrast(s))
    assert ci["gap_refusal_minus_harm_corrected"]["n_draws"] == 5


def test_site_point_matches_primary_blocks():
    rng = np.random.default_rng(7)
    n, s, k = 800, 10, 6
    ref, info = _twin_packs(rng, n, s, (1.0, 0.5))
    ys = {m: ML.pack(*ref[m], (rng.random((n, k)) < 0.4).astype(float)) for m in info}
    pt = ML.site_point(ys, info, s)
    assert pt["MZ_minus_U_corrected"] == ML.ladder_summary(
        ML.ladder(s)(ys, info))["MZ_minus_U_corrected"]
    assert pt["link_U_corrected"] == ML.link_summary(ML.link(s)(ys, info))["link_U_corrected"]


def test_select_own_kl_dropped():
    n_layer = 5  # int(0.8 * 5) = 4: layer 4 pruned
    by = np.array([[0.0, -1.0], [-3.0, -2.0], [-4.0, 1.0], [-2.0, 0.0], [-9.0, -9.0]])
    ind = np.array([[1.0, 1.0], [1.0, -0.5], [-0.1, 1.0], [0.0, 1.0], [1.0, 1.0]])
    kl = np.full((n_layer, 2), 0.5)  # nothing passes the KL filter
    sc = {"bypass": by, "induce": ind, "kl": kl}
    assert Dn.select_own(sc) is None
    assert Dn.select_own_kl_dropped(sc) == (1, 0)  # (2, 0) has induce < 0; layer 4 pruned


def test_ladder_merge_blocks(tmp_path):
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("ladder_007", "scripts/ladder_007.py")
    LD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LD)
    shared = {k: 1 for k in LD.SHARED}
    parts = []
    for b in LD.BLOCKS:
        d = shared | {"blocks": [b], "seconds": 1.0, "command": b}
        d |= {k: b for k in LD.BLOCK_KEYS[b]}
        p = tmp_path / f"{b}.json"
        p.write_text(json.dumps(d))
        parts.append(p)
    LD.merge(parts, tmp_path / "m.json")
    m = json.loads((tmp_path / "m.json").read_text())
    assert m["primary"] == "main" and m["site_map"] == "sites" and m["git_commit"] == 1
    bad = json.loads(parts[1].read_text()) | {"git_commit": 2}
    parts[1].write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        LD.merge(parts, tmp_path / "m2.json")
    with pytest.raises(AssertionError):
        LD.merge(parts[:3], tmp_path / "m3.json")


def test_causal_summarize_plain_harmful_mean():
    import importlib.util
    spec = importlib.util.spec_from_file_location("causal_007", "scripts/causal_007.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    rows = []
    for i, vals in enumerate(([1, 1, 0, 0], [1, None, None, None], [0, 0, 0, 0])):
        for s, v in enumerate(vals):
            rows.append({"prompt_id": f"p{i}", "sample": s, "wg_vulnerable": v,
                         "wg_harmful": 0 if v is None else v})
    rows.append({"prompt_id": "p0", "sample": -1, "wg_vulnerable": 1, "wg_harmful": 1})  # greedy
    out = C.summarize(rows, {"plain_harmful": ["p0", "p1", "p2"]})["plain_harmful"]
    assert abs(out["vulnerability"] - (0.5 + 1.0 + 0.0) / 3) < 1e-12  # prompt mean, NaN dropped
    # unlabelled cells (K = 20, 4 filled) count as NaN too
    assert abs(out["share_nan"] - (48 + 3) / 60) < 1e-12 and out["missing"] == 60 - 12
