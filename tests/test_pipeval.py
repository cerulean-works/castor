import json

import numpy as np
import pytest

from castor import pipeval


def _labels(p: dict[str, float], k: int = 20):
    """Deterministic labels: round(p*k) non-refusals per prompt."""
    out = []
    for pid, v in p.items():
        m = round(v * k)
        out += [(pid, False)] * m + [(pid, True)] * (k - m)
    return out


def test_per_prompt_counts_and_unparsed():
    pp = pipeval.per_prompt([("a", True), ("a", False), ("a", None), ("b", True)])
    assert pp["a"] == {"n": 2, "n_refusal": 1, "n_unparsed": 1, "p_nonrefusal": 0.5}
    assert pp["b"]["p_nonrefusal"] == 0.0


def test_all_unparsed_is_nan_and_ignored():
    pp = pipeval.per_prompt([("a", None)])
    assert np.isnan(pp["a"]["p_nonrefusal"])
    assert np.isnan(pipeval.refusal_rate(pp, ["a"]))


def test_pearson_ignores_nan_and_constant():
    x = np.array([0.0, 0.5, 1.0, np.nan])
    assert abs(pipeval.pearson(x, np.array([0.1, 0.6, 1.1, 0.3])) - 1.0) < 1e-12
    assert np.isnan(pipeval.pearson(np.ones(4), np.arange(4.0)))


def _panel(n: int = 60):
    rows = [{"id": f"h{i}", "source": "hb" if i % 2 else "wj", "is_harmful": True}
            for i in range(n)]
    rows += [{"id": f"b{i}", "source": "xs", "is_harmful": False} for i in range(10)]
    return rows


def test_validate_identical_models_is_go():
    panel = _panel()
    rng = np.random.default_rng(1)
    p = {r["id"]: float(rng.uniform()) for r in panel}
    pp = pipeval.per_prompt(_labels(p))
    res = pipeval.validate(pp, pp, pp, panel, n_boot=200)
    assert res["verdict"] == "GO"
    assert abs(res["r_pipe"] - 1) < 1e-12 and abs(res["diff"]) < 1e-12
    assert set(res["sources"]) == {"hb", "wj", "xs"}


def test_validate_fails_on_rate_shift_and_on_unrelated_reference():
    panel = _panel()
    rng = np.random.default_rng(2)
    p = {r["id"]: float(rng.uniform(0.2, 0.8)) for r in panel}
    ours = pipeval.per_prompt(_labels(p))
    shifted = pipeval.per_prompt(_labels({k: min(1.0, v + 0.1) for k, v in p.items()}))
    res = pipeval.validate(ours, ours, shifted, panel, n_boot=100)
    assert res["go_r"] and not res["go_rates"] and res["verdict"] == "FAIL"
    # the rate is refusal share: ref refuses 10 points less on every source
    assert all(abs(v["diff_seed1_ref"] - 0.1) < 0.051 for v in res["sources"].values())
    unrelated = pipeval.per_prompt(_labels({k: float(rng.uniform()) for k in p}))
    res2 = pipeval.validate(ours, ours, unrelated, panel, n_boot=100)
    assert not res2["go_r"]


def test_benign_prompts_excluded_from_correlations():
    panel = _panel()
    p = {r["id"]: (0.3 if r["is_harmful"] else 0.9) for r in panel}
    p.update({f"h{i}": i / 60 for i in range(60)})
    pp = pipeval.per_prompt(_labels(p))
    res = pipeval.validate(pp, pp, pp, panel, n_boot=50)
    assert res["n_harmful"] == 60


def test_refusals_from_labels_drops_greedy_and_maps_none():
    rows = [{"prompt_id": "a", "sample": 0, "v": 1}, {"prompt_id": "a", "sample": 1, "v": 0},
            {"prompt_id": "a", "sample": 2, "v": None}, {"prompt_id": "a", "sample": -1, "v": 1}]
    got = pipeval.refusals_from_labels(rows, "v")
    assert got == [("a", False), ("a", True), ("a", None)]
    pp = pipeval.per_prompt(got)["a"]
    assert (pp["n"], pp["n_refusal"], pp["n_unparsed"], pp["p_nonrefusal"]) == (2, 1, 1, 0.5)


def test_model_map_overrides():
    d = {"sft1": "a-sft-ours-seed1", "dpo_ai2": "b"}
    assert pipeval.model_map(d, []) == {"sft1": ("a-sft-ours-seed1", "004"), "dpo_ai2": ("b", "004")}
    m = pipeval.model_map(d, ["sft1=x-sft-ours-seed1/010"])
    assert m == {"sft1": ("x-sft-ours-seed1", "010"), "dpo_ai2": ("b", "004")}
    assert pipeval.model_map(d, ["sft1=x-sft-ours-seed1"])["sft1"] == ("x-sft-ours-seed1", "004")
    for bad in ("sfft1=x/010", "sft1=", "sft1", "sft1=/010", "sft1=x-sft-ours-seed1/",
                "sft1=allenai/OLMo-2-0425-1B-SFT", "sft1=x-sft-ours-seed1/010/y"):
        with pytest.raises(ValueError):
            pipeval.model_map(d, [bad])


def test_model_map_rejects_repeated_key():
    with pytest.raises(ValueError):
        pipeval.model_map(pipeval.MODELS, ["dpo1=x-dpo-ours-seed1/010", "dpo1=x-dpo-ours-seed2/010"])


OURS_010 = [f"{k}=olmo2-1b-ingr3-{k[:3]}-ours-seed{k[3]}/010" for k in pipeval.OURS]


def test_model_map_all_four_twins_move_together():
    m = pipeval.model_map(pipeval.MODELS, OURS_010)
    assert {m[k] for k in pipeval.OURS} == {(f"olmo2-1b-ingr3-{k[:3]}-ours-seed{k[3]}", "010")
                                            for k in pipeval.OURS}
    assert m["dpo_ai2"] == ("olmo2-1b-dpo-ai2", "004")
    for i in range(4):  # leaving out any one override mixes parents within the ours arms
        with pytest.raises(ValueError, match="mix"):
            pipeval.model_map(pipeval.MODELS, OURS_010[:i] + OURS_010[i + 1:])
    with pytest.raises(ValueError, match="mix"):  # same lineage, other prompt run
        pipeval.model_map(pipeval.MODELS, OURS_010[:3] + ["dpo2=olmo2-1b-ingr3-dpo-ours-seed2/011"])


def test_model_map_key_must_match_stage_and_seed():
    for bad in ("dpo1=x-dpo-ours-seed2/010", "dpo1=x-sft-ours-seed1/010", "sft1=olmo2-1b-sft-ai2"):
        with pytest.raises(ValueError, match="expected"):
            pipeval.model_map(pipeval.MODELS, [bad])


def _run(root, model, prompt="004", sampling_k=20, wg_rev="r1", sha="s"):
    d = root / model / prompt
    (d / "judge_v1").mkdir(parents=True)
    (d / "provenance.json").write_text(json.dumps({
        "chat_template": "tulu+bos", "sampling": {"k": sampling_k}, "seed_rule": "x",
        "eval_manifest": {"sha256": sha}, "git_commit": model}))
    for stage in ("wildguard", "qwen3guard", "coherence", "labels"):
        meta = {"judge_version": "judge_v1", "git_commit": model, "n": len(model),
                "command": model, "generations": [model], "throughput": {"s": len(model)}}
        if stage == "wildguard":
            meta["revision"] = wg_rev
        (d / "judge_v1" / f"{stage}.meta.json").write_text(json.dumps(meta))


def test_check_instrument(tmp_path):
    _run(tmp_path, "a")
    _run(tmp_path, "bb", "010")  # run-specific fields differ; the instrument does not
    models = {"x": ("a", "004"), "y": ("bb", "010")}
    got = pipeval.check_instrument(models, "s", tmp_path)
    assert got["sampling"] == {"k": 20} and "git_commit" not in got["wildguard"]
    with pytest.raises(ValueError, match="eval panel"):
        pipeval.check_instrument(models, "other", tmp_path)
    _run(tmp_path, "c", sampling_k=10)
    _run(tmp_path, "d", wg_rev="r2")
    for other in ("c", "d"):
        with pytest.raises(ValueError, match="different instruments"):
            pipeval.check_instrument(models | {"z": (other, "004")}, "s", tmp_path)
