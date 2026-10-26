"""scripts/floor_011.py --comparator: the default is 006's twins; `a0` is 008's stage-A0 reruns
of the same bases x seeds (one configuration, labels under run id 008)."""

import json

import pytest

from castor import floor, reliability, sanityb
from castor.provenance import load_script

F = load_script("floor_011")


def test_default_comparator_is_006_unchanged():
    ms = F.comparator_models("006")
    cells = sanityb.cells()
    assert list(ms) == [c.model_id for c in cells]
    for c in cells:
        info, pid, sft = ms[c.model_id]
        assert (info, pid, sft) == (c.info, sanityb.PROMPT_ID, c.out("sft"))


def test_a0_comparator_is_the_same_cells_on_prompt_008():
    ms = F.comparator_models("a0")
    assert len(ms) == 12
    seeds_006 = sorted((c.info.pretrain_seed, c.info.posttrain_seed) for c in sanityb.cells())
    assert sorted((i.pretrain_seed, i.posttrain_seed) for i, _, _ in ms.values()) == seeds_006
    for m, (info, pid, sft) in ms.items():
        assert info.recipe == sanityb.RECIPE
        assert pid == F.A0_PROMPT_ID == "008"
        assert m.startswith("dd-dclm-baseline-1b-") and m.endswith(f"-dpo-seed{info.posttrain_seed}")
        assert sft.parts[-3:] == ("dd-dclm-baseline-1B-seed-" + info.pretrain_seed, "sft",
                                  f"seed{info.posttrain_seed}")
    assert set(ms).isdisjoint(F.comparator_models("006"))  # 006's ids never enter the A0 run


def test_a0_cells_are_the_grid_stage_not_the_whole_grid():
    cells = F.a0_cells()
    assert len(cells) == 12
    assert {c.base.repo for c in cells} == {sanityb.REPO}
    assert sorted(c.k for c in cells) == sorted(sanityb.POSTTRAIN_SEEDS * 3)


def test_unknown_comparator_refused():
    with pytest.raises(ValueError):
        F.comparator_models("009")


def _fake_floor(tmp_path, seeds):
    """A floor.json-shaped file with one pair per subset x outcome."""
    models = {f"m-{s}-{k}": {"pretrain_seed": s, "posttrain_seed_k": k} for s in seeds for k in (1, 2)}
    ids = list(models)
    pair = {"a": ids[0], "b": ids[-1], "r": 0.5, "r_corrected": 0.6}
    block = {o: {"summary": {"x": 1.0}, "ci_prompts": {"x": {"lo": 0, "hi": 1}}, "pairs": [pair]}
             for o in F.OUTCOMES}
    f = {"models": models, "prediction": {"holds": True}, "git_commit": "abc"}
    f.update({s: block for s in sanityb.SUBSETS})
    p = tmp_path / "floor.json"
    p.write_text(json.dumps(f))
    return p


def test_reference_011_keys_on_bases_x_seeds_not_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "FLOOR_011", _fake_floor(tmp_path, ("default", "replicate")))
    monkeypatch.setattr(F, "ROOT", tmp_path)
    info = {f"other-{s}-{k}": reliability.ModelInfo(sanityb.RECIPE, s, str(k))
            for s in ("default", "replicate") for k in (1, 2)}
    ref = F.reference_011(info)
    assert ref["git_commit"] == "abc" and ref["prediction"] == {"holds": True}
    bp = ref["by_base_pair"]["adv_harmful"]["primary"]
    assert list(bp) == ["default|replicate"] and bp["default|replicate"]["group"] == "rung2"
    assert bp["default|replicate"]["r_corrected"] == 0.6
    # a different set of bases is refused
    bad = {f"x-{s}-{k}": reliability.ModelInfo(sanityb.RECIPE, s, str(k))
           for s in ("default", "large-aux-2") for k in (1, 2)}
    with pytest.raises(AssertionError):
        F.reference_011(bad)


def test_by_base_pair_group_of_consistency():
    assert floor.group_of("default", "replicate") == "rung2"
