"""castor.figdata and castor.reliability.by_group: the helpers behind the appendix
figures, on synthetic data with known answers."""

import numpy as np
import pytest

from castor import figdata, reliability


def test_lower_median_pair_picks_the_middle_never_the_best():
    pairs = [{"a": f"m{i}", "b": f"n{i}", "level": "MZ", "r_corrected": v}
             for i, v in enumerate([0.9, 0.5, 0.7, 0.8])] + \
            [{"a": "u", "b": "v", "level": "U", "r_corrected": 0.1}]
    assert figdata.lower_median_pair(pairs, "MZ")["r_corrected"] == 0.7  # 4 pairs -> 2nd of 4
    assert figdata.lower_median_pair(pairs[:3], "MZ")["r_corrected"] == 0.7  # 3 -> the median
    assert figdata.lower_median_pair(pairs, "U")["a"] == "u"
    with pytest.raises(ValueError):
        figdata.lower_median_pair(pairs, "R")


def test_prompt_range_and_shares():
    ps = {"a": np.array([1.0, 0.0, 0.5, np.nan]), "b": np.array([0.95, 0.05, 0.2, 0.3]),
          "c": np.array([0.9, 0.1, 0.8, 0.3])}
    rg = figdata.prompt_range(ps)
    assert rg["n_dropped"] == 1 and rg["keep"].tolist() == [True, True, True, False]
    assert np.allclose(rg["min"], [0.9, 0.0, 0.2]) and np.allclose(rg["max"], [1.0, 0.1, 0.8])
    assert np.allclose(rg["mean"], [0.95, 0.05, 0.5])
    s = figdata.stable_shares(rg["min"], rg["max"])
    assert (s["n"], s["n_every"], s["n_none"]) == (3, 1, 1)
    assert s["share_every"] == pytest.approx(1 / 3)


def test_confusion_counts_and_refuses_unknown_labels():
    m = figdata.confusion([("R", "R"), ("R", "N"), ("H", "H"), ("R", "N")], "RHN", "RHN")
    assert m["R"] == {"R": 1, "H": 0, "N": 2} and m["H"]["H"] == 1 and sum(m["N"].values()) == 0
    with pytest.raises(ValueError):
        figdata.confusion([("X", "R")], "RHN", "RHN")


def _twins(n_prompts: int, rng, shared_base: float):
    """3 bases x 2 post-training seeds; per-prompt rate = base signal (weight shared_base) + noise."""
    info, ys = {}, {}
    common = rng.uniform(0, 1, n_prompts)
    for s in ("a", "b", "c"):
        base = np.clip(0.5 * common + 0.5 * rng.uniform(0, 1, n_prompts), 0, 1)
        for k in ("1", "2"):
            p = np.clip(shared_base * base + (1 - shared_base) * rng.uniform(0, 1, n_prompts), 0, 1)
            m = f"{s}{k}"
            ys[m] = (rng.uniform(0, 1, (n_prompts, 20)) < p[:, None]).astype(float)
            info[m] = reliability.ModelInfo("r", s, k)
    return ys, info


def test_by_group_equals_the_statistics_on_each_group_alone():
    rng = np.random.default_rng(1)
    ys, info = _twins(120, rng, 0.9)
    groups = {"first": range(50), "rest": range(50, 120)}
    res = reliability.by_group(ys, info, groups, n_boot=50, seed=0)
    for name, rows in groups.items():
        idx = np.asarray(list(rows))
        sub = {m: y[idx] for m, y in ys.items()}
        alone = reliability.bootstrap_prompts(sub, info, 50, 0)
        assert res[name]["n_prompts"] == len(idx)
        for k in ("MZ_minus_U_corrected", "r_MZ", "r_U"):
            assert res[name]["ci_prompts"][k] == alone[k]
        # the twin structure is planted: MZ twins agree more than unrelated pairs in each group
        assert res[name]["ci_prompts"]["MZ_minus_U_corrected"]["estimate"] > 0
