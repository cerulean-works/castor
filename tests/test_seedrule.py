import math

import pandas as pd
import pytest

from castor.seedrule import passes, pooled_sd, z_score


def test_pooled_sd_matches_hand_computation():
    df = pd.DataFrame({"recipe": ["a"] * 3 + ["b"] * 3,
                       "value": [1.0, 2.0, 3.0, 10.0, 10.0, 13.0]})
    # SS_a = 2, SS_b = 6; dof = 4
    sd, dof = pooled_sd(df)
    assert dof == 4
    assert sd == pytest.approx(math.sqrt(8 / 4))


def test_pooled_sd_equal_sizes_is_rms_of_group_sds():
    df = pd.DataFrame({"recipe": list("aaabbbccc"),
                       "value": [0.1, 0.3, 0.2, 5, 5.5, 4.0, -1, -1.2, -0.7]})
    sds = df.groupby("recipe").value.std(ddof=1)
    sd, dof = pooled_sd(df)
    assert dof == 6
    assert sd == pytest.approx(math.sqrt((sds**2).mean()))


def test_pooled_sd_rejects_singletons_only():
    with pytest.raises(ValueError):
        pooled_sd(pd.DataFrame({"recipe": ["a", "b"], "value": [1.0, 2.0]}))


def test_z_score_uses_sqrt_4_over_3():
    z = z_score(1.0 + 2 * math.sqrt(4 / 3) * 0.5, [0.5, 1.0, 1.5], 0.5)
    assert z == pytest.approx(2.0)
    assert passes(z)
    assert not passes(z + 1e-9)


def test_z_score_needs_three_released():
    with pytest.raises(ValueError):
        z_score(1.0, [1.0, 2.0], 1.0)
