"""docs/compute_budget.py: the size-ladder projection and tier (a)'s block coverage."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "compute_budget", Path(__file__).resolve().parent.parent / "docs/compute_budget.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)


def test_ladder_projected_formula():
    # (SFT / 3.124 + DPO + start-up) x 8 + 1.83
    assert cb.ladder_projected("150M", 3.124) == pytest.approx((4.63 / 3.124 + 0.43 + 0.03) * 8 + 1.83)
    assert cb.ladder_projected("300M", 3.124) == pytest.approx((5.93 / 3.124 + 0.68 + 0.03) * 8 + 1.83)


def test_ladder_projected_without_speedup_reproduces_016():
    # at speed-up 1 the projection is the measured old-configuration cost (42.6 / 54.9)
    assert cb.ladder_projected("150M", 1.0) == pytest.approx(cb.POST_150M_ZERO3, abs=0.1)
    assert cb.ladder_projected("300M", 1.0) == pytest.approx(cb.POST_300M_ZERO3, abs=0.1)


def test_ladder_projected_is_cheaper_but_not_by_the_full_speedup():
    sp = 3.124
    for size, old in (("150M", cb.POST_150M_ZERO3), ("300M", cb.POST_300M_ZERO3)):
        new = cb.ladder_projected(size, sp)
        assert old / sp < new < old


def test_every_recorded_prompt_has_a_spent_block():
    assigned = {x for _, p, _, _ in cb.SPENT_BLOCKS for x in p}
    assert set(cb.SPENT_RECORDED) <= assigned
