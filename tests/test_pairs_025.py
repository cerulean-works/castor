import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("pairs_025", Path(__file__).parents[1] / "shell/olmo/pairs_025.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def test_bitwise_counts_and_max():
    a = {"w": torch.tensor([1.0, 2.0, 3.0]), "v": torch.tensor([0.5])}
    b = {"w": torch.tensor([1.0, 2.5, 3.0]), "v": torch.tensor([0.5])}
    r = p.bitwise(a, b)
    assert (r["n_unequal"], r["n_tensors"], r["max_abs"], r["n_unequal_elements"]) == (1, 2, 0.5, 1)
    assert r["unequal_tensors"] == ["w"]


def test_bitwise_one_ulp_and_negative_zero():
    x = torch.tensor([1.0])
    a, b = {"w": x}, {"w": torch.nextafter(x, torch.tensor([2.0]))}
    r = p.bitwise(a, b)
    assert r["n_unequal"] == 1 and 0 < r["max_abs"] < 1e-6
    # bits, not values: -0.0 and 0.0 differ
    assert p.bitwise({"w": torch.tensor([0.0])}, {"w": torch.tensor([-0.0])})["n_unequal"] == 1


def test_identical_is_zero():
    a = {"w": torch.randn(4, 4)}
    assert p.bitwise(a, {"w": a["w"].clone()}) | {"unequal_tensors": None} == {
        "n_tensors": 1, "n_unequal": 0, "max_abs": 0.0, "n_unequal_elements": 0, "n_elements": 16,
        "unequal_tensors": None}


def test_plan_pairs():
    pl = p.plan(["P1", "P2", "P3"])
    assert (1, "025-P1-s10", "025-P2-s10", "P1~P2") in pl
    assert (100, "025-P1-l100", "025-P3-l100", "P1~P3") in pl
    assert (10, "025-P2-s10", "025-P2-l100", "P2:s10~l100") in pl
    assert len(pl) == 3 * 11 + 3 * 11 + 3 * 2
