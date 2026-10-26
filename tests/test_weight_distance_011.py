import importlib.util
import json
import math
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "weight_distance", Path(__file__).parents[1] / "shell/olmo/weight_distance.py")
wd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wd)


def test_compare_known_values():
    a = {"w": torch.tensor([1.0, 0.0]), "v": torch.tensor([[0.0, 2.0]])}
    b = {"w": torch.tensor([0.0, 1.0]), "v": torch.tensor([[0.0, 2.0]])}
    r = wd.compare(a, b)
    assert math.isclose(r["rel_l2"], math.sqrt(2 / 5))
    assert math.isclose(r["cos"], 4 / 5)
    assert r["tensor_cos"] == {"v": 1.0, "w": 0.0}
    assert r["tensor_cos_mean"] == 0.5 and r["tensor_cos_min"] == 0.0


def test_compare_identical_and_orthogonal_scale():
    g = torch.Generator().manual_seed(0)
    a = {k: torch.randn(4096, generator=g) for k in "xyz"}
    b = {k: torch.randn(4096, generator=g) for k in "xyz"}
    assert wd.compare(a, a)["rel_l2"] == 0.0
    assert math.isclose(wd.compare(a, a)["cos"], 1.0)
    assert abs(wd.compare(a, b)["rel_l2"] - math.sqrt(2)) < 0.05  # independent draws: ≈ √2


def test_released_steps():
    assert len(wd.RELEASED_1B_STEPS) == 29
    assert wd.RELEASED_1B_STEPS[:3] == [0, 2500, 5000] and wd.RELEASED_1B_STEPS[-2:] == [67500, 69369]


def test_write_tsv_sorted_by_step(tmp_path):
    d = tmp_path / "divergence"
    d.mkdir()
    m = {"rel_l2": 1.0, "cos": 0.5, "tensor_cos_mean": 0.4, "tensor_cos_median": 0.4,
         "tensor_cos_min": 0.1, "tensor_cos": {}}
    for s in (10000, 2500):
        (d / f"step{s}.json").write_text(json.dumps(
            {"step": s, "git_commit": "abc", "pairs": {"ours~default": m}}))
    wd.write_tsv(d, tmp_path / "divergence.tsv")
    lines = (tmp_path / "divergence.tsv").read_text().splitlines()
    assert lines[0].split("\t")[:3] == ["step", "pair", "rel_l2"]
    assert [x.split("\t")[0] for x in lines[1:]] == ["2500", "10000"]


def test_trajectory_peer_pair(tmp_path, monkeypatch):
    g = torch.Generator().manual_seed(1)
    sd = {n: {"w": torch.randn(64, generator=g)} for n in ("r2", "r1", "default")}
    for n in ("r2", "r1"):
        d = tmp_path / n / "ckpt" / "step2500-unsharded"
        d.mkdir(parents=True)
        torch.save(sd[n], d / "model.pt")

    class Api:
        def repo_info(self, repo, revision):
            return type("I", (), {"sha": "abc"})()
    monkeypatch.setattr(wd, "HfApi", Api)
    monkeypatch.setattr(wd, "git_commit", lambda: "c0ffee")
    real = wd.load
    monkeypatch.setattr(wd, "load", lambda s: sd["default"] if "@" in s else real(s))
    a = wd.argparse.Namespace(run=str(tmp_path / "r2"), peer=[f"r1={tmp_path / 'r1'}"],
                              refs=["default"], steps=[2500], repo="x/y", outdir=None)
    wd.trajectory(a)
    res = json.loads((tmp_path / "r2" / "divergence" / "step2500.json").read_text())
    assert set(res["pairs"]) == {"ours~r1", "ours~default"}
    assert math.isclose(res["pairs"]["ours~r1"]["rel_l2"], wd.compare(sd["r2"], sd["r1"])["rel_l2"])
    assert res["ckpts"]["r1"].endswith("r1/ckpt/step2500-unsharded/model.pt")
