import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import colocate_025 as C


def _copy(t10: float, s: float) -> dict[int, dict]:
    """Steps 1..50 at a constant s/step, step 10 at t10."""
    return {k: {"t": t10 + (k - 10) * s, "t1": t10 + (k - 10) * s, "loss": 1.0} for k in range(1, 51)}


def test_stages_split_and_stamps():
    line = ("1790948400.250000 \r 10%| 5/50 [00:01<00:02]10/03/2026 01:00:00 - INFO - __main__ -   "
            "Step: {s}, LR: 1e-05, Loss: {l}, TPS: 1.0")
    text = "\n".join([line.format(s=1, l=2.5), "noise", line.format(s=2, l=2.0),
                      line.format(s=1, l=0.69)])
    st = C.stages(text)
    assert len(st) == 2 and sorted(st[0]) == [1, 2] and st[1][1]["loss"] == 0.69
    assert st[0][1]["t"] == 1790948400.25


def test_read_log_keeps_carriage_returns(tmp_path):
    f = tmp_path / "c1.log"
    f.write_bytes(b"1790948400.250000 \r 2%| 1/50 [00:01]10/03/2026 01:00:00 - INFO - __main__ -   "
                  b"Step: 1, LR: 1e-05, Loss: 2.5, TPS: 1.0\n")
    assert C.stages(C.read_log(f))[0][1]["t"] == 1790948400.25


def test_aggregate_full_overlap_two_copies():
    # two copies each at 2 s/step, fully overlapping; single job 1 s/step -> 1.0x
    r = C.aggregate([_copy(100, 2.0), _copy(100, 2.0)], 1.0)
    assert math.isclose(r["aggregate_x"], 1.0)
    assert math.isclose(r["overlap_aggregate_x"], 1.0)
    assert r["overlap_fraction_per_copy"] == [1.0, 1.0]


def test_aggregate_staggered_overlap_uses_shared_window():
    # copy 2 starts 40 s later; per-copy rates look like 2x, the window is 40 s
    r = C.aggregate([_copy(100, 2.0), _copy(140, 2.0)], 2.0)
    assert math.isclose(r["aggregate_x"], 2.0)
    assert math.isclose(r["overlap_window_s"], 40.0)
    assert math.isclose(r["overlap_aggregate_x"], 2.0)  # both progress 20 steps in 40 s
    assert math.isclose(r["rule_x"], 2.0)


def test_aggregate_short_overlap_gives_no_credit():
    r = C.aggregate([_copy(100, 1.0), _copy(135, 1.0)], 1.0)
    assert r["overlap_window_s"] == 5.0 and r["overlap_aggregate_x"] is None and r["rule_x"] is None


def test_steps_done_interpolates():
    c = _copy(0, 2.0)
    assert C.steps_done(c, 3.0) == 11.5
    assert C.steps_done(c, -5) == 10.0 and C.steps_done(c, 1e9) == 50.0


def test_projection_rule():
    p = C.projection(2.0, None)
    assert math.isclose(p["cost_h"], 17.04 / 2 + 12.74 + 1.77) and not p["fires"]
    assert not p["x_dpo_measured"]
    p = C.projection(2.0, 2.0)
    assert math.isclose(p["cost_h"], 29.78 / 2 + 1.77) and p["fires"]
    assert math.isclose(C.BOUND, 0.7 * 31.57)
    assert C.projection(None, 2.0)["fires"] is False


def test_peak_mem_and_transport(tmp_path):
    f = tmp_path / "smi.csv"
    f.write_text("2026/10/03 01:00:00.000, 0, 1000 MiB, 5 %\n2026/10/03 01:00:01.000, 0, 3000 MiB, 9 %\n"
                 "2026/10/03 01:00:01.000, 1, 2000 MiB, 9 %\nNo devices\n")
    assert C.peak_mem(f) == {0: 3000.0, 1: 2000.0}
    t = ("x NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read\n"
         "x NCCL INFO Channel 01/0 : 1[1] -> 2[2] via P2P/CUMEM/read\n"
         "x NCCL INFO Channel 00/0 : 0[0] -> 1[1] [send] via SHM/direct/direct\n")
    assert C.transports(t) == {"P2P/CUMEM/read": 2, "SHM/direct/direct": 1}
