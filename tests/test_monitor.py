"""Parsers of scripts/monitor_logs.py on synthetic snippets and on a real log of each type."""

import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from castor import grid

spec = importlib.util.spec_from_file_location("mon", Path(__file__).parents[1] / "scripts/monitor_logs.py")
mon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mon)

TRAIN = (
    "10/08/2026 11:00:00 - INFO - __main__ -   Step: 9, LR: 1, Loss: 5\n"  # before any header
    "open-instruct fc7748ea4 | sft | 8 GPUs | ds b615b6daa5ea | --x\n"
    "  0%|          | 1/13342 [00:00<3:16:36,  1.13it/s]10/08/2026 11:37:49 - INFO - __main__ -   "
    "Step: 1, LR: 7.5e-08, Loss: 1066.328125, TPS: 77893.8\r  0%|          | 2/13342 [00:01<2:51:15, "
    " 1.30it/s]10/08/2026 11:37:50 - INFO - __main__ -   Step: 2, LR: 1.5e-07, Loss: nan, TPS: 93339.5\n"
    "10/08/2026 11:37:52 - INFO - __main__ -   Step: 3, LR: 2e-07, Loss: 1000, TPS: 1\n"
    "open-instruct fc7748ea4 | dpo | 8 GPUs | --x\n"
    "100%|██████████| 50/50 [01:26<00:00,  1.60s/it]09/29/2026 23:06:09 - INFO - __main__ -   "
    "Step: 50, LR: 0.0, Loss: 0.6796842217445374"
)


def test_parse_train_stages_totals_and_nan():
    got = mon.parse_train(TRAIN)
    assert list(got) == ["sft", "dpo"]
    assert got["sft"]["total"] == 13342 and got["dpo"]["total"] == 50
    assert sorted(got["sft"]["steps"]) == [1, 2, 3]
    assert got["sft"]["steps"][1] == (1066.328125, datetime(2026, 10, 8, 11, 37, 49))  # noqa: DTZ001
    assert got["sft"]["steps"][2][0] != got["sft"]["steps"][2][0]  # nan kept, dropped by binned
    assert got["dpo"]["steps"] == {50: (0.6796842217445374, datetime(2026, 9, 29, 23, 6, 9))}  # noqa: DTZ001


def test_merge_later_segment_wins():
    a = mon.parse_train(TRAIN)
    b = mon.parse_train(TRAIN.replace("Loss: 1000", "Loss: 7").replace("/13342", "/99"))
    m = mon.merge([a, b])
    assert m["sft"]["steps"][3][0] == 7 and m["sft"]["total"] == 99 and m["sft"]["steps"][1][0] == 1066.328125


def test_binned_mean_and_rate():
    steps = mon.parse_train(TRAIN)["sft"]["steps"]
    [(step, loss, rate)] = mon.binned(steps, k=1)
    assert step == 3 and loss == pytest.approx((1066.328125 + 1000) / 2)  # nan dropped
    assert rate == pytest.approx(3 / 2)  # 3 s over steps 1..3
    assert mon.binned({}, k=5) == []
    three = mon.binned(steps, k=3)
    assert [s for s, _, _ in three] == [1, 2, 3] and all(r is None for _, _, r in three)


def test_parse_gj_bars_and_done():
    text = ("Processed prompts:   0%| | 0/800 [00:00<?]Processed prompts:  2%| | 20/800 [00:05<03:24]\n"
            "noise\nProcessed prompts: 100%|█| 40/40 [00:12<00:00]\n[2026-09-30T20:32:48+08:00] done\n")
    assert mon.parse_gj(text) == {"bars": [(20, 800), (40, 40)], "done": True}
    assert mon.parse_gj("Processed prompts: 5/9 [Processed prompts: 9/9 [Processed prompts: 1/9 [")["bars"] == [(9, 9), (1, 9)]
    assert mon.parse_gj("nothing") == {"bars": [], "done": False}


def test_chains_and_job_row(tmp_path):
    for n in ("008-train-c4-1B-default-s1-10.out", "008-train-c4-1B-default-s1-9.out",
              "008-gj-c4-1B-default-s1-11.out", "008-smoke-train-c4-1B-default-s1-7.out",
              "016-train-150M-s1-12.out", "008-refill-13.out"):
        (tmp_path / n).write_text("x")
    ch = mon.chains(tmp_path, {"008", "016"})
    assert sorted(ch) == ["008-gj-c4-1B-default-s1", "008-train-c4-1B-default-s1", "016-train-150M-s1"]
    assert [p.name for p in ch["008-train-c4-1B-default-s1"]] == [
        "008-train-c4-1B-default-s1-9.out", "008-train-c4-1B-default-s1-10.out"]
    assert sorted(mon.chains(tmp_path, {"008"})) == ["008-gj-c4-1B-default-s1", "008-train-c4-1B-default-s1"]
    now = datetime.now()  # noqa: DTZ005
    logs = ch["008-train-c4-1B-default-s1"]
    r = mon.job_row("008-train-c4-1B-default-s1", logs, [{"id": "10", "state": "RUNNING", "where": "n1"}],
                    {}, now)
    assert (r["job"], r["state"], r["where"], r["segments"], r["flags"]) == ("10", "RUNNING", "n1", 2, [])
    r = mon.job_row("008-train-c4-1B-default-s1", logs, [], {"10": "TIMEOUT"}, now)
    assert r["state"] == "TIMEOUT" and r["flags"] == ["FAILED"]
    r = mon.job_row("008-train-c4-1B-default-s1", logs, [{"id": "10", "state": "RUNNING", "where": "n1"}],
                    {}, now + timedelta(seconds=grid.STALL_S + 1))
    assert r["flags"] == ["STALLED"]
    r = mon.job_row("008-train-x-s2", [], [{"id": "14", "state": "PENDING", "where": "(Priority)"}], {}, now)
    assert (r["job"], r["state"], r["flags"], r["log_age_s"]) == ("14", "PENDING", ["PENDING"], None)
    assert mon.summary(r).startswith("008-train-x-s2") and "PENDING" in mon.summary(r)


def test_train_row_eta_and_summary(tmp_path):
    log = tmp_path / "008-train-c4-1B-default-s1-10.out"
    lines = ["open-instruct fc7748ea4 | sft | 8 GPUs | --x"]
    t0 = datetime(2026, 10, 8, 11, 0, 0)  # noqa: DTZ001
    for n in range(1, 402):  # 1 s/step: rate from step 200 is 1.0 s/step
        ts = t0 + timedelta(seconds=n)
        lines.append(f"  0%| | {n}/1000 [00:00<?]{ts:%m/%d/%Y %H:%M:%S} - INFO - __main__ -   "
                     f"Step: {n}, LR: 1e-5, Loss: {1000 - n}, TPS: 1")
    log.write_text("\n".join(lines))
    now = datetime.now()  # noqa: DTZ005
    r = mon.train_row("008-train-c4-1B-default-s1", [log], [{"id": "10", "state": "RUNNING", "where": "n1"}],
                      {}, now)
    assert (r["stage"], r["step"], r["total"], r["loss"]) == ("sft", 401, 1000, 599.0)
    assert r["s_per_step"] == pytest.approx(1.0) and r["eta_h"] == pytest.approx(599 / 3600)
    assert r["flags"] == [] and len(r["stages"]["sft"]) <= mon.POINTS
    s = mon.summary(r)
    assert "sft step 401/1000 loss 599 1.00 s/step ETA 0.2 h" in s and s.endswith("10 RUNNING n1")
    log.write_text("\n".join([*lines, "open-instruct fc7748ea4 | dpo | 8 GPUs | --x"]))  # dpo header, no step yet
    assert mon.train_row("008-train-c4-1B-default-s1", [log], [], {}, now)["stage"] == "sft"
    html = mon.page([r], now, tmp_path)
    assert html.count("<polyline") == 2 and "008-train-c4-1B-default-s1" in html


REAL = Path(__file__).parents[1] / "logs/nchc"
REAL_TRAIN = REAL / "008-smoke-train-c4-1B-default-s1-457357.out"
REAL_GJ = REAL / "008-smoke-gj-c4-1B-default-s1-464023.out"


@pytest.mark.skipif(not REAL_TRAIN.exists(), reason="real smoke log not on this machine")
def test_real_train_log():
    got = mon.parse_train(REAL_TRAIN.read_text(errors="replace"))
    assert list(got) == ["sft", "dpo"]
    assert got["sft"]["total"] == 50 and got["dpo"]["total"] == 50
    assert max(got["sft"]["steps"]) == 50 and max(got["dpo"]["steps"]) == 50
    assert got["dpo"]["steps"][50][0] == pytest.approx(0.6796842217445374)
    # the same step count grid.step_rate sees (one instrument for the SLOW flag)
    assert grid.step_rate(REAL_TRAIN.read_text(errors="replace"))["step"] == 50


@pytest.mark.skipif(not REAL_GJ.exists(), reason="real smoke log not on this machine")
def test_real_gj_log():
    got = mon.parse_gj(REAL_GJ.read_text(errors="replace"))
    assert got["done"] and len(got["bars"]) == 5  # sampled, greedy, three judge stages
    assert got["bars"][0] == (800, 800) and got["bars"][1] == (40, 40)
    assert all(n == d for n, d in got["bars"])


def _train_log(path: Path, start: int, stop: int, total: int, t0: datetime, dpo: bool = False) -> None:
    lines = ["open-instruct fc7748ea4 | sft | 8 GPUs | --x"]
    for n in range(start, stop + 1):  # 2 s/step
        ts = t0 + timedelta(seconds=2 * n)
        lines.append(f"  0%| | {n}/{total} [00:00<?]{ts:%m/%d/%Y %H:%M:%S} - INFO - __main__ -   "
                     f"Step: {n}, LR: {n * 1e-6}, Loss: {1000 - n}, TPS: {10 * n}")
    if dpo:
        lines += ["open-instruct fc7748ea4 | dpo | 8 GPUs | --x",
                  f"100%| | 1/1 [00:00<?]{t0:%m/%d/%Y %H:%M:%S} - INFO - __main__ -   Step: 1, LR: 0.0, Loss: 0.5"]
    path.write_text("\n".join(lines))


def test_run_name_from_chain_tag():
    """<prompt>/<base>/<pretraining seed>-s<k>, base as the manifest spells it."""
    assert mon.run_name("008-train-dclm-baseline-1B-default-s1") == "008/dclm-baseline-1B/default-s1"
    assert mon.run_name("008-train-c4-1B-large-aux-2-s3") == "008/c4-1B/large-aux-2-s3"
    assert (mon.run_name("008-train-dclm-baseline-25p-dolma1.7-75p-1B-large-aux-3-s1")
            == "008/dclm-baseline-25p-dolma1.7-75p-1B/large-aux-3-s1")
    assert mon.run_name("016-train-150M-small-aux-2-s4") == "016/dclm-baseline-150M/small-aux-2-s4"
    assert mon.run_name("008-train-oddtag") == "008/oddtag"


def test_write_tb_layout_append_and_retry(tmp_path):
    """One run per cell, tags per stage, append-only across passes, a retry opens -r2."""
    ea = pytest.importorskip("tensorboard.backend.event_processing.event_accumulator")

    def read(run):
        acc = ea.EventAccumulator(str(run), size_guidance={ea.SCALARS: 0})
        acc.Reload()
        return {t: [(e.step, e.value) for e in acc.Scalars(t)] for t in acc.Tags()["scalars"]}

    t0 = datetime(2026, 10, 8, 11, 0, 0)  # noqa: DTZ001
    logs, tb = tmp_path / "logs", tmp_path / "tb"
    logs.mkdir()
    name, seg1 = "008-train-c4-1B-default-s1", logs / "008-train-c4-1B-default-s1-10.out"
    total = 4 * mon.POINTS  # bins of 4 steps
    _train_log(seg1, 1, 10, total, t0)
    assert mon.write_tb(name, [seg1], tb) == 10 * 2 + 2 * 2  # lr+tps per step; bins 1-4, 5-8
    run = tb / "008/c4-1B/default-s1"
    got = read(run)
    assert sorted(got) == ["sft/loss", "sft/lr", "sft/s_per_step", "sft/tokens_per_s"]
    assert [s for s, _ in got["sft/lr"]] == list(range(1, 11)) and got["sft/tokens_per_s"][2][1] == 30
    assert got["sft/loss"] == [(4, pytest.approx(1000 - 2.5)), (8, pytest.approx(1000 - 6.5))]
    assert [v for _, v in got["sft/s_per_step"]] == [pytest.approx(2.0)] * 2
    assert mon.write_tb(name, [seg1], tb) == 0  # nothing new: nothing written
    _train_log(seg1, 1, 13, total, t0, dpo=True)  # the log grew; dpo started
    assert mon.write_tb(name, [seg1], tb) == 3 * 2 + 2 + 2  # steps 11-13, bin 9-12, dpo lr + its one-step loss bin
    got = read(run)
    assert [s for s, _ in got["sft/lr"]] == list(range(1, 14)) and [s for s, _ in got["sft/loss"]] == [4, 8, 12]
    assert got["dpo/lr"] == [(1, 0.0)]  # same run, dpo tags: a one-step bin has a loss but no rate yet
    assert sorted(t for t in got if t.startswith("dpo/")) == ["dpo/loss", "dpo/lr"]
    seg2 = logs / "008-train-c4-1B-default-s1-11.out"  # retry restarts at step 1
    _train_log(seg2, 1, 5, total, t0 + timedelta(hours=1))
    mon.write_tb(name, [seg1, seg2], tb)
    assert [s for s, _ in read(tb / "008/c4-1B/default-s1-r2")["sft/lr"]] == [1, 2, 3, 4, 5]
    assert [s for s, _ in read(run)["sft/lr"]] == list(range(1, 14))  # first run untouched
    assert json.loads((run / "last_step.json").read_text()) == {"sft": 13, "dpo": 1}


def test_write_tb_second_sink_backfills_with_own_marker(tmp_path):
    """A sink added after TensorBoard backfills from step 0 on its own marker, step-ordered, and
    leaves TensorBoard's marker alone."""
    class Rec:
        mark = "rec_step.json"

        def __init__(self):
            self.calls = []

        def write(self, d, run, stage, pts):
            self.calls.append((run, stage, [(s, v) for s, _, v in pts]))

    t0 = datetime(2026, 10, 8, 11, 0, 0)  # noqa: DTZ001
    logs, tb = tmp_path / "logs", tmp_path / "tb"
    logs.mkdir()
    name, seg = "008-train-c4-1B-default-s1", logs / "008-train-c4-1B-default-s1-10.out"
    _train_log(seg, 1, 10, 4 * mon.POINTS, t0)
    run = tb / "008/c4-1B/default-s1"
    run.mkdir(parents=True)
    (run / "last_step.json").write_text(json.dumps({"sft": 10}))  # TensorBoard already up to date
    rec = Rec()
    assert mon.write_tb(name, [seg], tb, [rec]) == 10 * 2 + 2 * 2
    (r, stage, pts), = rec.calls
    assert (r, stage) == ("008/c4-1B/default-s1", "sft")
    assert [s for s, _ in pts] == list(range(1, 11))  # one point per step, in order
    assert pts[3][1] == {"lr": pytest.approx(pts[3][1]["lr"]), "tokens_per_s": 40.0,
                         "loss": pytest.approx(1000 - 2.5), "s_per_step": pytest.approx(2.0)}
    assert json.loads((run / "rec_step.json").read_text()) == {"sft": 10}
    assert json.loads((run / "last_step.json").read_text()) == {"sft": 10}
    assert mon.write_tb(name, [seg], tb, [rec]) == 0 and len(rec.calls) == 1
