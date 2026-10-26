import importlib.util
import pathlib

import pytest

spec = importlib.util.spec_from_file_location(
    "smoke_cost_016", pathlib.Path(__file__).parent.parent / "scripts" / "smoke_cost_016.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

HDR = "open-instruct fc7748ea4 | {st} | 8 GPUs | ds {ds} | --foo bar"
LINE = ("{pct}%|xx| {n}/50 [00:00<00:00]10/08/2026 {t} - INFO - __main__ -   "
        "Step: {n}, LR: 1e-05, Loss: {loss}")


def _log(sft_s=1.25, dpo_s=0.425, cache_min=5.5):
    """A synthetic dry-run log with known rates: SFT steps every sft_s seconds from 12:00:00,
    DPO 'Num examples' at 13:00:00, DPO step 1 cache_min later, then every dpo_s seconds."""
    from datetime import datetime, timedelta
    out = [HDR.format(st="sft", ds="b615b6daa5ea")]
    t0 = datetime(2026, 10, 8, 12, 0, 0)  # noqa: DTZ001
    for n in range(1, 51):
        out.append(LINE.format(pct=2 * n, n=n, t=(t0 + timedelta(seconds=sft_s * n)).strftime(
            "%H:%M:%S"), loss=1000 + n) + f", TPS: {n}")
    out.append(HDR.format(st="dpo", ds="bbe8f49732e8"))
    t1 = datetime(2026, 10, 8, 13, 0, 0)  # noqa: DTZ001
    out.append(f"10/08/2026 {t1:%H:%M:%S} - INFO - __main__ -   Num examples = 370845")
    t2 = t1 + timedelta(minutes=cache_min)
    for n in range(1, 51):
        out.append(LINE.format(pct=2 * n, n=n, t=(t2 + timedelta(seconds=dpo_s * (n - 1))).strftime(
            "%H:%M:%S"), loss=0.6931 - 0.0001 * n))
    return "\n".join(out) + "\n"


def test_reproduces_p016_1_from_the_zero3_rates():
    """The ZeRO-3 projection: 150M at 1.25 / 0.425 s/step and 5.5 min caching gives 42.6
    H200-h; 300M at 1.60 / 0.675 and 8.1 min gives 54.9. Whole-second timestamps limit the
    synthetic rates, so 300M is checked with 1.6 / 0.675 rounded by the log's resolution."""
    st = C.parse_log(_log())
    assert st["sft"]["ds"] == "b615b6daa5ea" and st["dpo"]["gpus"] == 8
    r = C.cost(st, projection=17.37)
    assert r["sft_s_per_step"] == pytest.approx(1.25) and r["dpo_s_per_step"] == pytest.approx(
        0.425)
    assert r["dpo_caching_min"] == pytest.approx(5.5)
    assert r["sft_h"] == pytest.approx(4.63, abs=0.01)
    assert r["dpo_h_incl_caching"] == pytest.approx(0.43, abs=0.01)
    assert r["all_in_h200h"] == pytest.approx(42.6, abs=0.1)
    assert r["su_at_60"] == pytest.approx(60 * r["all_in_h200h"])
    assert r["measured_over_projection"] == pytest.approx(r["all_in_h200h"] / 17.37)
    assert r["loss_step1_step50"]["dpo"] == (pytest.approx(0.6930), pytest.approx(0.6881))
    assert C.SFT_STEPS == 13342 and C.DPO_STEPS == 2898


def test_rate_is_steps_10_to_50_and_refuses_a_short_log():
    st = C.parse_log(_log(sft_s=2.0))
    assert C.step_time(st["sft"]) == pytest.approx(2.0)
    short = {k: v for k, v in st["sft"]["steps"].items() if k < 50}
    with pytest.raises(AssertionError, match="not both in the log"):
        C.step_time(st["sft"] | {"steps": short})
    with pytest.raises(AssertionError, match="stages found"):
        C.parse_log(_log().split("| dpo |")[0])


def test_summary_needs_one_log_per_ladder_size():
    """The `total_24_*` keys are 12 twins x both ladder sizes; one log, or a
    misspelt size, is an error rather than a null projection or a mislabelled total."""
    both = C.summarise({"150M": _log(), "300M": _log(sft_s=1.6, dpo_s=0.675, cache_min=8.1)})
    assert set(both["sizes"]) == {"150M", "300M"} and C.TWINS_PER_SIZE == 12
    assert (C.TOTAL_H200H, C.TOTAL_SU) == ("total_24_h200h", "total_24_su_at_60")  # one key
    assert both[C.TOTAL_H200H] == pytest.approx(
        12 * (both["sizes"]["150M"]["all_in_h200h"] + both["sizes"]["300M"]["all_in_h200h"]))
    assert both[C.TOTAL_SU] == pytest.approx(60 * both[C.TOTAL_H200H])
    assert both["sizes"]["300M"]["projection_d067c_h200h"] == 22.69
    with pytest.raises(AssertionError, match="one log per ladder size"):
        C.summarise({"300M": _log()})
    with pytest.raises(AssertionError, match="one log per ladder size"):
        C.summarise({"150m": _log(), "300M": _log()})


def test_log_lines_are_read_through_castor_grid():
    """One copy of open-instruct's log-line contract. The synthetic log's
    steps are what grid.step_lines (the monitor's SLOW/STALLED instrument) reads."""
    from castor import grid
    text = _log()
    steps = grid.step_lines(text)
    assert [n for n, _ in steps] == list(range(1, 51)) * 2  # SFT then DPO
    st = C.parse_log(text)
    assert st["sft"]["steps"][50] == steps[49][1] and st["dpo"]["steps"][1] == steps[50][1]
    assert grid.step_rate(text)["stage_index"] == 2


def test_log_option_names_its_contract():
    """`--log PATH` without the size, a size twice, a size that is not a ladder size (`150m`),
    and an empty or blank path (`150M= `) are refused with the SIZE=PATH wording before any log
    is read; a missing ladder size is named here, not by summarise; surrounding whitespace is
    stripped."""
    assert C.log_paths(["150M=a.out", "300M=b.out"]) == {"150M": "a.out", "300M": "b.out"}
    assert C.log_paths([" 150M = a.out ", "300M=b.out"]) == {"150M": "a.out", "300M": "b.out"}
    for bad in ("logs/nchc/016-smoke-150M.out", "150m=a.out", "150M=", "=a.out", "150M= ",
                "150M=\t"):
        with pytest.raises(AssertionError, match="SIZE=PATH"):
            C.log_paths([bad, "300M=b.out"])
    with pytest.raises(AssertionError, match="given twice"):
        C.log_paths(["150M=a.out", "150M=b.out", "300M=b.out"])
    with pytest.raises(AssertionError, match="one log per ladder size"):
        C.log_paths(["150M=a.out"])
    with pytest.raises(AssertionError, match="one log per ladder size"):
        C.log_paths([])
