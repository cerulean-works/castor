import hashlib
import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


s1a = _load("stage1a_compare")
zt = _load("stage1_ztable")
dl = _load("download_recipe_data")

SETS = [f"set{i:02d}" for i in range(11)]


def _rel(steps, val=10.0):
    return pd.DataFrame({f"eval/{s}-validation/Perplexity": [val] * len(steps) for s in SETS},
                        index=[float(x) for x in steps])


def _ours(steps, val=10.0, drop=()):
    return {st: {s: {"Perplexity": val, "CrossEntropyLoss": math.log(val)}
                 for s in SETS if (st, s) not in drop} for st in steps}


# --- stage 1a ---------------------------------------------------------------------------

def test_log_order_numeric_job_id():
    logs = [Path("003-1a-1000000.out"), Path("003-1a-999999.out"), Path("003-1a-444.out")]
    assert [p.name for p in s1a.log_order(logs)] == \
        ["003-1a-444.out", "003-1a-999999.out", "003-1a-1000000.out"]


def test_log_order_rejects_name_without_job_id():
    with pytest.raises(AssertionError):
        s1a.log_order([Path("run.log")])


def test_compare_pass_and_rel_diff():
    ours = _ours([1250, 2500], val=10.05)
    ours[99] = ours[1250]  # a step Ai2 did not evaluate is ignored
    df, missing, incomplete = s1a.compare(ours, _rel([1250, 2500]))
    assert len(df) == 22 and not missing and not incomplete
    assert df.rel_diff.round(6).eq(0.005).all()
    assert s1a.passes(df, missing, incomplete)


def test_compare_fails_above_tolerance():
    ours = _ours([1250])
    ours[1250]["set03"]["Perplexity"] = 10.2
    df, missing, incomplete = s1a.compare(ours, _rel([1250]))
    assert not s1a.passes(df, missing, incomplete)


def test_compare_missing_set_fails():
    df, missing, incomplete = s1a.compare(_ours([1250, 2500], drop={(2500, "set07")}),
                                          _rel([1250, 2500]))
    assert incomplete == {2500: ["set07"]} and not missing and len(df) == 21
    assert not s1a.passes(df, missing, incomplete)


def test_compare_missing_step_fails():
    df, missing, incomplete = s1a.compare(_ours([1250]), _rel([1250, 2500]))
    assert missing == [2500] and not s1a.passes(df, missing, incomplete)


def test_compare_requires_11_reference_sets():
    with pytest.raises(AssertionError):
        s1a.compare(_ours([1250]), _rel([1250]).iloc[:, :10])


# --- stage 1b z-table ---------------------------------------------------------------------

def test_recipe_of():
    assert zt.recipe_of("allenai/DataDecide-dclm-baseline-150M", "150M") == "dclm-baseline"
    assert zt.recipe_of("allenai/DataDecide-dclm-baseline-1B", "1B") == "dclm-baseline"
    with pytest.raises(AssertionError):
        zt.recipe_of("allenai/DataDecide-dclm-baseline-1B", "150M")


def test_ai2_mean_ce_is_mean_log_ppl():
    pp = pd.DataFrame({f"eval/{s}-validation/Perplexity": [math.e ** (i + 1)] for i, s in enumerate(SETS)})
    pp["params"] = "150M"
    assert zt.ai2_mean_ce(pp).iloc[0] == pytest.approx(6.0)


def test_check_instrument_same_code_passes(castor_history):
    # 38f39ef and c7e204d: the two 003 eval commits, instrument code identical.
    zt.check_instrument({"38f39ef139d1f15dfe99ef8371981e0146db0145",
                         "c7e204dbbfbc89793e4d627492e4272e28f4bfd0"})


def test_check_instrument_different_code_fails(castor_history):
    # 20bce43 predates 4d6b88b, which changed shell/olmo/eval_ppl.py.
    with pytest.raises(AssertionError, match="instrument code differs"):
        zt.check_instrument({"20bce43", "4d6b88b"})


# --- downloader -------------------------------------------------------------------------

def test_quota_gate_does_not_count_failed_fetch(tmp_path, monkeypatch):
    gate = dl.QuotaGate(tmp_path, limit_bytes=100)
    gate.t = float("inf")  # never refresh from du

    def boom(*a, **k):
        raise RuntimeError("503")
    monkeypatch.setattr(dl, "_download", boom)
    with pytest.raises(RuntimeError):
        dl.fetch("a.npy", {"size": 60, "sha256": "x"}, tmp_path, gate)
    assert gate.usage == 0 and gate.reserved == 0
    monkeypatch.setattr(dl, "_download", lambda *a, **k: "ok")
    assert dl.fetch("a.npy", {"size": 60, "sha256": "x"}, tmp_path, gate) == "ok"
    assert gate.usage == 60 and gate.reserved == 0


def test_range_ignored_restarts_from_zero(tmp_path, monkeypatch):
    body = b"0123456789"
    want = {"size": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    dst = tmp_path / "f.npy"
    (tmp_path / "f.npy.part").write_bytes(body[:4])
    calls = []

    class Resp:
        def __init__(self, headers):
            calls.append(headers)
            self.status_code = 200  # ignores Range every time: full body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, n):
            yield body

    monkeypatch.setattr(dl.requests, "get", lambda url, headers, **k: Resp(headers))
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    assert dl._download("f.npy", want, dst, tmp_path / "f.npy.sha256", retries=3) == "ok"
    assert dst.read_bytes() == body
    assert calls == [{"Range": "bytes=4-"}, {}]
