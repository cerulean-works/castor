import json

import pytest

from castor.olmes10 import N_MMLU, TASKS9, olmes10


def _write(d, i, name, primary, acc_raw, n):
    (d / f"task-{i:03d}-{name}-metrics.json").write_text(json.dumps({
        "task_name": name, "task_hash": f"h{i}", "num_instances": n,
        "model_config": {"model": "m"},
        "metrics": {"primary_score": primary, "acc_raw": acc_raw}}))


def test_macro_is_mean_of_ten(tmp_path):
    for i, t in enumerate(TASKS9):
        _write(tmp_path, i, t, primary=0.1 * (i + 1), acc_raw=0.0, n=100)
    for j in range(N_MMLU):
        _write(tmp_path, 100 + j, f"mmlu_s{j}", primary=0.9, acc_raw=0.25 if j else 0.82, n=10 if j else 67)
    r = olmes10(tmp_path)
    mmlu_macro = (0.82 + 0.25 * 56) / 57
    assert r["scores"]["mmlu"] == pytest.approx(mmlu_macro)
    assert r["olmes_10_macro_avg"] == pytest.approx((sum(0.1 * (i + 1) for i in range(9)) + mmlu_macro) / 10)
    micro = olmes10(tmp_path, mmlu_agg="micro")["scores"]["mmlu"]
    assert micro == pytest.approx((0.82 * 67 + 0.25 * 560) / (67 + 560))


def test_missing_task_raises(tmp_path):
    for i, t in enumerate(TASKS9[:-1]):
        _write(tmp_path, i, t, 0.5, 0.5, 10)
    with pytest.raises(ValueError, match="missing"):
        olmes10(tmp_path)


def test_wrong_mmlu_count_raises(tmp_path):
    for i, t in enumerate(TASKS9):
        _write(tmp_path, i, t, 0.5, 0.5, 10)
    _write(tmp_path, 100, "mmlu_x", 0.5, 0.5, 10)
    with pytest.raises(ValueError, match="57"):
        olmes10(tmp_path)
