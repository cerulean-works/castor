import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("lc", Path(__file__).parents[1] / "scripts/loss_curves_004.py")
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)


def test_parse_splits_stages():
    lines = [
        "Step: 9, LR: 1, Loss: 5",  # before any header: ignored
        "open-instruct fc7748ea4 | sft | 8 GPUs | --x",
        "INFO   Step: 1, LR: 7.4e-08, Loss: 1028.96, TPS: 19488.5",
        "  Step: 2, LR: 1.4e-07, Loss: nan, TPS: 1",
        "open-instruct fc7748ea4 | dpo | 8 GPUs | --x",
        "  Step: 1, LR: 2.5e-09, Loss: 0.6931",
    ]
    log = "\n".join(lines)
    got = lc.parse(log)
    assert got["sft"][0] == (1, 7.4e-08, 1028.96)
    assert len(got["sft"]) == 2 and got["dpo"] == [(1, 2.5e-09, 0.6931)]


def test_rolling_mean_warmup():
    import numpy as np
    assert np.allclose(lc.rolling(np.array([1.0, 3.0, 5.0]), 2), [1.0, 2.0, 4.0])
