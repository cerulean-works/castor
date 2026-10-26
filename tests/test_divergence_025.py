import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import divergence_025 as D

LOG = """2026-10-02 23:12:52.135	n:0	olmo.train:880	INFO	[step=1/69359]
    train/masked_instances_local_rank=0
    optim/total_grad_norm=4.521
    train/CrossEntropyLoss=11.21
2026-10-02 23:12:56.684	n:0	olmo.train:880	INFO	[step=2/69359]
    optim/total_grad_norm=1,004.6
    train/CrossEntropyLoss=11.07
"""


def test_train_metrics():
    m = D.train_metrics(LOG)
    assert m == {1: {"optim/total_grad_norm": 4.521, "train/CrossEntropyLoss": 11.21},
                 2: {"optim/total_grad_norm": 1004.6, "train/CrossEntropyLoss": 11.07}}


def test_first_divergence_and_metric_difference():
    rows = [{"pair": "P1~P2", "step": s, "n_unequal": n} for s, n in [(0, 0), (1, 0), (2, 5), (10, 99)]]
    assert D.first_divergence(rows, "P1~P2") == 2
    assert D.first_divergence(rows, "P1~P3") is None
    a = {1: {"l": 1.0}, 2: {"l": 2.0}, 3: {"l": 3.0}}
    b = {1: {"l": 1.0}, 2: {"l": 2.0}, 3: {"l": 3.1}}
    assert D.first_metric_difference(a, b, "l") == 3
    assert D.first_metric_difference(a, a, "l") is None


def test_svg_renders():
    rows = [{"pair": "P1~P2", "step": s, "rel_l2": 1e-4 * s} for s in (1, 10, 100)]
    out = D.svg(rows)
    assert out.startswith("<svg") and out.count("<circle") == 3
