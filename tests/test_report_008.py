"""report_008's sacct parsers (the cost it reports)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "report_008", Path(__file__).resolve().parent.parent / "scripts" / "report_008.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def test_elapsed_s():
    assert m.elapsed_s("00:01:05") == 65
    assert m.elapsed_s("1-02:00:00") == 93600
    assert m.elapsed_s("19:59:59") == 71999


def test_gpus():
    assert m.gpus("billing=96,cpu=96,gres/gpu=8,mem=1600G,node=1") == 8
    assert m.gpus("cpu=12,gres/gpu=1,mem=200G,node=1") == 1
    assert m.gpus("cpu=2,mem=4G,node=1") == 0
