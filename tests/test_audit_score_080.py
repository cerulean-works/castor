import importlib.util
import pathlib

import pytest

from castor import audit008 as A8
from castor import auditgate

ROOT = pathlib.Path(__file__).parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S80 = _load("audit_score_080", ROOT / "scripts" / "audit_score_080.py")
T8 = _load("test_audit008", ROOT / "tests" / "test_audit008.py")


def one_recipe(seed=0):
    return [x for x in T8.synth(seed=seed, per_cell=40) if x["model"] == "dclm-baseline"]


@pytest.mark.parametrize("items", [T8.synth(seed=1), one_recipe()])
def test_stat_equals_051s_judge_error_statistic(items):
    ours = S80.stat(items)
    theirs = A8.make_stat()(items)
    for k in S80.KEYS:
        assert ours[f"pooled:{k}"] == pytest.approx(theirs[f"pooled:{k}"], abs=1e-15)
    for s in A8.SOURCES:
        assert ours[f"source:{s}:precision"] == pytest.approx(theirs[f"source:{s}"], abs=1e-15)
        assert ours[f"source:{s}:recall"] == pytest.approx(
            theirs[f"pooled_source:{s}:recall"], abs=1e-15)


def test_bootstrap_matches_051s_on_shared_keys():
    items = one_recipe(seed=2)
    ours = S80.score(items, 200, 0)
    theirs = auditgate.stratified_bootstrap(items, A8.make_stat(), 200, 0,
                                            collapse=auditgate.source_cell)
    for k in ("precision", "recall"):
        for e in ("point", "lo", "hi"):
            assert ours["pooled"][k][e] == pytest.approx(theirs[f"pooled:{k}"][e], abs=1e-15)


def test_precision_population_drops_greedy_and_incoherent_judge_calls():
    items = one_recipe(seed=3)
    out = S80.score(items, 20, 0)
    assert out["n_population"] == len(A8.population(items))
    assert out["n_greedy_excluded"] == sum(x["sample"] < 0 for x in items)
    assert out["counts"]["precision"] == pytest.approx(out["pooled"]["precision"]["point"])
