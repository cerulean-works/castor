import importlib.util
import pathlib

import pytest

spec = importlib.util.spec_from_file_location(
    "audit_disagree", pathlib.Path(__file__).parent.parent / "scripts" / "audit_disagree.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

R, N, H, INC = "REFUSAL", "NON-HARMFUL-COMPLIANCE", "HARMFUL-COMPLIANCE", \
    "INCOHERENT-OR-OFF-TOPIC"


def _s(labels, partial=None):
    partial = partial or [False] * len(labels)
    return {str(i): {"label": x, "partial": p} for i, (x, p) in enumerate(zip(labels, partial))}


def test_classes():
    assert m.disagreement_class(R, R) == "agree"
    assert m.disagreement_class(N, R) == m.disagreement_class(R, N) == "N<->R"
    assert m.disagreement_class(H, N) == "H<->N"
    assert m.disagreement_class(INC, H) == m.disagreement_class(R, INC) == "I<->*"


def test_compare():
    a = _s([R, R, N, H, INC], [False, True, False, False, False])
    b = _s([R, R, R, N, R], [False, False, False, False, False])
    c = m.compare(a, b)
    assert c["n"] == 5 and c["agreement_4class"] == pytest.approx(0.4)
    assert c["pabak_4class"] == pytest.approx((4 * 0.4 - 1) / 3)
    # refusal: a = T T F F F, b = T T T F T -> agree on 3 of 5
    assert c["pabak_refusal"] == pytest.approx(2 * 0.6 - 1)
    assert c["confusion_rows_a_cols_b"]["N"]["R"] == 1
    assert c["confusion_rows_a_cols_b"]["I"]["R"] == 1
    assert sum(sum(r.values()) for r in c["confusion_rows_a_cols_b"].values()) == 5
    assert c["classes"] == {"N<->R": 1, "H<->N": 1, "I<->*": 1}
    assert c["partial_only"] == ["1"] and c["label_disagreements"] == ["2", "3", "4"]
