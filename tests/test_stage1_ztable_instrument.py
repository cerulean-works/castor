import pytest

from castor import provenance

Z = provenance.load_script("stage1_ztable")


def test_code_ignores_comments_and_docstrings():
    a = '"""doc one."""\nimport os\n\n\ndef f():\n    """x"""\n    return 1  # note\n'
    b = '"""doc two, longer."""\nimport os\n\n\ndef f():\n    return 1\n'
    assert Z._code(a, "m.py") == Z._code(b, "m.py")
    assert Z._code(a, "m.py") != Z._code(a.replace("return 1", "return 2"), "m.py")
    sh1, sh2 = "#!/bin/bash\n# one\nset -e\nrun x\n", "#!/bin/bash\n# two\nset -e\nrun x\n"
    assert Z._code(sh1, "s.sh") == Z._code(sh2, "s.sh")
    assert Z._code(sh1, "s.sh") != Z._code(sh1.replace("run x", "run y"), "s.sh")


def test_086_cleanup_is_code_identical(castor_history):
    # 004efa3: an instrument commit of the released evaluations; 9ffc0f8: R2's (after 086)
    out = Z.check_instrument({"004efa33ec1fabafd1ac78d2e21ba09309da6cf9",
                              "9ffc0f8e135b83f80d931a227e79656a9f28cbcd"})
    assert out == {"9ffc0f8e135b83f80d931a227e79656a9f28cbcd": "code-identical"}


def test_real_code_change_fails(castor_history):
    with pytest.raises(AssertionError, match="instrument code differs"):
        Z.check_instrument({"20bce43", "1469ee6"})  # 003's instrument before and after OLMES aggregation
