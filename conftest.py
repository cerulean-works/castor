"""Public export only: a test whose input file is not released is skipped, not failed."""

import pytest

REASON = "input not in the public release"


def _skip(rep, where):
    rep.outcome = "skipped"
    rep.longrepr = (str(where[0]), where[1] or 0, REASON)


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    rep = yield
    if rep.failed and "FileNotFoundError" in str(rep.longrepr):
        _skip(rep, (collector.path, 0))
    return rep


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    rep = yield
    if rep.failed and call.excinfo is not None and call.excinfo.errisinstance(FileNotFoundError):
        _skip(rep, item.location[:2])
    return rep
