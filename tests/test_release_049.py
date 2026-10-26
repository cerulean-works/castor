import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import requests
from huggingface_hub.utils import HfHubHTTPError

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("release_049", ROOT / "scripts/release_049.py")
R = importlib.util.module_from_spec(spec)
sys.modules["release_049"] = R
spec.loader.exec_module(R)


def _http(status):
    resp = requests.Response()
    resp.status_code = status
    return HfHubHTTPError("x", response=resp)


def test_retry_only_on_transient(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    calls = []

    def flaky(err, n):
        def fn():
            calls.append(1)
            if len(calls) <= n:
                raise err
            return "ok"
        return fn

    assert R._retry(flaky(requests.ConnectionError("down"), 2)) == "ok" and len(calls) == 3
    calls.clear()
    assert R._retry(flaky(_http(503), 1)) == "ok" and len(calls) == 2
    for err in (_http(403), _http(404), TypeError("bug")):
        calls.clear()
        with pytest.raises(type(err)):
            R._retry(flaky(err, 3))
        assert len(calls) == 1


def test_models_resume(tmp_path):
    args = Namespace(upload=True, verify=True, force=False)
    assert not R._done(tmp_path, "m", args)
    R._record(tmp_path, "models", "m", {"hub_commit": "abc"})
    assert not R._done(tmp_path, "m", args)
    assert R._done(tmp_path, "m", Namespace(upload=True, verify=False, force=False))
    R._record(tmp_path, "models", "m", {"hub_commit": "abc", "verify": {"digest_match": True}})
    assert R._done(tmp_path, "m", args)
    assert not R._done(tmp_path, "m", Namespace(upload=True, verify=True, force=True))
    rec = json.loads((tmp_path / "manifest/models/m.json").read_text())
    assert rec["verify"]["digest_match"]


def test_n_labels_counts_hub_and_staged(tmp_path):
    d = tmp_path / "labels"
    for f in ("data/008/a.parquet", "data/008/b.parquet"):
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_bytes(b"")

    class Api:
        def repo_exists(self, repo, repo_type=None):
            return True

        def list_repo_files(self, repo, repo_type=None):
            return ["README.md", "data/008/b.parquet", "data/016/c.parquet"]

    assert R._n_labels(None, d) == 2
    assert R._n_labels(Api(), d) == 3
