import subprocess
import sys

import pytest

from castor import provenance


def _repo(tmp_path):
    def run(*a):
        subprocess.run(a, cwd=tmp_path, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@t")
    run("git", "config", "user.name", "t")
    (tmp_path / "castor").mkdir()
    (tmp_path / "castor/x.py").write_text("a = 1\n")
    (tmp_path / "notes.md").write_text("n\n")
    run("git", "add", ".")
    run("git", "commit", "-qm", "init")


def test_git_commit_refuses_dirty_code(tmp_path):
    _repo(tmp_path)
    head = provenance.git_commit(tmp_path)
    (tmp_path / "notes.md").write_text("changed\n")  # non-code: fine
    assert provenance.git_commit(tmp_path) == head
    (tmp_path / "castor/x.py").write_text("a = 2\n")
    with pytest.raises(RuntimeError, match="castor/x.py"):
        provenance.git_commit(tmp_path)
    assert provenance.git_commit(tmp_path, allow_dirty=True) == head + "-dirty"


def test_git_commit_refuses_untracked_code(tmp_path):
    _repo(tmp_path)
    (tmp_path / "castor/new_module.py").write_text("b = 1\n")  # untracked, but importable
    with pytest.raises(RuntimeError, match="castor/new_module.py"):
        provenance.git_commit(tmp_path)
    (tmp_path / "castor/new_module.py").unlink()
    (tmp_path / "castor/__pycache__").mkdir()
    (tmp_path / "castor/__pycache__/x.pyc").write_bytes(b"0")
    (tmp_path / ".gitignore").write_text("__pycache__/\n")
    assert not provenance.git_commit(tmp_path).endswith("-dirty")  # ignored files are not code


def test_command_keeps_arguments():
    argv = [str(provenance.ROOT / "scripts/build_panel.py"), "--out", "/tmp/a b", "--workers", "3"]
    assert provenance.command(argv) == (
        "uv run --frozen python scripts/build_panel.py --out '/tmp/a b' --workers 3")


def _scripts(tmp_path):
    (tmp_path / "scripts").mkdir()
    return tmp_path / "scripts"


def test_load_script_is_one_module_per_name(tmp_path, monkeypatch):
    (_scripts(tmp_path) / "ls_one.py").write_text("N = 1\n")
    monkeypatch.delitem(sys.modules, "ls_one", raising=False)
    a = provenance.load_script("ls_one", tmp_path)
    assert provenance.load_script("ls_one", tmp_path) is a is sys.modules["ls_one"]


def test_load_script_refuses_to_shadow_an_imported_module(tmp_path, monkeypatch):
    """A script named like a module already imported (a package, or the same script from another
    root) is refused rather than replacing it; the existing module is left in place."""
    (_scripts(tmp_path) / "json.py").write_text("x = 1\n")
    with pytest.raises(ImportError, match="json is already imported"):
        provenance.load_script("json", tmp_path)
    assert sys.modules["json"].dumps({"a": 1}) == '{"a": 1}'


def test_load_script_failure_leaves_no_entry(tmp_path, monkeypatch):
    (_scripts(tmp_path) / "ls_bad.py").write_text("raise RuntimeError('boom')\n")
    monkeypatch.delitem(sys.modules, "ls_bad", raising=False)
    with pytest.raises(RuntimeError, match="boom"):
        provenance.load_script("ls_bad", tmp_path)
    assert "ls_bad" not in sys.modules
