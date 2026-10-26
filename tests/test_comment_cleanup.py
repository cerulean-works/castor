"""Comment-only cleanup commits change no behaviour.

Every commit whose subject starts with "086: clean" is checked file by file against its parent:
Python ASTs equal once docstrings are dropped, shell code lines equal once comments are dropped
(heredoc bodies compared verbatim), YAML and TOML equal as parsed data. Markdown is prose.

    uv run --frozen python tests/test_comment_cleanup.py [PATH ...]   # working tree vs HEAD
"""

import ast
import io
import re
import subprocess
import sys
import tokenize
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = "086: clean"
TOOL_COMMENT = re.compile(r"#\s*(noqa(:\s*[A-Z]+\d+(,\s*[A-Z]+\d+)*)?|type:\s*ignore(\[[^\]]*\])?"
                          r"|pragma:\s*\S+|fmt:\s*\w+|ruff:\s*\S+|-\*-.*-\*-)")
HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?")
TRAILING = re.compile(r"\s+#[^'\"`]*$")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout


def blobs(specs: list[str]) -> list[str]:
    """`rev:path` contents in one `git cat-file --batch` call (one git call per file is slow here)."""
    out = subprocess.run(["git", "cat-file", "--batch"], cwd=ROOT, check=True, capture_output=True,
                         input="".join(f"{s}\n" for s in specs).encode()).stdout
    res, i = [], 0
    for _ in specs:
        head, i = out[i:out.index(b"\n", i)], out.index(b"\n", i) + 1
        n = int(head.split()[2])
        res.append(out[i:i + n].decode())
        i += n + 1
    return res


def _drop_docstrings(tree: ast.AST) -> ast.AST:
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = n.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                n.body = b[1:] or [ast.Pass()]
    return tree


def _tool_comments(src: str) -> list[str]:
    return [m.group(0) for t in tokenize.generate_tokens(io.StringIO(src).readline)
            if t.type == tokenize.COMMENT for m in [TOOL_COMMENT.search(t.string)] if m]


def py_same(old: str, new: str) -> bool:
    return (ast.dump(_drop_docstrings(ast.parse(old))) == ast.dump(_drop_docstrings(ast.parse(new)))
            and _tool_comments(old) == _tool_comments(new))


def sh_code(src: str) -> list[str]:
    out, end = [], None
    for i, line in enumerate(src.splitlines()):
        if end is not None:
            out.append(line)
            if line.strip() == end:
                end = None
            continue
        if (i == 0 and line.startswith("#!")) or line.startswith("#SBATCH"):
            out.append(line)
            continue
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        out.append(TRAILING.sub("", line))
        if (m := HEREDOC.search(line)):
            end = m.group(1)
    return out


def same(path: str, old: str, new: str) -> bool:
    if path.endswith(".py"):
        return py_same(old, new)
    if path.endswith(".sh") or path == "shell/nchc/exclude_nodes":
        return sh_code(old) == sh_code(new)
    if path.endswith((".yaml", ".yml")):
        return yaml.safe_load(old) == yaml.safe_load(new)
    if path.endswith(".toml"):
        return tomllib.loads(old) == tomllib.loads(new)
    return path.endswith(".md")


def test_submit_help_is_its_header():
    src = (ROOT / "shell/nchc/submit.sh").read_text().splitlines()
    m = re.search(r"sed -n '2,(\d+)p'", "\n".join(src))
    end = int(m.group(1))
    assert all(x.startswith("#") for x in src[1:end]) and not src[end].startswith("#")


def cleanup_commits() -> list[str]:
    return [c for c in git("log", "--format=%H %s").splitlines()
            if c.split(" ", 1)[1].startswith(SUBJECT)]


def test_checker_sees_code_changes():
    assert py_same('"""a, long."""\nx = 1  # aside\n', '"""a."""\nx = 1\n')
    assert py_same('def f():\n    """only a docstring."""\n', "def f():\n    pass\n")
    assert not py_same('x = "a \u2014 b"\n', 'x = "a, b"\n')
    assert not py_same("x = 1  # noqa: E501\n", "x = 1\n")
    assert not py_same("x = 1  # noqa: E501\n", "x = 1  # noqa: E50\n")
    assert py_same("x = 1  # noqa: E501 \u2014 long\n", "x = 1  # noqa: E501, long\n")
    assert sh_code("#!/bin/bash\n# aside\nx=1  # aside\n") == sh_code("#!/bin/bash\nx=1\n")
    assert sh_code("cat <<EOF\n# kept\nEOF\n") != sh_code("cat <<EOF\nEOF\n")
    assert sh_code('S="#!/bin/bash\n#SBATCH -N 1\n"\n') != sh_code('S="#!/bin/bash\n#SBATCH -N 2\n"\n')
    assert same("a.yaml", "a: 1  # aside\n", "a: 1\n")
    assert not same("a.yaml", "a: 1\n", "a: 2\n")


@pytest.mark.parametrize("commit", [c.split(" ", 1)[0] for c in cleanup_commits()])
def test_cleanup_commit_is_comment_only(commit):
    paths = git("diff-tree", "--no-commit-id", "--name-only", "-r", "--diff-filter=M",
                commit).split()
    assert paths
    old = blobs([f"{commit}^:{p}" for p in paths])
    new = blobs([f"{commit}:{p}" for p in paths])
    bad = [p for p, a, b in zip(paths, old, new) if not same(p, a, b)]
    assert not bad, bad


if __name__ == "__main__":
    paths = sys.argv[1:] or git("diff", "--name-only", "HEAD").split()
    old = blobs([f"HEAD:{p}" for p in paths])
    bad = [p for p, a in zip(paths, old) if not same(p, a, (ROOT / p).read_text())]
    print(f"{len(paths) - len(bad)}/{len(paths)} comment-only", *bad, sep="\n")
    sys.exit(bool(bad))
