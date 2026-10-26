"""Provenance for artifacts: the git commit whose code produced them, and the exact command.

Every artifact records script + command + git commit. A commit only identifies the code if the
code is committed, so `git_commit` refuses to run with uncommitted changes to
code (castor/, scripts/, configs/, shell/, pyproject.toml, uv.lock), including untracked
files there (a new module a job imports is code too). Other files may be dirty; they do not
change what a script computes.

Jobs submitted with `shell/nchc/submit.sh` run from a snapshot worktree of HEAD (a detached
checkout under $CASTOR_WORK/tmp/jobs/), so the commit recorded here is the code that ran even
if the working tree changes while the job waits in the queue.
"""

import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
CODE_PATHS = ["castor", "scripts", "configs", "shell", "pyproject.toml", "uv.lock"]


def dirty_code(root: Path = ROOT) -> list[str]:
    """Code files that differ from HEAD: modified, staged, or untracked (ignored files such
    as __pycache__ are not listed by git status)."""
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--",
                          *CODE_PATHS], cwd=root, capture_output=True, text=True, check=True)
    return [line[3:] for line in out.stdout.splitlines() if line.strip()]


def git_commit(root: Path = ROOT, allow_dirty: bool = False) -> str:
    """HEAD commit; raises if tracked code differs from it (unless allow_dirty, which marks
    the result `-dirty` so it can never be mistaken for a clean commit)."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, check=True).stdout.strip()
    dirty = dirty_code(root)
    if dirty and not allow_dirty:
        raise RuntimeError(f"uncommitted code changes, commit first: {dirty}")
    return f"{head}-dirty" if dirty else head


def rel(path: Path, root: Path = ROOT) -> str:
    """Repo-relative where possible, absolute otherwise: a job's snapshot path is deleted when
    the job ends, so an artifact records paths under the repo relative to it."""
    path = Path(path).resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def command(argv: list[str] | None = None) -> str:
    """The command line that reproduces this run, as run from the repo root."""
    argv = sys.argv if argv is None else argv
    return shlex.join(["uv", "run", "--frozen", "python", rel(Path(argv[0])), *argv[1:]])


def load_script(name: str, root: Path = ROOT) -> ModuleType:
    """scripts/<name>.py as a module: entry points are not package members, and an analysis
    that builds on another script must load the one beside it. Registered
    in sys.modules under `name`, as a normal import would be, so repeat calls (and a script's own
    `import <name>` from scripts/) share one module object: one execution, one set of classes.
    A `name` already bound to a different file is refused (ImportError), never replaced: a script
    must not shadow a package, and a failed load must not evict a module others hold."""
    path = root / "scripts" / f"{name}.py"
    mod = sys.modules.get(name)
    if mod is not None:
        if getattr(mod, "__file__", None) == str(path):
            return mod
        raise ImportError(f"{name} is already imported from {mod.__file__!r}, not {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod
