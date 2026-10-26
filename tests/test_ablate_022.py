"""022: own-site ablation reads the twin's committed own site, nothing else."""

import importlib.util
import json
from pathlib import Path


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_site_for_own_site_reads_selection_own():
    A = _load("ablate_007", "scripts/ablate_007.py")
    sel = {"common": {"layer": 11, "pos_index": 5},
           "own": {"a": {"layer": 9, "pos_index": 0}, "b": {"layer": 10, "pos_index": 6}}}
    assert A.site_for(sel, "own-site", "a") == (9, 0)
    assert A.site_for(sel, "own-site", "b") == (10, 6)
    assert A.site_for(sel, "own", "a") == (11, 5)  # the common site
    assert A.PROMPT_OF_KIND == {"own": "007", "random": "007", "own-site": "022"}


def test_site_for_committed_selection():
    A = _load("ablate_007", "scripts/ablate_007.py")
    sel = json.loads(Path("runs/007/selection.json").read_text())
    # runs/007/selection.json: default twins at (9, -7) x3 and (10, -1)
    got = [A.site_for(sel, "own-site", f"dd-dclm-1b-default-dpo-seed{i}") for i in range(1, 5)]
    assert [(lay, pos - 7) for lay, pos in got] == [(9, -7)] * 3 + [(10, -1)]


def test_causal_labels_path_own_site():
    C = _load("causal_007", "scripts/causal_007.py")
    assert C.labels_path("m", "own-site").parts[-4:] == (
        "m-ablate-ownsite", "022", "judge_v1", "labels.jsonl")
    assert C.labels_path("m", "random").parts[-4:-2] == ("m-ablate-random", "007")
    assert C.labels_path("m", None).parts[-4:-2] == ("m", "006")
