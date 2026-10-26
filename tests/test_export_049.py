import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("export_049", ROOT / "scripts/export_049.py")
E = importlib.util.module_from_spec(spec)
sys.modules["export_049"] = E
spec.loader.exec_module(E)


def test_where_pattern():
    pat = E.where_pattern("runs/061/compute/<file>.psv")
    assert pat.match("runs/061/compute/a100.psv")
    assert not pat.match("runs/061/compute/a100.json")
    assert not pat.match("runs/061/compute/sub/a.psv")
    pat = E.where_pattern("runs/<id>/x.json")
    assert pat.match("runs/008/x.json") and not pat.match("runs/008/y.json")
    assert E.where_pattern("runs/008/a.json").match("runs/008/a.json")
    assert not E.where_pattern("runs/008/a.json").match("runs/008/a.jsonl")


def test_readme_has_paper_and_citation():
    inv = json.loads((ROOT / "runs/049/inventory.json").read_text())
    text = E.readme(E.release.card_config(), inv)
    assert "208 post-trained" in text and "25 DataDecide" in text and "\u2014" not in text
    assert "12 1B models whose SFT" in text and "16\n  models built on our own" in text
    assert "**Paper:** *Identically Trained" in text and "```bibtex" in text and "## Citation" in text
    assert text.startswith("![Cerulean Labs Castor banner](assets/banner.png)\n") and (ROOT / "configs/release/banner.png").is_file()
    assert "paper/" not in text and "runs/numbers.json" in text and "LaTeX source is on arXiv" in text


def test_paper_tools_are_not_exported():
    for p in ("scripts/paper_lint.py", "scripts/bib_audit.py", "tests/test_paper_numbers.py",
              "tests/test_bib_audit.py"):
        assert E.DENY.search(p), p
    for p in ("scripts/release_049.py", "tests/test_060.py", "tests/test_stage1.py"):
        assert not E.DENY.search(p), p


def test_long_strings_in_every_aggregate_format():
    long = "x" * (E.LONG_STRING + 1)
    assert E.long_strings("runs/a.json", f'{{"k": "{long}"}}'.encode()) == 1
    assert E.long_strings("runs/a.tsv", f"id\t{long}\nb\t1\n".encode()) == 1
    assert E.long_strings("runs/a.psv", f"id|{long}\n".encode()) == 1
    assert E.long_strings("runs/a.csv", f'id,"{long}, with a comma"\n'.encode()) == 1
    assert E.long_strings("runs/a.txt", f"{long}\nshort\n".encode()) == 1
    assert E.long_strings("runs/a.tsv", b"a\tb\n") == 0
    assert E.long_strings("castor/a.py", long.encode()) == 0


def test_free_text_columns():
    assert E.free_text_columns("runs/a.tsv", b"prompt_id\tlabel\tnotes\n1\t0\tx\n") == ["notes"]
    assert E.free_text_columns("runs/a.csv", b'id,"Comment",label\n') == ["Comment"]
    assert E.free_text_columns("runs/a.jsonl", b'{"id": 1, "label": 0}\n{"id": 2, "text": "x"}\n') == ["text"]
    assert E.free_text_columns("runs/a.psv", b"1|job|COMPLETED\n") == []
    assert E.free_text_columns("runs/a.json", b'{"note": "aggregate metadata"}') == []
    assert E.free_text_columns("runs/a.tsv", b"prompt_id\tsource\twg_label\n") == []


def test_readme_counts_add_up_with_035():
    inv = {"n_models": 208, "repos": ["r1", "r2"],
           "by_prompt": {"008": {"n": 156}, "009": {"n": 8}, "011": {"n": 4}, "016": {"n": 24},
                         "035": {"n": 4}, "080": {"n": 12}},
           "models": [{"prompt": "008", "recipe": "a"}, {"prompt": "008", "recipe": "b"}]}
    c = E.readme_counts(inv)
    assert c["n_grid"] + c["n_small"] + c["n_rerun"] + c["n_swap"] == c["n_models"] == 208
    assert c["n_rerun"] == 16 and c["n_swap"] == 12 and c["n_recipes"] == 2 and c["n_repos"] == 2
    assert E.readme_counts(inv | {"by_prompt": {"008": {"n": 156}}})["n_rerun"] == 0


def test_commands_and_reviewed_strings_pass():
    cmd = "uv run --frozen python scripts/x.py " + "--flag " * 60
    assert E.long_strings("runs/a.json", f'{{"command": "{cmd}"}}'.encode()) == 0
    log = "node 25a-hgpn029 job 488193 mps 1 " + "y" * 300
    assert E.long_strings("runs/a.json", f'{{"log": "{log}"}}'.encode()) == 0
    long = "z" * (E.LONG_STRING + 1)
    assert E.long_strings("runs/080/posthoc.json", f'{{"note": "{long}"}}'.encode()) == 0
    two = f'{{"a": "{long}", "b": "{long}"}}'.encode()
    assert E.long_strings("runs/080/posthoc.json", two) == 2
    assert E.DENY.search("data/manifests/fallback12_recipes.json")
