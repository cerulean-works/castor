"""008's audit supplement: recipe-keyed strata over the two adversarial sources."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

from castor import audit

_spec = importlib.util.spec_from_file_location(
    "audit_sample_008", Path(__file__).resolve().parent.parent / "scripts" / "audit_sample_008.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def _labels(path: Path, model: str, prompts: list[str]) -> None:
    rows = []
    for i, p in enumerate(prompts):
        for s in range(-1, 4):
            rows.append({"model": model, "prompt_id": p, "sample": s, "coherent": True,
                         "coherence_rule": None, "llm_invalid": False, "wg_label": "REFUSAL",
                         "wg_partial": False, "q3g_label": "REFUSAL", "q3g_partial": False,
                         "refusal_agree": (i + s) % 3 != 0})
    path.mkdir(parents=True)
    (path / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_population_strata_are_recipe_by_source(tmp_path):
    source = {"wj1": "wildjailbreak", "wj2": "wildjailbreak", "hj1": "harmbench_humanjb",
              "hj2": "harmbench_humanjb", "sb1": "sorrybench"}
    adv = {"wj1", "wj2", "hj1", "hj2"}  # sb1 is not adversarial-harmful
    models = {}
    for recipe in ("c4", "falcon"):
        for k in (1, 2):
            mid = f"dd-{recipe}-1b-default-dpo-seed{k}"
            _labels(tmp_path / mid / "judge_v1", mid, sorted(source))
            models[mid] = (recipe, tmp_path / mid)
    rows = m.population(models, adv, source)
    assert {r["model"] for r in rows} == {"c4", "falcon"}  # the stratum key is the recipe
    assert {r["model_id"] for r in rows} == set(models)
    assert {r["source"] for r in rows} == set(m.SOURCES)
    assert len(rows) == 4 * 4 * 5  # models x adversarial prompts x (greedy + 4 samples)
    drawn = audit.draw(rows, 16, seed=2008)
    for pool in ("disagree", "random"):  # each part spreads equally over recipe x source
        per_cell = Counter((d["recipe"], d["source"]) for d in drawn if d["pool"] == pool)
        assert len(per_cell) == 4 and max(per_cell.values()) - min(per_cell.values()) <= 1
    assert len(drawn) == 16
