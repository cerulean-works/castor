"""055: the seed-over-recipe ratio's recipe bootstrap, and runs/055/ratio.json against its inputs."""

import json
from pathlib import Path

import numpy as np
import pytest

from castor import gridstats

ROOT = Path(__file__).resolve().parent.parent
RATIO = ROOT / "runs" / "055" / "ratio.json"


def _per(n: int, n_mz: int, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    per = {}
    for i in range(n):
        has = i < n_mz
        mz = 0.98 + 0.01 * rng.standard_normal()
        u = 0.81 + 0.02 * rng.standard_normal()
        r = 0.78 + 0.02 * rng.standard_normal()
        per[f"q{i:02d}"] = {"MZ": {"n_pairs": 3 if has else 0, "r": mz, "r_corrected": mz},
                            "U": {"n_pairs": 6, "r": u, "r_corrected": u},
                            "R": {"n_pairs": 9, "r": r, "r_corrected": r},
                            "MZ_minus_U_corrected": mz - u if has else float("nan"),
                            "U_minus_R_corrected": u - r, "MZ_minus_U": mz - u if has else float("nan"),
                            "U_minus_R": u - r}
    return per


@pytest.mark.parametrize("n_mz", [25, 9])
def test_ratio_bootstrap_uses_recipe_bootstraps_draws(n_mz):
    per = _per(25, n_mz)
    rb = gridstats.recipe_bootstrap(per, 200, 0)
    bs = gridstats.ratio_bootstrap(per, 200, 0)
    for part, k in (("numerator", "MZ_minus_U_corrected"), ("denominator", "U_minus_R_corrected")):
        for e in ("estimate", "lo", "hi"):
            assert bs[part][e] == rb[k][e]
    assert bs["ratio"]["estimate"] == pytest.approx(rb["MZ_minus_U_corrected"]["estimate"]
                                                    / rb["U_minus_R_corrected"]["estimate"], abs=1e-15)
    assert bs["ratio"]["lo"] <= bs["ratio"]["estimate"] <= bs["ratio"]["hi"]


def test_ratio_bootstrap_counts_nonpositive_denominators():
    per = _per(25, 25)
    for q in per:
        per[q]["U_minus_R_corrected"] = 0.0
    bs = gridstats.ratio_bootstrap(per, 50, 0)
    assert bs["frac_den_nonpositive"] == 1.0 and bs["ratio"]["n_draws"] == 0


@pytest.mark.skipif(not RATIO.exists(), reason="runs/055/ratio.json not built")
def test_ratio_json_against_its_inputs():
    r = json.loads(RATIO.read_text())
    g = json.loads((ROOT / "runs" / "008" / "grid.json").read_text())["adv_harmful"]["primary"]["levels"]
    au = json.loads((ROOT / "runs" / "008" / "audit_correct.json").read_text())
    raw = r["variants"]["raw"]["ratio"]["estimate"]
    assert abs(raw - g["MZ_minus_U_corrected"] / g["U_minus_R_corrected"]) < 1e-12
    assert f"{raw:.1f}" == "5.4"  # the value the paper prints
    # the ratio of record is raw; the audit-corrected levels of record stay per_source
    assert r["of_record"] == "raw" and au["levels"]["of_record"] == "per_source"
    ps = r["variants"]["per_source"]
    assert abs(ps["numerator"]["estimate"] - au["contrast"]["per_source"]["MZ_minus_U_corrected"]) < 1e-9
    assert abs(ps["denominator"]["estimate"] - au["contrast"]["per_source"]["U_minus_R_corrected"]) < 1e-9
    for v, o in r["variants"].items():
        assert o["ratio"]["lo"] <= o["ratio"]["estimate"] <= o["ratio"]["hi"], v
        assert o["ratio"]["estimate"] == pytest.approx(o["numerator"]["estimate"] / o["denominator"]["estimate"])


@pytest.mark.skipif(not RATIO.exists(), reason="runs/055/ratio.json not built")
def test_ratio_json_names_raw_of_record_and_current_inputs():
    """The ratio of record is raw; the recorded input hashes match the files on disk."""
    import hashlib
    r = json.loads(RATIO.read_text())
    assert r["of_record"] == "raw"
    for path, sha in r["inputs"].items():
        p = ROOT / path  # repo-relative
        assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, path
