import subprocess

import numpy as np
import pytest

from castor import reliability
from castor.provenance import ROOT, load_script


@pytest.fixture(scope="session")
def analyze_008():
    return load_script("analyze_008")


@pytest.fixture
def castor_history():
    """Skip a test that reads old commits when this checkout lacks them (the public export has
    one commit)."""
    gone = subprocess.run(["git", "cat-file", "-e", "20bce43^{commit}"], cwd=ROOT, capture_output=True,
                          check=False)
    if gone.returncode:
        pytest.skip("castor's git history is not in this checkout")


@pytest.fixture
def synthetic_twins():
    """Factory fixture returning the shared synthetic twin generator (a fixture because conftest
    is importable only under pytest's default import mode)."""
    return _synthetic_twins


def _synthetic_twins(seeds, ancestry: dict[str, float] | None = None, seed: int = 0,
                     n: int = 200, k: int = 20, recipe: str = "dclm-baseline"):
    """A synthetic twin panel with a known answer: 4 post-trained models per base label in
    `seeds`, per-prompt logit = recipe latent + 0.7 x base latent + noise, k Bernoulli draws
    per prompt. `ancestry` gives bases that share part of `default`'s latent (label -> weight),
    so their pairs with `default` correlate between the unrelated-seed (U) and same-base (MZ)
    levels."""
    ancestry = ancestry or {}
    assert set(ancestry) <= set(seeds) - {"default"}, (ancestry, seeds)
    assert not ancestry or "default" in seeds, seeds
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    base = {s: rng.normal(size=n) for s in seeds}
    for s, w in ancestry.items():
        base[s] = w * base["default"] + base[s]
    ys, info = {}, {}
    for s in seeds:
        for kk in range(1, 5):
            logit = latent + 0.7 * base[s] + 0.2 * rng.normal(size=n)
            p = 1 / (1 + np.exp(-logit))
            m = f"{s}-s{kk}"
            ys[m] = (rng.random((n, k)) < p[:, None]).astype(float)
            info[m] = reliability.ModelInfo(recipe, s, str(kk))
    return ys, info
