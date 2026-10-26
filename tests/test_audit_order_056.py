"""056: gold labels (and the adjudication alternatives) come out in an order that does not depend
on Python's per-process hash seed, so every resampler over audit items is reproducible."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHILD = r"""
import importlib.util, json, random, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location("audit_score", root / "scripts" / "audit_score.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
from castor import audit008 as A8
from castor.judge import labels as L
rng = random.Random(0)
ids = [f"{i:06x}" for i in rng.sample(range(16 ** 6), 300)]
labs = [L.REFUSAL, L.HARMFUL, L.NONHARMFUL]
a = {i: {"label": rng.choice(labs), "partial": False} for i in ids}
b = {i: ({"label": rng.choice(labs), "partial": False} if rng.random() < 0.3 else dict(v))
     for i, v in a.items()}
adj = {i: {"label": rng.choice(labs), "partial": False} for i in ids if a[i] != b[i]}
gold, info = S.gold_labels([a, b], adj)
alt = A8.rejected_alternatives([a, b], adj)
print(json.dumps({"gold": list(gold), "alt": list(alt), "set_order": list(set(a) & set(b))}))
"""


def _run(seed: int) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    out = subprocess.run([sys.executable, "-c", CHILD, str(ROOT)], env=env, check=True,
                         capture_output=True, text=True).stdout
    return json.loads(out)


def test_gold_order_is_independent_of_the_hash_seed():
    r0, r1 = _run(0), _run(1)
    # the test is sensitive: a bare set of these ids iterates differently under the two seeds
    assert r0["set_order"] != r1["set_order"]
    assert r0["gold"] == r1["gold"] == sorted(r0["gold"])
    assert len(r0["gold"]) == 300
    assert r0["alt"] == r1["alt"] == sorted(r0["alt"])
    assert r0["alt"]  # some adjudicated disagreements differ on the vulnerable call
