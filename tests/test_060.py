"""060: the fifth-base rerun of record (made at or after 9d5d90c) and the trajectory copies.

The 4-base blocks of `runs/009/factorial_fifth.json` must equal `runs/009/factorial.json` (the
pre-registered table is unchanged by `--extra-grid`); R's pairs sit under `replicate` / `R_*` and are
never pooled into LEVELS; the trajectory aggregates are byte-identical copies of the /work
originals."""
import hashlib
import json
import subprocess
from pathlib import Path

from castor import factorial

ROOT = Path(__file__).resolve().parents[1]
FIFTH = ROOT / "runs/009/factorial_fifth.json"
FOUR = ROOT / "runs/009/factorial.json"
SUBSETS = ("adv_harmful", "plain_harmful", "harmful", "benign")


def _load(p):
    return json.loads(p.read_text())


def test_four_base_blocks_unchanged():
    fifth, four = _load(FIFTH), _load(FOUR)
    for s in SUBSETS:
        assert json.dumps(fifth[s]) == json.dumps(four[s]), s


def test_rerun_at_reviewed_code(castor_history):
    fifth = _load(FIFTH)
    assert fifth["n_boot"] == 1000 and fifth["seed"] == 0
    assert fifth["extra_grid"] == "data/manifests/grid_011.json" and fifth["allow_mixed_sft"]
    commit = fifth["git_commit"]
    assert not commit.endswith("-dirty")
    # the artifact was made at a commit that contains the fixes in 9d5d90c
    subprocess.run(["git", "merge-base", "--is-ancestor", "9d5d90c", commit], cwd=ROOT, check=True)


def test_r_pairs_never_pooled_into_levels():
    fifth = _load(FIFTH)
    r_models = set(fifth["fifth_base"]["models"])
    assert len(r_models) == 4
    for s in SUBSETS:
        for oc, b in fifth["fifth_base"][s].items():
            assert set(b["levels"]) == set(factorial.FIFTH_LEVELS), (s, oc)
            # the four LEVELS over the 20 models are the 4-base table's levels exactly
            for lvl in factorial.LEVELS:
                assert b["levels"][lvl] == fifth[s][oc]["levels"][lvl], (s, oc, lvl)
            for p in b["pairs"]:
                has_r = bool({p["a"], p["b"]} & r_models)
                assert has_r == (p["level"] not in factorial.LEVELS), (s, oc, p)
            # the bootstrap reads the fifth-base pairs' summary: every CI key is a summary key
            assert set(b["ci_prompts"]) == set(b["summary"]), (s, oc)
            for k in ("bp_RX", "bp_RY", "bp_XY", "hw_init_minus_order"):
                assert f"{k}_corrected" in b["ci_prompts"], (s, oc, k)


TRAJ = {
    "runs/009/trajectory/deviation_from_released_mean.tsv":
        "6b2198c09e1c70f3f9468b339e63b3d3381327cac00f761b185b38bdc20dcbf1",
    "runs/009/trajectory/summary.json":
        "fb549d62ae63dd971a9292e87ba22397d4b3843eaef713f77cfec443408de9a1",
    "runs/011/trajectory/deviation_from_released_mean.tsv":
        "6639653bb34ad52e54667ebe160a7c59534ddfce5761334457484d92408c981b",
    "runs/011/trajectory/summary.json":
        "9dcb2256c0fa5231d27da2a2e1b0b2ba0b24503a4695f569d818b78968022d56",
}


def test_trajectory_copies_byte_identical():
    work = Path("/work/harrychang08/castor")
    for rel, h in TRAJ.items():
        data = (ROOT / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == h, rel
        if (work / rel).exists():
            assert (work / rel).read_bytes() == data, rel


def _max_dev_after_10k(tsv: Path) -> dict[str, float]:
    lines = tsv.read_text().splitlines()
    cols = lines[0].split("\t")[1:]
    out = dict.fromkeys(cols, 0.0)
    for line in lines[1:]:
        step, *vals = line.split("\t")
        if int(step) >= 10000:
            for c, v in zip(cols, vals):
                out[c] = max(out[c], abs(float(v)))
    return out


def test_trajectory_summary_matches_tsv():
    """summary.json's max |deviation| after step 10k (step 10,000 included) is the TSV's."""
    for r in ("009", "011"):
        d = ROOT / "runs" / r / "trajectory"
        s, m = _load(d / "summary.json"), _max_dev_after_10k(d / "deviation_from_released_mean.tsv")
        assert max(m[k] for k in ("default", "large aux 2", "large aux 3")) == \
            s["released_seed_abs_dev_max_after_10k"]
        for run, v in s["ours_abs_dev_max_after_10k"].items():
            assert m[run] == v, (r, run)


NUMBERS_JSON = ROOT / "paper" / "numbers.json"


def _filled():
    import pytest
    if not NUMBERS_JSON.exists():
        pytest.skip("paper/ is not checked out next to castor")
    return json.loads(NUMBERS_JSON.read_text())["filled"]


def test_trajectory_and_largest_z_macros():
    f = _filled()
    m9 = _max_dev_after_10k(ROOT / "runs/009/trajectory/deviation_from_released_mean.tsv")
    m11 = _max_dev_after_10k(ROOT / "runs/011/trajectory/deviation_from_released_mean.tsv")
    assert f["nineTrajDevX"]["value"] == m9["castor-init-default-data-aux2"]
    assert f["nineTrajDevY"]["value"] == m9["castor-init-aux2-data-default"]
    assert f["nineTrajDevR"]["value"] == m11["castor-dclm-1B-i6198-d6198"]
    assert f["nineTrajDevReleased"]["value"] == max(m9[c] for c in ("default", "large aux 2", "large aux 3"))
    zs = [max(abs(r["z_mean_ce"]), abs(r["z_olmes10"])) for p in ("009", "011")
          for r in _load(ROOT / f"runs/{p}/ztable/summary.json")["ztable"]]
    assert f["nineZLargest"]["value"] == max(zs)


def test_fifth_base_macros_read_the_rerun():
    f, fb = _filled(), _load(FIFTH)["fifth_base"]
    for sub, oc, tag in (("adv_harmful", "primary", ""), ("adv_harmful", "harm", "Harm"),
                         ("harmful", "primary", "AllHarmful"), ("benign", "primary", "Benign")):
        k = "hw_init_minus_order_corrected"
        assert f[f"nineFifthInitMinusOrder{tag}"]["value"] == fb[sub][oc]["summary"][k]
        ci = fb[sub][oc]["ci_prompts"][k]
        assert f[f"nineFifthInitMinusOrder{tag}CI"]["value"] == [ci["lo"], ci["hi"]]
    commit = _load(FIFTH)["git_commit"][:7]
    assert all(f"@ {commit}" in v["where"] for n, v in f.items() if n.startswith("nineFifth"))
