import functools
import importlib.util
import json
import pathlib

import numpy as np
import pytest

from castor import grid, ladder, posttrain, reliability, sanityb

spec = importlib.util.spec_from_file_location(
    "launch_016", pathlib.Path(__file__).parent.parent / "scripts" / "launch_016.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

CELL = ladder.Cell("150M", "small-aux-2", 3)
# trend_rule's flags are keyword-only without a default: the complete run's values,
# overridden by the tests of INCOMPLETE
RULE = functools.partial(ladder.trend_rule, labels_complete=True, anchor_checked=True)


def test_cells_and_naming():
    cs = ladder.cells()
    assert len(cs) == 24 and len({c.model_id for c in cs}) == 24
    assert len({c.tag for c in cs}) == 24 and len({c.group for c in cs}) == 6
    assert CELL.base_ref == "allenai/DataDecide-dclm-baseline-150M@step37500-seed-small-aux-2"
    assert ladder.Cell("300M", "default", 1).base_ref == \
        "allenai/DataDecide-dclm-baseline-300M@step45000-seed-default"
    assert CELL.model_id == "dd-dclm-150m-small-aux-2-dpo-seed3"
    assert CELL.out("dpo").parts[-3:] == ("dd-dclm-150M-seed-small-aux-2", "dpo", "seed3")
    # never collides with 006's 1B twins
    assert not {c.model_id for c in cs} & {c.model_id for c in sanityb.cells()}
    # same interface as sanityb.Cell, so 006's launcher functions take it
    for attr in ("k", "revision", "base_ref", "base_dir", "group", "model_id", "tag", "info"):
        assert hasattr(CELL, attr)
    # MZ within a base, U across bases of one size (castor.reliability levels)
    a, b, c = ladder.Cell("150M", "default", 1), ladder.Cell("150M", "default", 2), \
        ladder.Cell("150M", "small-aux-3", 1)
    assert reliability.pair_level(a.info, b.info) == "MZ"
    assert reliability.pair_level(a.info, c.info) == "U"


def _twins(rng, n, base_sd, post_sd, k=20, prefix="m"):
    """12 twins (3 bases × 4 seeds) whose per-prompt logit = shared + base effect + seed effect;
    K Bernoulli samples each."""
    shared = rng.normal(0, 1.5, n)
    ys, info = {}, {}
    for s in ("a", "b", "c"):
        base = shared + rng.normal(0, base_sd, n)
        for k_ in (1, 2, 3, 4):
            p = 1 / (1 + np.exp(-(base + rng.normal(0, post_sd, n))))
            m = f"{prefix}-{s}-{k_}"
            ys[m] = (rng.random((n, k)) < p[:, None]).astype(float)
            info[m] = reliability.ModelInfo("r", s, str(k_))
    return ys, info


def _ladder(n=400, order=ladder.TREND):
    """A synthetic ladder with known ordering: base effect (the pretraining-seed effect) large at
    150M, smaller at 300M, smallest at 1B. Returned in the dict order `order`."""
    rng = np.random.default_rng(0)
    sd = {"150M": (1.2, "s"), "300M": (0.7, "m"), "1B": (0.3, "l")}
    data = {z: _twins(rng, n, sd[z][0], 0.1, prefix=sd[z][1]) for z in ladder.TREND}
    ids = [f"p{i}" for i in range(n)]
    return ({z: data[z][0] for z in order}, {z: list(ids) for z in order},
            {z: data[z][1] for z in order})


def test_paired_bootstrap_and_rule():
    ys, ids, info = _ladder()
    boot = ladder.paired_size_bootstrap(ys, ids, info, n_boot=50, seed=0)
    g = {z: boot[f"gap_{z}"]["estimate"] for z in ladder.TREND}
    assert g["150M"] > g["300M"] > g["1B"] > 0
    d = boot["diff_150M_1B"]
    assert d["estimate"] == pytest.approx(g["150M"] - g["1B"]) and d["lo"] > 0
    assert g["1B"] == pytest.approx(ladder.gap(reliability.twin_correlations(
        ys["1B"], info["1B"])["levels"]))
    ok = {z: True for z in ladder.TREND}
    r = RULE(boot, ok)
    assert r["holds"] and r["ci_150M_1B_above_0"]
    # a failed gate at any size takes it out of the trend: the prediction cannot hold
    r = RULE(boot, ok | {"300M": False})
    assert not r["holds"] and r["sizes_in_trend"] == ["150M", "1B"]
    # order violated
    swapped = boot | {"gap_300M": boot["gap_300M"] | {"estimate": g["150M"] + 1}}
    assert not RULE(swapped, ok)["holds"]
    # ordered but the interval includes 0
    wide = boot | {"diff_150M_1B": d | {"lo": -0.01, "hi": 0.5}}
    r = RULE(wide, ok)
    assert r["ordered"] and not r["ci_150M_1B_above_0"] and not r["holds"]
    assert r["full_bootstrap"] and r["n_draws"] == r["n_boot"] == 50
    assert r["n_draws_per_size"] == {z: 50 for z in ladder.TREND}
    assert ladder.TREND_PAIRS == {"MZ": 18, "U": 48}  # 3 bases x 4 post-training seeds


# the twin of each synthetic size that _noisy replaces, and the seeds it is built with: seed
# WEAK gives a small positive split-half reliability on the full sample (so the point estimate
# is defined) that is <= 0 on some prompt resamples; seed NEG gives reliability <= 0 on the
# full sample itself. Both are preconditions asserted where used, not magnitudes.
NOISY = {"150M": "s-a-1", "300M": "m-a-1", "1B": "l-a-1"}
WEAK, NEG = 0, 2


def _noisy(ys, info, seed, n, size="1B"):
    """One twin of `size` measured with two Bernoulli samples of a weak signal, so its split-half
    reliability is small (`disattenuate` NaN for its pairs wherever it is <= 0). The noise
    stream depends on `seed` only, not on the size."""
    rng = np.random.default_rng(seed)
    p = 1 / (1 + np.exp(-0.25 * rng.normal(0, 1, n)))
    y = (rng.random((n, 2)) < p[:, None]).astype(float)
    return ys | {size: ys[size] | {NOISY[size]: y}}


def test_gap_every_pair_is_undefined_when_any_pair_is():
    """twin_correlations averages r_corrected with nanmean, so a twin whose reliability is ≤ 0
    drops out of the level mean and the gap stays finite over a subset of pairs; the trend's gap
    is NaN instead, and the levels say how many pairs were defined. On a clean sample the two
    agree."""
    ys, _, info = _ladder(n=200)
    clean = reliability.twin_correlations(ys["1B"], info["1B"])
    assert ladder.gap_every_pair(clean) == pytest.approx(ladder.gap(clean["levels"]))
    for lvl, n in ladder.TREND_PAIRS.items():
        assert clean["levels"][lvl]["n_pairs"] == clean["levels"][lvl]["n_pairs_defined"] == n
    tw = reliability.twin_correlations(_noisy(ys, info, NEG, 200)["1B"], info["1B"])
    assert tw["reliability"]["l-a-1"] <= 0, "precondition: seed NEG gives reliability <= 0"
    bad = [p for p in tw["pairs"] if "l-a-1" in (p["a"], p["b"])]
    assert len(bad) == 11 and all(np.isnan(p["r_corrected"]) for p in bad)  # 11 = its pairs
    assert all(np.isfinite(p["r_corrected"]) for p in tw["pairs"] if p not in bad)
    assert tw["levels"]["MZ"]["n_pairs_defined"] == 18 - 3  # 3 MZ, 8 U pairs of l-a-1
    assert tw["levels"]["U"]["n_pairs_defined"] == 48 - 8
    assert np.isfinite(ladder.gap(tw["levels"]))  # 006's statistic, over the other pairs
    assert np.isnan(ladder.gap_every_pair(tw))


def test_gap_every_pair_is_undefined_with_a_twin_missing():
    """The pre-registered statistic averages 18 MZ and 48 U pairs; with a
    twin missing (--allow-incomplete) the levels are finite over fewer pairs, and the trend's
    gap is NaN whatever signal_016's labels_complete flag says."""
    ys, ids, info = _ladder(n=100)
    short = {m: y for m, y in ys["1B"].items() if m != "l-a-1"}
    tw = reliability.twin_correlations(short, info["1B"])
    assert tw["levels"]["MZ"]["n_pairs"] == 15 and tw["levels"]["U"]["n_pairs"] == 40
    assert np.isfinite(ladder.gap(tw["levels"])) and np.isnan(ladder.gap_every_pair(tw))
    boot = ladder.paired_size_bootstrap(ys | {"1B": short}, ids, info, n_boot=3)
    assert np.isnan(boot["gap_1B"]["estimate"]) and boot["gap_1B"]["n_draws"] == 0
    r = RULE(boot, {z: True for z in ladder.TREND})
    assert r["holds"] is None and r["verdict"] == "undecided"
    assert r["undefined_point_estimates"] == ["1B"] and "pair missing" in r["reason"]


def _partial(ys, info, ids, size):
    boot = ladder.paired_size_bootstrap(_noisy(ys, info, WEAK, 200, size), ids, info, n_boot=50)
    nd = boot[f"gap_{size}"]["n_draws"]
    assert 0 < nd < 50, f"precondition: seed WEAK undefined on some draws at {size}, not all"
    assert all(boot[f"gap_{z}"]["n_draws"] == 50 for z in ladder.TREND if z != size)
    return boot


def test_rule_is_undecided_on_a_partial_bootstrap():
    """End to end: a resample on which one twin's correction is undefined makes that draw's gap
    NaN (gap_every_pair), _percentile_ci drops it, n_draws < n_boot, and the rule reports that
    instead of a verdict either way."""
    ys, ids, info = _ladder(n=200)
    ok = {z: True for z in ladder.TREND}
    assert RULE(ladder.paired_size_bootstrap(ys, ids, info, 50, 0), ok)["holds"]
    boot = _partial(ys, info, ids, "1B")
    g = {z: boot[f"gap_{z}"]["estimate"] for z in ladder.TREND}
    assert all(np.isfinite(v) for v in g.values()) and g["150M"] > g["300M"] > g["1B"]
    nd = boot["gap_1B"]["n_draws"]
    assert boot["diff_150M_1B"]["n_draws"] == nd  # the drop is 1B's alone
    r = RULE(boot, ok)
    assert r["holds"] is None and r["verdict"] == "undecided" and not r["full_bootstrap"]
    assert r["ordered"] and r["n_draws"] == nd and r["n_boot"] == 50
    assert r["n_draws_per_size"] == {"150M": 50, "300M": 50, "1B": nd}
    assert f"{nd}/50 bootstrap draws finite" in r["reason"] and "'1B': " in r["reason"]
    with pytest.raises(KeyError, match="n_draws"):
        RULE(boot | {"diff_150M_1B": {"estimate": 0.4, "lo": 0.1, "hi": 0.6}}, ok)


def test_rule_is_undecided_on_a_partial_bootstrap_at_the_middle_size():
    """A draw on which only gap(300M) is undefined leaves diff_150M_1B's
    n_draws == n_boot; the rule is still undecided, as its `rule` text says (every size on
    every draw)."""
    ys, ids, info = _ladder(n=200)
    ok = {z: True for z in ladder.TREND}
    boot = _partial(ys, info, ids, "300M")
    assert boot["diff_150M_1B"]["n_draws"] == 50
    r = RULE(boot, ok)
    assert not r["full_bootstrap"] and r["holds"] is None
    assert r["verdict"] == ("undecided" if r["ordered"] else "prediction fails")
    # the same from the rule alone, with the point estimates in order
    clean = ladder.paired_size_bootstrap(ys, ids, info, n_boot=20)
    assert RULE(clean, ok)["holds"] is True
    mid = clean | {"gap_300M": clean["gap_300M"] | {"n_draws": 19}}
    r = RULE(mid, ok)
    assert r["ordered"] and r["n_draws"] == 20 and not r["full_bootstrap"]
    assert r["holds"] is None and r["verdict"] == "undecided"
    assert r["n_draws_per_size"]["300M"] == 19 and "'300M': 19" in r["reason"]


def test_rule_is_undecided_on_an_undefined_point_estimate():
    """A correction undefined on the full sample is not evidence against the prediction (NaN
    compares False, which would otherwise read as 'not ordered' → fails)."""
    ys, ids, info = _ladder(n=200)
    boot = ladder.paired_size_bootstrap(_noisy(ys, info, NEG, 200), ids, info, n_boot=20)
    assert np.isnan(boot["gap_1B"]["estimate"]) and np.isnan(boot["diff_150M_1B"]["estimate"])
    assert np.isfinite(boot["gap_150M"]["estimate"])
    r = RULE(boot, {z: True for z in ladder.TREND})
    assert r["holds"] is None and r["verdict"] == "undecided"
    assert r["undefined_point_estimates"] == ["1B"] and not r["ordered"]
    assert "point estimate undefined at ['1B']" in r["reason"]


def test_partial_bootstrap_does_not_override_a_decided_failure():
    """A failed gate or unordered point estimates fail the conjunction on
    their own; the state of the bootstrap cannot turn that into 'undecided'."""
    ys, ids, info = _ladder(n=200)
    boot = _partial(ys, info, ids, "1B")
    ok = {z: True for z in ladder.TREND}
    r = RULE(boot, ok | {"300M": False})
    assert r["holds"] is False and r["verdict"] == "prediction fails" and "gate" in r["reason"]
    assert r["sizes_in_trend"] == ["150M", "1B"] and not r["full_bootstrap"]
    assert r["ordered"] and not r["complete"]  # the order test on its own
    g150 = boot["gap_150M"]["estimate"]
    swapped = boot | {"gap_300M": boot["gap_300M"] | {"estimate": g150 + 1}}
    r = RULE(swapped, ok)
    assert r["holds"] is False and r["verdict"] == "prediction fails"
    assert "not ordered" in r["reason"] and not r["ordered"]


def test_rule_needs_at_least_one_finite_draw():
    """n_draws == n_boot == 0 (or a NaN interval) is not a full bootstrap."""
    ys, ids, info = _ladder(n=100)
    ok = {z: True for z in ladder.TREND}
    boot = ladder.paired_size_bootstrap(ys, ids, info, n_boot=0)
    d = boot["diff_150M_1B"]
    assert d["n_draws"] == 0 and np.isnan(d["lo"]) and boot["n_boot"] == 0
    r = RULE(boot, ok)
    assert r["holds"] is None and r["verdict"] == "undecided" and not r["full_bootstrap"]
    assert r["ordered"] and not r["ci_150M_1B_above_0"]
    boot = ladder.paired_size_bootstrap(ys, ids, info, n_boot=5)
    nan_lo = boot | {"diff_150M_1B": boot["diff_150M_1B"] | {"lo": float("nan")}}
    assert RULE(nan_lo, ok)["holds"] is None


def test_incomplete_labels_keep_the_rule_shape():
    """--allow-incomplete runs write the same keys as a complete run, with
    verdict INCOMPLETE and holds None, whatever the data say."""
    ys, ids, info = _ladder(n=100)
    boot = ladder.paired_size_bootstrap(ys, ids, info, n_boot=5)
    ok = {z: True for z in ladder.TREND}
    full, part = RULE(boot, ok), RULE(boot, ok, labels_complete=False)
    assert full["holds"] is True and full["labels_complete"]
    assert part["holds"] is None and part["verdict"] == "INCOMPLETE" and not part["labels_complete"]
    assert set(part) == set(full) and part["rule"] == full["rule"] and part["ordered"]
    assert all(r["verdict"] in ladder.VERDICTS for r in (full, part))


def test_rule_is_directional():
    """An interval entirely below 0 contradicts the prediction, even with the point estimates
    (inconsistently) ordered."""
    boot = {"gap_150M": {"estimate": 0.5, "n_draws": 10}, "gap_300M": {"estimate": 0.3,
            "n_draws": 10}, "gap_1B": {"estimate": 0.1, "n_draws": 10}, "n_boot": 10,
            "diff_150M_1B": {"estimate": 0.4, "lo": -0.3, "hi": -0.1, "n_draws": 10}}
    r = RULE(boot, {z: True for z in ladder.TREND})
    assert r["ordered"] and not r["ci_150M_1B_above_0"] and r["holds"] is False
    assert r["verdict"] == "prediction fails" and "does not lie above 0" in r["reason"]


@pytest.mark.parametrize("order", [("1B", "150M", "300M"), ("300M", "1B", "150M")])
def test_bootstrap_keys_do_not_depend_on_dict_order(order):
    """Keys are canonical (TREND order), so the rule finds diff_150M_1B whatever
    order the sizes arrive in, and the numbers are those of the canonical order."""
    ref = ladder.paired_size_bootstrap(*_ladder(n=200), n_boot=20, seed=3)
    shuf = ladder.paired_size_bootstrap(*_ladder(n=200, order=order), n_boot=20, seed=3)
    assert set(shuf) == set(ref) and "diff_150M_1B" in shuf and "diff_1B_150M" not in shuf
    for k in ref:
        if k.startswith(("gap_", "diff_")):
            assert shuf[k] == pytest.approx(ref[k])
    assert RULE(shuf, {z: True for z in ladder.TREND})["holds"]


def test_rule_fails_loudly_on_a_missing_key():
    ys, ids, info = _ladder(n=100)
    boot = ladder.paired_size_bootstrap(ys, ids, info, n_boot=5)
    del boot["diff_150M_1B"]
    with pytest.raises(KeyError, match="diff_150M_1B"):
        RULE(boot, {z: True for z in ladder.TREND})


def test_paired_bootstrap_refuses_unpaired_panels():
    """Same prompts in the same order at every size, not only the same count."""
    ys, ids, info = _ladder(n=60)
    reordered = ids | {"300M": ids["300M"][::-1]}
    with pytest.raises(AssertionError, match="300M: prompt ids differ"):
        ladder.paired_size_bootstrap(ys, reordered, info, n_boot=2)
    other = ids | {"1B": ids["1B"][:-1] + ["zzz"]}
    with pytest.raises(AssertionError, match="1B: prompt ids differ"):
        ladder.paired_size_bootstrap(ys, other, info, n_boot=2)
    short = ys | {"1B": {m: y[:-1] for m, y in ys["1B"].items()}}
    with pytest.raises(AssertionError, match="rows for 60 prompt ids"):
        ladder.paired_size_bootstrap(short, ids, info, n_boot=2)
    with pytest.raises(AssertionError, match="sizes"):
        ladder.paired_size_bootstrap({z: ys[z] for z in ("150M", "1B")}, ids, info, n_boot=2)


# ---- launcher ---------------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder.convert, "converted_dir", lambda repo, rev: tmp_path / "base" / rev)
    monkeypatch.setattr(sanityb, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(ladder, "MODELS_ROOT", tmp_path / "models")
    return tmp_path / "models"


def _write(path, d):
    path.mkdir(parents=True, exist_ok=True)
    (path / "provenance.json").write_text(json.dumps(d))


def test_006_status_and_train_cmd_take_a_ladder_cell(root, monkeypatch):
    assert L.L6.status(CELL, root) == "todo"
    _write(CELL.out("sft", root), {"post_training_seed_k": 3, "base_model": str(CELL.base_dir),
                                   "base_ref": CELL.base_ref, "finished": "x"})
    assert L.L6.status(CELL, root) == "started"
    ckpt = CELL.out("dpo", root) / "oi" / "ck"
    ckpt.mkdir(parents=True)
    _write(CELL.out("dpo", root), {"post_training_seed_k": 3, "finished": "x",
                                   "base_model": str(CELL.out("sft", root) / "model"),
                                   "checkpoint": str(ckpt)})
    assert L.L6.status(CELL, root) == "done"
    monkeypatch.setattr(sanityb, "check_base", lambda c: {})
    cmd = L.L6.train_cmd(CELL, root, None)
    assert f"--base-ref {CELL.base_ref}" in cmd and "max_train_steps" not in cmd
    assert sanityb.SFT_CONFIG in cmd and sanityb.DPO_CONFIG in cmd  # the 1B recipe unchanged
    # gen+judge goes to runs/<id>/016 and keys on the checkpoint
    out = sanityb.RUNS / CELL.model_id / "016"
    assert L.genjudge_status(CELL, root) == "todo"
    _write(out, {"limit": None, "model_resolved": "/old"})
    assert L.genjudge_status(CELL, root) == "stale"
    _write(out, {"limit": None, "model_resolved": str(ckpt.resolve())})
    (out / "judge_v1").mkdir()
    (out / "judge_v1" / "labels.jsonl").write_text("")
    assert L.genjudge_status(CELL, root) == "done"
    gj = L.genjudge_cmd(CELL, root)
    assert gj.index("check-trained") < gj.index("gen_judge.sh") and " 016" in gj


@pytest.mark.parametrize("argv, msg", [
    (["train", "--size", "150M", "--smoke", "0"], "positive"),
    (["train", "--size", "150M", "--dev"], "smoke only"),
    (["genjudge", "--size", "150M", "--smoke", "5"], "training only"),
    (["train", "--size", "150M", "--smoke", "5", "--chain"], "not generated"),
])
def test_main_rejects_unsafe_flags(monkeypatch, argv, msg):
    monkeypatch.setattr("sys.argv", ["launch_016.py", *argv])
    monkeypatch.setattr(L.L6, "queued", lambda: pytest.fail("checks must run before squeue"))
    with pytest.raises(AssertionError, match=msg):
        L.main()


def test_main_chains_pins_commit_and_caps(root, monkeypatch, capsys):
    monkeypatch.setattr(sanityb, "check_base", lambda c: {})
    monkeypatch.setattr(L, "head", lambda: "abc123")
    monkeypatch.setattr(L.L6, "queued", lambda: {"016-train-150M-default-s1", "x"})
    calls = []

    def fake(opts, cmd, dry, commit=None, run_dry=False):  # launch_006.submit's signature
        assert commit == "abc123", "every job must be pinned to this invocation's HEAD"
        assert dry is False
        calls.append((opts, cmd))
        return str(100 + len(calls))
    monkeypatch.setattr(L.L6, "submit", fake)
    monkeypatch.setattr("sys.argv", ["launch_016.py", "train", "--size", "150M", "--chain",
                                     "--max-queued", "5"])
    L.main()
    out = capsys.readouterr().out
    assert "skip 150M-default-s1" in out and "stop at" in out
    # 1 already queued + 2 chains × 2 jobs = 5 = the cap
    assert len(calls) == 4
    assert calls[1][0][calls[1][0].index("--after") + 1] == "101"
    assert calls[0][0][calls[0][0].index("--name") + 1] == "016-train-150M-default-s2"
    assert L.HEAD == "abc123"


def test_head_refuses_any_dirty_code_path(monkeypatch):
    """Configs, train.sh, sanityb.py, convert.py ... (provenance.CODE_PATHS), not only the
    launcher's own files, must be committed before a job is pinned to HEAD."""
    monkeypatch.setattr(L.provenance, "dirty_code", lambda root: ["configs/posttrain/x.yaml"])
    with pytest.raises(RuntimeError, match="configs/posttrain/x.yaml"):
        L.head()
    monkeypatch.setattr(L.provenance, "dirty_code", lambda root: [])
    assert len(L.head()) == 40


def test_check_trained_does_not_need_a_clean_tree(root, monkeypatch):
    """check-trained runs inside a job (its snapshot); it never pins HEAD or reads squeue."""
    monkeypatch.setattr(L, "head", lambda: pytest.fail("check-trained must not pin HEAD"))
    monkeypatch.setattr("sys.argv", ["launch_016.py", "check-trained", "--size", "150M",
                                     "--only", CELL.tag])
    with pytest.raises(AssertionError, match="training todo"):
        L.main()



def test_dry_run_executes_the_real_paths_argv(monkeypatch, capsys):
    """A dry run goes through launch_006.submit (one submit path) which runs
    `submit.sh --dry-run` on the argv its real path builds, never a rebuilt one; launch_006's
    own dry run (run_dry False) only prints it."""
    monkeypatch.setattr(L, "HEAD", "abc123")
    runs = []
    monkeypatch.setattr(L.L6.subprocess, "run", lambda argv, **kw: runs.append((argv, kw)))
    opts = ["--gpus", "1", "--time", "12:00:00", "--name", "016-genjudge-x", "--after", "77"]
    assert L.submit(opts, "echo hi", True) == L.DRY_JOBID
    (argv, kw), = runs
    assert argv == L.L6.submit_argv(opts, "echo hi", True, commit="abc123")
    assert [x for x in argv if x != "--dry-run"] == \
        L.L6.submit_argv(opts, "echo hi", False, commit="abc123")
    assert argv[0] == str(L.L6.SUBMIT) and argv[1:3] == ["--commit", "abc123"]
    assert argv.index("--dry-run") < argv.index("--") < argv.index("bash")
    assert argv[argv.index("--") + 1:] == ["bash", "-c", "echo hi"]
    assert kw == {"check": True, "cwd": L.REPO_ROOT}
    assert capsys.readouterr().out.startswith("$ ")
    # launch_006's own dry path prints and returns without running anything
    assert L.L6.submit(opts, "echo hi", True, commit="abc123") == "<jobid>"
    assert len(runs) == 1
    # the real path is launch_006.submit with the same options and the pin
    monkeypatch.setattr(L.L6, "submit",
                        lambda o, c, d, commit=None, run_dry=False: f"{o}|{c}|{d}|{commit}")
    assert L.submit(opts, "echo hi", False) == f"{opts}|echo hi|False|abc123"


def test_submit_refuses_an_unset_pin(monkeypatch):
    """HEAD is "" until main() pins it; a submit before that (a REPL, a
    future entry point, a test without the monkeypatch) must fail, not run unpinned, dry or
    not. submit_argv itself refuses an empty pin (None is 006's unpinned path)."""
    monkeypatch.setattr(L.L6.subprocess, "run",
                        lambda *a, **k: pytest.fail("unpinned argv must never run"))
    monkeypatch.setattr(L, "HEAD", "")  # the module default; main() of an earlier test set it
    for dry in (True, False):
        with pytest.raises(AssertionError, match="HEAD not pinned"):
            L.submit(["--gpus", "1"], "echo hi", dry)
    with pytest.raises(AssertionError, match="empty --commit"):
        L.L6.submit_argv(["--gpus", "1"], "echo hi", True, commit="")
    assert "--commit" not in L.L6.submit_argv(["--gpus", "1"], "echo hi", True)
    assert L.L6.submit_argv([], "x", False, commit="abc")[1:3] == ["--commit", "abc"]


def test_dry_chain_passes_a_numeric_placeholder_after(root, monkeypatch):
    """With --chain --dry-run the gen+judge job's --after is DRY_JOBID
    (digits), so the printed sbatch script carries the real --dependency form, not '<jobid>'."""
    monkeypatch.setattr(sanityb, "check_base", lambda c: {})
    monkeypatch.setattr(L, "head", lambda: "abc123")
    monkeypatch.setattr(L.L6, "queued", lambda: set())
    runs = []
    monkeypatch.setattr(L.L6.subprocess, "run", lambda argv, **kw: runs.append(argv))
    monkeypatch.setattr("sys.argv", ["launch_016.py", "train", "--size", "150M", "--chain",
                                     "--dry-run", "--only", CELL.tag])
    L.main()
    assert len(runs) == 2 and all("--dry-run" in argv for argv in runs)
    train, gj = runs
    assert "--after" not in train and train[train.index("--gpus") + 1] == "8"
    assert gj[gj.index("--after") + 1] == L.DRY_JOBID and L.DRY_JOBID.isdigit()
    assert gj[gj.index("--name") + 1] == f"016-genjudge-{CELL.tag}"
    assert "check-trained" in gj[-1] and "gen_judge.sh" in gj[-1]


# ---- the launcher configuration of record and the A0 anchor -----------------------------------

spec11 = importlib.util.spec_from_file_location(
    "launch_011", pathlib.Path(__file__).parent.parent / "scripts" / "launch_011.py")
L11 = importlib.util.module_from_spec(spec11)
spec11.loader.exec_module(L11)
spec6s = importlib.util.spec_from_file_location(
    "signal_006", pathlib.Path(__file__).parent.parent / "scripts" / "signal_006.py")
S6 = importlib.util.module_from_spec(spec6s)
spec6s.loader.exec_module(S6)

ZERO1 = ladder.SFT_DS_SHA256
ID_006 = "dd-dclm-1b-default-dpo-seed1"  # a 006 model id: the by-name refusal
ENV = {"TOKENIZERS_PARALLELISM": "false", "BEAKER_ASSIGNED_CPU_COUNT": "96"}


def _sft_prov(cell, root, ds_sha=ZERO1, ds_path=ladder.SFT_DS, default=False, pre024=False):
    oi = {"deepspeed_config": ds_path}
    if not pre024:
        oi |= {"deepspeed_config_sha256": ds_sha, "deepspeed_config_default": default}
    d = {"stage": "sft", "post_training_seed_k": cell.k, "base_model": str(cell.base_dir),
         "base_ref": cell.base_ref, "finished": "x", "open_instruct": oi}
    if not pre024:
        d["launcher"] = {"env": ENV}
    _write(cell.out("sft", root), d)


def _dpo_prov(cell, root, pre024=False):
    oi = {"deepspeed_config": "configs/ds_configs/stage2_accelerate.conf"}
    if not pre024:
        oi |= {"deepspeed_config_sha256":
               ladder.Launcher().stages["dpo"]["deepspeed_config_sha256"],
               "deepspeed_config_default": True}
    ckpt = cell.out("dpo", root) / "oi" / "ck"
    ckpt.mkdir(parents=True, exist_ok=True)
    d = {"stage": "dpo", "post_training_seed_k": cell.k, "finished": "x",
         "base_model": str(cell.out("sft", root) / "model"), "checkpoint": str(ckpt),
         "open_instruct": oi}
    if not pre024:
        d["launcher"] = {"env": ENV}
    _write(cell.out("dpo", root), d)


def test_launcher_constants_are_the_ratified_configuration(tmp_path):
    """One configuration of record for 011, 008 and the ladder: the file 011 launched on, by
    path and by content. Launcher() reads the file, not the constant, and
    check_launcher_request compares a request's bytes with it."""
    assert (ladder.SFT_DS, ladder.SFT_DS_SHA256) == (L11.SFT_DS, L11.SFT_DS_SHA256)
    assert ladder.Launcher().stages["sft"]["deepspeed_config_sha256"] == ZERO1
    other = tmp_path / "ds.json"
    other.write_text('{"zero_optimization": {"stage": 3}}')
    with pytest.raises(AssertionError, match="configuration of record"):
        ladder.check_launcher_request(str(other), None)
    # a byte-identical copy elsewhere is the same configuration (sha, as castor.grid reads it)
    copy = tmp_path / "copy.json"
    copy.write_bytes(posttrain.ds_config_path("sft", ladder.SFT_DS).read_bytes())
    assert ladder.check_launcher_request(str(copy), None)["ds_config"] == {"sft": str(copy)}


def test_check_launcher_request_refuses_any_other_configuration():
    """`--ds-config` / `--launch-env` are plumbing, not a choice: the request must resolve to
    the configuration of record at both stages (same DeepSpeed bytes, same environment)."""
    assert ladder.check_launcher_request(ladder.SFT_DS, None) == ladder.LAUNCHER
    assert ladder.check_launcher_request(ladder.SFT_DS, []) == ladder.LAUNCHER
    with pytest.raises(AssertionError, match="configuration of record"):
        ladder.check_launcher_request(None, None)  # the current (ZeRO-3) configuration
    with pytest.raises(AssertionError, match="configuration of record"):
        ladder.check_launcher_request(ladder.SFT_DS, ["NCCL_CUMEM_ENABLE=0"])
    with pytest.raises(ValueError, match="not a launcher setting"):
        ladder.check_launcher_request(ladder.SFT_DS, ["PYTHONHASHSEED=3"])


def test_check_cell_launcher_reads_provenance(root):
    assert ladder.check_cell_launcher(CELL, root) == {}  # nothing trained yet
    _sft_prov(CELL, root)
    got = ladder.check_cell_launcher(CELL, root)
    assert set(got) == {"sft"} and got["sft"]["deepspeed_config_sha256"] == ZERO1
    assert "BEAKER_ASSIGNED_CPU_COUNT" not in got["sft"]["env"]  # follows the allocation
    _dpo_prov(CELL, root)
    assert set(ladder.check_cell_launcher(CELL, root)) == {"sft", "dpo"}
    # a chain whose SFT ran on another configuration is refused, never read as done
    other = ladder.Cell("150M", "default", 2)
    _sft_prov(other, root, ds_sha="0" * 64, ds_path="configs/posttrain/ds/sft_zero3_tuned.json")
    with pytest.raises(AssertionError, match="launcher configuration"):
        ladder.check_cell_launcher(other, root)
    # 006's twins: pre-024 provenance = the open-instruct default (ZeRO-3), refused
    old = ladder.Cell("150M", "default", 3)
    _sft_prov(old, root, ds_path="configs/ds_configs/stage3_no_offloading_accelerate.conf",
              pre024=True)
    with pytest.raises(AssertionError, match="launcher configuration"):
        ladder.check_cell_launcher(old, root)


def test_anchor_cells_are_the_a0_rerun(tmp_path):
    """The 1B point on the new configuration: 008's stage-A0 cells, 006's bases x seeds 1-4,
    with ids and paths disjoint from 006's and the ladder's."""
    cs = ladder.anchor_cells()
    assert len(cs) == 12 and len({c.model_id for c in cs}) == 12
    assert {c.base_ref for c in cs} == {c.base_ref for c in sanityb.cells()}
    assert {c.k for c in cs} == {1, 2, 3, 4} and all(c.size == "1B" for c in cs)
    assert cs[0].model_id == "dd-dclm-baseline-1b-default-dpo-seed1"
    assert cs[0].out("sft").parts[-3:] == ("dd-dclm-baseline-1B-seed-default", "sft", "seed1")
    ids = {c.model_id for c in cs}
    assert not ids & {c.model_id for c in sanityb.cells()}
    assert not ids & {c.model_id for c in ladder.cells()}
    for attr in ("k", "revision", "base_ref", "base_dir", "group", "model_id", "tag", "info"):
        assert hasattr(cs[0], attr)
    a, b, c = cs[0], cs[1], next(x for x in cs if x.pretrain_seed == "large-aux-2")
    assert reliability.pair_level(a.info, b.info) == "MZ"
    assert reliability.pair_level(a.info, c.info) == "U"
    assert a.info.recipe == sanityb.RECIPE  # the same level as 006's twins (gridstats/analyze)
    # the manifest must be 006's instrument and hold exactly those 12 cells
    base = json.loads(ladder.GRID_008.read_text())
    for kw, msg in [({"prompt_id": "009"}, "prompt_id"),
                    ({"sft_config": "configs/posttrain/x.yaml", "variant": "x"}, "post-training configs"),
                    ({"bases": [b for b in base["bases"] if "dclm-baseline-1B" not in b["repo"]
                                or b["revision"].endswith("default")]}, "main-recipe cells"),
                    ({"bases": [b | {"posttrain_seeds": [1, 2]} if "posttrain_seeds" in b else b
                                for b in base["bases"]]}, "main-recipe cells")]:
        p = tmp_path / "m.json"
        p.write_text(json.dumps(base | kw))
        with pytest.raises(ValueError, match=msg):
            ladder.anchor_cells(p)


def test_check_anchor_signal_refuses_006_and_a_different_gap(tmp_path):
    """The anchor file must be the A0 twins' (by model id) and agree with the recomputation;
    runs/006/signal.json (006's ZeRO-3 twins) is refused by name."""
    cs = ladder.anchor_cells()
    s006 = json.loads((pathlib.Path(__file__).parent.parent / "runs/006/signal.json").read_text())
    with pytest.raises(AssertionError, match=ID_006):  # by model id, not by the file's name
        ladder.Anchor(s006, tmp_path / "x.json", cs)
    with pytest.raises(AssertionError, match="not an Anchor"):  # a raw dict never reaches it
        ladder.check_anchor_signal(s006, 0.2)
    sig = {"models": {c.model_id: {"base": c.base_ref} for c in cs}
           | {"olmo2-1b-dpo-ai2": {"reference": True}},
           sanityb.PRIMARY_SUBSET: {"primary": {"levels": {"MZ": {"r_corrected": 0.9},
                                                           "U": {"r_corrected": 0.7}}}},
           "git_commit": "abc", "script": "scripts/signal_006.py"}
    anchor = ladder.Anchor(sig, tmp_path / "a0.json", cs)
    r = ladder.check_anchor_signal(anchor, 0.2 + 1e-12)
    assert r["gap_signal"] == pytest.approx(0.2) and r["signal_git_commit"] == "abc"
    assert r["twins"] == anchor.twins == sorted(c.model_id for c in cs)
    assert r["signal"] == str(tmp_path / "a0.json")
    with pytest.raises(AssertionError, match="gap"):
        ladder.check_anchor_signal(anchor, 0.21)
    short = sig | {"models": {k: v for k, v in sig["models"].items() if not k.endswith("seed4")}}
    with pytest.raises(AssertionError, match="A0 rerun cells"):
        ladder.Anchor(short, tmp_path / "a0.json", cs)


def test_signal_006_twins_option_default_unchanged():
    """`--twins a0` = the same script on the A0 twins (labels under runs/<id>/008/); the default
    is 006's cells and prompt id, as before."""
    assert S6.TWINS["006"] == (sanityb.cells, sanityb.PROMPT_ID)
    assert S6.TWINS["a0"] == (ladder.anchor_cells, ladder.ANCHOR_PROMPT_ID)
    assert ladder.ANCHOR_PROMPT_ID == "008" and sanityb.PROMPT_ID == "006"
    assert ladder.ANCHOR_SIGNAL.parts[-3:] == ("runs", "008", "signal_a0.json")


def test_main_train_passes_the_configuration_through_006_train_cmd(root, monkeypatch):
    """--ds-config rides on launch_006.train_cmd's ds_config: the SFT command
    is 006's plus `--ds-config <ratified file>`, the DPO command is 006's unchanged; the
    default is the configuration of record without any flag on the command line."""
    monkeypatch.setattr(sanityb, "check_base", lambda c: {})
    monkeypatch.setattr(L, "head", lambda: "abc123")
    monkeypatch.setattr(L.L6, "queued", lambda: set())
    calls = []
    monkeypatch.setattr(L.L6, "submit",
                        lambda opts, cmd, dry, commit=None, run_dry=False:
                        calls.append((opts, cmd)) or "7")
    monkeypatch.setattr("sys.argv", ["launch_016.py", "train", "--size", "150M",
                                     "--only", CELL.tag])
    L.main()
    (opts, cmd), = calls
    sft, dpo = cmd.split(" && ")
    psft, pdpo = L.L6.train_cmd(CELL, root, None).split(" && ")
    assert sft == psft + " --ds-config " + ladder.SFT_DS and dpo == pdpo
    assert "--launch-env" not in cmd
    assert cmd == L.L6.train_cmd(CELL, root, None, ds_config={"sft": ladder.SFT_DS})
    # the explicit flag gives the same command; the smoke root is per configuration
    calls.clear()
    monkeypatch.setattr("sys.argv", ["launch_016.py", "train", "--size", "150M", "--only",
                                     CELL.tag, "--smoke", "50", "--dev", "--ds-config",
                                     ladder.SFT_DS])
    L.main()
    (opts, cmd), = calls
    assert "--ds-config " + ladder.SFT_DS in cmd and "max_train_steps=50" in cmd
    assert str(L.smoke_root(ladder.SFT_DS)) in cmd and "--dev" in opts
    assert L.smoke_root(ladder.SFT_DS).parts[-3:] == ("smoke", "016", "sft_zero1")
    assert str(ladder.MODELS_ROOT / "smoke" / "016" / CELL.group) not in cmd  # ZeRO-3 dry runs


@pytest.mark.parametrize("argv, exc, msg", [
    (["--ds-config", "configs/posttrain/ds/sft_zero3_tuned.json"], AssertionError,
     "configuration of record"),
    (["--launch-env", "NCCL_CUMEM_ENABLE=0"], AssertionError, "configuration of record"),
    (["--launch-env", "SEED=3"], ValueError, "not a launcher setting"),
])
def test_main_refuses_another_configuration_before_squeue(monkeypatch, argv, exc, msg):
    monkeypatch.setattr("sys.argv", ["launch_016.py", "train", "--size", "150M", *argv])
    monkeypatch.setattr(L, "head", lambda: pytest.fail("configuration checks come first"))
    monkeypatch.setattr(L.L6, "queued", lambda: pytest.fail("configuration checks come first"))
    with pytest.raises(exc, match=msg):
        L.main()


def test_status_and_check_trained_refuse_a_chain_on_another_configuration(root, monkeypatch):
    """A ladder cell with provenance from another launcher configuration is never 'done' (so
    never skipped at launch) and never generated (check-trained inside the gen + judge job)."""
    _sft_prov(CELL, root)
    _dpo_prov(CELL, root)
    assert L.status(CELL, root) == "done"
    bad = ladder.Cell("150M", "default", 1)
    _sft_prov(bad, root, ds_path="configs/ds_configs/stage3_no_offloading_accelerate.conf",
              pre024=True)
    _dpo_prov(bad, root, pre024=True)
    assert L.L6.status(bad, root) == "done"  # 006's rule alone would skip it as done
    with pytest.raises(AssertionError, match="launcher configuration") as e:
        L.status(bad, root)
    assert "mappingproxy" not in str(e.value)  # the record printed as a plain dict
    monkeypatch.setattr(L, "head", lambda: pytest.fail("check-trained must not pin HEAD"))
    monkeypatch.setattr("sys.argv", ["launch_016.py", "check-trained", "--size", "150M",
                                     "--only", bad.tag])
    with pytest.raises(AssertionError, match="launcher configuration"):
        L.main()
    monkeypatch.setattr("sys.argv", ["launch_016.py", "check-trained", "--size", "150M",
                                     "--only", CELL.tag])
    L.main()  # the ZeRO-1 chain passes


def test_grid_008_is_repo_anchored(tmp_path, monkeypatch):
    """The manifest path is repo-anchored like ANCHOR_SIGNAL, so `anchor_cells()` (signal_006
    --twins a0, signal_016) works from /work or a job snapshot."""
    assert ladder.GRID_008.is_absolute()
    assert ladder.GRID_008 == grid.ROOT / "data/manifests/grid_008.json"
    monkeypatch.chdir(tmp_path)
    assert len(ladder.anchor_cells()) == 12


def test_launcher_pins_the_configuration_of_record(monkeypatch):
    """The ratified sha is enforced where every entry point resolves the configuration of
    record, not only in the launcher's request check. An edited sft_zero1.json is refused by
    Launcher() (the type's constructor is the one pin); a request for another configuration
    still resolves (castor.grid, and is then refused by the comparison)."""
    assert ladder.Launcher().stages["sft"]["deepspeed_config_sha256"] == ZERO1
    monkeypatch.setattr(ladder, "SFT_DS_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="not the ratified configuration"):
        ladder.Launcher()
    other = {"ds_config": None, "launch_env": None}
    assert grid.resolve_launcher(other)["sft"]["deepspeed_config_sha256"] != ZERO1
    with pytest.raises(ValueError, match="not the ratified configuration"):
        ladder.check_launcher_request(ladder.SFT_DS, None)
    with pytest.raises(ValueError, match="not the ratified configuration"):
        ladder.check_cell_launcher(CELL)


def test_launcher_is_resolved_once_per_invocation(root, monkeypatch):
    """`want=` passes the resolved configuration through, so a survey of many cells hashes the
    DeepSpeed files once (castor.grid.resolve_launcher's contract), and launch_016.main
    resolves once in all, whatever the cell count."""
    want = ladder.Launcher()
    _sft_prov(CELL, root)
    _dpo_prov(CELL, root)
    calls = []
    real = grid.resolve_launcher
    monkeypatch.setattr(grid, "resolve_launcher",
                        lambda launcher: calls.append(launcher) or real(launcher))
    assert set(ladder.check_cell_launcher(CELL, root, want=want)) == {"sft", "dpo"}
    assert L.status(CELL, root, want=want) == "done"
    assert calls == []  # the cell checks hash nothing
    assert ladder.check_launcher_request(ladder.SFT_DS, None, want) == ladder.LAUNCHER
    assert calls == []  # the default request is the record: nothing hashed again
    for k in (1, 2):
        c = ladder.Cell("150M", "default", k)
        _sft_prov(c, root)
        _dpo_prov(c, root)  # resolves once itself (the DPO sha of the fixture)
    calls.clear()
    monkeypatch.setattr(L, "head", lambda: "abc123")
    monkeypatch.setattr(L.L6, "queued", lambda: set())
    monkeypatch.setattr("sys.argv", ["launch_016.py", "genjudge", "--size", "150M", "--only",
                                     "150M-default-s1", "150M-default-s2", CELL.tag])
    monkeypatch.setattr(L.L6, "genjudge_status", lambda c, root, pid: "done")
    L.main()
    assert len(calls) == 1


def test_anchor_block_skips_only_under_allow_incomplete(tmp_path):
    """Before the A0 twins are judged, runs/008/signal_a0.json does not exist and the 1B labels
    are partial; --allow-incomplete records that instead of raising after every size's
    statistics. Without the flag, both are errors; with the file and complete labels, the
    cross-check runs."""
    cs = ladder.anchor_cells()
    where = tmp_path / "signal_a0.json"
    sig = {"models": {c.model_id: {"base": c.base_ref} for c in cs},
           sanityb.PRIMARY_SUBSET: {"primary": {"levels": {"MZ": {"r_corrected": 0.9},
                                                           "U": {"r_corrected": 0.7}}}}}
    anchor, missing = ladder.Anchor(sig, where, cs), ladder.MissingAnchor(where)
    assert missing != ladder.MissingAnchor(where) and hash(missing)  # identity, as Anchor
    with pytest.raises(AssertionError, match="does not exist"):
        ladder.anchor_block(missing, 0.2, True, False)
    with pytest.raises(AssertionError, match="not all labelled"):
        ladder.anchor_block(anchor, 0.2, False, False)
    r = ladder.anchor_block(missing, 0.2, False, True)
    assert r["checked"] is False and "does not exist" in r["reason"] \
        and "not all labelled" in r["reason"] and r["gap_recomputed"] == 0.2
    r = ladder.anchor_block(anchor, 0.2, False, True)
    assert r["checked"] is False and "does not exist" not in r["reason"]
    r = ladder.anchor_block(anchor, 0.2, True, True)
    assert r["checked"] is True and r["gap_signal"] == pytest.approx(0.2)
    with pytest.raises(AssertionError, match="gap"):  # the file and complete labels: checked
        ladder.anchor_block(anchor, 0.3, True, True)
    # the source is the object's own, on either branch; there is no
    # second `where` to disagree with it, and nothing but read_anchor's two results is taken
    other = tmp_path / "other.json"
    assert ladder.anchor_block(ladder.MissingAnchor(other), 0.2, False, True)["signal"] \
        == str(other)
    assert ladder.anchor_block(anchor, 0.2, True, True)["signal"] == str(where)
    for bad in (None, sig, str(where), where):
        with pytest.raises(AssertionError, match="not an Anchor or MissingAnchor"):
            ladder.anchor_block(bad, 0.2, True, True)


def test_rule_is_incomplete_while_the_anchor_is_unchecked():
    """Under --allow-incomplete with every label complete but no anchor file, the rule is
    INCOMPLETE until the recomputed 1B gap is confirmed against the file of record; the shape
    is the complete run's."""
    ys, ids, info = _ladder(n=100)
    boot = ladder.paired_size_bootstrap(ys, ids, info, n_boot=5)
    ok = {z: True for z in ladder.TREND}
    full = RULE(boot, ok)
    part = RULE(boot, ok, anchor_checked=False)
    assert full["holds"] is True and full["anchor_checked"]
    assert part["holds"] is None and part["verdict"] == "INCOMPLETE"
    assert "anchor" in part["reason"] and not part["anchor_checked"]
    assert set(part) == set(full) and part["ordered"]
    # labels first: a run with both missing names the labels
    both = RULE(boot, ok, labels_complete=False, anchor_checked=False)
    assert both["verdict"] == "INCOMPLETE" and "labelled" in both["reason"]


def test_anchor_twins_are_checked_whenever_the_file_exists(tmp_path):
    """The by-name refusal of a wrong anchor file (runs/006/signal.json) does not depend on the
    labels, so it runs under --allow-incomplete with partial 1B labels too; only the gap
    comparison waits for complete labels. anchor_block takes an `Anchor`, which a wrong file
    cannot become, and refuses a raw dict. The match is a 006 model id."""
    cs = ladder.anchor_cells()
    where = tmp_path / "signal_a0.json"
    s006 = (pathlib.Path(__file__).parent.parent / "runs/006/signal.json").read_text()
    where.write_text(s006)
    with pytest.raises(AssertionError, match=ID_006):
        ladder.read_anchor(cs, where, allow_incomplete=True)
    with pytest.raises(AssertionError, match=ID_006):
        ladder.check_anchor_twins(json.loads(s006), cs, where)
    with pytest.raises(AssertionError, match=ID_006):
        ladder.Anchor(json.loads(s006), where, cs)
    with pytest.raises(AssertionError, match="not an Anchor"):  # partial labels, wrong file
        ladder.anchor_block(json.loads(s006), 0.3, False, True)
    sig = {"models": {c.model_id: {"base": c.base_ref} for c in cs},
           sanityb.PRIMARY_SUBSET: {"primary": {"levels": {"MZ": {"r_corrected": 0.9},
                                                           "U": {"r_corrected": 0.7}}}}}
    assert ladder.check_anchor_twins(sig, cs, where) == sorted(c.model_id for c in cs)
    where.write_text(json.dumps(sig))
    anchor = ladder.read_anchor(cs, where, allow_incomplete=True)
    assert isinstance(anchor, ladder.Anchor) and anchor.twins == sorted(c.model_id for c in cs)
    r = ladder.anchor_block(anchor, 0.3, False, True)  # gap differs: not compared yet
    assert r["checked"] is False and "not all labelled" in r["reason"]


def test_anchor_twins_are_checked_once_per_run(tmp_path, monkeypatch):
    """read_anchor -> anchor_block -> check_anchor_signal runs the twin check exactly once (at
    read_anchor, through Anchor's constructor)."""
    cs = ladder.anchor_cells()
    where = tmp_path / "signal_a0.json"
    where.write_text(json.dumps({
        "models": {c.model_id: {"base": c.base_ref} for c in cs},
        sanityb.PRIMARY_SUBSET: {"primary": {"levels": {"MZ": {"r_corrected": 0.9},
                                                        "U": {"r_corrected": 0.7}}}}}))
    calls = []
    real = ladder.check_anchor_twins
    monkeypatch.setattr(ladder, "check_anchor_twins",
                        lambda *a, **k: calls.append(a) or real(*a, **k))
    anchor = ladder.read_anchor(cs, where)
    r = ladder.anchor_block(anchor, 0.2, True, False)
    assert r["checked"] and r["twins"] == anchor.twins
    assert len(calls) == 1


def test_read_anchor_owns_the_missing_file_policy(tmp_path):
    """One home for 'the anchor file does not exist' (signal_016's early assertion and
    anchor_block's reason both come from ladder); ANCHOR_SIGNAL is anchored on the same root as
    GRID_008."""
    cs = ladder.anchor_cells()
    where = tmp_path / "signal_a0.json"
    with pytest.raises(AssertionError, match="does not exist"):
        ladder.read_anchor(cs, where)
    missing = ladder.read_anchor(cs, where, allow_incomplete=True)
    assert isinstance(missing, ladder.MissingAnchor) and missing.where == where
    sig = {"models": {c.model_id: {} for c in cs}}
    where.write_text(json.dumps(sig))
    assert ladder.read_anchor(cs, where).signal == sig
    r = ladder.anchor_block(missing, 0.2, True, True)
    assert r["reason"] == ladder._anchor_missing(where) and r["signal"] == str(where)
    assert ladder.ANCHOR_SIGNAL == grid.ROOT / "runs" / "008" / "signal_a0.json"
    assert ladder.ANCHOR_SIGNAL.parents[2] == ladder.GRID_008.parents[2] == grid.ROOT


def test_anchor_check_refuses_empty_cells(tmp_path):
    """check_anchor_twins, read_anchor and Anchor refuse empty cells; read_anchor refuses them
    before the file is read, so an existing file that is not JSON is never parsed."""
    where = tmp_path / "signal_a0.json"
    where.write_text('{"models": {}}')
    not_json = tmp_path / "not_json.json"
    not_json.write_text("{")
    for cells in ([], ()):
        with pytest.raises(AssertionError, match="no A0 cells"):
            ladder.read_anchor(cells, where)
        with pytest.raises(AssertionError, match="no A0 cells"):  # not JSONDecodeError
            ladder.read_anchor(cells, not_json)
        with pytest.raises(AssertionError, match="no A0 cells"):
            ladder.read_anchor(cells, tmp_path / "absent.json", allow_incomplete=True)
        with pytest.raises(AssertionError, match="no A0 cells"):
            ladder.check_anchor_twins({"models": {}}, cells, where)
        with pytest.raises(AssertionError, match="no A0 cells"):
            ladder.Anchor({"models": {}}, where, cells)


def test_anchor_is_hashable_identity(tmp_path):
    """An Anchor is an identity (eq=False): hashable, never compared field-wise."""
    cs = ladder.anchor_cells()
    sig = {"models": {c.model_id: {} for c in cs}}
    a, b = ladder.Anchor(sig, tmp_path / "a.json", cs), ladder.Anchor(sig, tmp_path / "a.json", cs)
    same = a
    assert hash(a) == hash(same) and a == same and a != b and len({a, b, same}) == 2
    assert a.twins == b.twins


def test_launcher_is_the_type_whose_constructor_is_the_pin(root, monkeypatch):
    """`want` must be a `ladder.Launcher`, which only `Launcher()` makes (resolve + pin): a dict
    carrying the record's own values is refused by every taker. `stages` is a deep read-only
    view of exactly the record's shape (`as_dict()` is a copy); `_launcher` repeats the pin on
    `want`, so a Launcher built before the constant changed is refused and named as stale; the
    checks raise (they must hold under python -O); a refusal prints the record as a plain
    dict."""
    want = ladder.Launcher()
    assert ladder.check_launcher_request(ladder.SFT_DS, None, want) == ladder.LAUNCHER
    with pytest.raises(AssertionError, match="configuration of record") as e:
        ladder.check_launcher_request(None, None, want)
    assert "mappingproxy" not in str(e.value)
    same = want
    assert hash(want) == hash(same) and want == same and want != ladder.Launcher()  # identity
    plain = want.as_dict()  # the record's values, as a plain dict: JSON-serialisable, a copy
    assert json.loads(json.dumps(plain)) == plain == want.stages \
        and set(plain) == {"sft", "dpo"} and set(plain["sft"]) == {"deepspeed_config_sha256", "env"}
    for mutate in (lambda: want.stages.__setitem__("sft", {}),
                   lambda: want.stages["sft"].__setitem__("deepspeed_config_sha256", "0" * 64),
                   lambda: want.stages["dpo"]["env"].__setitem__("X", "y"),
                   lambda: want.stages["sft"].pop("env")):
        with pytest.raises((TypeError, AttributeError)):
            mutate()
    plain["sft"]["deepspeed_config_sha256"] = "0" * 64  # the copy; the Launcher is untouched
    assert want.stages["sft"]["deepspeed_config_sha256"] == ZERO1 \
        and want.as_dict() != plain and want.as_dict() is not want.as_dict()
    forged = {st: dict(v) for st, v in want.stages.items()}  # the record's values, as a dict
    stale = forged | {"sft": forged["sft"] | {"deepspeed_config_sha256": "0" * 64}}
    _sft_prov(CELL, root)
    for bad in (want.stages, forged, stale):
        with pytest.raises(TypeError, match="not a ladder.Launcher"):
            ladder.check_launcher_request(ladder.SFT_DS, None, bad)
        with pytest.raises(TypeError, match="not a ladder.Launcher"):
            ladder.check_launcher_request(None, None, bad)
        with pytest.raises(TypeError, match="not a ladder.Launcher"):
            ladder.check_cell_launcher(CELL, root, want=bad)
        with pytest.raises(TypeError, match="not a ladder.Launcher"):
            L.status(CELL, root, bad)
    with pytest.raises(TypeError):  # no way to build one from values
        ladder.Launcher(want.stages)
    monkeypatch.setattr(ladder, "SFT_DS_SHA256", "0" * 64)
    with pytest.raises(ValueError, match=f"{ladder.SFT_DS} is .* not the ratified configuration"):
        ladder.Launcher()
    # a Launcher built before the constant changed is stale, and every taker says so, naming
    # the Launcher (the file on disk may already be the new configuration)
    stale_msg = "Launcher handed as want .* not the ratified configuration"
    with pytest.raises(ValueError, match=stale_msg):
        ladder.check_launcher_request(ladder.SFT_DS, None, want)
    with pytest.raises(ValueError, match=stale_msg):
        ladder.check_launcher_request(None, None, want)
    with pytest.raises(ValueError, match=stale_msg):
        ladder.check_cell_launcher(CELL, root, want=want)
    with pytest.raises(ValueError, match=stale_msg):
        L.status(CELL, root, want)


def test_cell_launcher_check_has_no_launcher_argument(root):
    """The ladder has one configuration, so the check takes only `want` (= ladder.Launcher()),
    never a `launcher` argument that could disagree with it."""
    import inspect
    assert "launcher" not in inspect.signature(ladder.check_cell_launcher).parameters
    assert "launcher" not in inspect.signature(L.status).parameters
    _sft_prov(CELL, root)
    _dpo_prov(CELL, root)
    assert L.status(CELL, root, ladder.Launcher()) == "done"
