"""Forecast each run's post-trained per-prompt jailbreak residual from readouts of its base model
(definitions in castor/forecast.py). CPU; reads the runs/008 judge_v1 labels, the base readouts of
scripts/extract_077.py and the refusal directions stored by runs/021.

    uv run --frozen python scripts/forecast_077.py --out /abs/path/runs/077 \
        [--readouts /work/harrychang08/castor/analysis/077] [--n-boot 1000] [--n-sub 200]

Readouts per base and prompt:
  a    prefix log-odds: logsumexp over the five compliance prefixes' log-probabilities minus the
       same over the five refusal prefixes (primary)
  b    base harm-direction projection at (13, −1): resid_pre on the unit difference of means
       harmful − harmless (021's extraction training prompts) fitted in the same base
  b8   the same at (13, −8)
  c    prompt loss: mean NLL of the templated prompt's tokens
  d    base resid_pre at (13, −1) on the unit refusal direction 021 stores for post-training
       seed s of that base (candidates.safetensors mean_diff[13, −1]); target = seed s′ ≠ s
Targets: per-prompt P(non-refusal | coherent), judge_v1 `wg_vulnerable`, adversarial-harmful,
averaged over post-training seeds {1, 2} (grid.json's models); residual = minus the recipe mean.
Writes summary.json (every number, with `where`), per_run.tsv and curve.tsv (figure data).
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from castor import evalset, fingerprint, forecast, grid, gridstats, provenance, sanityb
from castor.judge.common import read_jsonl

FIELD, SUBSET = "wg_vulnerable", sanityb.PRIMARY_SUBSET
GRID_JSON = grid.ROOT / "runs" / "008" / "grid.json"
ACT021 = Path("/work/harrychang08/castor/activations/021")
SEEDS = (1, 2)
COUNTS = (50, 200, 1000, 3211)  # pre-registered
READOUTS = ("a", "b", "b8", "c")
SIGN_NOTE = ("signed Pearson r; a: compliance minus refusal, so + = base favours compliance on "
             "the prompts its run later finds vulnerable; b/b8/d: + = projects more on the "
             "harmful-minus-harmless direction; c: + = higher prompt loss")


def _y(args):
    path, ids = args
    y, c = sanityb.outcome_matrix(read_jsonl(path), ids, FIELD)
    assert c["missing"] == 0, f"{path}: {c['missing']} missing label cells"
    return y


def lse(a: np.ndarray) -> np.ndarray:
    m = a.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=1, keepdims=True)))[:, 0]


def unit(v):
    return v / np.linalg.norm(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--readouts", type=Path, default=Path("/work/harrychang08/castor/analysis/077"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-sub", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    assert a.out.is_absolute()
    commit, t0 = provenance.git_commit(), time.time()
    cmd = provenance.command()
    panel = evalset.load_eval()
    ids = sanityb.subset_ids(panel, SUBSET)
    assert len(ids) == 3212

    gj = json.loads(GRID_JSON.read_text())
    gm = gj["models"]
    assert len(gm) == 150 and {v["posttrain_seed_k"] for v in gm.values()} == set(SEEDS)
    extra = {f"dd-dclm-baseline-1b-{b}-dpo-seed{s}": Path(
        f"/work/harrychang08/castor/runs/dd-dclm-baseline-1b-{b}-dpo-seed{s}/008/judge_v1/"
        "labels.jsonl") for b in ("default", "large-aux-2", "large-aux-3") for s in (3, 4)}
    paths = {m: Path(v["labels"]) for m, v in gm.items()} | extra
    names = sorted(paths)
    with ProcessPoolExecutor(a.workers) as ex:
        ys = dict(zip(names, ex.map(_y, [(str(paths[m]), ids) for m in names]), strict=True))
    P, Po, Pe = gridstats.halves(ys, names)
    row = {m: i for i, m in enumerate(names)}
    pm = gj[SUBSET]["primary"]["per_model"]
    for m in gm:  # instrument check against runs/008/grid.json
        assert abs(float(np.nanmean(P[row[m]])) - pm[m]["mean_vulnerability"]) < 1e-12, m
    grid_rows = [row[m] for m in gm]
    keep = np.isfinite(P[grid_rows]).all(axis=0)
    dropped = [i for i, k in zip(ids, keep, strict=True) if not k]
    assert dropped == ["wildjailbreak-00924"], dropped
    P, Po, Pe = P[:, keep], Po[:, keep], Pe[:, keep]
    kept_ids = [i for i, k in zip(ids, keep, strict=True) if k]
    n_p = len(kept_ids)
    key = {(gm[m]["recipe"], gm[m]["pretrain_seed"], gm[m]["posttrain_seed_k"]): m for m in gm}
    recipes = sorted({k[0] for k in key})
    bases = sorted({k[1] for k in key})
    runs = [(r, b) for r in recipes for b in bases]
    g = np.array([r for r, _ in runs])

    def prof(seeds, extra_main=()):
        out = []
        for r, b in runs:
            ss = list(seeds) + (list(extra_main) if r == "dclm-baseline" else [])
            mm = [key[(r, b, s)] if s in SEEDS else f"dd-dclm-baseline-1b-{b}-dpo-seed{s}"
                  for s in ss]
            out.append([np.mean([M[row[m]] for m in mm], axis=0) for M in (P, Po, Pe)])
        return tuple(np.stack([o[i] for o in out]) for i in range(3))

    Y, Yo, Ye = prof(SEEDS)

    gr = grid.load(grid.ROOT / "data/manifests/grid_008.json")
    bmap = {(b.recipe, b.pretrain_seed): b for b in gr.bases}
    X = {k: [] for k in READOUTS}
    acts_m1, prov = [], {}
    for r, b in runs:
        d = a.readouts / bmap[(r, b)].group
        p = json.loads((d / "provenance.json").read_text())
        assert p["ids"] == ids and p["limit"] is None and p["tail_mismatch"] == 0
        prov[f"{r}/{b}"] = {k: p[k] for k in ("base", "git_commit", "slurm_job", "seconds",
                                              "align_max_abs_first_token")}
        t = load_file(str(d / "readouts.safetensors"))
        lp = t["logp"][keep]
        X["a"].append(lse(lp[:, :5]) - lse(lp[:, 5:]))
        acts = t["acts"][keep].astype(np.float64)
        hd = t["harm_dir"]
        for k, j in (("b", 1), ("b8", 0)):
            u = unit(hd[j])
            assert (hd[j] @ u) > 0
            X[k].append(acts[:, j] @ u)
        X["c"].append(t["prompt_nll"][keep])
        acts_m1.append(acts[:, 1])
    X = {k: np.stack(v) for k, v in X.items()}
    commits = sorted({v["git_commit"] for v in prov.values()})

    out = {"design": {"n_runs": len(runs), "n_recipes": len(recipes), "n_prompts": n_p,
                      "dropped": dropped, "posttrain_seeds": list(SEEDS), "sign": SIGN_NOTE,
                      "bases_order": bases},
           "extraction": {"commits": commits, "per_base": prov,
                          "align_max_abs_first_token": max(v["align_max_abs_first_token"]
                                                           for v in prov.values())}}

    def block(arms, groups, residual=True):
        pr = forecast.per_run_avg(arms, groups, residual)
        rb = forecast.recipe_bootstrap(pr, groups, a.n_boot, a.seed)
        pb, per_rec = forecast.prompt_bootstrap(arms, groups, a.n_boot, a.seed, residual)
        return pr, {"point": forecast.summarise(pr), "ci_recipes": rb, "ci_prompts": pb,
                    "per_recipe": per_rec,
                    "n_recipes_ci_excludes_zero": int(sum(
                        (v["lo"] > 0) or (v["hi"] < 0) for v in per_rec.values()))}

    res, per_run_rows = {}, {}
    for k in READOUTS:
        pr, res[k] = block([(X[k], Y, Yo, Ye)], g)
        _, res[f"{k}_level"] = block([(X[k], Y, Yo, Ye)], g, residual=False)
        per_run_rows[k] = pr
        print(k, json.dumps(res[k]["point"]), flush=True)

    # sensitivity: the main recipe's runs averaged over its four seeds
    Y4 = prof(SEEDS, (3, 4))
    _, res["a_main4"] = block([(X["a"], *Y4)], g)

    # (d) descendant refusal direction, leave-one-descendant-out
    rd = {}
    for (r, b) in runs:
        for s in SEEDS:
            f = ACT021 / key[(r, b, s)] / "candidates.safetensors"
            if f.exists():
                rd[(r, b, s)] = unit(load_file(str(f))["mean_diff"][13, -1])
    d_runs = [i for i, (r, b) in enumerate(runs) if all((r, b, s) in rd for s in SEEDS)]
    # a recipe enters only with all three runs (its residual needs the recipe mean)
    d_rec = [q for q in recipes if sum(g[i] == q for i in d_runs) == 3]
    d_runs = [i for i in d_runs if g[i] in d_rec]
    gd = g[d_runs]
    single = {s: prof((s,)) for s in SEEDS}
    arms_d, arms_a, arms_b = [], [], []
    for s, s2 in ((1, 2), (2, 1)):
        Xd = np.stack([acts_m1[i] @ rd[(*runs[i], s)] for i in d_runs])
        T = tuple(M[d_runs] for M in single[s2])
        arms_d.append((Xd, *T))
        arms_a.append((X["a"][d_runs], *T))
        arms_b.append((X["b"][d_runs], *T))
    _, res["d"] = block(arms_d, gd)
    _, res["d_same_subset_a"] = block(arms_a, gd)
    _, res["d_same_subset_b"] = block(arms_b, gd)
    res["d"]["n_runs"], res["d"]["recipes"] = len(d_runs), d_rec
    print("d", json.dumps(res["d"]["point"]), flush=True)

    # reverse fingerprint (a)
    Rx = forecast.recipe_residual(X["a"], g)
    qk = [(r, b, s) for r, b in runs for s in SEEDS]
    rmean = Y - forecast.recipe_residual(Y, g)  # each run's row: its recipe's mean profile
    # base readouts use no label, so no query can leak into a reference
    Q = np.stack([P[row[key[k]]] - rmean[runs.index(k[:2])] for k in qk])
    truth = [runs.index(k[:2]) for k in qk]
    qg = [k[0] for k in qk]
    rev = forecast.reverse_fingerprint(Q, Rx, truth, qg, list(g))
    curve = forecast.reverse_curve(Q, Rx, truth, qg, list(g), COUNTS, a.n_sub, a.seed)
    res["reverse_a"] = rev | {"curve": curve}
    # cross-check with castor.fingerprint
    assert rev["all"]["n_correct"] == int(sum(fingerprint.assign(
        fingerprint.corr_cross(Q, Rx)) == np.array(truth)))
    print("reverse", json.dumps({k: rev[k] for k in ("all", "within_recipe")}), flush=True)
    # post hoc: the two seeds of a run are near-duplicate queries, so the binomial interval over
    # 150 overstates n; run-level queries (seed mean) and seed agreement
    C = fingerprint.corr_cross(Q, Rx)
    same = np.array(qg)[:, None] == g[None, :]
    got = forecast.assign_within(C, same)
    by_run = got.reshape(len(runs), len(SEEDS))
    rq = forecast.reverse_fingerprint(Y - rmean, Rx, list(range(len(runs))), list(g), list(g))
    res["reverse_a_posthoc"] = {
        "run_level": rq,
        "seeds_pick_same_base": int((by_run[:, 0] == by_run[:, 1]).sum()),
        "runs_both_correct": int(((by_run == np.arange(len(runs))[:, None]).all(1)).sum()),
        "runs_one_correct": int(((by_run == np.arange(len(runs))[:, None]).sum(1) == 1).sum()),
        "n_runs": len(runs)}
    print("reverse_posthoc", json.dumps(res["reverse_a_posthoc"]), flush=True)

    out["results"] = res
    w = f"runs/077/summary.json → results.<readout>; scripts/forecast_077.py @ {commit}"
    out |= {"where": w, "script": "scripts/forecast_077.py", "git_commit": commit,
            "command": cmd, "n_boot": a.n_boot, "n_sub": a.n_sub, "seed": a.seed,
            "seconds": time.time() - t0}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    with open(a.out / "per_run.tsv", "w") as f:
        f.write("recipe\tbase\treadout\trel\tmatched_c\tmismatched_c\n")
        for k, pr in per_run_rows.items():
            for i, (r, b) in enumerate(runs):
                f.write(f"{r}\t{b}\t{k}\t{pr['rel'][i]:.6f}\t{pr['matched_c'][i]:.6f}\t"
                        f"{pr['mismatched_c'][i]:.6f}\n")
    with open(a.out / "curve.tsv", "w") as f:
        f.write("n_prompts\tset\tmean\tlo\thi\n")
        for c in curve:
            for k in ("all", "within_recipe"):
                f.write(f"{c['n_prompts']}\t{k}\t{c[k]['mean']:.6f}\t{c[k]['lo']:.6f}\t"
                        f"{c[k]['hi']:.6f}\n")
    print(json.dumps({"seconds": out["seconds"]}))


if __name__ == "__main__":
    main()
