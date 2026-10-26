"""Attack leaderboards under reseeding (definitions in castor/leaderboard.py). CPU only; reads the
150 grid models' judge_v1 labels on the HarmBench human-jailbreak prompts.

    uv run --frozen python scripts/leaderboard_074.py --out /abs/path/runs/074 [--workers 4]

Template shares as in runs/044 (castor.stability.template_shares, adversarial-harmful x
`wg_vulnerable`, 91 templates); the grid template levels of runs/044/summary.json and their
per-recipe means are asserted reproduced to 1e-9 before anything else is computed. Writes
summary.json (flip rates per gap bin x pair kind with recipe-bootstrap intervals, the per-recipe
rates with template-bootstrap intervals, top-1 / top-5 change rates, per-recipe Spearman, the
observed-gap secondary, provenance) and figure tsvs (level/label/x/value/lo/hi)."""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

from castor import evalset, grid, provenance, reliability, sanityb, stability
from castor import leaderboard as lb
from castor.judge.common import read_jsonl

FIELD, SUBSET, HJ = "wg_vulnerable", sanityb.PRIMARY_SUBSET, "harmbench_humanjb"
GRID_JSON = grid.ROOT / "runs" / "008" / "grid.json"
SUMMARY_044 = grid.ROOT / "runs" / "044" / "summary.json"
KINDS = (("MZ", "MZ twins"), ("U", "unrelated pairs"), ("R", "cross-recipe pairs"))


def _shares(args: tuple[str, list[str], list[int]]) -> dict:
    path, ids, col = args
    y, c = sanityb.outcome_matrix(read_jsonl(path), ids, FIELD)
    assert c["missing"] == 0, f"{path}: {c['missing']} missing label cells"
    col = np.asarray(col)
    return stability.template_shares(y, col, int(col.max()) + 1)


def interval(point, draws) -> dict:
    return {"value": float(point), "ci": lb.percentile(draws)}


def per_bin(point: np.ndarray, draws: np.ndarray, extra: dict | None = None) -> dict:
    out = {}
    for b, lab in enumerate(lb.BIN_LABELS):
        out[lab] = interval(point[b], draws[:, b]) | ({k: v[b] for k, v in extra.items()}
                                                      if extra else {})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute (submit.sh snapshots are deleted)"
    commit, t0 = provenance.git_commit(), time.time()

    panel = evalset.load_eval()
    ids = sanityb.subset_ids(panel, SUBSET)
    row = {r["id"]: r for r in panel}
    hj = [i for i in ids if row[i]["source"] == HJ]
    tnames, col = stability.template_index([row[i]["source_id"].split("::")[1] for i in hj])
    n_t = len(tnames)
    gj = json.loads(GRID_JSON.read_text())
    gm = gj["models"]
    assert len(gm) == 150
    names = sorted(gm)
    with ProcessPoolExecutor(a.workers) as ex:
        sh = list(ex.map(_shares, [(gm[m]["labels"], hj, col.tolist()) for m in names]))
    S, E, O = (np.vstack([s[k] for s in sh]) for k in ("full", "even", "odd"))
    assert np.isfinite(S).all() and np.isfinite(E).all() and np.isfinite(O).all()
    info = [reliability.ModelInfo(gm[m]["recipe"], gm[m]["pretrain_seed"],
                                  str(gm[m]["posttrain_seed_k"])) for m in names]
    t_load = time.time() - t0

    # ---- check first: runs/044 template levels, on the grid and as per-recipe means ----------
    ref = json.loads(SUMMARY_044.read_text())["grid"]["templates"]["levels"]
    assert json.loads(SUMMARY_044.read_text())["grid"]["templates"]["n_templates"] == n_t == 91
    full = stability.template_levels(S, E, O, stability.level_masks(info))
    recipes = sorted({i.recipe for i in info})
    rows = {r: [j for j, i in enumerate(info) if i.recipe == r] for r in recipes}
    assert all(len(v) == 6 for v in rows.values()) and len(recipes) == 25
    per_rec = {r: stability.template_levels(S[v], E[v], O[v],
                                            stability.level_masks([info[j] for j in v]))
               for r, v in rows.items()}
    check = {}
    for k in ("r_MZ", "r_MZ_corrected", "r_U", "r_U_corrected", "r_R", "r_R_corrected"):
        assert abs(full[k] - ref[k]["value"]) < 1e-9, (k, full[k], ref[k])
        check[k] = {"074": full[k], "044": ref[k]["value"]}
    for k in ("r_MZ", "r_MZ_corrected", "r_U", "r_U_corrected"):
        m = float(np.mean([per_rec[r][k] for r in recipes]))
        assert abs(m - ref[k]["value"]) < 1e-9, (k, m, ref[k])
        check[f"{k} (mean of per-recipe)"] = {"074": m, "044": ref[k]["value"]}

    # ---- signs, bins, flips --------------------------------------------------------------------
    iu = lb.template_pairs(n_t)
    n_tp = len(iu[0])
    sg = lb.signs(S, iu)
    nb, R = len(lb.EDGES), len(recipes)
    mean_r = {r: S[v].mean(axis=0) for r, v in rows.items()}
    bins_r = {r: lb.gap_bins(mean_r[r], iu) for r in recipes}
    f_w = {k: np.zeros((R, nb)) for k in ("MZ", "U")}
    c_w = {k: np.zeros((R, nb)) for k in ("MZ", "U")}
    ftp = {k: np.zeros((R, n_tp)) for k in ("MZ", "U")}  # per template pair, summed over pairs
    npairs = {}
    of_w = {k: np.zeros((R, nb)) for k in ("MZ", "U")}  # observed-gap secondary
    oc_w = {k: np.zeros((R, nb)) for k in ("MZ", "U")}
    ties = {"cells_one_tie": 0, "cells_both_tie": 0, "cells": 0}
    pairs_r = {}
    for x, r in enumerate(recipes):
        v = rows[r]
        prs = lb.within_pairs(v, [info[j].pretrain_seed for j in v])
        pairs_r[r] = prs
        for k, pp in prs.items():
            pa, pb = np.array(pp).T
            F = lb.flip(sg[pa], sg[pb])
            f_w[k][x], c_w[k][x] = lb.bin_sums(F, bins_r[r])
            ftp[k][x] = F.sum(axis=0)
            npairs[k] = len(pp)
            of_w[k][x], oc_w[k][x] = lb.observed_gap_sums(S, sg, pp, iu)
            z = (sg[pa] == 0), (sg[pb] == 0)
            ties["cells_one_tie"] += int((z[0] ^ z[1]).sum())
            ties["cells_both_tie"] += int((z[0] & z[1]).sum())
            ties["cells"] += F.size
    f_x, c_x = np.zeros((R, R, nb)), np.zeros((R, R, nb))
    of_x, oc_x = np.zeros((R, R, nb)), np.zeros((R, R, nb))
    for x, y in combinations(range(R), 2):
        va, vb = rows[recipes[x]], rows[recipes[y]]
        bins = lb.gap_bins(S[va + vb].mean(axis=0), iu)
        pp = [(p, q) for p in va for q in vb]
        pa, pb = np.array(pp).T
        F = lb.flip(sg[pa], sg[pb])
        f_x[x, y], c_x[x, y] = lb.bin_sums(F, bins)
        of_x[x, y], oc_x[x, y] = lb.observed_gap_sums(S, sg, pp, iu)
    for A in (f_x, c_x, of_x, oc_x):
        A += A.transpose(1, 0, 2)

    # ---- pooled rates with recipe-bootstrap intervals ------------------------------------------
    W = lb.recipe_draws(R, a.n_boot, a.seed)
    flips, obs, draws_k, odraws_k = {}, {}, {}, {}
    for k, _ in KINDS:
        if k == "R":
            pt, dr = lb.pooled_cross(f_x, c_x), np.vstack([lb.pooled_cross(f_x, c_x, w)
                                                            for w in W])
            opt = lb.pooled_cross(of_x, oc_x)
            odr = np.vstack([lb.pooled_cross(of_x, oc_x, w) for w in W])
            cells = c_x.sum(axis=(0, 1)) / 2
        else:
            pt, dr = lb.pooled_within(f_w[k], c_w[k]), np.vstack(
                [lb.pooled_within(f_w[k], c_w[k], w) for w in W])
            opt = lb.pooled_within(of_w[k], oc_w[k])
            odr = np.vstack([lb.pooled_within(of_w[k], oc_w[k], w) for w in W])
            cells = c_w[k].sum(axis=0)
        draws_k[k], odraws_k[k] = dr, odr
        flips[k] = per_bin(pt, dr, {"cells": cells.astype(int).tolist()})
        obs[k] = per_bin(opt, odr, {"cells": ((oc_x.sum(axis=(0, 1)) / 2) if k == "R"
                                              else oc_w[k].sum(axis=0)).astype(int).tolist()})
    # all-bins rates with their own draws
    for k in ("MZ", "U"):
        fa, ca = f_w[k].sum(axis=1, keepdims=True), c_w[k].sum(axis=1, keepdims=True)
        flips[k]["all bins"] = interval(lb.pooled_within(fa, ca)[0],
                                        [lb.pooled_within(fa, ca, w)[0] for w in W])
    fa, ca = f_x.sum(axis=2, keepdims=True), c_x.sum(axis=2, keepdims=True)
    flips["R"]["all bins"] = interval(lb.pooled_cross(fa, ca)[0],
                                      [lb.pooled_cross(fa, ca, w)[0] for w in W])
    pb = lb.PRIMARY_BIN
    u_mz = {"U / MZ": interval(flips["U"][lb.BIN_LABELS[pb]]["value"]
                               / flips["MZ"][lb.BIN_LABELS[pb]]["value"],
                               draws_k["U"][:, pb] / draws_k["MZ"][:, pb]),
            "U - MZ": interval(flips["U"][lb.BIN_LABELS[pb]]["value"]
                               - flips["MZ"][lb.BIN_LABELS[pb]]["value"],
                               draws_k["U"][:, pb] - draws_k["MZ"][:, pb]),
            "R - U": interval(flips["R"][lb.BIN_LABELS[pb]]["value"]
                              - flips["U"][lb.BIN_LABELS[pb]]["value"],
                              draws_k["R"][:, pb] - draws_k["U"][:, pb])}

    ov = {k: obs[k][lb.BIN_LABELS[pb]]["value"] for k in ("MZ", "U", "R")}
    od = {k: odraws_k[k][:, pb] for k in ("MZ", "U", "R")}
    obs["contrasts (2-5 bin)"] = {"U / MZ": interval(ov["U"] / ov["MZ"], od["U"] / od["MZ"]),
                                  "U - MZ": interval(ov["U"] - ov["MZ"], od["U"] - od["MZ"]),
                                  "R - U": interval(ov["R"] - ov["U"], od["R"] - od["U"])}

    # ---- per-recipe rates (2-5 bin) with template-bootstrap intervals --------------------------
    Wt = lb.template_pair_weights(n_t, iu, a.n_boot, a.seed)
    per_recipe, pooled_t = {}, {}
    for k in ("MZ", "U"):
        num_all, den_all = np.zeros(a.n_boot), np.zeros(a.n_boot)
        per_recipe[k] = {}
        for x, r in enumerate(recipes):
            m = bins_r[r] == pb
            num = Wt[:, m] @ ftp[k][x][m]
            den = Wt[:, m].sum(axis=1) * npairs[k]
            num_all += num
            den_all += den
            per_recipe[k][r] = interval(f_w[k][x, pb] / c_w[k][x, pb], lb.ratio(num, den)) | {
                "n_template_pairs": int(m.sum())}
        pooled_t[k] = interval(flips[k][lb.BIN_LABELS[pb]]["value"], lb.ratio(num_all, den_all))

    # ---- top-1 / top-5 -------------------------------------------------------------------------
    def topk(pairs, tiebreak):
        t1, t5, ov, nt = [], [], [], 0
        for p, q in pairs:
            t1.append(lb.top1_change(S[p], S[q]))
            c5, o5, n5 = lb.top5_change(S[p], S[q], tiebreak)
            t5.append(c5)
            ov.append(o5)
            nt += n5
        return np.array([np.sum(t1), np.sum(t5), np.sum(ov), len(pairs)]), nt
    tk_w = {k: np.zeros((R, 4)) for k in ("MZ", "U")}
    tk_x = np.zeros((R, R, 4))
    tied5 = {"MZ": 0, "U": 0, "R": 0}
    n_max_ties = int(sum((S[j] == S[j].max()).sum() > 1 for j in range(len(names))))
    per_recipe_top1 = {}
    for x, r in enumerate(recipes):
        for k in ("MZ", "U"):
            tk_w[k][x], nt = topk(pairs_r[r][k], mean_r[r])
            tied5[k] += nt
        per_recipe_top1[r] = {"U_top1_change": tk_w["U"][x, 0] / tk_w["U"][x, 3],
                              "U_top5_change": tk_w["U"][x, 1] / tk_w["U"][x, 3],
                              "top1_template_by_model": {names[j]: tnames[int(np.argmax(S[j]))]
                                                         for j in rows[r]}}
    for x, y in combinations(range(R), 2):
        va, vb = rows[recipes[x]], rows[recipes[y]]
        tk_x[x, y], nt = topk([(p, q) for p in va for q in vb], S[va + vb].mean(axis=0))
        tied5["R"] += nt
    tk_x += tk_x.transpose(1, 0, 2)
    top = {}
    for k, _ in KINDS:
        top[k] = {}
        for j, lab in enumerate(("top1_change", "top5_change", "top5_overlap")):
            if k == "R":
                f, c = tk_x[..., j:j + 1], tk_x[..., 3:4]
                pt = lb.pooled_cross(f, c)[0]
                dr = [lb.pooled_cross(f, c, w)[0] for w in W]
            else:
                f, c = tk_w[k][:, j:j + 1], tk_w[k][:, 3:4]
                pt = lb.pooled_within(f, c)[0]
                dr = [lb.pooled_within(f, c, w)[0] for w in W]
            top[k][lab] = interval(pt, dr)
        top[k]["n_pairs"] = int(tk_x[..., 3].sum() / 2 if k == "R" else tk_w[k][:, 3].sum())
        top[k]["n_models_tied_at_5th"] = tied5[k]

    # ---- per-recipe Spearman with template-bootstrap intervals ---------------------------------
    tb = stability.boot_indices(n_t, a.n_boot, a.seed)
    spearman = {}
    for r, v in rows.items():
        mk = stability.level_masks([info[j] for j in v])
        dr = [stability.template_levels(S[v], E[v], O[v], mk, i) for i in tb]
        spearman[r] = {k: interval(per_rec[r][k], [d[k] for d in dr])
                       for k in ("r_MZ", "r_MZ_corrected", "r_U", "r_U_corrected")}

    # ---- pre-registered readings (descriptive; no rule) ----------------------------------------
    lab = lb.BIN_LABELS[pb]
    u, mz = flips["U"][lab]["value"], flips["MZ"][lab]["value"]
    t1 = top["U"]["top1_change"]["value"]
    prediction = {
        "U flip (2-5 bin) >= 0.25": bool(u >= 0.25),
        "U flip (2-5 bin) >= 2 x MZ flip (2-5 bin)": bool(u >= 2 * mz),
        "top-1 differs between two runs of one recipe in >= 1/3 of run pairs": bool(t1 >= 1 / 3),
        "values": {"U": u, "MZ": mz, "U/MZ": u / mz, "U top1 change": t1}}
    summary = {
        "subset": SUBSET, "field": FIELD, "source": HJ, "n_templates": n_t,
        "n_prompts": len(hj), "n_behaviours": len({row[i]["source_id"].split("::")[0]
                                                   for i in hj}),
        "n_template_pairs": n_tp, "n_models": len(names), "n_recipes": R,
        "model_pairs": {"MZ": npairs["MZ"] * R, "U": npairs["U"] * R,
                        "R": int(sum(len(rows[p]) * len(rows[q])
                                     for p, q in combinations(recipes, 2)))},
        "bins_points": {"edges": list(lb.EDGES), "labels": list(lb.BIN_LABELS),
                        "primary": lab},
        "check_044": check,
        "flip": flips, "primary": {"U": flips["U"][lab], "MZ": flips["MZ"][lab],
                                   "template_bootstrap (secondary)": pooled_t, **u_mz},
        "per_recipe_flip_2_5": per_recipe,
        "top": top, "top1_template_ties (models with a tied maximum)": n_max_ties,
        "per_recipe_spearman": spearman,
        "spearman_mean_over_recipes": {k: float(np.mean([per_rec[r][k] for r in recipes]))
                                       for k in ("r_MZ", "r_MZ_corrected", "r_U",
                                                 "r_U_corrected")},
        "secondary_observed_gap": obs, "ties": ties,
        "per_recipe_top": per_recipe_top1,
        "prediction": prediction,
        "script": "scripts/leaderboard_074.py", "git_commit": commit,
        "command": provenance.command(), "n_boot": a.n_boot, "seed": a.seed,
        "grid_json": str(GRID_JSON.relative_to(grid.ROOT)), "grid_json_git_commit":
            gj["git_commit"], "seconds_load": t_load, "seconds": time.time() - t0}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    np.save(a.out / "shares.npy", S)  # (150 models x 91 templates); order in summary/models.txt
    (a.out / "models.txt").write_text("\n".join(names) + "\n")
    (a.out / "templates.txt").write_text("\n".join(tnames) + "\n")
    # figure data
    with open(a.out / "flip_bins.tsv", "w") as f:
        f.write("level\tlabel\tx\tbin\tvalue\tlo\thi\tcells\n")
        for k, kl in KINDS:
            for b, bl in enumerate(lb.BIN_LABELS):
                d = flips[k][bl]
                f.write(f"{k}\t{kl}\t{b + 1}\t{bl}\t{d['value']}\t{d['ci'][0]}\t{d['ci'][1]}\t"
                        f"{d['cells']}\n")
    with open(a.out / "top_change.tsv", "w") as f:
        f.write("level\tlabel\tx\ttop1\ttop1_lo\ttop1_hi\ttop5\ttop5_lo\ttop5_hi\n")
        for x, (k, kl) in enumerate(KINDS, 1):
            t, s5 = top[k]["top1_change"], top[k]["top5_change"]
            f.write(f"{k}\t{kl}\t{x}\t{t['value']}\t{t['ci'][0]}\t{t['ci'][1]}\t{s5['value']}\t"
                    f"{s5['ci'][0]}\t{s5['ci'][1]}\n")
    print(json.dumps({"prediction": prediction, "seconds": summary["seconds"]}, default=float))


if __name__ == "__main__":
    main()
