"""The harm-direction control per data recipe, and what separates the recipes whose harm
direction is base-specific from those whose harm direction is shared across bases.

Pure helpers (no torch): the cell list, the group rule, the prediction check, the diagnostics'
arithmetic and the part merge. `scripts/harm_046.py` does the I/O with the per-model instrument
of `scripts/ladder_007.py`.

Cells (one per recipe of the 25-recipe grid, from `data/manifests/mech_021.json`):
  B:<recipe>      the 8 stage-B recipes: set B, 6 models (3 bases x post-training seeds 1, 2):
                  3 MZ and 12 U pairs -> levels and gaps for both readouts
  A:<recipe>      the 16 stage-A-only recipes: set A, 3 models (post-training seed 1): 3 U pairs
                  -> the unrelated-pair level only (no gap)
  A0:dclm-baseline   the main recipe's 12 reruns (second set-up): 18 MZ, 48 U
  S300M:dclm-baseline  300M twins of the main recipe (labelled size rung, never pooled)

Group rule (operationalising the pre-registered "≈ 0.99 or ≈ 0.4, nothing between"): a recipe
is in the *shared* group when the harm direction's corrected unrelated-pair level at layer 13 is
>= SHARED_MIN, in the *base-specific* group when it is <= SPECIFIC_MAX, and *between*
otherwise; "bimodal" = no recipe between.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SHARED_MIN = 0.95
SPECIFIC_MAX = 0.55
PREDICTED_SPECIFIC = ("dclm-baseline", "dclm-baseline-qc-7p-fw3")  # pre-registered
PREDICTED_EXTRA_MAX = 2  # "plus at most two more"
MAIN_RECIPE = "dclm-baseline"
LAYERS = (13, 11)  # the fixed site's layer and the common site's layer
PUNCT = ".?!:"
SOURCES = ("advbench", "malicious_instruct", "tdc2023")  # extraction_v1 harmful training sources


# ---- cells ---------------------------------------------------------------------------------


def cells(manifest: dict) -> dict[str, dict]:
    """Cell id -> {set, recipe, size, models}; see the module docstring."""
    sets, by_id = manifest["sets"], {m["model_id"]: m for m in manifest["models"]}
    out: dict[str, dict] = {}

    def add(set_name: str, recipe: str, models: list[str]) -> None:
        sizes = {by_id[m]["size"] for m in models}
        assert len(sizes) == 1 and all(by_id[m]["recipe"] == recipe for m in models)
        out[f"{set_name}:{recipe}"] = {"set": set_name, "recipe": recipe, "size": sizes.pop(),
                                       "models": sorted(models)}

    b_recipes = sorted({by_id[m]["recipe"] for m in sets["B"]["models"]})
    for rc in b_recipes:
        add("B", rc, [m for m in sets["B"]["models"] if by_id[m]["recipe"] == rc])
    for rc in sorted({by_id[m]["recipe"] for m in sets["A"]["models"]} - set(b_recipes)):
        add("A", rc, [m for m in sets["A"]["models"] if by_id[m]["recipe"] == rc])
    for s in ("A0", "S300M"):
        add(s, MAIN_RECIPE, list(sets[s]["models"]))
    return out


def grid_cells(cell_ids) -> list[str]:
    """The 25 recipe cells of the 1B grid (A0 for the main recipe; S300M is a size rung)."""
    return [c for c in cell_ids if not c.startswith("S300M:")]


# ---- groups and the prediction -------------------------------------------------------------


def classify(r_u: dict[str, float], shared_min: float = SHARED_MIN,
             specific_max: float = SPECIFIC_MAX) -> dict[str, list[str]]:
    """Recipe -> group by the harm direction's unrelated-pair level (layer 13)."""
    out: dict[str, list[str]] = {"shared": [], "base_specific": [], "between": []}
    for rc in sorted(r_u):
        v = r_u[rc]
        key = ("shared" if v >= shared_min else "base_specific" if v <= specific_max
               else "between")
        out[key].append(rc)
    return out


def prediction_check(r_u13: dict[str, float], gaps: dict[str, dict[int, dict[str, float]]]
                     ) -> dict:
    """The pre-registered descriptive prediction, part by part (values and booleans).
    `r_u13`: recipe -> corrected harm r_U at layer 13 over the 25 grid recipes.
    `gaps`: recipe -> layer -> {refusal_gap, harm_gap} for recipes with MZ pairs."""
    groups = classify(r_u13)
    spec = set(groups["base_specific"])
    predicted = set(PREDICTED_SPECIFIC)
    extra = sorted(spec - predicted)
    shared_mz = [rc for rc in groups["shared"] if rc in gaps]
    below = {rc: {int(la): bool(gaps[rc][la]["harm_gap"] < gaps[rc][la]["refusal_gap"])
                  for la in gaps[rc]} for rc in shared_mz}
    values = sorted(r_u13.values())
    jumps = np.diff(values) if len(values) > 1 else np.array([0.0])
    return {
        "rule": {"shared_min": SHARED_MIN, "specific_max": SPECIFIC_MAX,
                 "predicted_base_specific": list(PREDICTED_SPECIFIC),
                 "extra_allowed": PREDICTED_EXTRA_MAX},
        "groups": groups,
        "part1_bimodal": {"predicted": "every recipe >= 0.95 or <= 0.55 (none between)",
                          "n_between": len(groups["between"]), "between": groups["between"],
                          "largest_gap_between_sorted_values": float(jumps.max()),
                          "holds": len(groups["between"]) == 0},
        "part2_membership": {"predicted": f"{list(PREDICTED_SPECIFIC)} plus at most "
                                          f"{PREDICTED_EXTRA_MAX} more",
                             "base_specific": sorted(spec),
                             "predicted_present": sorted(spec & predicted),
                             "predicted_missing": sorted(predicted - spec), "extra": extra,
                             "holds": predicted <= spec and len(extra) <= PREDICTED_EXTRA_MAX},
        "part3_shared_gap_below_refusal": {
            "predicted": "in every shared-group recipe with MZ pairs the harm gap is below the "
                         "refusal gap at both layers",
            "per_recipe": below, "n_recipes": len(shared_mz),
            "n_holding": sum(all(v.values()) for v in below.values()),
            "holds": bool(shared_mz) and all(all(v.values()) for v in below.values())},
    }


# ---- diagnostics' arithmetic -----------------------------------------------------------------


def last_char_class(prompt: str) -> str:
    """The class of a prompt's last non-space character: one of PUNCT, else 'word'."""
    s = prompt.rstrip()
    return s[-1] if s and s[-1] in PUNCT else "word"


def eta_squared(x: np.ndarray, groups) -> float:
    """Share of the variance of x between the groups (one-way ANOVA eta^2) over finite x."""
    x = np.asarray(x, float)
    g = np.asarray(groups)
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    if len(x) < 3:
        return float("nan")
    tot = float(((x - x.mean()) ** 2).sum())
    if tot <= 0:
        return float("nan")
    between = 0.0
    for v in np.unique(g):
        sel = x[g == v]
        between += len(sel) * (sel.mean() - x.mean()) ** 2
    return float(between / tot)


def demean_within(a: np.ndarray, groups) -> np.ndarray:
    """Subtract each group's column means (rows = prompts): the within-group part of every
    column of a packed array (projection columns; outcome columns are passed through by the
    caller)."""
    a = np.asarray(a, float).copy()
    g = np.asarray(groups)
    for v in np.unique(g):
        sel = g == v
        a[sel] -= np.nanmean(a[sel], axis=0, keepdims=True)
    return a


def shares(labels) -> dict[str, float]:
    labels = list(labels)
    n = len(labels)
    return {str(k): round(sum(1 for v in labels if v == k) / n, 4) for k in sorted(set(labels))}


def loo_keep(sources, drop: str) -> np.ndarray:
    """Row indices of the harmful training prompts not from `drop` (leave-one-source-out)."""
    return np.array([i for i, s in enumerate(sources) if s != drop], dtype=int)


def matched_random_keep(n: int, n_drop: int, seed: int) -> np.ndarray:
    """Row indices after dropping `n_drop` rows at random (the size-matched control)."""
    rng = np.random.default_rng(seed)
    drop = set(rng.choice(n, size=n_drop, replace=False).tolist())
    return np.array([i for i in range(n) if i not in drop], dtype=int)


# ---- merge ---------------------------------------------------------------------------------

SHARED = ("prompt", "manifest_sha256", "extraction_sha256", "n_splits", "split_seed",
          "n_boot_sec", "layers", "harm_position", "sites", "n_prompts", "token_groups")


def merge(parts: list[Path]) -> dict:
    """Join the cell files of one run: shared fields agree, every cell appears once; the
    summary blocks (groups, prediction) are computed here from the merged cells."""
    ds = [json.loads(p.read_text()) for p in parts]
    out: dict = {k: ds[0][k] for k in SHARED}
    cells_out: dict[str, dict] = {}
    commits = set()
    for p, d in zip(parts, ds):
        for k in SHARED:
            assert d[k] == out[k], f"{p}: {k} differs from {parts[0]}"
        commits.add(d["git_commit"])
        for c, v in d["cells"].items():
            assert c not in cells_out, f"cell {c} twice"
            cells_out[c] = v
    out["cells"] = dict(sorted(cells_out.items()))
    out["git_commits"] = sorted(commits)
    out["parts"] = {str(p): {"seconds": d["seconds"], "command": d["command"],
                             "git_commit": d["git_commit"], "cells": sorted(d["cells"]),
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p, d in zip(parts, ds)}
    out["summary"] = summarise(out["cells"])
    return out


def summarise(cells_out: dict[str, dict]) -> dict:
    """Groups, the prediction check and the per-group ranges from the merged cells."""
    grid = grid_cells(cells_out)
    r_u13 = {cells_out[c]["recipe"]: cells_out[c]["layers"]["13"]["point"]["harm_r_U_corrected"]
             for c in grid}
    gaps = {}
    for c in grid:
        v = cells_out[c]
        if v["n_MZ"] == 0:
            continue
        gaps[v["recipe"]] = {int(la): {"refusal_gap": v["layers"][la]["point"]
                                       ["refusal_MZ_minus_U_corrected"],
                                       "harm_gap": v["layers"][la]["point"]
                                       ["harm_MZ_minus_U_corrected"]} for la in v["layers"]}
    pred = prediction_check(r_u13, gaps)
    by_recipe = {cells_out[c]["recipe"]: c for c in grid}
    ranges = {}
    for la in ("13", "11"):
        ranges[la] = {}
        for g, members in pred["groups"].items():
            if not members:
                ranges[la][g] = {"n": 0}
                continue
            def col(key, members=members, la=la):
                return [cells_out[by_recipe[rc]]["layers"][la]["point"][key] for rc in members]
            h, r = col("harm_r_U_corrected"), col("refusal_r_U_corrected")
            entry = {"n": len(members), "harm_r_U": {"min": min(h), "max": max(h)},
                     "refusal_r_U": {"min": min(r), "max": max(r)}}
            mz = [rc for rc in members if rc in gaps]
            if mz:
                hg = [gaps[rc][int(la)]["harm_gap"] for rc in mz]
                rg = [gaps[rc][int(la)]["refusal_gap"] for rc in mz]
                entry["mz_recipes"] = {"n": len(mz), "recipes": mz,
                                       "harm_gap": {"min": min(hg), "max": max(hg)},
                                       "refusal_gap": {"min": min(rg), "max": max(rg)},
                                       "n_harm_gap_below_refusal":
                                           sum(a < b for a, b in zip(hg, rg))}
            ranges[la][g] = entry
    return {"n_grid_recipes": len(grid), "harm_r_U_layer13": dict(sorted(r_u13.items())),
            "prediction": pred, "groups": pred["groups"], "ranges": ranges,
            "mz_gaps": {rc: {str(k): v for k, v in d.items()} for rc, d in gaps.items()}}


# ---- per pretraining run ------------------------------------------------------------------

ODD_ETA2_MAX = 0.5  # a run "does not read the last token" when its mean eta^2 is below this


def per_run(cells_out: dict[str, dict], run_of: dict[str, str]) -> dict:
    """The harm direction per pretraining run (descriptive, defined after the per-recipe numbers
    existed). `run_of`: model id -> pretraining seed (the manifest's
    `pretrain_seed`). Per cell and layer: each run's eta^2 of the harm projection on the
    last-token id (mean and values over its post-training seeds); a run is *odd* when that mean
    is below ODD_ETA2_MAX at layer 13 (the observed values are bimodal, 0.03-0.05 vs >= 0.97,
    so the threshold is not load-bearing). Per cell with an odd run: the corrected per-pair
    levels split into pairs between token-reading runs and pairs involving the odd run, for both
    readouts, and the MZ - U gap of each kind where MZ pairs exist."""
    out_cells: dict[str, dict] = {}
    grid = set(grid_cells(cells_out))
    counts = {"n_runs": 0, "n_token": 0, "n_odd": 0, "odd": []}
    for c, v in cells_out.items():
        runs = sorted({run_of[m] for m in v["models"]})
        entry: dict = {"recipe": v["recipe"], "runs": {}, "odd_runs": [], "layers": {}}
        for la in v["layers"]:
            pm = v["layers"][la]["diagnosis_c"]["per_model"]
            for r in runs:
                e = [pm[m]["eta2_harm_last_token"] for m in v["models"] if run_of[m] == r]
                entry["runs"].setdefault(r, {})[la] = {
                    "eta2_harm_last_token_mean": float(np.mean(e)),
                    "eta2_harm_last_token": [float(x) for x in e]}
        entry["odd_runs"] = [r for r in runs if entry["runs"][r]["13"]
                             ["eta2_harm_last_token_mean"] < ODD_ETA2_MAX]
        if c in grid:
            counts["n_runs"] += len(runs)
            counts["n_odd"] += len(entry["odd_runs"])
            counts["odd"] += [{"cell": c, "recipe": v["recipe"], "run": r}
                              for r in entry["odd_runs"]]
        for la in v["layers"]:
            if not entry["odd_runs"]:
                break
            lay: dict = {}
            for ro in ("harm", "refusal"):
                kinds: dict[str, list[float]] = {}
                for p in v["layers"][la]["pairs"][ro]:
                    ra, rb = run_of[p["a"]], run_of[p["b"]]
                    odd = ra in entry["odd_runs"] or rb in entry["odd_runs"]
                    k = f"{p['level']}_{'odd' if odd else 'token'}"
                    kinds.setdefault(k, []).append(p["r_corrected"])
                lv = {k: {"n": len(x), "mean": float(np.mean(x)), "min": float(min(x)),
                          "max": float(max(x))} for k, x in sorted(kinds.items())}
                gaps = {kind: lv[f"MZ_{kind}"]["mean"] - lv[f"U_{kind}"]["mean"]
                        for kind in ("token", "odd") if f"MZ_{kind}" in lv and f"U_{kind}" in lv}
                lay[ro] = {"levels": lv, "gap": gaps}
            entry["layers"][la] = lay
        out_cells[c] = entry
    counts["n_token"] = counts["n_runs"] - counts["n_odd"]
    return {"rule": {"odd_eta2_max": ODD_ETA2_MAX, "layer": 13,
                     "written": "after the 046 numbers (D096a re-read); descriptive"},
            **counts, "cells": out_cells}
