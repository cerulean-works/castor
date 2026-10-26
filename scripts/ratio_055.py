"""Recipe-bootstrap interval for the seed-over-recipe ratio: the mean over recipes of the
ceiling-corrected MZ − U over the mean of U − R, on each level variant. Pure CPU; reads
runs/008/grid.json, runs/008/audit_correct.json and the 150 grid models' judge labels.

    uv run --frozen python scripts/ratio_055.py --out /abs/path/runs/055/ratio.json

Variants: raw (grid.json's primary block, asserted bit for bit; the ratio of record), per_source
and per_group_source (descriptive only). audit_correct.json keeps only the means over recipes of
its corrected contrast, so the per-recipe differences are rebuilt here with
scripts/audit_correct_008.py's own code path (`load_grid` / `factor_matrix`, rates x per-prompt
factor at the audit's point factors, then `gridstats.levels`), and both means are asserted equal
to audit_correct.json's `contrast` block. Interval: `gridstats.ratio_bootstrap` (the resampling of
`gridstats.recipe_bootstrap`, 1,000 draws, seed 0), percentile; the numerator's interval is
asserted equal to the stored `ci_recipes` (same draws). The audit's own uncertainty in the
factors is not propagated.
"""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

from castor import gridstats, provenance

VARIANTS = {"raw": "none", "per_source": "per_source", "per_group_source": "per_group_source"}
KEYS = ("MZ_minus_U_corrected", "U_minus_R_corrected")
_spec = importlib.util.spec_from_file_location(
    "audit_correct_008", Path(__file__).parent / "audit_correct_008.py")
AC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AC)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=Path, default=AC.GRID)
    ap.add_argument("--audit", type=Path, default=AC.ROOT / "runs" / "008" / "audit_correct.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    AC.require(a.out.is_absolute(), "--out must be absolute")
    commit, t0 = provenance.git_commit(), time.time()
    au = json.loads(a.audit.read_text())
    AC.require(au["levels"]["of_record"] == "per_source", "of record")
    g, names, infos, src, ys = AC.load_grid(a.grid)
    AC.require(au["grid_git_commit"] == g.get("git_commit"), "audit read another grid.json")
    AC.require((a.n_boot, a.seed) == (g["n_boot"], g["seed"]) == (au["n_boot_grid"], au["seed"]),
               "n_boot / seed differ from 008's and 051's recipe bootstrap")
    recipe_of = dict(zip(names, (i.recipe for i in infos)))
    masks = gridstats.pair_masks(infos)
    point = {f"source:{s}": v["point"] for s, v in au["factors"]["per_source"].items()}
    point |= {f"group_source:{k}": v["point"] for k, v in au["factors"]["per_group_source"].items()}
    ref = g["adv_harmful"]["primary"]

    out = {}
    for v, av in VARIANTS.items():
        F = AC.factor_matrix(av, names, recipe_of, src, point)
        sy = {m: ys[m] * F[i][:, None] for i, m in enumerate(names)}  # as audit_correct_008.contrast
        P, Po, Pe = gridstats.halves(sy, names)
        lv = gridstats.levels(P, gridstats.reliabilities(Po, Pe), infos, masks)
        stored = au["contrast"][av]
        for k in KEYS:
            AC.require(AC.close(lv[k], stored[k]), ("audit_correct.json contrast", v, k, lv[k], stored[k]))
            if v == "raw":
                AC.require(lv[k] == ref["levels"][k], ("grid.json", k))
                AC.require(all(lv["per_recipe"][q][k] == ref["levels"]["per_recipe"][q][k]
                               for q in lv["per_recipe"]), ("grid.json per_recipe", k))
        bs = gridstats.ratio_bootstrap(lv["per_recipe"], a.n_boot, a.seed)
        civ = stored["ci_recipes"]
        AC.require(AC.close(bs["numerator"]["lo"], civ["lo"]) and AC.close(bs["numerator"]["hi"], civ["hi"]),
                   ("same draws as the stored recipe interval", v, bs["numerator"], civ))
        lo, hi, est = bs["ratio"]["lo"], bs["ratio"]["hi"], bs["ratio"]["estimate"]
        AC.require(lo <= est <= hi, ("interval brackets the point", v))
        out[v] = {"ratio": bs["ratio"], "frac_den_nonpositive": bs["frac_den_nonpositive"],
                  "difference": bs["difference"], "numerator": bs["numerator"],
                  "denominator": bs["denominator"],
                  "per_recipe": {q: {k: d[k] for k in KEYS} for q, d in sorted(lv["per_recipe"].items())},
                  "n_recipes": len(lv["per_recipe"])}
    raw = ref["levels"]["MZ_minus_U_corrected"] / ref["levels"]["U_minus_R_corrected"]
    AC.require(abs(out["raw"]["ratio"]["estimate"] - raw) < 1e-12, ("raw ratio", raw))
    res = {"of_record": "raw", "variants": out,
           "descriptive_only": ["per_group_source"],
           "statistic": "mean over recipes of MZ_minus_U_corrected / mean over recipes of U_minus_R_corrected",
           "interval": "recipe bootstrap (gridstats.recipe_bootstrap's resampling), percentile 95 %; "
                       "factors held at the audit's point values",
           "subset": "adv_harmful", "outcome": "primary", "field": AC.FIELD, "n_models": len(names),
           "n_boot": a.n_boot, "seed": a.seed,
           "inputs": {provenance.rel(a.grid): sha256(a.grid), provenance.rel(a.audit): sha256(a.audit)},
           "grid_git_commit": g.get("git_commit"), "audit_git_commit": au["git_commit"],
           "script": "scripts/ratio_055.py", "git_commit": commit,
           "command": provenance.command(), "seconds": round(time.time() - t0, 2)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({v: {"ratio": [round(o["ratio"][k], 3) for k in ("estimate", "lo", "hi")],
                          "frac_den_nonpositive": o["frac_den_nonpositive"]} for v, o in out.items()}))


if __name__ == "__main__":
    main()
