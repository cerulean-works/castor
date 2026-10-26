"""Print the tables of runs/021/ladder.json (formatting only; every number is read from the
merged artifact, nothing is recomputed).

    uv run --frozen python scripts/report_021.py [--ladder runs/021/ladder.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def ci(block: dict, key: str, nd: int = 3) -> str:
    c = block["ci_prompts"][key]
    return f"{c['estimate']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}]"


def f(x, nd=3) -> str:
    return "\u2014" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def link_table(sets: dict) -> None:
    print("\n## Link at the fixed site (13, −1) vs 007's common site (11, −2), corrected, "
          "different-base pairs, adv-harmful (prompt bootstrap)\n")
    print("| set | level | n pairs | link (13, −1) | link (11, −2) | paired difference | "
          "reading at (13, −1) | draws |")
    print("|---|---|---|---|---|---|---|---|")
    for name, s in sets.items():
        L = s.get("link")
        if not L or "undefined" in L:
            print(f"| {name} | \u2014 | \u2014 | undefined ({L.get('undefined') if L else 'no main block'}) "
                  f"| | | | |")
            continue
        levels = [k[len("site_link_"):-len("_corrected")] for k in L["point"]
                  if k.startswith("site_link_") and k.endswith("_corrected")]
        for lvl in levels:
            n = L["point"].get(f"site_n_{lvl}", 0)
            if not n:
                continue
            print(f"| {name} | {lvl} | {n} | {ci(L, f'site_link_{lvl}_corrected')} | "
                  f"{ci(L, f'compare_link_{lvl}_corrected')} | {ci(L, f'diff_link_{lvl}_corrected')} "
                  f"| {L['reading']['site'] if lvl == 'U' else '\u2014'} | {L['n_draws']} |")
        for lvl in ("MZ", "both"):
            n = L["point"].get(f"site_n_{lvl}", 0)
            if n:
                print(f"| {name} | {lvl} (raw) | {n} | {ci(L, f'site_link_{lvl}')} | "
                      f"{ci(L, f'compare_link_{lvl}')} | {ci(L, f'diff_link_{lvl}')} | \u2014 | "
                      f"{L['n_draws']} |")


def prediction(sets: dict) -> None:
    for name, s in sets.items():
        p = (s.get("link") or {}).get("prediction_d053d")
        if not p:
            continue
        tag = "CONFIRMATORY" if p.get("confirmatory") else "descriptive"
        print(f"\n### D053d rule on set {name} ({tag}): part 1 {'HOLDS' if p['part1']['holds'] else 'FAILS'}"
              f" (link {p['part1']['link']['estimate']:.3f}, CI lo {p['part1']['link']['lo']:.3f}); "
              f"part 2 {'HOLDS' if p['part2']['holds'] else 'FAILS'} (difference "
              f"{p['part2']['difference']['estimate']:.3f}, CI lo {p['part2']['difference']['lo']:.3f})"
              f" → {'HOLDS' if p['holds'] else 'FAILS'}; reading at the site: {p['reading_site']}")
        jk = s["link"].get("jackknife_recipes") or s["link"].get("jackknife_bases")
        if jk:
            which = "recipes" if "jackknife_recipes" in s["link"] else "bases"
            print(f"    jackknife over {which} (n = {jk['site_link_U_corrected']['n_groups']}): SE "
                  f"site {jk['site_link_U_corrected']['se']:.3f}, compare "
                  f"{jk['compare_link_U_corrected']['se']:.3f}, difference "
                  f"{jk['diff_link_U_corrected']['se']:.3f}")


def per_recipe(sets: dict) -> None:
    for name, s in sets.items():
        pr = s.get("per_recipe")
        if not pr:
            continue
        print(f"\n## Set {name}: per recipe (3 pure U pairs each; 200 prompt draws)\n")
        print("| recipe | link (13, −1) | link (11, −2) | difference |")
        print("|---|---|---|---|")
        for rc, v in sorted(pr["recipes"].items()):
            print(f"| {rc} | {ci(v, 'site_link_U_corrected')} | {ci(v, 'compare_link_U_corrected')} | "
                  f"{ci(v, 'diff_link_U_corrected')} |")
        sp, sc = pr["spread_site"], pr["spread_compare"]
        print(f"\nSpread across {sp['n']} recipes at (13, −1): mean {sp['mean']:.3f}, SD {sp['sd']:.3f}, "
              f"min {sp['min']:.3f}, median {sp['median']:.3f}, max {sp['max']:.3f}; ≥ 0.5 in "
              f"{sp['n_runs_through']}, 0.3–0.5 in {sp['n_partly']}, < 0.3 in {sp['n_elsewhere']}. "
              f"At (11, −2): mean {sc['mean']:.3f}, SD {sc['sd']:.3f}, min {sc['min']:.3f}, max "
              f"{sc['max']:.3f}; ≥ 0.5 in {sc['n_runs_through']}, 0.3–0.5 in {sc['n_partly']}, "
              f"< 0.3 in {sc['n_elsewhere']}. Difference > 0 in {pr['n_recipes_diff_positive']} "
              f"recipes; site CI lower bound > 0.3 in {pr['n_recipes_site_ci_above_0.3']}.")


def ladder(sets: dict) -> None:
    print("\n## Projection ladder at both sites (corrected; 200 prompt draws)\n")
    print("| set | quantity | (13, −1) | (11, −2) | difference |")
    print("|---|---|---|---|---|")
    for name, s in sets.items():
        G = s.get("ladder")
        if not G:
            continue
        keys = [k[len("site_"):] for k in G["point"] if k.startswith("site_") and
                (k.endswith("_corrected") or k == "site_mean_reliability")]
        for k in keys:
            if not np.isfinite(G["point"][f"site_{k}"]):
                continue
            d = f"diff_{k}"
            print(f"| {name} | {k} | {ci(G, f'site_{k}')} | {ci(G, f'compare_{k}')} | "
                  f"{ci(G, d) if d in G['ci_prompts'] else '\u2014'} |")


def harm(sets: dict) -> None:
    print("\n## Harm-direction control (D051a) at each site's layer, last instruction token\n")
    print("| set | layer | gap_refusal − gap_harm (corrected) | refusal gap | harm gap | harm-direction link |")
    print("|---|---|---|---|---|---|")
    for name, s in sets.items():
        H = s.get("harm_control")
        if not H:
            continue
        for which in ("site", "compare"):
            e = H[which]
            hl = e["harm_link"]
            if "contrast" in e:
                c = e["contrast"]
                print(f"| {name} | {e['layer']} | {ci(c, 'gap_refusal_minus_harm_corrected')} | "
                      f"{f(c['point']['refusal_MZ_minus_U_corrected'])} | "
                      f"{f(c['point']['harm_MZ_minus_U_corrected'])} | "
                      f"{ci(hl, 'link_U_corrected')} |")
            else:
                lk = [k for k in hl["point"] if k.startswith("link_") and k.endswith("_corrected")
                      and np.isfinite(hl["point"][k])]
                print(f"| {name} | {e['layer']} | \u2014 (no MZ pairs) | \u2014 | \u2014 | "
                      + "; ".join(f"{k[5:-10]} {ci(hl, k)}" for k in lk) + " |")


def harm_per_recipe(sets: dict) -> None:
    print("\n## Harm-direction control per recipe (corrected means over pairs; point estimates)\n")
    print("| set | recipe | layer | n MZ / U | refusal r_MZ | refusal r_U | refusal gap | harm r_MZ | "
          "harm r_U | harm gap | refusal − harm |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, s in sets.items():
        H = s.get("harm_control")
        if not H:
            continue
        for which in ("site", "compare"):
            e = H[which]
            pr = e.get("contrast", {}).get("per_recipe")
            if not pr:
                continue
            for rc, v in sorted(pr.items()):
                print(f"| {name} | {rc} | {e['layer']} | {v['n_MZ']} / {v['n_U']} | "
                      f"{f(v['refusal_r_MZ_corrected'])} | {f(v['refusal_r_U_corrected'])} | "
                      f"{f(v['refusal_MZ_minus_U_corrected'])} | {f(v['harm_r_MZ_corrected'])} | "
                      f"{f(v['harm_r_U_corrected'])} | {f(v['harm_MZ_minus_U_corrected'])} | "
                      f"{f(v['gap_refusal_minus_harm_corrected'])} |")


def validity_cosines(sets: dict) -> None:
    print("\n## Validity (projection vs P per twin, adv-harmful, corrected Pearson) and cosines at (13, −1)\n")
    for name, s in sets.items():
        V = s.get("validity")
        if V:
            vals = [v for v in V.values()]
            cols = {"rel_proj": [v["rel_proj"] for v in vals],
                    "pearson r_corrected": [v["pearson"]["r_corrected"] for v in vals],
                    "spearman r_corrected": [v["spearman"]["r_corrected"] for v in vals]}
            line = "; ".join(f"{k} {min(x):.3f} to {max(x):.3f} (mean {np.mean(x):.3f})"
                             for k, x in cols.items())
            print(f"- {name} ({len(vals)} twins): {line}")
        C = s.get("cosines")
        if C:
            print(f"  cosines: {json.dumps(C['levels'])}")


def link_cell(row: dict) -> float:
    """The corrected different-base link of a site-map row: U for pair-level sets, `neither`
    (different init and order) for the factorial set; absent keys are NaN, not missing."""
    for k in ("link_U_corrected", "link_neither_corrected"):
        if k in row and np.isfinite(row[k]):
            return row[k]
    return float("nan")


def site_map(sets: dict) -> None:
    print("\n## Site map (point estimates, corrected different-base link: U, or `neither` for the "
          "factorial set; layers 0–L by column)\n")
    for name, s in sets.items():
        sm = s.get("site_map")
        if not sm:
            continue
        rows = {(r["layer"], r["position"]): r for r in sm}
        L = max(k[0] for k in rows) + 1
        print(f"\n### Set {name} ({L} layers) \u2014 link")
        print("| position | " + " | ".join(str(l) for l in range(L)) + " |")
        print("|---|" + "---|" * L)
        for pos in range(-7, 0):
            print(f"| {pos} | " + " | ".join(f(link_cell(rows[(l, pos)]), 2)
                                             for l in range(L)) + " |")
        if any(np.isfinite(r.get("MZ_minus_U_corrected", np.nan)) for r in sm):
            print(f"\n### Set {name} \u2014 gap MZ − U")
            print("| position | " + " | ".join(str(l) for l in range(L)) + " |")
            print("|---|" + "---|" * L)
            for pos in range(-7, 0):
                print(f"| {pos} | " + " | ".join(f(rows[(l, pos)].get("MZ_minus_U_corrected"), 2)
                                                 for l in range(L)) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", type=Path, default=ROOT / "runs/021/ladder.json")
    a = ap.parse_args()
    d = json.loads(a.ladder.read_text())
    sets = d["sets"]
    print(f"# 021 report from {a.ladder}; sets {list(sets)}")
    for name, s in sets.items():
        print(f"- {name}: models {len(s['models'])}, levels {s['levels']}, blocks {s['blocks']}, "
              f"commit {s['git_commit'][:7]}, manifest sha {s['manifest_sha256'][:8]}")
    link_table(sets)
    prediction(sets)
    per_recipe(sets)
    ladder(sets)
    harm(sets)
    harm_per_recipe(sets)
    validity_cosines(sets)
    site_map(sets)


if __name__ == "__main__":
    main()
