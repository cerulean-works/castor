"""Print every table of runs/046/harm_per_recipe.json (formatting only).

    uv run --frozen python scripts/report_046.py [--artifact runs/046/harm_per_recipe.json]
"""

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def f3(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "\u2014"
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def ci(d, nd=3):
    return f"{f3(d['estimate'], nd)} [{f3(d['lo'], nd)}, {f3(d['hi'], nd)}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=ROOT / "runs/046/harm_per_recipe.json")
    a = ap.parse_args()
    d = json.loads(a.artifact.read_text())
    cells, s = d["cells"], d["summary"]
    group_of = {rc: g for g, rcs in s["groups"].items() for rc in rcs}
    print(f"# 046 report from {a.artifact}; {len(cells)} cells; commits {d['git_commits']}; "
          f"draws {d['n_boot_sec']}; adv-harmful prompts {d['n_prompts']['adv_harmful']}")
    print(f"\n## Groups (rule: shared >= {s['prediction']['rule']['shared_min']}, base-specific <= "
          f"{s['prediction']['rule']['specific_max']}, layer-13 harm r_U)")
    for g, rcs in s["groups"].items():
        print(f"- {g} ({len(rcs)}): {', '.join(rcs)}")
    p = s["prediction"]
    print(f"- part 1 (bimodal): n between = {p['part1_bimodal']['n_between']}, largest jump between "
          f"sorted values {p['part1_bimodal']['largest_gap_between_sorted_values']:.3f} -> "
          f"{p['part1_bimodal']['holds']}")
    print(f"- part 2 (membership): base-specific = {p['part2_membership']['base_specific']}, predicted "
          f"missing {p['part2_membership']['predicted_missing']}, extra {p['part2_membership']['extra']} "
          f"-> {p['part2_membership']['holds']}")
    p3 = p["part3_shared_gap_below_refusal"]
    print(f"- part 3 (shared-group harm gap < refusal gap at both layers): {p3['n_holding']} of "
          f"{p3['n_recipes']} -> {p3['holds']}")

    order = sorted(cells, key=lambda c: (cells[c]["set"] == "S300M",
                                         -cells[c]["layers"]["13"]["point"]["harm_r_U_corrected"]))
    for la in ("13", "11"):
        site = d["sites"][la]
        print(f"\n## Per recipe, layer {la} (refusal at ({site['layer']}, {site['position']}), harm at "
              f"position {d['harm_position']}; corrected; {d['n_boot_sec']} prompt draws)")
        print("| cell | group | n MZ / U | refusal r_MZ | refusal r_U | refusal gap | harm r_MZ | "
              "harm r_U | harm gap | refusal − harm gap |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for c in order:
            v = cells[c]
            pt, cp = v["layers"][la]["point"], v["layers"][la]["ci_prompts"]
            print(f"| {c} | {group_of[v['recipe']]} | {v['n_MZ']} / {v['n_U']} | "
                  f"{f3(pt['refusal_r_MZ_corrected'])} | {ci(cp['refusal_r_U_corrected'])} | "
                  f"{ci(cp['refusal_MZ_minus_U_corrected']) if v['n_MZ'] else '\u2014'} | "
                  f"{f3(pt['harm_r_MZ_corrected'])} | {ci(cp['harm_r_U_corrected'])} | "
                  f"{ci(cp['harm_MZ_minus_U_corrected']) if v['n_MZ'] else '\u2014'} | "
                  f"{ci(cp['diff_MZ_minus_U_corrected']) if v['n_MZ'] else '\u2014'} |")

    print("\n## Harm r_U by layer (position −8; corrected point estimates; layers 0–15)")
    print("| cell | " + " | ".join(str(i) for i in range(16)) + " |")
    print("|---|" + "---|" * 16)
    for c in order:
        bl = cells[c]["harm_by_layer"]
        print(f"| {c} | " + " | ".join(f3(bl[str(i)].get("r_U_corrected"), 2) if str(i) in bl else "\u2014"
                                       for i in range(16)) + " |")

    for la in ("13", "11"):
        print(f"\n## Diagnosis (a), layer {la}: direction reliability and cosines (harm; refusal)")
        print("| cell | harm dir. reliability (mean) | harm cos MZ | harm cos U (mean, max |.|) | "
              "refusal dir. reliability | refusal cos MZ | refusal cos U |")
        print("|---|---|---|---|---|---|---|")
        for c in order:
            da = cells[c]["layers"][la]["diagnosis_a"]
            h, r = da["harm"], da["refusal"]
            hm = h["levels"]["MZ"]
            rm = r["levels"]["MZ"]
            rel = lambda x: f3(sum(x["reliability"].values()) / len(x["reliability"]), 4)
            print(f"| {c} | {rel(h)} | {f3(hm.get('cos')) if hm['n_pairs'] else '\u2014'} | "
                  f"{f3(h['levels']['U']['cos'])}, {f3(h['levels']['U']['cos_abs_max'])} | {rel(r)} | "
                  f"{f3(rm.get('cos')) if rm['n_pairs'] else '\u2014'} | {f3(r['levels']['U']['cos'])} |")

    for la in ("13", "11"):
        print(f"\n## Diagnosis (b), layer {la}: harm r_U with one harmful training source left out "
              f"(and the size-matched random drop, mean of 3 seeds); cos = cosine of the reduced "
              f"direction to the full one (first model)")
        print("| cell | full | −advbench (194) | random −194 | cos | −malicious_instruct (42) | "
              "random −42 | cos | −tdc2023 (20) | random −20 | cos |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for c in order:
            v = cells[c]["layers"][la]
            row = [c, f3(v["point"]["harm_r_U_corrected"])]
            for src in ("advbench", "malicious_instruct", "tdc2023"):
                b = v["diagnosis_b"][src]
                row += [f3(b["harm_r_U_corrected"]), f3(b["random_drop_control"]["harm_r_U_corrected_mean"]),
                        f3(min(b["cos_to_full_direction"].values()), 2)]
            print("| " + " | ".join(row) + " |")

    for la in ("13", "11"):
        print(f"\n## Diagnosis (c), layer {la}: the last instruction token (adv-harmful prompts; "
              f"means over the cell's models; within-group levels corrected)")
        print("| cell | group | η² harm ~ last char | η² harm ~ last token | η² refusal ~ last char | "
              "η² refusal ~ last token | r(harm, ‖x‖) | harm r_U within char | harm r_U within token | "
              "harm gap within token | refusal r_U within token |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for c in order:
            v = cells[c]["layers"][la]["diagnosis_c"]
            m = v["mean"]
            print(f"| {c} | {group_of[cells[c]['recipe']]} | {f3(m['eta2_harm_last_char'])} | "
                  f"{f3(m['eta2_harm_last_token'])} | {f3(m['eta2_refusal_last_char'])} | "
                  f"{f3(m['eta2_refusal_last_token'])} | {f3(m['r_harm_norm'])} | "
                  f"{f3(v['harm_within_last_char']['r_U_corrected'])} | "
                  f"{f3(v['harm_within_last_token']['r_U_corrected'])} | "
                  f"{f3(v['harm_within_last_token']['MZ_minus_U_corrected']) if cells[c]['n_MZ'] else '\u2014'} | "
                  f"{f3(v['refusal_within_last_token']['r_U_corrected'])} |")

    tg = d["token_groups"]
    print("\n## Token groups (data only)")
    print(f"- panel last-character shares: {tg['panel_last_char_shares']}; distinct last tokens "
          f"{tg['panel_n_distinct_last_tokens']}")
    for k, v in tg["extraction"].items():
        print(f"- {k}: n {v['n']}, last-char shares {v['last_char_shares']}, distinct last tokens "
              f"{v['n_distinct_last_tokens']}, top tokens {v['top_last_tokens']}")
    print(f"- extraction_last_char (all four sets): {d['extraction_last_char']}")
    print("\n## Timing")
    for pth, v in d["parts"].items():
        print(f"- {pth}: {v['seconds']:.0f} s, commit {v['git_commit'][:7]}, cells {len(v['cells'])}")


if __name__ == "__main__":
    main()
