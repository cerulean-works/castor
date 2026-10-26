"""Same-seed replicate divergence: table and figure from pairs_025.py's JSON, the six step-0 checks
and the six training logs (no GPU). Rules, fixed before data: first divergence = smallest saved
step with >= 1 bit-unequal tensor; rel L2 at 100 from the l100 pairs; same- vs cross-node factor
= rel L2(P1~P3) / rel L2(P1~P2) at step 100; losses / grad norms as OLMo prints them.

    uv run --frozen python scripts/divergence_025.py --out /home/harrychang08/castor/runs/025/probe1
"""

import argparse
import json
import re
from pathlib import Path

from castor import provenance

RUNS = Path("/work/harrychang08/castor/runs/003")
LOGS = Path("/work/harrychang08/castor/runs/025/probe1-logs")
REPS = ["P1", "P2", "P3"]
TAGS = ["s10", "l100"]
STEP = re.compile(r"\[step=(\d+)/\d+\]")
METRIC = re.compile(r"^\s+(train/CrossEntropyLoss|optim/total_grad_norm)=([\d,.eE+-]+)\s*$")


def train_metrics(text: str) -> dict[int, dict[str, float]]:
    """step -> {train/CrossEntropyLoss, optim/total_grad_norm} from OLMo's console block after `[step=N/M]`."""
    out: dict[int, dict[str, float]] = {}
    step = None
    for line in text.splitlines():
        if m := STEP.search(line):
            step = int(m.group(1))
        elif (m := METRIC.match(line)) and step is not None:
            out.setdefault(step, {})[m.group(1)] = float(m.group(2).replace(",", ""))
    return out


def first_divergence(rows: list[dict], pair: str) -> int | None:
    steps = sorted(r["step"] for r in rows if r["pair"] == pair and r["n_unequal"] > 0)
    return steps[0] if steps else None


def first_metric_difference(a: dict[int, dict], b: dict[int, dict], key: str) -> int | None:
    for s in sorted(set(a) & set(b)):
        if a[s].get(key) != b[s].get(key):
            return s
    return None


def row(rows: list[dict], pair: str, step: int, tag: str) -> dict:
    hit = [r for r in rows if r["pair"] == pair and r["step"] == step and r["a"].endswith(tag)]
    assert len(hit) == 1, (pair, step, tag, len(hit))
    return hit[0]


def svg(rows: list[dict]) -> str:
    """Rel L2 vs step, log-log, one line per replicate pair (hand-written SVG: no plotting library
    in castor's env, as divergence_011.py)."""
    import math
    style = {"P1~P2": ("#1f77b4", "same node (P1~P2)"), "P1~P3": ("#d62728", "cross node (P1~P3)"),
             "P2~P3": ("#ff7f0e", "cross node (P2~P3)")}
    series = {pr: sorted({(r["step"], r["rel_l2"]) for r in rows if r["pair"] == pr and r["step"] > 0
                          and r["rel_l2"] > 0}) for pr in style}
    ys = [y for v in series.values() for _, y in v] or [1e-6, 1.0]
    y0, y1 = math.floor(math.log10(min(ys))), math.ceil(math.log10(max(ys)))
    y1 = max(y1, y0 + 1)
    W, H, L, T, R, B = 560, 360, 70, 30, 170, 45
    def px(x: float) -> float:
        return L + (W - L - R) * math.log10(x) / 2  # steps 1..100
    def py(y: float) -> float:
        return T + (H - T - B) * (y1 - math.log10(y)) / (y1 - y0)
    parts = [f'<rect x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}" fill="none" stroke="#999"/>']
    for x in (1, 10, 100):
        parts.append(f'<text x="{px(x):.1f}" y="{H - B + 18}" font-size="12" text-anchor="middle">{x}</text>')
    for e in range(y0, y1 + 1):
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{py(10**e):.1f}" y2="{py(10**e):.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{L - 6}" y="{py(10**e) + 4:.1f}" font-size="12" text-anchor="end">1e{e}</text>')
    for i, (pr, (color, label)) in enumerate(style.items()):
        pts = series[pr]
        if pts:
            d = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
            parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="1.8"/>')
            parts += [f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.5" fill="{color}"/>' for x, y in pts]
        ly = T + 15 + 18 * i
        parts.append(f'<line x1="{W - R + 12}" x2="{W - R + 30}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{W - R + 35}" y="{ly + 4}" font-size="11">{label}</text>')
    parts.append(f'<text x="{(L + W - R) / 2}" y="{H - 8}" font-size="12" text-anchor="middle">training step (log)</text>')
    parts.append(f'<text x="16" y="{(T + H - B) / 2}" font-size="12" text-anchor="middle" '
                 f'transform="rotate(-90 16 {(T + H - B) / 2})">relative L2 ‖a − b‖/‖b‖ (log)</text>')
    parts.append(f'<text x="{L}" y="18" font-size="13">Same-seed 1B OLMo replicates (init = data seed 6198)</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'style="background:#fff;font-family:sans-serif">' + "".join(parts) + "</svg>\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, help="default <out>/pairs.json")
    a = ap.parse_args()
    assert a.out.is_absolute()
    pj = json.loads((a.pairs or a.out / "pairs.json").read_text())
    rows = pj["rows"]
    # step 0 vs released
    step0 = {}
    for p in REPS:
        for t in TAGS:
            c = json.loads((RUNS / f"025-{p}-{t}" / "compare-step0.json").read_text())["vs"]["step0-seed-default"]
            step0[f"{p}-{t}"] = c
    metrics = {f"{p}-{t}": train_metrics((LOGS / f"025-{p}-{t}.log").read_text(errors="replace"))
               for p in REPS for t in TAGS}
    pairs = ["P1~P2", "P1~P3", "P2~P3"]
    summary = {
        "step0_all_exact": all(v["exact"] for v in step0.values()), "step0": step0,
        "first_divergence_step": {pr: first_divergence(rows, pr) for pr in
                                  [*pairs, *(f"{p}:s10~l100" for p in REPS)]},
        "rel_l2_step100": {pr: row(rows, pr, 100, "l100")["rel_l2"] for pr in pairs},
        "rel_l2_step1": {pr: row(rows, pr, 1, "s10")["rel_l2"] for pr in pairs},
        "first_loss_difference": {}, "first_gradnorm_difference": {},
    }
    r100 = summary["rel_l2_step100"]
    summary["cross_over_same_node_step100"] = r100["P1~P3"] / r100["P1~P2"]
    summary["cross_over_same_node_step100_P2P3"] = r100["P2~P3"] / r100["P1~P2"]
    for pr in pairs:
        x, y = pr.split("~")
        for t in TAGS:
            summary["first_loss_difference"][f"{pr}:{t}"] = first_metric_difference(
                metrics[f"{x}-{t}"], metrics[f"{y}-{t}"], "train/CrossEntropyLoss")
            summary["first_gradnorm_difference"][f"{pr}:{t}"] = first_metric_difference(
                metrics[f"{x}-{t}"], metrics[f"{y}-{t}"], "optim/total_grad_norm")
    a.out.mkdir(parents=True, exist_ok=True)
    cols = ["step", "pair", "a", "b", "n_unequal", "n_tensors", "n_unequal_elements", "n_elements",
            "max_abs", "rel_l2", "cos"]
    lines = ["\t".join(cols)] + ["\t".join(str(r[c]) for c in cols)
                                 for r in sorted(rows, key=lambda r: (r["pair"], r["a"][-4:], r["step"]))]
    (a.out / "table.tsv").write_text("\n".join(lines) + "\n")
    (a.out / "losses.json").write_text(json.dumps(metrics, indent=1) + "\n")
    (a.out / "rel_l2_vs_step.svg").write_text(svg(rows))
    meta = {"script": "scripts/divergence_025.py", "git_commit": provenance.git_commit(),
            "command": provenance.command(), "pairs_json_commit": pj["git_commit"]}
    (a.out / "summary.json").write_text(json.dumps(summary | {"_meta": meta}, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "step0"}, indent=1))


if __name__ == "__main__":
    main()
