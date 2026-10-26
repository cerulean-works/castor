"""Loss curves of the 004 post-training runs, parsed from the Slurm logs (tracking is off).

    uv run --frozen python scripts/loss_curves_004.py --log 1=logs/nchc/004-sftdpo-s1-442791.out \
        --log 2=logs/nchc/004-sftdpo-s2-442802.out --out runs/posttrain/004

Each log holds one SFT then one DPO run (shell/posttrain/train.sh prints an "open-instruct <sha> |
<stage> |" header before each). open-instruct logs "Step: N, LR: x, Loss: y" every step on rank 0
(finetune.py:835-840, dpo_tune_cache.py:982-987 @ fc7748ea4). SFT loss is the sum-reduced token
loss (reduce_loss sum), DPO loss is dpo_norm. Writes loss_<stage>_seed<k>.csv, loss_summary.json
and loss_curves.svg (two panels, the two post-training seeds overlaid, 100-step rolling mean).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from castor import provenance

HEADER = re.compile(r"open-instruct \w+ \| (sft|dpo) \|")
STEP = re.compile(r"Step: (\d+), LR: ([0-9.eE+-]+), Loss: ([0-9.eE+-]+|nan)")
SERIES = {1: "#2a78d6", 2: "#eb6834"}  # dataviz reference palette, categorical slots 1-2
WINDOW = 100


def parse(text: str) -> dict[str, list[tuple[int, float, float]]]:
    """stage -> [(step, lr, loss)] in log order; lines before the first header are ignored."""
    out, stage = {}, None
    for line in text.splitlines():
        h = HEADER.search(line)
        if h:
            stage = h.group(1)
            out[stage] = []
            continue
        m = STEP.search(line)
        if m and stage:
            out[stage].append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return out


def rolling(y: np.ndarray, w: int = WINDOW) -> np.ndarray:
    c = np.cumsum(np.insert(y, 0, 0.0))
    return np.array([(c[i + 1] - c[max(0, i + 1 - w)]) / min(w, i + 1) for i in range(len(y))])


PANEL_W, PANEL_H, PAD = 420, 280, (58, 16, 30, 40)  # left, right, top, bottom


def svg_panel(x0: int, title: str, ylabel: str, series: dict[int, np.ndarray]) -> str:
    """One panel as SVG: raw per-step loss (faint) and its rolling mean (solid) per seed.
    The y range is the 1st-99th percentile of the raw values, so a few spikes do not flatten it."""
    left, right, top, bottom = PAD
    w, h = PANEL_W - left - right, PANEL_H - top - bottom
    out = [f'<text x="{x0 + left}" y="18" font-weight="bold" fill="#0b0b0b">{title}</text>']
    if not series:
        return "".join(out)
    allx = np.concatenate([s[:, 0] for s in series.values()])
    ally = np.concatenate([s[:, 1] for s in series.values()])
    ally = ally[np.isfinite(ally)]
    x_lo, x_hi = allx.min(), max(allx.max(), allx.min() + 1)
    y_lo, y_hi = np.percentile(ally, [1, 99])
    y_hi = max(y_hi, y_lo + 1e-9)

    def pt(x, y):
        yy = min(max(y, y_lo), y_hi)
        return (x0 + left + (x - x_lo) / (x_hi - x_lo) * w, top + (y_hi - yy) / (y_hi - y_lo) * h)

    for t in np.linspace(y_lo, y_hi, 5):
        _, yy = pt(x_lo, t)
        out.append(f'<line x1="{x0 + left}" x2="{x0 + left + w}" y1="{yy:.1f}" y2="{yy:.1f}" '
                   f'stroke="#e6e5e1" stroke-width="0.5"/><text x="{x0 + left - 4}" y="{yy + 4:.1f}" '
                   f'text-anchor="end" fill="#52514e">{t:.4g}</text>')
    for t in np.linspace(x_lo, x_hi, 5):
        xx, _ = pt(t, y_lo)
        out.append(f'<text x="{xx:.1f}" y="{top + h + 16}" text-anchor="middle" fill="#52514e">'
                   f'{int(t)}</text>')
    out.append(f'<text x="{x0 + left + w / 2}" y="{top + h + 32}" text-anchor="middle" '
               f'fill="#52514e">optimizer step</text>')
    out.append(f'<text transform="translate({x0 + 12},{top + h / 2}) rotate(-90)" '
               f'text-anchor="middle" fill="#52514e">{ylabel}</text>')
    for j, (k, s) in enumerate(series.items()):
        stride = max(1, len(s) // 1500)
        for y, style in ((s[:, 1], 'stroke-opacity="0.18" stroke-width="0.6"'),
                         (rolling(s[:, 1]), 'stroke-width="2"')):
            pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in
                           (pt(x, v) for x, v in zip(s[::stride, 0], y[::stride]) if np.isfinite(v)))
            out.append(f'<polyline fill="none" stroke="{SERIES[k]}" {style} points="{pts}"/>')
        lx = x0 + left + w - 70
        out.append(f'<line x1="{lx}" x2="{lx + 16}" y1="{top + 8 + 14 * j}" y2="{top + 8 + 14 * j}" '
                   f'stroke="{SERIES[k]}" stroke-width="2"/><text x="{lx + 20}" '
                   f'y="{top + 12 + 14 * j}" fill="#0b0b0b">seed {k}</text>')
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", required=True, help="K=path (post-training seed K)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for spec in a.log:
        k, path = spec.split("=", 1)
        runs[int(k)] = (path, parse(Path(path).read_text(errors="replace")))

    summary = {}
    for k, (path, stages) in runs.items():
        for stage, rows in stages.items():
            (a.out / f"loss_{stage}_seed{k}.csv").write_text(
                "step,lr,loss\n" + "".join(f"{s},{lr},{loss}\n" for s, lr, loss in rows))
            y = np.array([r[2] for r in rows])
            summary[f"{stage}_seed{k}"] = {
                "log": path, "n_steps": len(rows), "last_step": rows[-1][0] if rows else None,
                "first_100_mean": float(y[:WINDOW].mean()) if len(y) else None,
                "last_100_mean": float(y[-WINDOW:].mean()) if len(y) else None,
                "n_nan": int(np.isnan(y).sum())}

    panels = []
    for i, (stage, ylabel) in enumerate((("sft", "SFT loss (sum over tokens)"),
                                         ("dpo", "DPO loss (dpo_norm)"))):
        series = {k: np.array([(r[0], r[2]) for r in st.get(stage, [])])
                  for k, (_, st) in sorted(runs.items()) if st.get(stage)}
        panels.append(svg_panel(i * PANEL_W, stage.upper(), ylabel, series))
    (a.out / "loss_curves.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{2 * PANEL_W}" height="{PANEL_H + 30}" '
        f'font-family="sans-serif" font-size="11"><rect width="100%" height="100%" fill="#fcfcfb"/>'
        + "".join(panels)
        + f'<text x="12" y="{PANEL_H + 22}" fill="#52514e">faint: per step; solid: {WINDOW}-step '
        "mean. OLMo-2-0425-1B, post-training seeds 1 (Ai2's) and 2.</text></svg>\n")
    summary["git_commit"] = provenance.git_commit()
    summary["command"] = provenance.command()
    (a.out / "loss_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
