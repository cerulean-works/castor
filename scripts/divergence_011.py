"""Figure of the weight-space divergence trajectory.

Reads <run>/divergence.tsv (written by shell/olmo/weight_distance.py, trajectory mode) and draws
two panels against step: relative L2 ||a − b|| / ||b|| and global cosine, for ours ~ released
`default` (same init, same data order, our numerics), released `large-aux-2` ~ `default` (the
different-seed reference) and ours ~ `large-aux-2`. Descriptive, not a gate. Writes
divergence.svg and a copy of the TSV into --out.

    uv run --frozen python scripts/divergence_011.py \
        --tsv /work/harrychang08/castor/runs/003/castor-dclm-1B-i6198-d6198/divergence.tsv \
        --out /home/harrychang08/castor/runs/011/divergence
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from castor import provenance

LABEL = {"ours~default": "ours ~ released default (same seed)",
         "large-aux-2~default": "released aux-2 ~ default (different seed)",
         "ours~large-aux-2": "ours ~ released aux-2 (different seed)"}
# dataviz reference palette, categorical slots 1-2; ours ~ aux-2 dashed in slot 2 (same contrast)
COLOR = {"ours~default": "#2a78d6", "large-aux-2~default": "#eb6834", "ours~large-aux-2": "#eb6834"}
DASH = {"ours~large-aux-2"}
W, H, L, R, T, GAP = 760, 220, 70, 250, 30, 50


def _panel(d: pd.DataFrame, y0: float, ylabel: str, lo: float, hi: float) -> list[str]:
    xs = d.index.to_numpy(dtype=float)
    x0, x1 = xs.min(), xs.max()

    def X(v):
        return L + (v - x0) / (x1 - x0 or 1) * (W - L - R)

    def Y(v):
        return y0 + H - (v - lo) / (hi - lo) * H
    out = [f'<line x1="{L}" y1="{y0 + H}" x2="{W - R}" y2="{y0 + H}" stroke="#bbb"/>',
           (f'<text x="{L - 50}" y="{y0 + H / 2}" font-size="11" transform="rotate(-90 {L - 50} '
            f'{y0 + H / 2})" text-anchor="middle" fill="#444">{ylabel}</text>')]
    for i in range(5):
        v = lo + i * (hi - lo) / 4
        out.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W - R}" y2="{Y(v):.1f}" stroke="#eee"/>'
                   f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" font-size="10" text-anchor="end" '
                   f'fill="#666">{v:.2f}</text>')
    for c in d.columns:
        ok = d[c].notna()
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs[ok], d[c][ok]))
        dash = ' stroke-dasharray="6 4"' if c in DASH else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{COLOR[c]}" stroke-width="2"'
                   f'{dash}><title>{LABEL[c]}</title></polyline>')
        out += [f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.5" fill="{COLOR[c]}"/>'
                for x, y in zip(xs[ok], d[c][ok]) if c not in DASH]
    return out


def svg(l2: pd.DataFrame, cos: pd.DataFrame) -> str:
    parts = [(f'<text x="{L}" y="18" font-size="14" fill="#222">DCLM-Baseline 1B: weight-space '
              f'divergence, steps {l2.index.min()}–{l2.index.max()}</text>')]
    parts += _panel(l2, T, "relative L2 ||a − b|| / ||default||", 0.0, 1.5)
    parts += _panel(cos, T + H + GAP, "global cosine", 0.0, 1.0)
    xa = T + 2 * H + GAP
    x0, x1 = l2.index.min(), l2.index.max()
    for i in range(5):
        v = x0 + i * (x1 - x0) / 4
        x = L + i * (W - L - R) / 4
        parts.append(f'<text x="{x:.1f}" y="{xa + 16}" font-size="10" text-anchor="middle" '
                     f'fill="#666">{v:,.0f}</text>')
    parts.append(f'<text x="{(W - R + L) / 2}" y="{xa + 32}" font-size="11" text-anchor="middle" '
                 f'fill="#444">step</text>')
    for i, c in enumerate(l2.columns):
        y = T + 10 + 20 * i
        dash = ' stroke-dasharray="6 4"' if c in DASH else ""
        parts.append(f'<line x1="{W - R + 12}" y1="{y}" x2="{W - R + 40}" y2="{y}" '
                     f'stroke="{COLOR[c]}" stroke-width="2"{dash}/>'
                     f'<text x="{W - R + 46}" y="{y + 4}" font-size="11" fill="#222">{LABEL[c]}</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{xa + 45}" '
            f'style="background:#fff;font-family:sans-serif">' + "".join(parts) + "</svg>\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.tsv, sep="\t")
    assert set(d.pair) == set(LABEL), set(d.pair)
    l2 = d.pivot(index="step", columns="pair", values="rel_l2")[list(LABEL)]
    cos = d.pivot(index="step", columns="pair", values="cos")[list(LABEL)]
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.tsv, a.out / "divergence.tsv")
    (a.out / "divergence.svg").write_text(svg(l2, cos))
    ratio = l2["ours~default"] / l2["large-aux-2~default"]
    summary = {"steps": [int(s) for s in l2.index], "tsv": str(a.tsv),
               "l2_ratio_same_over_different_seed": {int(s): round(float(v), 4)
                                                     for s, v in ratio.items() if s > 0},
               "git_commit": provenance.git_commit(), "command": provenance.command()}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
