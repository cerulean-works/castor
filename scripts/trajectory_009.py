"""In-loop held-out loss of the two stage-2 1B runs, overlaid on the released DCLM-Baseline 1B
seeds (default, large-aux-2, large-aux-3).

Metric: mean held-out CE = equal-weight mean over the 11 sets of log(perplexity), at every step
where Ai2 evaluated. Ours: OLMo console logs (castor.olmolog, the same in-loop evaluator); released:
DataDecide-ppl-results (Ai2's in-loop values). Descriptive, not a gate (in-loop trajectories are not
reproducible to 1%); the z-rule at the final step is the gate. Also reported: each run's deviation
from the released 3-seed mean per step, next to the released seeds' own deviations (their seed
spread).

    uv run --frozen python scripts/trajectory_009.py --out /abs/runs/003/stage2/trajectory
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from castor import provenance
from castor.olmolog import in_loop_evals

PPL = "/work/harrychang08/castor/data/datadecide_evals/ppl_results.parquet"
LOGS = Path(__file__).resolve().parent.parent / "logs" / "nchc"
RUNS = {"castor-init-default-data-aux2": "003-s2-i6198-d4-seg*.out",
        "castor-init-aux2-data-default": "003-s2-i4-d6198-seg*.out"}
LABEL = {"default": "released default (i6198 d6198)", "large aux 2": "released aux-2 (i4 d4)",
         "large aux 3": "released aux-3 (i5 d5)",
         "castor-init-default-data-aux2": "ours i6198 d4", "castor-init-aux2-data-default": "ours i4 d6198",
         "castor-dclm-1B-i6198-d6198": "ours i6198 d6198 (011)",
         "castor-dclm-1B-i6198-d6198-r2": "ours i6198 d6198 R2 (035)"}
# dataviz reference palette, categorical slots 1-5 in fixed order; ours dashed (secondary encoding)
COLOR = {"default": "#2a78d6", "large aux 2": "#eb6834", "large aux 3": "#1baf7a",
         "castor-init-default-data-aux2": "#4a3aa7", "castor-init-aux2-data-default": "#e34948",
         "castor-dclm-1B-i6198-d6198": "#4a3aa7", "castor-dclm-1B-i6198-d6198-r2": "#e34948"}


W, H1, H2, L, R, T, GAP = 760, 300, 200, 70, 200, 30, 50


def _panel(d: pd.DataFrame, y0: float, h: float, ours: list[str], ylabel: str,
           zero: bool = False) -> list[str]:
    xs = d.index.to_numpy(dtype=float)
    lo, hi = float(d.min().min()), float(d.max().max())
    pad = 0.05 * (hi - lo or 1)
    lo, hi = lo - pad, hi + pad
    x0, x1 = xs.min(), xs.max()

    def X(v):
        return L + (v - x0) / (x1 - x0) * (W - L - R)

    def Y(v):
        return y0 + h - (v - lo) / (hi - lo) * h
    out = [f'<line x1="{L}" y1="{y0 + h}" x2="{W - R}" y2="{y0 + h}" stroke="#bbb"/>',
           (f'<text x="{L - 55}" y="{y0 + h / 2}" font-size="11" transform="rotate(-90 {L - 55} '
            f'{y0 + h / 2})" text-anchor="middle" fill="#444">{ylabel}</text>')]
    for v in (lo + pad, (lo + hi) / 2, hi - pad):
        out.append(f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" font-size="10" text-anchor="end" '
                   f'fill="#666">{v:.3f}</text>')
    if zero and lo < 0 < hi:
        out.append(f'<line x1="{L}" y1="{Y(0):.1f}" x2="{W - R}" y2="{Y(0):.1f}" stroke="#999"/>')
    for c in d.columns:
        ok = d[c].notna()
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs[ok], d[c][ok]))
        dash = ' stroke-dasharray="6 4"' if c in ours else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{COLOR[c]}" '
                   f'stroke-width="2"{dash}><title>{LABEL[c]}</title></polyline>')
    return out


def svg(df: pd.DataFrame, dev: pd.DataFrame, ours: list[str]) -> str:
    """Two panels (mean CE; deviation from the released 3-seed mean), steps >= 10k, legend."""
    parts = [(f'<text x="{L}" y="18" font-size="14" fill="#222">DCLM-Baseline 1B: stage-2 runs vs '
              f'released seeds (in-loop, steps {df.index.min()}–{df.index.max()})</text>')]
    parts += _panel(df, T, H1, ours, "mean held-out CE (11 sets)")
    parts += _panel(dev, T + H1 + GAP, H2, ours, "minus released 3-seed mean", zero=True)
    xa = T + H1 + GAP + H2
    for v in (df.index.min(), (df.index.min() + df.index.max()) // 2, df.index.max()):
        x = L + (v - df.index.min()) / (df.index.max() - df.index.min()) * (W - L - R)
        parts.append(f'<text x="{x:.1f}" y="{xa + 16}" font-size="10" text-anchor="middle" '
                     f'fill="#666">{v}</text>')
    parts.append(f'<text x="{(W - R + L) / 2}" y="{xa + 32}" font-size="11" text-anchor="middle" '
                 f'fill="#444">step</text>')
    for i, c in enumerate(df.columns):
        y = T + 10 + 20 * i
        dash = ' stroke-dasharray="6 4"' if c in ours else ""
        parts.append(f'<line x1="{W - R + 12}" y1="{y}" x2="{W - R + 40}" y2="{y}" '
                     f'stroke="{COLOR[c]}" stroke-width="2"{dash}/>'
                     f'<text x="{W - R + 46}" y="{y + 4}" font-size="11" fill="#222">{LABEL[c]}</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{xa + 45}" '
            f'style="background:#fff;font-family:sans-serif">' + "".join(parts) + "</svg>\n")


def job_id(p: Path) -> int:
    return int(p.stem.rsplit("-", 1)[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", metavar="RUN=LOG_GLOB", default=None,
                    help="our runs and their log globs (default: 009's two; 011 passes its replicate)")
    a = ap.parse_args()
    assert a.out.is_absolute()
    runs = dict(r.split("=", 1) for r in a.runs) if a.runs else RUNS
    pp = pd.read_parquet(PPL)
    pp = pp[(pp.params == "1B") & (pp.data == "DCLM-Baseline")
            & pp.seed.isin(["default", "large aux 2", "large aux 3"])]
    cols = [c for c in pp.columns if c.endswith("-validation/Perplexity")]
    assert len(cols) == 11, cols
    rel = pp.assign(mean_ce=pp[cols].map(math.log).mean(axis=1))[["seed", "step", "mean_ce"]]
    rel = rel.pivot(index="step", columns="seed", values="mean_ce")
    sets = [c.removeprefix("eval/").removesuffix("/Perplexity") for c in cols]

    ours, logs_used = {}, {}
    for run, pat in runs.items():
        logs = sorted(LOGS.glob(pat), key=job_id)
        logs_used[run] = [p.name for p in logs]
        ev = in_loop_evals(logs)
        s = {}
        for step, got in ev.items():
            names = {k.removesuffix("-validation") if k.endswith("-validation") else k for k in got}
            keys = [k for k in got if k in sets or f"{k}-validation" in sets]
            if len(keys) == 11:
                s[step] = sum(math.log(got[k]["Perplexity"]) for k in keys) / 11
            else:
                assert len(names) < 11, (step, names)
        ours[run] = pd.Series(s)
    df = rel.copy()
    for run, s in ours.items():
        df[run] = s.reindex(df.index)
    df.index = df.index.astype(int)
    released = ["default", "large aux 2", "large aux 3"]
    mean = df[released].mean(axis=1)
    dev = df.sub(mean, axis=0)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "mean_ce_by_step.tsv", sep="\t")
    dev.to_csv(a.out / "deviation_from_released_mean.tsv", sep="\t")

    (a.out / "trajectory.svg").write_text(svg(df[df.index >= 10000], dev[dev.index >= 10000],
                                              list(ours)))

    last = int(max(s.index.max() for s in ours.values()))
    summary = {
        "steps_compared": int(dev[list(ours)].notna().all(axis=1).sum()),
        "last_common_step": last,
        "released_seed_abs_dev_max_after_10k": float(dev.loc[dev.index >= 10000, released].abs().max().max()),
        "ours_abs_dev_max_after_10k": {r: float(dev.loc[dev.index >= 10000, r].abs().max()) for r in ours},
        "at_final": {c: (float(df.loc[69369, c]) if 69369 in df.index and pd.notna(df.loc[69369, c])
                         else None) for c in df.columns},
        "logs": logs_used, "git_commit": provenance.git_commit(), "command": provenance.command(),
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
