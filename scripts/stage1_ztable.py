"""z-rule for new pretraining runs against the released seeds.

Every number here comes from our own evaluators, run identically on the 75 released checkpoints
of one size (25 recipes x 3 seeds; --size/--step/--released-list select them) and on our runs at
the same step:
  - mean held-out CE = equal-weight mean of the 11 per-set CEs of shell/olmo/eval_ppl.py
    (aggregate fixed before any run of ours was evaluated);
  - OLMES-10 macro = castor.olmes10 over shell/olmes/olmes10.sh outputs.
σ_seed is pooled over all 25 recipes (50 df); m̄ is the mean of the recipe's 3 released seeds;
z = |x − m̄| / (σ_seed √(4/3)) ≤ 2 on both metrics (castor.seedrule).
Also written: our recompute vs Ai2's released table for all 75 (instrument check, not a gate).
Every evaluation (released and ours, ppl.json and olmes/provenance.json) must come from commits
whose instrument code is identical (comments and docstrings aside).

    uv run --frozen python scripts/stage1_ztable.py \
        --size 150M --step 37500 --released-list configs/eval/released-150M-step37500.tsv \
        --released-dir /work/harrychang08/castor/runs/003/eval/released-150M \
        --ours castor-dclm-150M-s1001__step37500=1b castor-dclm-150M-s6198__step37500=1a \
        --ours-dir /work/harrychang08/castor/runs/003/eval/ours-150M \
        --out /work/harrychang08/castor/runs/003/stage1b
"""

import argparse
import ast
import json
import math
import subprocess
from pathlib import Path

import pandas as pd

from castor import provenance
from castor.olmes10 import olmes10
from castor.seedrule import passes, pooled_sd, z_score

INSTRUMENT = ["shell/olmo/eval_ppl.py", "castor/olmes10.py", "shell/olmes/olmes10.sh"]
EVALS = "/work/harrychang08/castor/data/datadecide_evals"
SEED_NAMES = {"seed-default": "default", "seed-small-aux-2": "small aux 2",
              "seed-small-aux-3": "small aux 3", "seed-large-aux-2": "large aux 2",
              "seed-large-aux-3": "large aux 3"}


def measure(d: Path) -> dict:
    ppl = json.loads((d / "ppl.json").read_text())
    ol = olmes10(d / "olmes")
    return {"mean_ce": ppl["mean_ce_11"], "olmes10": ol["olmes_10_macro_avg"],
            "ppl_commit": ppl["git_commit"], "n_seq": ppl["n_sequences_total"],
            "olmes_commit": json.loads((d / "olmes" / "provenance.json").read_text())["git_commit"],
            **{f"ce/{k}": v for k, v in ppl["ce"].items()},
            **{f"olmes/{k}": v for k, v in ol["scores"].items()}}


def recipe_of(repo: str, size: str) -> str:
    """allenai/DataDecide-<recipe>-<size> -> <recipe>."""
    stem = repo.split("/")[1]
    assert stem.startswith("DataDecide-") and stem.endswith(f"-{size}"), (repo, size)
    return stem.removeprefix("DataDecide-").removesuffix(f"-{size}")


def _code(text: str, path: str) -> object:
    """A file's code without comments and docstrings (Python: the AST; shell: non-comment lines)."""
    if not path.endswith(".py"):
        return [ln for ln in text.splitlines() if ln.startswith("#!") or not ln.lstrip().startswith("#")]
    tree = ast.parse(text)
    for n in ast.walk(tree):
        if (isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and n.body and isinstance(n.body[0], ast.Expr)
                and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str)):
            n.body = n.body[1:]
    return ast.dump(tree)


def check_instrument(commits: set[str]) -> dict[str, str]:
    """The instrument code must be identical across every commit that produced an evaluation.
    Byte-identical files pass as "identical"; files that differ only in comments and docstrings
    (086's cleanup) pass as "code-identical"; anything else fails."""
    cs = sorted(commits)

    def show(c: str, f: str) -> str | None:
        r = subprocess.run(["git", "show", f"{c}:{f}"], capture_output=True, text=True, check=False)
        return r.stdout if r.returncode == 0 else None

    def code(c: str, f: str) -> object:
        text = show(c, f)
        return None if text is None else _code(text, f)

    out = {}
    for c in cs[1:]:
        diff = subprocess.run(["git", "diff", "--quiet", cs[0], c, "--", *INSTRUMENT], check=False)
        if diff.returncode == 0:
            out[c] = "identical"
            continue
        bad = [f for f in INSTRUMENT if code(cs[0], f) is None or code(cs[0], f) != code(c, f)]
        assert not bad, f"instrument code differs between {cs[0]} and {c}: {bad}"
        out[c] = "code-identical"
    return out


def ai2_mean_ce(pp: pd.DataFrame) -> pd.Series:
    """Ai2's table stores per-set perplexity; mean held-out CE = mean over the 11 sets of log ppl."""
    cols = [c for c in pp.columns if c.endswith("/Perplexity")]
    assert len(cols) == 11, cols
    return pp[cols].map(math.log).mean(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--released-dir", type=Path, required=True)
    ap.add_argument("--size", required=True, help="DataDecide size, e.g. 150M, 1B")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--released-list", type=Path, required=True)
    ap.add_argument("--ours", nargs="+", required=True, help="<label>=<name>")
    ap.add_argument("--ours-dir", type=Path, required=True)
    ap.add_argument("--recipe", default="dclm-baseline", help="recipe of our runs (repo stem)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows = []
    for line in a.released_list.read_text().splitlines():
        label, repo, rev, *_ = line.split("\t")
        recipe = recipe_of(repo, a.size)
        step, seed = rev.split("-", 1)
        assert step == f"step{a.step}", (rev, a.step)
        rows.append({"label": label, "recipe": recipe, "seed": seed, "repo": repo, "revision": rev,
                     **measure(a.released_dir / label)})
    rel = pd.DataFrame(rows)
    assert len(rel) == 75 and rel.recipe.nunique() == 25, (len(rel), rel.recipe.nunique())
    sig, dof = {}, {}
    for m in ["mean_ce", "olmes10"]:
        sig[m], dof[m] = pooled_sd(rel.rename(columns={m: "value"}))
    base = rel[rel.recipe == a.recipe]
    assert len(base) == 3, base

    ours = []
    for spec in a.ours:
        label, name = spec.split("=")
        ours.append((label, name, measure(a.ours_dir / label)))
    # Evaluations may span commits (reruns); the instrument code must be identical across all.
    instrument = check_instrument(set(rel.ppl_commit) | set(rel.olmes_commit)
                     | {x[k] for *_, x in ours for k in ["ppl_commit", "olmes_commit"]})

    ztab = []
    for label, name, x in ours:
        row = {"run": name, "label": label}
        for m in ["mean_ce", "olmes10"]:
            z = z_score(x[m], base[m].tolist(), sig[m])
            row |= {m: x[m], f"{m}_mbar": base[m].mean(), f"{m}_sigma": sig[m], f"z_{m}": z,
                    f"pass_{m}": passes(z)}
        row["pass"] = row["pass_mean_ce"] and row["pass_olmes10"]
        ztab.append(row)
    zt = pd.DataFrame(ztab)

    # Instrument check against Ai2's released table (not a gate).
    ev = pd.read_parquet(f"{EVALS}/macro_avg-00000-of-00001.parquet")
    ev = ev[(ev.params == a.size) & (ev.step == a.step) & (ev.task == "olmes_10_macro_avg")]
    pp = pd.read_parquet(f"{EVALS}/ppl_results.parquet")
    pp = pp[(pp.params == a.size) & (pp.step == a.step)]
    chk = rel[["label", "recipe", "seed"]].copy()
    # repo stem -> Ai2 table name (configs/eval/datadecide-recipe-names.json, one-to-one)
    mapping = json.loads(Path("configs/eval/datadecide-recipe-names.json").read_text())
    chk["data"] = chk.recipe.map(mapping)
    assert chk.data.notna().all(), chk[chk.data.isna()]
    chk["ai2_seed"] = chk.seed.map(lambda s: SEED_NAMES[s])
    ai2_olmes = ev.assign(v=ev.metrics.map(lambda s: json.loads(s)["primary_metric"])) \
        .set_index(["data", "seed"]).v
    ai2_ce = pp.assign(v=ai2_mean_ce(pp)).set_index(["data", "seed"]).v
    chk["ai2_olmes10"] = [ai2_olmes.get((d, s)) for d, s in zip(chk.data, chk.ai2_seed)]
    chk["ai2_mean_ce"] = [ai2_ce.get((d, s)) for d, s in zip(chk.data, chk.ai2_seed)]
    chk = chk.merge(rel[["label", "mean_ce", "olmes10"]], on="label")
    chk["d_olmes10"] = chk.olmes10 - chk.ai2_olmes10
    chk["d_mean_ce"] = chk.mean_ce - chk.ai2_mean_ce
    sig_ai2 = {m: pooled_sd(chk.rename(columns={f"ai2_{m}": "value"}))[0]
               for m in ["mean_ce", "olmes10"]}

    a.out.mkdir(parents=True, exist_ok=True)
    rel.to_csv(a.out / f"released_{a.size}_step{a.step}.tsv", sep="\t", index=False)
    zt.to_csv(a.out / "ztable.tsv", sep="\t", index=False)
    chk.to_csv(a.out / "instrument_vs_ai2.tsv", sep="\t", index=False)
    summary = {
        "size": a.size, "step": a.step, "sigma_seed": sig, "dof": dof, "sigma_seed_from_ai2_table": sig_ai2,
        "mbar_recipe": {m: float(base[m].mean()) for m in ["mean_ce", "olmes10"]},
        "ztable": ztab,
        "instrument_vs_ai2": {"max_abs_d_olmes10": float(chk.d_olmes10.abs().max()),
                              "max_abs_d_mean_ce": float(chk.d_mean_ce.abs().max()),
                              "mean_d_olmes10": float(chk.d_olmes10.mean()),
                              "mean_d_mean_ce": float(chk.d_mean_ce.mean())},
        "instrument_commits": instrument,
        "git_commit": provenance.git_commit(), "command": provenance.command(),
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(zt.to_string())
    print(json.dumps({k: summary[k] for k in ["sigma_seed", "dof", "sigma_seed_from_ai2_table",
                                              "instrument_vs_ai2"]}, indent=1))


if __name__ == "__main__":
    main()
