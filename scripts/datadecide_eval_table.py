"""Tabulate DataDecide 1B final-checkpoint (step 69369) OLMES scores per seed and seed spread.

Inputs (downloaded from HF, pinned revisions):
  allenai/DataDecide-eval-results @ 9919b5a0e61e57a85021263918fa82d6ceaee038
      data/macro_avg-00000-of-00001.parquet
  allenai/DataDecide-ppl-results  @ c4a9fa360a0c8351e71f3ede04dd165995fab68c
      data/train-00000-of-00001.parquet  (saved as ppl_results.parquet)

Metric: the `primary_metric` field of each row's `metrics` JSON (OLMES primary metric per task;
`olmes_10_macro_avg` is Ai2's released macro average over the 10 OLMES tasks).
Spread: max-min and sample SD (ddof=1) over the 3 released 1B seeds
(default, large aux 2, large aux 3).

Usage:
  uv run --frozen python scripts/datadecide_eval_table.py \
      [--dir /work/harrychang08/castor/data/datadecide_evals]
"""

import argparse
from pathlib import Path

import pandas as pd

from castor.datadecide import FINAL_STEP, SEEDS, seed_spread

TASKS = ["olmes_10_macro_avg", "mmlu", "arc_challenge", "hellaswag"]
PPL_COLS = ["eval/c4_en-validation/Perplexity", "eval/pile-validation/Perplexity"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/work/harrychang08/castor/data/datadecide_evals")
    args = ap.parse_args()
    d = Path(args.dir)

    tab = seed_spread(pd.read_parquet(d / "macro_avg-00000-of-00001.parquet"))

    pp = pd.read_parquet(d / "ppl_results.parquet")
    pp = pp[(pp.params == "1B") & (pp.step == FINAL_STEP) & pp.seed.isin(SEEDS)]
    ppl = pp.groupby("data")[PPL_COLS].agg(["mean", lambda x: x.max() - x.min()])
    ppl.columns = [f"{c.split('/')[1].split('-')[0]}_{'mean' if a == 'mean' else 'range'}"
                   for c, a in ppl.columns]
    nppl = pp.groupby("data").size().rename("ppl_n")

    pd.set_option("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.4f}".format)
    out = []
    for task in TASKS:
        t = tab[tab.task == task].sort_values("mean", ascending=False).set_index("data")
        print(f"\n=== {task} (1B, step {FINAL_STEP}) ===")
        print(t.drop(columns="task").to_string())
        out.append(t)
    # BoolQ dominates seed range for several recipes; show the 9-task macro without it
    nb = tab[~tab.task.isin(["boolq", "olmes_10_macro_avg"])].groupby("data")[SEEDS].mean()
    nb["mean"] = nb.mean(axis=1)
    nb["range"] = nb[SEEDS].max(axis=1) - nb[SEEDS].min(axis=1)
    print("\n=== 9-task macro excluding BoolQ (1B, final) ===")
    print(nb.sort_values("mean", ascending=False).to_string())
    print("\n=== seed noise vs recipe spread per task (Magnusson et al. 2025 Fig. 5 style, at 1B) ===")
    summ = tab.groupby("task").agg(mean_seed_sd=("sd", "mean"), max_seed_range=("range", "max"),
                                   recipe_sd=("mean", "std"))
    print(summ.to_string())
    print("\n=== 1B final perplexity (mean over seeds, range) ===")
    print(ppl.join(nppl).sort_values("c4_en_mean").to_string())
    tab.to_csv(d / "datadecide_1B_final_table.csv", index=False)
    ppl.join(nppl).to_csv(d / "datadecide_1B_final_ppl.csv")


if __name__ == "__main__":
    main()
