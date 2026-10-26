"""Write an OLMo TrainConfig YAML exactly as DataDecide's ladder builds it, with local paths.

Runs in the OLMo env. The config comes from OLMo's own `scripts/ladder.py
config_from_args` @ ccabb54e with Ai2's launch arguments (scripts/beaker/create_ladder_launches.py:
`--model {size} --data {mix} --length 5xC --name {mix} [--seed N]`). Only site-specific fields
are changed afterwards, and each change is listed in the YAML's sidecar `<out>.changes.json`:
  - data / eval-data prefix  /weka/oe-training-default/ai2-llm -> local mirror (same relative paths)
  - save_folder -> our run dir; remote_save_folder, wandb -> None (no Ai2 S3 / W&B)
  - data.seed (optional) -> separate data-order seed for the init x order factorial
Nothing in OLMo's source is edited; `find_latest_checkpoint` is bypassed only because the
ladder calls it on Ai2's private S3 bucket.

  uv run --frozen --project $CASTOR_WORK/third_party/olmo-env python shell/olmo/make_config.py \
      --model 150M --data DCLM-baseline --seed 6198 --run-name castor-150M-s6198 \
      --out $CASTOR_WORK/runs/003/castor-150M-s6198/config.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import olmo

OLMO_DIR = Path(olmo.__file__).resolve().parent.parent
sys.path.insert(0, str(OLMO_DIR))
from scripts import ladder

AI2_PREFIX = "/weka/oe-training-default/ai2-llm"
DATA_ROOT = "/work/harrychang08/castor/data/datadecide-recipes"
EVAL_ROOT = "/work/harrychang08/castor/data/datadecide-recipes"  # eval-data/perplexity/... below it


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=ladder.DEFAULT_SEED)
    ap.add_argument("--data-seed", type=int, default=None,
                    help="data-order seed if different from --seed (DataConfig.seed)")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--save-folder", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--eval-root", default=EVAL_ROOT)
    ap.add_argument("--test-existing-only", action="store_true",
                    help="TEST ONLY: keep only data files already verified on disk (step-0 init tests;"
                         " never for a reported run)")
    a = ap.parse_args()

    ns = argparse.Namespace(model=a.model, data=a.data, length="5xC", name=a.data, seed=a.seed,
                            s3=False, wandb=False, stop_at=None, epoch=None, write_location=None,
                            save_overwrite=False, load_path="<none>", eval_on_load=False)
    cfg = ladder.config_from_args(ns)
    ai2 = {"run_name": cfg.run_name, "save_folder": cfg.save_folder,
           "remote_save_folder": cfg.remote_save_folder, "load_path": None}

    changes = []
    cfg.load_path = None
    cfg.save_folder = str(a.save_folder)
    cfg.remote_save_folder = None
    cfg.run_name = a.run_name
    changes += [f"save_folder {ai2['save_folder']} -> {cfg.save_folder}",
                f"remote_save_folder {ai2['remote_save_folder']} -> None",
                f"run_name {ai2['run_name']} -> {cfg.run_name}"]

    def local(p: str, root: str) -> str:
        assert p.startswith(AI2_PREFIX), p
        return root + p[len(AI2_PREFIX):]

    cfg.data.paths = [local(p, a.data_root) for p in cfg.data.paths]
    for ev in cfg.evaluators:
        if ev.data.datasets:
            ev.data.datasets = {k: [local(p, a.eval_root) for p in v] for k, v in ev.data.datasets.items()}
    changes.append(f"path prefix {AI2_PREFIX} -> {a.data_root} (train), {a.eval_root} (eval)")
    if a.test_existing_only:
        n = len(cfg.data.paths)
        cfg.data.paths = [p for p in cfg.data.paths if Path(p + ".sha256").exists()]
        changes.append(f"TEST ONLY: data.paths restricted to {len(cfg.data.paths)}/{n} verified files")
    if a.data_seed is not None:
        cfg.data.seed = a.data_seed
        changes.append(f"data.seed None (falls back to seed={a.seed}) -> {a.data_seed}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.save(a.out)
    (a.out.parent / (a.out.name + ".changes.json")).write_text(
        json.dumps({"ladder_args": vars(ns), "ai2_values": ai2, "changes": changes}, indent=1) + "\n")
    print(a.out, cfg.run_name, "seed", cfg.seed, "data.seed", cfg.data.seed,
          "global_bs", cfg.global_train_batch_size, "max_duration", cfg.max_duration)


if __name__ == "__main__":
    main()
