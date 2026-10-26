"""Held-out perplexity of one checkpoint with OLMo's own in-loop LM evaluator, on 1 GPU.

Runs in the OLMo env. The perplexity instrument, applied identically to released DataDecide
checkpoints and to ours:
  - the evaluator is `all-small-ppl-validation` from the ladder config (11 v3_small sets),
    built by `olmo.eval.build_evaluator` and run by an unmodified `olmo.train.Trainer.eval()`
    (same autocast, same fused loss, same per-sequence MeanMetric as in training);
  - world size 1, so `drop_last` drops nothing and every sequence of every set is scored.
    (In-loop values at world size 8 drop up to 7 seed-dependent sequences per set; those are
    compared only against each other.)
  - the sampler seed is fixed (`--sampler-seed`, default 6198) for every model, so the batch
    composition is identical across models too.
The only site changes to the config are `seed` (sampler) and `model.init_device=cpu`.

  uv run --frozen --project $CASTOR_WORK/third_party/olmo-env python shell/olmo/eval_ppl.py \
      --config $CASTOR_WORK/runs/003/eval/config-150M.yaml \
      --hf allenai/DataDecide-dclm-baseline-150M@step37500-seed-default --out ppl.json
  ... --olmo-ckpt $RUN/step38157-unsharded/model.pt --out ppl.json
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from olmo.config import EvaluatorType, TrainConfig
from olmo.eval import build_evaluator
from olmo.model import OLMo
from olmo.optim import build_optimizer
from olmo.tokenizer import Tokenizer
from olmo.train import Trainer
from safetensors.torch import load_file

LABEL = "all-small-ppl-validation"


def released_state(spec: str) -> tuple[dict[str, torch.Tensor], dict]:
    repo, rev = spec.split("@")
    path = Path(snapshot_download(repo, revision=rev))
    sd = {}
    for shard in sorted(path.glob("*.safetensors")):
        sd.update({k.removeprefix("model."): v for k, v in load_file(str(shard)).items()})
    return sd, {"repo": repo, "revision": rev, "snapshot": path.name}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", help="REPO@REVISION of a released hf_olmo checkpoint")
    src.add_argument("--olmo-ckpt", type=Path, help="an OLMo unsharded model.pt")
    ap.add_argument("--sampler-seed", type=int, default=6198)
    ap.add_argument("--git-commit", required=True, help="castor commit (from the wrapper)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cfg = TrainConfig.load(a.config)
    cfg.seed = a.sampler_seed
    cfg.model.init_device = "cpu"
    assert cfg.precision == "amp_bf16" and cfg.fused_loss, (cfg.precision, cfg.fused_loss)
    ev_cfg = [e for e in cfg.evaluators if e.type == EvaluatorType.lm]
    assert [e.label for e in ev_cfg] == [LABEL], [e.label for e in ev_cfg]

    if a.hf:
        sd, source = released_state(a.hf)
    else:
        sd = torch.load(a.olmo_ckpt, map_location="cpu")
        source = {"olmo_ckpt": str(a.olmo_ckpt), "sha256": sha256(a.olmo_ckpt)}

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model = OLMo(cfg.model)
    model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    model = model.to(device)
    tok = Tokenizer.from_train_config(cfg)
    evaluator = build_evaluator(cfg, ev_cfg[0], tok, device)
    trainer = Trainer(cfg=cfg, model=model, dist_model=model, optim=build_optimizer(cfg, model),
                      scheduler=None, train_loader=None, device=device, evaluators=[evaluator])
    t0 = time.time()
    metrics = trainer.eval()
    n_seq = len(evaluator.eval_loader.dataset)

    ce = {k.split("/")[1]: v for k, v in metrics.items() if k.endswith("/CrossEntropyLoss")}
    assert len(ce) == 11, sorted(ce)
    mean_ce = sum(ce.values()) / len(ce)
    out = {
        "source": source, "config": str(a.config), "sampler_seed": a.sampler_seed,
        "world_size": 1, "n_sequences_total": n_seq,
        "ce": ce, "ppl": {k: math.exp(v) for k, v in ce.items()},
        "mean_ce_11": mean_ce, "seconds": time.time() - t0,
        "git_commit": a.git_commit, "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"mean_ce_11": mean_ce, "c4_en": ce.get("c4_en-validation")}), flush=True)


if __name__ == "__main__":
    main()
