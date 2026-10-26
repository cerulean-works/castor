"""Is arm A (the candidate launcher configuration) inside the identical-seed floor that arm B
(the current configuration rerun) defines against R (006's default/s1)? Pre-registered;
operationalisations fixed before any data.

    uv run --frozen python scripts/equiv_024.py --out $PWD/runs/024/equiv.json \
        --job-a <sacct job id of arm A> --job-b <arm B>

All three models: DataDecide-dclm-baseline-1B@step69369-seed-default, SFT seed 1 / DPO 111,
configs {sft,dpo}_datadecide_eosbos.yaml. R = 006's twin, B = 024/rep, A = 024/cfgA.
1. Weights: L2 over every tensor in fp32, after DPO (the rule) and after SFT (reported):
   L2(A,R) <= 1.25 L2(R,B) and L2(A,B) <= 1.25 L2(R,B). L2(R, 006 default/s2) is the
   seed-to-seed reference.
2. Per-prompt behaviour: r* = 006's corrected correlation (subset adv_harmful, outcome
   wg_vulnerable, per-prompt P(non-refusal) Pearson r / sqrt(split-half Spearman-Brown
   reliabilities), castor.reliability). Pass iff the paired prompt-bootstrap (2,000 draws,
   reliabilities recomputed per draw) 95 % interval of r*(B,R) - r*(A,R) includes 0, or the
   point difference is <= 0.01.
3. Levels: per panel source, the pooled refusal rate of 010/012 (pipeval.refusal_rate,
   wg_vulnerable_incl, every prompt of the source); |A-R| <= max(3 pt, |B-R|).
Speed-up = training-job elapsed of B / A (sacct, 8 GPUs each, SFT + DPO); GO iff >= 1.3 and
1-3 hold.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from castor import evalset, grid, pipeval, provenance, reliability, sanityb
from castor.judge.common import read_jsonl

WEIGHT_FACTOR = 1.25
R_TOL = 0.01
RATE_MIN = 0.03
MIN_SPEEDUP = 1.3
N_BOOT = 2000
SUBSET, FIELD, LEVEL_FIELD = "adv_harmful", "wg_vulnerable", "wg_vulnerable_incl"

M = sanityb.MODELS_ROOT
GROUP = "dd-dclm-1B-seed-default"
MODELS = {  # key -> (model root, model id, prompt id of its generations)
    "R": (M, "dd-dclm-1b-default-dpo-seed1", "006"),
    "B": (M / "024" / "rep", "dd-dclm-1b-default-dpo-seed1-rep", "024"),
    "A": (M / "024" / "cfgA", "dd-dclm-1b-default-dpo-seed1-cfgA", "024"),
}
SEED2 = M  # 006 default/s2 (same root as R, post-training seed 2): the seed-to-seed reference


def ckpt(root: Path, stage: str, k: int = 1) -> Path:
    return root / GROUP / stage / f"seed{k}" / "model"


def load_state(path: Path) -> dict[str, torch.Tensor]:
    """name -> tensor of a saved open-instruct checkpoint (pytorch_model*.bin or safetensors)."""
    from safetensors.torch import load_file

    st = sorted(path.glob("*.safetensors"))
    if st:
        return {k: v for f in st for k, v in load_file(str(f)).items()}
    sd = {}
    for f in sorted(path.glob("pytorch_model*.bin")):
        sd |= torch.load(f, map_location="cpu", mmap=True, weights_only=True)
    assert sd, f"no weights in {path}"
    return sd


def l2(x: dict[str, torch.Tensor], y: dict[str, torch.Tensor]) -> float:
    """Global L2 norm of the parameter difference, every tensor, in fp32 (as 010)."""
    assert x.keys() == y.keys(), f"tensor names differ: {sorted(x.keys() ^ y.keys())[:5]}"
    s = 0.0
    for k, xk in x.items():
        assert xk.shape == y[k].shape, f"{k}: {xk.shape} vs {y[k].shape}"
        s += float(torch.sum((xk.float() - y[k].float()) ** 2))
    return float(np.sqrt(s))


def weight_rule(d: dict[str, float]) -> dict:
    """d: 'A|R', 'A|B', 'R|B' distances. Both of A's distances within 1.25x the floor."""
    floor = d["R|B"]
    ok = d["A|R"] <= WEIGHT_FACTOR * floor and d["A|B"] <= WEIGHT_FACTOR * floor
    return {"distances": d, "floor": floor, "ratio_AR": d["A|R"] / floor,
            "ratio_AB": d["A|B"] / floor, "bound": WEIGHT_FACTOR * floor, "pass": bool(ok)}


def rstar(ya: np.ndarray, yb: np.ndarray) -> float:
    """006's corrected correlation of two (n_prompts, K) outcome matrices."""
    pa, ra = reliability.outcome_measure(ya)
    pb, rb = reliability.outcome_measure(yb)
    return reliability.disattenuate(reliability.corr(pa, pb), ra, rb)


def behaviour_rule(ys: dict[str, np.ndarray], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Paired prompt bootstrap of r*(B,R) - r*(A,R): one resample of prompts per draw, used for
    both terms (reliabilities recomputed inside each draw)."""
    point = {"A|R": rstar(ys["A"], ys["R"]), "B|R": rstar(ys["B"], ys["R"]),
             "A|B": rstar(ys["A"], ys["B"])}
    diff = point["B|R"] - point["A|R"]
    rng = np.random.default_rng(seed)
    n = ys["R"].shape[0]
    draws = []
    for _ in range(n_boot):
        j = rng.integers(0, n, n)
        draws.append(rstar(ys["B"][j], ys["R"][j]) - rstar(ys["A"][j], ys["R"][j]))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    ok = (lo <= 0 <= hi) or diff <= R_TOL
    raw = {k: reliability.corr(reliability.per_prompt_p(ys[a]), reliability.per_prompt_p(ys[b]))
           for k, (a, b) in {"A|R": ("A", "R"), "B|R": ("B", "R"), "A|B": ("A", "B")}.items()}
    rel = {k: reliability.split_half(y)["reliability"] for k, y in ys.items()}
    return {"r_star": point, "r_raw": raw, "reliability": rel, "diff_B_minus_A": diff,
            "diff_ci": [float(lo), float(hi)], "n_prompts": n, "n_boot": n_boot,
            "pass": bool(ok)}


def level_rule(rates: dict[str, dict[str, float]]) -> dict:
    """rates: source -> {'A','B','R'} -> refusal rate. The 3-point band with B as the spread."""
    out, ok = {}, True
    for s, r in rates.items():
        band = max(RATE_MIN, abs(r["B"] - r["R"]))
        within = abs(r["A"] - r["R"]) <= band
        ok &= within
        out[s] = r | {"A_minus_R": r["A"] - r["R"], "B_minus_R": r["B"] - r["R"],
                      "band": band, "within": bool(within)}
    return {"sources": out, "pass": bool(ok), "n_within": sum(v["within"] for v in out.values())}


def verdict(speedup: float, w: dict, b: dict, lv: dict) -> str:
    return "GO" if speedup >= MIN_SPEEDUP and w["pass"] and b["pass"] and lv["pass"] else "KILL"


def sacct_hours(job: str) -> dict:
    """GPU-hours of a finished training job (castor.grid.sacct: elapsed x allocated GPUs)."""
    s = grid.sacct(job)
    assert s.get("state") == "COMPLETED", f"job {job}: {s}"
    assert s["gpus"] > 0, f"job {job}: no GPUs in AllocTRES"
    assert s["elapsed_h"] is not None, f"job {job}: unparsed Elapsed {s['elapsed']!r}"
    return {k: s[k] for k in ("job", "elapsed_h", "gpus", "node", "state")} | \
        {"h200_h": s["elapsed_h"] * s["gpus"]}


def _epoch(stamp: str) -> float:
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S%z").timestamp()


def stage_hours(root: Path, gpus: int) -> dict:
    """Per-stage wall-clock from the run provenances (render start -> finish) x the job's GPUs."""
    out = {}
    for st in ("sft", "dpo"):
        p = json.loads((root / GROUP / st / "seed1" / "provenance.json").read_text())
        h = (_epoch(p["finished"]) - _epoch(p["started"])) / 3600
        out[st] = {"h200_h": h * gpus, "started": p["started"], "finished": p["finished"],
                   "deepspeed_config": p["open_instruct"].get("deepspeed_config"),
                   "deepspeed_config_sha256": p["open_instruct"].get("deepspeed_config_sha256"),
                   "launcher_env": (p.get("launcher") or {}).get("env")}
    return out


def oi_args(root: Path, stage: str) -> list[str]:
    """open-instruct argument list with the run-specific paths and names blanked."""
    a = (root / GROUP / stage / "seed1" / "oi_args.txt").read_text().split("\n")
    out, skip = [], {"--output_dir", "--exp_name", "--model_name_or_path"}
    for i, t in enumerate(a):
        out.append("<run>" if i and a[i - 1] in skip else t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--job-a", required=True)
    ap.add_argument("--job-b", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    assert a.out.is_absolute(), "--out must be absolute"
    commit = provenance.git_commit()

    # the open-instruct argument lists must be identical across the three runs
    args_same = {st: oi_args(MODELS["A"][0], st) == oi_args(MODELS["B"][0], st) == oi_args(M, st)
                 for st in ("sft", "dpo")}
    assert all(args_same.values()), f"open-instruct arguments differ: {args_same}"

    weights = {}
    for st in ("dpo", "sft"):
        s = {k: load_state(ckpt(root, st)) for k, (root, _, _) in MODELS.items()}
        d = {"A|R": l2(s["A"], s["R"]), "A|B": l2(s["A"], s["B"]), "R|B": l2(s["R"], s["B"])}
        d["R|seed2"] = l2(s["R"], load_state(ckpt(SEED2, st, 2)))
        weights[st] = weight_rule(d)
        del s

    panel = evalset.load_eval()
    eval_sha = json.loads(evalset.EVAL_MANIFEST.read_text())["eval"]["sha256"]
    instrument = pipeval.check_instrument({k: (mid, pid) for k, (_, mid, pid) in MODELS.items()}, eval_sha)
    rows = {k: list(read_jsonl(sanityb.RUNS / mid / pid / "judge_v1" / "labels.jsonl"))
            for k, (_, mid, pid) in MODELS.items()}
    ids = sanityb.subset_ids(panel, SUBSET)
    ys, counts = {}, {}
    for k, r in rows.items():
        ys[k], counts[k] = sanityb.outcome_matrix(r, ids, FIELD)
    beh = behaviour_rule(ys, a.n_boot)

    pp = {k: pipeval.per_prompt(pipeval.refusals_from_labels(r, LEVEL_FIELD)) for k, r in rows.items()}
    by_src = {}
    for r in panel:
        by_src.setdefault(r["source"], []).append(r["id"])
    rates = {s: {k: pipeval.refusal_rate(pp[k], i) for k in pp} for s, i in sorted(by_src.items())}
    lv = level_rule(rates)

    cost = {}
    for k, job in (("A", a.job_a), ("B", a.job_b)):
        cost[k] = sacct_hours(job)
        cost[k]["stages"] = stage_hours(MODELS[k][0], cost[k]["gpus"])
    speedup = cost["B"]["h200_h"] / cost["A"]["h200_h"]
    res = {"verdict": verdict(speedup, weights["dpo"], beh, lv), "speedup": speedup,
           "speedup_sft": cost["B"]["stages"]["sft"]["h200_h"] / cost["A"]["stages"]["sft"]["h200_h"],
           "speedup_dpo": cost["B"]["stages"]["dpo"]["h200_h"] / cost["A"]["stages"]["dpo"]["h200_h"],
           "weights": weights, "behaviour": beh, "levels": lv, "cost": cost,
           "oi_args_identical": args_same, "label_counts": counts, "instrument": instrument,
           "models": {k: {"root": str(root), "model_id": mid, "prompt_id": pid}
                      for k, (root, mid, pid) in MODELS.items()},
           "git_commit": commit, "command": provenance.command()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(res["verdict"], f"speedup {speedup:.2f}", "weights", weights["dpo"]["pass"],
          "behaviour", beh["pass"], "levels", lv["pass"])


if __name__ == "__main__":
    main()
