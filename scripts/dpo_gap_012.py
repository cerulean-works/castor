"""Is 010's DPO refusal-level gap seed noise or a systematic DPO-stage difference?

    uv run --frozen python scripts/dpo_gap_012.py --out $PWD/runs/posttrain/012/dpo_gap.json

Arms (pre-registered): s1 = 010's DPO seed 1 on our SFT s1;
A = DPO k=1 from the released SFT@0d85a3d0; B = DPO k=1 again on our SFT s1 (identical seed);
C, D = DPO k=3, 4 on our SFT s1; ref = released OLMo-2-0425-1B-DPO (labels from 004).
Statistics are 010's (castor.pipeval): per-source refusal rate = pooled share of parsed sampled
responses labelled refusal (`wg_vulnerable_incl`) over every prompt of the source, as in
pipeval.validate; per-prompt P(non-refusal) Pearson r over harmful prompts; ceiling = split-half
Spearman-Brown reliability as in scripts/pipeval_004_diag.py.
Rule (fixed before data): W_s = max pairwise |rate diff| among {s1, C, D} on source s (per
source); reading (a) iff |A - ref| <= max(3 pt, W_s) on every source, else (b).
F_s = |s1 - B| (identical-seed floor) is reported, not part of the rule.
"""

import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path

import numpy as np

from castor import evalset, pipeval, provenance

FIELD = "wg_vulnerable_incl"
RATE_MIN = 0.03
M = "/work/harrychang08/castor/models/posttrain"
SFT1 = f"{M}/olmo2-1b-ingr3/sft/seed1/model"
SFT_AI2 = ("allenai/OLMo-2-0425-1B-SFT", "0d85a3d037876ce6ac7d4311d994400fc66ac27f")
# key -> (model id, prompt id), expected oi seed, expected parent (local path or HF repo@rev)
ARMS = {
    "s1": (("olmo2-1b-ingr3-dpo-ours-seed1", "010"), 111, SFT1),
    "A": (("olmo2-1b-sftai2-dpo-seed1", "012"), 111, SFT_AI2),
    "B": (("olmo2-1b-ingr3-dpo-rerun-seed1", "012"), 111, SFT1),
    "C": (("olmo2-1b-ingr3-dpo-seed3", "012"), 113, SFT1),
    "D": (("olmo2-1b-ingr3-dpo-seed4", "012"), 114, SFT1),
}
REF = ("olmo2-1b-dpo-ai2", "004")
REF_MODEL = "allenai/OLMo-2-0425-1B-DPO"
SPREAD = ("s1", "C", "D")
PAIRS = [("A", "ref"), ("B", "s1"), ("C", "s1"), ("D", "s1"), ("C", "D"),
         ("s1", "ref"), ("B", "ref"), ("C", "ref"), ("D", "ref")]

_diag = importlib.util.spec_from_file_location(
    "pipeval_004_diag", Path(__file__).parent / "pipeval_004_diag.py")
diag = importlib.util.module_from_spec(_diag)
_diag.loader.exec_module(diag)


def check_lineage(runs: Path = pipeval.RUNS) -> dict:
    """Each arm's generations came from the DPO run it names: oi seed and parent checkpoint,
    read from the generation provenance -> the model's post-training provenance."""
    out = {}
    for k, (model, seed, parent) in ARMS.items():
        gen = json.loads((pipeval.run_dir(model, runs) / "provenance.json").read_text())
        run = json.loads((Path(gen["model"]).parent / "provenance.json").read_text())
        got = (run["base_model"], run["base_revision"])
        want = parent if isinstance(parent, tuple) else (os.path.realpath(parent), None)
        if not isinstance(parent, tuple):
            got = (os.path.realpath(got[0]), got[1])
        assert run["stage"] == "dpo" and run["oi_seed"] == seed, f"{k}: {run['stage']} seed {run['oi_seed']}"
        assert got == want, f"{k} was trained from {got}, expected {want}"
        out[k] = {"model": gen["model"], "oi_seed": seed, "parent": got,
                  "parent_base": (run.get("parent") or {}).get("base_revision"),
                  "train_git_commit": run["git_commit"], "gen_git_commit": gen["git_commit"]}
    gen = json.loads((pipeval.run_dir(REF, runs) / "provenance.json").read_text())
    assert gen["model"] == REF_MODEL, f"ref labels are from {gen['model']}"
    out["ref"] = {"model": gen["model"], "revision": gen.get("revision")}
    return out


def gap_rule(rates: dict[str, dict[str, float]]) -> dict:
    """rates: source -> arm -> refusal rate. Per source W_s, F_s, A - ref and the band."""
    sources, ok = {}, True
    for s, r in rates.items():
        w = max(abs(r[x] - r[y]) for x, y in itertools.combinations(SPREAD, 2))
        band = max(RATE_MIN, w)
        gap = r["A"] - r["ref"]
        within = abs(gap) <= band
        ok &= within
        sources[s] = r | {"W": w, "F": abs(r["s1"] - r["B"]), "A_minus_ref": gap,
                          "s1_minus_ref": r["s1"] - r["ref"], "band": band, "within": within}
    return {"sources": sources, "reading": "a" if ok else "b",
            "n_within": sum(v["within"] for v in sources.values()), "n_sources": len(sources)}


def rates_by_source(pp: dict, panel: list[dict], harmful_only: bool) -> dict:
    ids = {}
    for r in panel:
        if r["is_harmful"] or not harmful_only:
            ids.setdefault(r["source"], []).append(r["id"])
    return {s: {k: pipeval.refusal_rate(v, i) for k, v in pp.items()} for s, i in sorted(ids.items())}


def correlations(vec: dict, rel: dict, n_boot: int, seed: int = 0) -> dict:
    n = len(next(iter(vec.values())))
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, n, n) for _ in range(n_boot)]
    out = {}
    for x, y in PAIRS:
        r = pipeval.pearson(vec[x], vec[y])
        boot = [pipeval.pearson(vec[x][j], vec[y][j]) for j in idx]
        ceil = float(np.sqrt(rel[x] * rel[y]))
        out[f"{x}|{y}"] = {"r": r, "r_ci": np.nanpercentile(boot, [2.5, 97.5]).tolist(),
                           "ceiling": ceil, "r_corrected": r / ceil}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    commit = provenance.git_commit()
    models = {k: v[0] for k, v in ARMS.items()} | {"ref": REF}
    lineage = check_lineage()
    panel = evalset.load_eval()
    instrument = pipeval.check_instrument(
        models, json.loads(evalset.EVAL_MANIFEST.read_text())["eval"]["sha256"])
    harmful = [r["id"] for r in panel if r["is_harmful"]]
    rows = {k: pipeval.load_rows(v) for k, v in models.items()}
    pp = {k: pipeval.per_prompt(pipeval.refusals_from_labels(v, FIELD)) for k, v in rows.items()}
    vec = {k: pipeval.vector(v, harmful) for k, v in pp.items()}
    rel = {k: diag.split_half(rows[k], harmful)["spearman_brown"] for k in models}
    res = {"rule": gap_rule(rates_by_source(pp, panel, harmful_only=False)),
           "harmful_only": gap_rule(rates_by_source(pp, panel, harmful_only=True)),
           "correlations": correlations(vec, rel, a.n_boot), "reliability": rel,
           "unparsed": {k: sum(x["n_unparsed"] for x in v.values()) for k, v in pp.items()},
           "field": FIELD, "n_harmful": len(harmful), "lineage": lineage,
           "models": {k: "/".join(v) for k, v in models.items()}, "instrument": instrument,
           "git_commit": commit, "command": provenance.command()}
    res["reading"] = res["rule"]["reading"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"reading": res["reading"], "n_within": res["rule"]["n_within"],
                      "r": {k: round(v["r"], 3) for k, v in res["correlations"].items()}}))


if __name__ == "__main__":
    main()
