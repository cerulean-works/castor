"""Refusal-direction extraction inputs, fixed before any selection score exists (CPU only).

    uv run --frozen python scripts/prep_007.py [--arditi DIR] [--what all|prompts|models|tokens]

Writes (committed):
  data/directions/extraction_v1.jsonl   harmful / harmless training and validation prompts
  data/directions/MANIFEST.json         sources, Arditi commit, decontamination, draw, sha256
  data/manifests/mech_007.json          the 12 main-recipe twins (model manifest, 006 paths)
  data/directions/refusal_tokens_v1.json  the one refusal-token set for every twin

Extraction prompts: Arditi et al.'s (2024) released data at andyrdt/refusal_direction@9d852fa.
Harmful = the processed files of the three training sources (advbench, malicious_instruct, tdc2023:
720 prompts; Arditi's own split subsamples <= 128 per source, which leaves too few after
decontamination). Harmless training = `splits/harmless_train.json`, harmless validation =
`splits/harmless_val.json` (Alpaca, no input field). Exact duplicate texts are dropped (first kept).
A prompt is removed if it matches the panel (`data/panel/panel_v1.jsonl`, all 6,290) under the
decontamination rule (castor.decontam: > 50 % of a panel prompt's tokens covered by 8-grams of the
extraction prompt, or an exact normalized match) or the reverse (> 50 % of the extraction prompt's
tokens inside panel 8-grams); the reverse catches short instructions embedded in long jailbreak
prompts. Draw: numpy default_rng(2027); harmful pool permuted, first 256 = train, next 32 = val;
harmless train = 256 drawn from its pool, harmless val = 32 from its pool, in that order from the
same generator.

Refusal tokens: the first token (twin tokenizer) of every judge_v1 REFUSAL-labelled greedy
response to a plain-harmful panel prompt, pooled over the 12 twins; the smallest set covering
>= 80 % (castor.directions.refusal_token_set).
"""

import argparse
import hashlib
import json
from pathlib import Path

from castor import decontam as D
from castor import directions, evalset, provenance, sanityb
from castor.judge.common import read_jsonl

ROOT = Path(__file__).resolve().parent.parent
ARDITI = Path("/work/harrychang08/castor/third_party/src/refusal_direction")
ARDITI_REPO = "https://github.com/andyrdt/refusal_direction"
ARDITI_COMMIT = "9d852fae1a9121c78b29142de733cb1340770cc3"
HARMFUL_FILES = ("advbench", "malicious_instruct", "tdc2023")
N_TRAIN, N_VAL, SEED = 256, 32, 2027
OUT_DIR = ROOT / "data" / "directions"
MODELS = ROOT / "data" / "manifests" / "mech_007.json"
SOURCE_PROMPT = "006"


def matches(index: D.PanelIndex, texts: list[str]) -> list[dict]:
    """Per text: forward hit, reverse hit, exact hit (empty dict = clean)."""
    docs, why = [], []
    for i, t in enumerate(texts):
        g, ex = index.hits(t)
        docs.append((i, g))
        toks = D.tokenize(t)
        cov = set()
        for pos, gram in D.ngrams(toks):
            if gram in g:
                cov.update(range(pos, pos + D.N))
        w = {}
        if ex:
            w["exact"] = len(ex)
        if toks and len(cov) / len(toks) > D.THRESHOLD:
            w["reverse"] = round(len(cov) / len(toks), 4)
        why.append(w)
    for s in D.score(index, docs):
        if s["score"] > D.THRESHOLD:
            why[s["doc"]]["forward"] = max(why[s["doc"]].get("forward", 0), round(s["score"], 4))
    return why


def load(name: str) -> list[dict]:
    return json.loads((ARDITI / "dataset" / f"{name}.json").read_text())


def build_prompts() -> None:
    import subprocess

    import numpy as np

    head = subprocess.run(["git", "-C", str(ARDITI), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert head == ARDITI_COMMIT, f"Arditi checkout at {head}, expected {ARDITI_COMMIT}"
    panel = evalset.load_panel()
    index = D.PanelIndex([r["prompt"] for r in panel])

    pools, report = {}, {}
    harmful = [(f, i, x["instruction"]) for f in HARMFUL_FILES
               for i, x in enumerate(load(f"processed/{f}"))]
    for name, rows in (("harmful", harmful),
                       ("harmless_train", [("alpaca/harmless_train", i, x["instruction"]) for i, x
                                           in enumerate(load("splits/harmless_train"))]),
                       ("harmless_val", [("alpaca/harmless_val", i, x["instruction"]) for i, x
                                         in enumerate(load("splits/harmless_val"))])):
        seen, dedup = set(), []
        for r in rows:
            if r[2] not in seen:
                seen.add(r[2])
                dedup.append(r)
        why = matches(index, [r[2] for r in dedup])
        clean = [r for r, w in zip(dedup, why) if not w]
        pools[name] = clean
        report[name] = {"n_raw": len(rows), "n_dedup": len(dedup),
                        "n_removed": len(dedup) - len(clean), "n_pool": len(clean),
                        "removed_forward": sum("forward" in w for w in why),
                        "removed_reverse": sum("reverse" in w for w in why),
                        "removed_exact": sum("exact" in w for w in why)}
        if name == "harmful":
            report[name]["pool_by_source"] = {f: sum(r[0] == f for r in clean)
                                              for f in HARMFUL_FILES}

    rng = np.random.default_rng(SEED)
    h = [pools["harmful"][i] for i in rng.permutation(len(pools["harmful"]))]
    assert len(h) >= N_TRAIN + N_VAL, f"harmful pool {len(h)} < {N_TRAIN + N_VAL}"
    draw = {("harmful", "train"): h[:N_TRAIN], ("harmful", "val"): h[N_TRAIN:N_TRAIN + N_VAL]}
    ht = pools["harmless_train"]
    draw[("harmless", "train")] = [ht[i] for i in rng.choice(len(ht), N_TRAIN, replace=False)]
    hv = pools["harmless_val"]
    draw[("harmless", "val")] = [hv[i] for i in rng.choice(len(hv), N_VAL, replace=False)]

    lines = []
    for (cls, split), rows in draw.items():
        for j, (src, idx, text) in enumerate(rows):
            lines.append(json.dumps({"id": f"{cls}-{split}-{j:03d}", "class": cls,
                                     "split": split, "source": src, "source_index": idx,
                                     "prompt": text}, ensure_ascii=False))
    data = ("\n".join(lines) + "\n").encode()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "extraction_v1.jsonl").write_bytes(data)
    counts = {f"{c}_{s}": len(r) for (c, s), r in draw.items()}
    by_src = {}
    for (c, s), rows in draw.items():
        for src, _, _ in rows:
            by_src.setdefault(f"{c}_{s}", {}).setdefault(src, 0)
            by_src[f"{c}_{s}"][src] += 1
    manifest = {"file": "data/directions/extraction_v1.jsonl",
                "sha256": hashlib.sha256(data).hexdigest(), "counts": counts,
                "by_source": by_src, "pools": report,
                "source": {"repo": ARDITI_REPO, "commit": ARDITI_COMMIT,
                           "harmful": [f"dataset/processed/{f}.json" for f in HARMFUL_FILES],
                           "harmless_train": "dataset/splits/harmless_train.json",
                           "harmless_val": "dataset/splits/harmless_val.json",
                           "licenses": {"advbench": "MIT", "malicious_instruct": "see source",
                                        "tdc2023": "MIT", "alpaca": "CC BY-NC 4.0"}},
                "decontamination": {"panel": "data/panel/panel_v1.jsonl",
                                    "panel_sha256": evalset.PANEL_SHA256,
                                    "rule": "D007 forward (castor.decontam, > 0.5) or exact, "
                                            "or reverse coverage > 0.5"},
                "draw": {"rng": f"numpy.random.default_rng({SEED})",
                         "order": "harmful permutation (train = first 256, val = next 32); "
                                  "harmless_train choice(256); harmless_val choice(32)"},
                "git_commit": provenance.git_commit(allow_dirty=True),
                "command": provenance.command()}
    (OUT_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("counts", "pools", "sha256")}, indent=1))


def build_models() -> None:
    models = []
    for c in sanityb.cells():
        run = sanityb.RUNS / c.model_id / SOURCE_PROMPT
        prov = json.loads((run / "provenance.json").read_text())
        assert prov["model_id"] == c.model_id and prov["limit"] is None
        base = sanityb.check_base(c)
        models.append({"model_id": c.model_id, "checkpoint": prov["model"],
                       "checkpoint_resolved": prov["model_resolved"],
                       "base": c.base_ref, "base_converted": str(c.base_dir),
                       "recipe": sanityb.RECIPE, "pretrain_seed": c.pretrain_seed,
                       "posttrain_seed": str(c.k),
                       "generations": str(run), "labels": str(run / "judge_v1" / "labels.jsonl"),
                       "base_provenance_source": base["source"]})
    MODELS.write_text(json.dumps({"prompt": "007", "source_prompt": SOURCE_PROMPT,
                                  "models": models}, indent=1) + "\n")
    print(f"{len(models)} models -> {MODELS}")


def build_tokens() -> None:
    from transformers import AutoTokenizer

    models = json.loads(MODELS.read_text())["models"]
    panel = {r["id"]: r for r in evalset.load_eval()}
    plain = {i for i, r in panel.items() if sanityb.SUBSETS["plain_harmful"](r)}
    tok_sha, tok = set(), None
    firsts, per_model = [], {}
    for m in models:
        tdir = Path(m["checkpoint"])
        tok_sha.add(hashlib.sha256((tdir / "tokenizer.json").read_bytes()).hexdigest())
        if tok is None:
            tok = AutoTokenizer.from_pretrained(tdir)
        refused = {r["prompt_id"] for r in read_jsonl(Path(m["labels"]))
                   if r["sample"] == -1 and r["prompt_id"] in plain and r["wg_label"] == "REFUSAL"}
        n = 0
        for r in read_jsonl(Path(m["generations"]) / "greedy.jsonl"):
            if r["prompt_id"] in refused:
                ids = tok.encode(r["response"], add_special_tokens=False)
                if ids:
                    firsts.append(ids[0])
                    n += 1
        per_model[m["model_id"]] = n
    assert len(tok_sha) == 1, "twins do not share one tokenizer"
    res = directions.refusal_token_set(firsts)
    res |= {"decoded": [tok.decode([t]) for t in res["tokens"]],
            "decoded_top": {tok.decode([int(t)]): k for t, k in res["counts"].items()},
            "per_model_n": per_model, "tokenizer_sha256": tok_sha.pop(),
            "rule": "first token of judge_v1 REFUSAL greedy responses to plain-harmful prompts, "
                    "12 twins pooled; smallest set covering >= 0.8",
            "git_commit": provenance.git_commit(allow_dirty=True),
            "command": provenance.command()}
    (OUT_DIR / "refusal_tokens_v1.json").write_text(json.dumps(res, indent=1) + "\n")
    print(res["tokens"], res["decoded"], res["coverage"], res["n_responses"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all", choices=("all", "prompts", "models", "tokens"))
    a = ap.parse_args()
    if a.what in ("all", "prompts"):
        build_prompts()
    if a.what in ("all", "models"):
        build_models()
    if a.what in ("all", "tokens"):
        build_tokens()


if __name__ == "__main__":
    main()
