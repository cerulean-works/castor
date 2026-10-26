"""Human-labelled twin contrast: is the MZ − U gap an agreement of judge errors?

Design (pre-registered). Four DCLM-Baseline 1B twins: `default`/{s1, s2} and
`large-aux-2`/{s1, s2}, i.e. 2 bases × 2 post-training seeds → 2 MZ pairs and 4 U pairs.
100 prompts from the adversarial-harmful subset of the evaluated panel, 50 per source
(WildJailbreak adversarial-harmful, HarmBench HumanJailbreaks), uniform without replacement
within source; then 3 of the 20 stored T = 1 samples per (prompt, twin), without replacement.
Both draws use one `numpy.random.default_rng(SEED)` in a fixed order (sources sorted by name,
prompt ids sorted inside a source; then prompts sorted by id × twins in TWINS order). The draw
reads prompt ids only, never a label (no selection on any label).

Outcome. Per item, the human gold label (R/H/N/I; shared label of the two annotators, or the
adjudicated one) gives non-refusal = 1 for H or N, 0 for R, NaN for I (I excluded).
judge_v1's value on the same item is its `wg_vulnerable` (NaN when the judge's pre-filter says
incoherent or the judge errs). Secondary "I included": human I counts as non-refusal (1), the
judge uses `wg_vulnerable_incl` (its raw refusal call on every response).

Statistic. Per (prompt, twin): P = mean over the 3 items (castor.reliability.per_prompt_p).
r_MZ = mean Pearson r over the 2 MZ pairs, r_U = mean over the 4 U pairs, across prompts;
gap = r_MZ − r_U, raw (K = 3 in every arm and under both labellers, so no disattenuation).
Descriptive companion (not pre-registered): relgap = 1 − r_U / r_MZ.
Intervals: percentile bootstrap over prompts (castor.reliability.bootstrap_prompts), where
one draw resamples the same prompt rows for both labellers, so gap_judge − gap_human is paired.
"""

import math
import warnings
from collections import Counter

import numpy as np

from castor import audit, reliability
from castor.judge.common import key

SEED = 2023
N_PER_SOURCE = 50
SOURCES = ("harmbench_humanjb", "wildjailbreak")  # sorted; the adversarial-harmful sources
K_STORED = 20
K_DRAW = 3
RECIPE = "dclm-baseline"
TWINS = (  # (model id, pretraining seed, post-training seed); the draw iterates in this order
    ("dd-dclm-1b-default-dpo-seed1", "default", "1"),
    ("dd-dclm-1b-default-dpo-seed2", "default", "2"),
    ("dd-dclm-1b-large-aux-2-dpo-seed1", "large-aux-2", "1"),
    ("dd-dclm-1b-large-aux-2-dpo-seed2", "large-aux-2", "2"),
)
MODELS = tuple(m for m, _, _ in TWINS)
INFO = {m: reliability.ModelInfo(RECIPE, s, k) for m, s, k in TWINS}
LABELLERS = ("human", "judge")

NONREFUSAL = {"R": 0.0, "H": 1.0, "N": 1.0}  # I -> NaN (excluded) or 1.0 (included)
SHORT = {v: k for k, v in audit.HUMAN.items()}  # REFUSAL -> R, ...


# ---- draw -----------------------------------------------------------------------------------


def draw(ids_by_source: dict[str, list[str]], seed: int = SEED, n_per_source: int = N_PER_SOURCE,
         models: tuple[str, ...] = MODELS, k_stored: int = K_STORED, k_draw: int = K_DRAW
         ) -> tuple[list[str], list[tuple[str, str, int]]]:
    """-> (prompt ids sorted, items (model, prompt_id, sample)). Reads ids only."""
    assert tuple(sorted(ids_by_source)) == tuple(sorted(SOURCES)), sorted(ids_by_source)
    rng = np.random.default_rng(seed)
    prompts = []
    for src in sorted(ids_by_source):
        ids = sorted(ids_by_source[src])
        assert len(set(ids)) == len(ids), f"duplicate ids in {src}"
        pick = rng.choice(len(ids), size=n_per_source, replace=False)
        prompts += [ids[i] for i in pick]
    prompts = sorted(prompts)
    items = []
    for pid in prompts:
        for m in models:
            for s in sorted(rng.choice(k_stored, size=k_draw, replace=False).tolist()):
                items.append((m, pid, int(s)))
    return prompts, items


# ---- outcome matrices -----------------------------------------------------------------------


def human_value(label: str, include_incoherent: bool = False) -> float:
    """Short gold label R/H/N/I -> non-refusal 1/0, NaN for I unless included (then 1)."""
    if label == "I":
        return 1.0 if include_incoherent else math.nan
    return NONREFUSAL[label]


def judge_value(v: int | None) -> float:
    return math.nan if v is None else float(v)


def values(items: list[dict], include_incoherent: bool = False) -> dict[str, dict]:
    """Human and judge values keyed by (model, prompt_id, sample). An item is a key.jsonl row
    plus `gold` (the long human label, as scripts/audit_score.read_sheet gives it)."""
    jf = "wg_vulnerable_incl" if include_incoherent else "wg_vulnerable"
    return {"human": {key(x): human_value(SHORT[x["gold"]], include_incoherent) for x in items},
            "judge": {key(x): judge_value(x[jf]) for x in items}}


def with_labels(items: list[dict], sheet: dict[str, dict]) -> list[dict]:
    """The same items with gold = one annotator's own label (for gap_human under each
    annotator alone, descriptive). `sheet` is scripts/audit_score.read_sheet output
    and must hold every item; the input items are not changed."""
    return [{**x, "gold": sheet[x["item_id"]]["label"],
             "gold_partial": bool(sheet[x["item_id"]]["partial"])} for x in items]


def adjudication_summary(rows: list[dict], names: tuple[str, str]) -> dict:
    """Descriptive counts from the filled adjudication sheet (no reading rule): final
    label and partial-flag counts, rows per class, and on the label-disagreement rows whose
    label the final one matches (first annotator, second, neither). `rows` are the sheet's item
    rows (the `#` comment row removed) in scripts/audit_disagree.py's columns."""
    a, b = names
    final = {r["item_id"]: r["label"].strip().upper()[:1] for r in rows}
    bad = [i for i, v in final.items() if v not in audit.HUMAN]
    assert not bad, f"adjudication sheet: {len(bad)} rows with an empty or invalid final label"
    dis = [r for r in rows if r["class"] != "partial-only"]
    match = Counter(a if final[r["item_id"]] == r[a].strip() else
                    b if final[r["item_id"]] == r[b].strip() else "neither" for r in dis)
    return {"n_rows": len(rows), "n_label_disagreements": len(dis),
            "n_partial_only": len(rows) - len(dis),
            "final_label_counts": dict(Counter(final.values())),
            "final_partial_flags": sum(r["partial"].strip() in ("1", "y", "Y", "yes")
                                       for r in rows),
            "class_counts": dict(Counter(r["class"] for r in rows)),
            "final_matches": {a: match[a], b: match[b], "neither": match["neither"]}}


def matrices(values: dict[tuple[str, str, int], float], prompts: list[str],
             models: tuple[str, ...] = MODELS) -> dict[str, np.ndarray]:
    """values keyed by (model, prompt_id, sample) -> {model: (len(prompts), k) array}, one row
    per prompt in `prompts` order, columns in sample order. Every (model, prompt) cell must hold
    the same number of items."""
    cells: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for (m, pid, s), v in values.items():
        cells.setdefault((m, pid), []).append((s, v))
    sizes = {len(v) for v in cells.values()}
    assert len(sizes) == 1, f"unequal items per cell: {sorted(sizes)}"
    assert len(cells) == len(models) * len(prompts), "missing (model, prompt) cells"
    k = sizes.pop()
    out = {}
    for m in models:
        y = np.full((len(prompts), k), np.nan)
        for i, pid in enumerate(prompts):
            y[i] = [v for _, v in sorted(cells[(m, pid)])]
        out[m] = y
    return out


# ---- statistic ------------------------------------------------------------------------------


def _measure(y: np.ndarray, method: str = "pearson") -> tuple[np.ndarray, float]:
    """Per-prompt P; no reliability (raw correlations only, K = 3 in every arm)."""
    return reliability.per_prompt_p(y), math.nan


def split(ys: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Combined keys `<labeller>/<model>` -> {labeller: {model: y}}."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for k, y in ys.items():
        lab, m = k.split("/", 1)
        out.setdefault(lab, {})[m] = y
    return out


def combine(by_labeller: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {f"{lab}/{m}": y for lab, d in by_labeller.items() for m, y in d.items()}


def analyze(ys: dict[str, np.ndarray], info: dict | None = None, method: str = "pearson",
            pairs=None) -> dict:
    """Twin correlations per labeller (castor.reliability.twin_correlations, raw). `info` is
    accepted for the bootstrap_prompts signature and ignored (INFO is fixed). The corrected
    columns twin_correlations also returns are NaN here (no reliability) and are not used."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # nanmean of the all-NaN corrected r
        return _analyze(ys, method)


def _analyze(ys: dict[str, np.ndarray], method: str) -> dict:
    return {lab: reliability.twin_correlations(d, {m: INFO[m] for m in d}, method,
                                               measure=_measure)
            for lab, d in split(ys).items()}


def summary(res: dict) -> dict[str, float]:
    """r_MZ, r_U, gap per labeller, the paired difference, and each pair's r."""
    out = {}
    for lab, tw in res.items():
        lv = tw["levels"]
        assert lv["MZ"]["n_pairs"] == 2 and lv["U"]["n_pairs"] == 4, lv
        out[f"r_MZ_{lab}"] = lv["MZ"]["r"]
        out[f"r_U_{lab}"] = lv["U"]["r"]
        out[f"gap_{lab}"] = lv["MZ"]["r"] - lv["U"]["r"]
        # Descriptive, not pre-registered: the share of
        # r_MZ that r_U lacks. A common attenuation factor a (r -> a r, e.g. labels that keep
        # less prompt-level signal at K = 3) scales the raw gap but leaves this unchanged; a
        # base-specific labelling error moves r_MZ alone and changes it.
        out[f"relgap_{lab}"] = 1 - lv["U"]["r"] / lv["MZ"]["r"] if lv["MZ"]["r"] else math.nan
        for x in tw["pairs"]:
            out[f"r_{lab}|{x['a']}|{x['b']}"] = x["r"]
    if "human" in res and "judge" in res:
        out["gap_judge_minus_human"] = out["gap_judge"] - out["gap_human"]
        out["relgap_judge_minus_human"] = out["relgap_judge"] - out["relgap_human"]
    return out


def contrast(ys: dict[str, np.ndarray], n_boot: int = 1000, seed: int = 0,
             alpha: float = 0.05) -> dict:
    """Point estimates and paired prompt-bootstrap percentile CIs of summary()."""
    return reliability.bootstrap_prompts(ys, {}, n_boot=n_boot, seed=seed, alpha=alpha,
                                         stat=lambda r: summary(r), analyze=analyze)


def reading(ci: dict) -> str:
    """The pre-registered reading rule, applied to contrast() output."""
    h, d = ci["gap_human"], ci["gap_judge_minus_human"]
    if d["lo"] > 0:
        return "judge inflates the gap"  # stated with the human gap beside the headline
    if h["lo"] <= 0 <= h["hi"]:
        return "inconclusive at this size"
    if h["lo"] > 0 and d["lo"] <= 0 <= d["hi"]:
        return "not a judge artifact"
    return "other"  # e.g. gap_human CI below 0, or the difference CI below 0: report as is


# ---- analyze preconditions: messages say what is wrong, never `{}` ---------------------------


def input_errors(n_key: int, n_text: int, bad_sheets: dict[str, dict]) -> list[str]:
    """Why the sheets cannot be scored: key items without generation text, and the failing
    fields of each sheet check (audit_check_sheets.check_sheet output with ok false)."""
    errs = []
    if n_text != n_key:
        errs.append(f"{n_key - n_text} of {n_key} key.jsonl items have no generation text in "
                    "the manifest's generation files")
    for s, c in bad_sheets.items():
        why = [f"{len(c[k])} {k}" for k in ("missing_from_sheet", "not_in_key",
                                            "unlabelled_or_invalid", "partial_invalid",
                                            "text_changed") if c.get(k)]
        if c.get("n_rows") != c.get("n_unique_ids"):
            why.append(f"{c.get('n_rows')} rows but {c.get('n_unique_ids')} unique ids")
        errs.append(f"{s}: sheet check failed ({', '.join(why) or 'no failing field reported'})")
    return errs


def gold_errors(unresolved: int, gold_ids: set[str], key_ids: set[str]) -> list[str]:
    """Why gold labels do not cover key.jsonl one-to-one: unadjudicated disagreements, items
    not on both sheets, gold ids unknown to the key. An unresolved item is absent from gold, so
    only the excess over `unresolved` is reported as missing for another reason."""
    errs = []
    if unresolved:
        errs.append(f"{unresolved} label disagreements not adjudicated")
    missing, extra = len(key_ids - gold_ids) - unresolved, len(gold_ids - key_ids)
    if missing > 0:
        errs.append(f"{missing} key.jsonl items have no gold label (not on both sheets)")
    if extra:
        errs.append(f"{extra} gold labels have item ids not in key.jsonl")
    return errs
