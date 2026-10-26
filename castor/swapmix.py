"""The 080 swap mixture: the Tulu 3 SFT mix with its safety slice replaced (D125-D128).

The three safety sources of `tulu-3-sft-olmo-2-mixture-0225` (WildGuardMix, WildJailbreak,
CoCoNot; 110,800 rows) are dropped and 110,800 replacement examples are added from five public
sets in a fixed order: Nemotron-SFT-Safety-v2 (English rows), Llama-Nemotron post-training
`safety` (one "thinking off" copy per prompt), PKU-SafeRLHF (Ji et al. 2024, 2406.15513;
responses labelled safe), BeaverTails (Ji et al. 2023, 2307.04657; `is_safe`) and HH-RLHF
harmless-base chosen (Bai et al. 2022, 2204.05862). Each set is put in a fixed shuffle (seed
2026) and exact duplicate (user turns, response) examples are dropped before labelling.

Selection reads WildGuard's raw labels (judge_v1's primary) of every candidate and every slice
row, and runs two passes over the sets in order (D126 (d), D127):
- pass 1 takes harmful-prompt refusals that are not harmful responses, up to the slice's own
  count under the same labels and eligibility rules (the cap);
- pass 2 fills to 110,800 with benign-prompt examples that are neither refusals nor harmful
  responses.
A label that did not parse makes a row ineligible. Each pass has two tiers (D128). Tier 1 skips a
candidate if its normalized prompt (decontam's normalized user turns) was already taken or occurs
in the kept Tulu rows, or if it fails the panel decontamination. Tier 2 runs only if tier 1 falls
short of the pass's target: it adds second-and-later responses to prompts tier 1 took in the same
pass, by rounds, each passing the pass's predicate, so the replacement repeats prompts as the removed
slice does (26 % of its rows are repeats). A tier-2 row needs no decontamination of its own: the
check reads only the user turns, which norm_prompt determines, and tier 1 passed that prompt. A
label computed on a response cut to fit the judge (`truncated`) makes a row ineligible.

SFT trains on every assistant turn, so each earlier assistant turn of a multi-turn example (HH-RLHF)
is judged too, as the pair (the user turn before it, the turn), and the example is eligible only if
every such turn is non-empty, labelled, uncut, not a harmful response (`earlier_clean`).

Decontamination reuses castor.decontam (D007's 8-gram rule, threshold 0.5, exact match), on
user turns, in both directions:
- panel -> candidate: a panel prompt whose tokens are more than half covered by 8-grams of one
  candidate's user turn (the D007 direction, here removing the candidate, since the panel is
  frozen);
- candidate -> panel: a candidate user turn whose tokens are more than half covered by 8-grams
  of one panel prompt;
- either side's normalized full string equal to the other's.
"""

import hashlib
import json
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Iterator

from castor import decontam

SAFETY_SOURCES = (
    "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k",
    "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k",
    "ai2-adapt-dev/coconot_converted",
)
SLICE_N = 110_800
SEED = 2026
SWAP_SOURCE = {"nemotron_v2": "castor-swap/Nemotron-SFT-Safety-v2",
               "llama_nemotron": "castor-swap/Llama-Nemotron-safety",
               "pku": "castor-swap/PKU-SafeRLHF", "beavertails": "castor-swap/BeaverTails",
               "hh": "castor-swap/hh-rlhf-harmless-base"}
ORDER = ("nemotron_v2", "llama_nemotron", "pku", "beavertails", "hh")
LABELS = ("prompt_harmful", "refusal", "response_harmful")

_HH_TURN = re.compile(r"\n\n(Human|Assistant): ")
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def is_true(v) -> bool:
    """PKU-SafeRLHF stores its safety labels as the strings 'True'/'False'."""
    return v is True or v == "True"


def pair(prompt: str, response: str) -> list[dict]:
    return [{"content": prompt, "role": "user"}, {"content": response, "role": "assistant"}]


def nemotron_v2_examples(rows: Iterable[tuple[int, dict]]) -> Iterator[tuple[str, list[dict], dict]]:
    """English rows (no `translation_languages`) of Nemotron-SFT-Safety-v2 -> (id, messages, own
    labels); turns keep role and content only (`reasoning_content` dropped)."""
    for line, r in rows:
        if r.get("translation_languages"):
            continue
        msgs = [{"content": m["content"], "role": m["role"]} for m in r["messages"]
                if m["role"] != "system"]
        if not msgs or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant" \
                or not msgs[-1]["content"].strip():
            continue
        yield (f"nemotron_v2/train/{line}", msgs,
               {"prompt_label": r["metadata"].get("prompt_label"),
                "response_policy": r["response_policy"]})


def llama_nemotron_examples(rows: Iterable[tuple[int, dict]]) -> Iterator[tuple[str, list[dict]]]:
    """The first "detailed thinking off" row per distinct input of Llama-Nemotron `safety`, with
    any <think> block stripped and no system prompt."""
    seen: set = set()
    for line, r in rows:
        if r["system_prompt"] != "detailed thinking off":
            continue
        msgs = [{"content": m["content"], "role": m["role"]} for m in r["input"]]
        if not msgs or msgs[0]["role"] != "user" or msgs[-1]["role"] != "user":
            continue
        k = json.dumps([(m["role"], m["content"].strip()) for m in msgs], ensure_ascii=False)
        out = _THINK.sub("", r["output"]).strip()
        if k in seen or not out:
            continue
        seen.add(k)
        yield f"llama_nemotron/safety/{line}", msgs + [{"content": out, "role": "assistant"}]


def pku_examples(rows: Iterable[tuple[str, int, dict]]) -> Iterator[tuple[str, list[dict]]]:
    """(file tag, line, row) -> (id, messages) for every response labelled safe."""
    for tag, line, r in rows:
        for i in (0, 1):
            if is_true(r[f"is_response_{i}_safe"]):
                yield f"pku/{tag}/{line}/{i}", pair(r["prompt"], r[f"response_{i}"])


def beavertails_examples(rows: Iterable[tuple[int, dict]]) -> Iterator[tuple[str, list[dict]]]:
    for line, r in rows:
        if is_true(r["is_safe"]):
            yield f"beavertails/330k_train/{line}", pair(r["prompt"], r["response"])


def parse_hh(text: str) -> list[dict] | None:
    """An HH-RLHF transcript ('\\n\\nHuman: ... \\n\\nAssistant: ...') as messages; None unless
    the turns alternate from a user turn to a final, non-empty assistant turn."""
    parts = _HH_TURN.split(text)
    if parts[0].strip():
        return None
    msgs = []
    for who, content in zip(parts[1::2], parts[2::2], strict=True):
        role = "user" if who == "Human" else "assistant"
        if msgs and msgs[-1]["role"] == role:
            return None
        msgs.append({"content": content.strip(), "role": role})
    if not msgs or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant" \
            or not msgs[-1]["content"]:
        return None
    return msgs


def hh_examples(rows: Iterable[tuple[int, dict]]) -> Iterator[tuple[str, list[dict]]]:
    for line, r in rows:
        msgs = parse_hh(r["chosen"])
        if msgs is not None:
            yield f"hh/harmless-base/{line}", msgs


def user_turns(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "user"]


def last_turns(messages: list[dict]) -> tuple[str, str]:
    """(last user turn, final assistant turn), the pair the judge sees."""
    return user_turns(messages)[-1], messages[-1]["content"]


def earlier_turns(messages: list[dict]) -> list[tuple[int, str, str]]:
    """(message index, the user turn before it, content) of every assistant turn but the last."""
    out = []
    for k, m in enumerate(messages[:-1]):
        if m["role"] == "assistant":
            prev = [x["content"] for x in messages[:k] if x["role"] == "user"]
            out.append((k, prev[-1], m["content"]))
    return out


def turn_id(rec_id: str, k: int) -> str:
    return f"{rec_id}#{k}"


def example_key(messages: list[dict]) -> bytes:
    """Digest of the (role, stripped content) turns: exact duplicates share it."""
    turns = [(m["role"], m["content"].strip()) for m in messages]
    return hashlib.sha1(json.dumps(turns, ensure_ascii=False).encode()).digest()


def prompt_key(messages: list[dict]) -> tuple[str, ...]:
    return tuple(t.strip() for t in user_turns(messages))


def shuffled(xs: list, seed: int = SEED) -> list:
    out = list(xs)
    random.Random(seed).shuffle(out)
    return out


def covered_by_one(index: decontam.PanelIndex, toks: list[str]) -> dict[int, float]:
    """Fraction of `toks` inside 8-grams that also occur in indexed document pid, per pid."""
    cover: dict[int, set[int]] = defaultdict(set)
    for i, g in decontam.ngrams(toks, index.n):
        for pid in {p for p, _ in index.grams.get(g, ())}:
            cover[pid].update(range(i, i + index.n))
    return {pid: len(c) / len(toks) for pid, c in cover.items()}


def covering(index: decontam.PanelIndex, toks: list[str]) -> dict[int, float]:
    """Fraction of indexed document pid's tokens covered by 8-grams of `toks`, per pid
    (decontam.score for one training document)."""
    cover: dict[int, set[int]] = defaultdict(set)
    for _, g in decontam.ngrams(toks, index.n):
        for pid, pos in index.grams.get(g, ()):
            cover[pid].update(range(pos, pos + index.n))
    return {pid: len(c) / len(index.tokens[pid]) for pid, c in cover.items()}


def contamination(index: decontam.PanelIndex, turns: list[str]) -> dict:
    """Why a candidate fails the panel decontamination ({} if clean). index = the panel."""
    why: dict = {}
    for t in turns:
        toks = decontam.tokenize(t)
        a = {p: s for p, s in covering(index, toks).items() if s > decontam.THRESHOLD}
        b = {p: s for p, s in covered_by_one(index, toks).items() if s > decontam.THRESHOLD}
        ex = index.exact.get(decontam.exact_key(toks), [])
        if a:
            why.setdefault("panel_in_candidate", {}).update(a)
        if b:
            why.setdefault("candidate_in_panel", {}).update(b)
        if ex:
            why.setdefault("exact", []).extend(ex)
    return why


def norm_prompt(messages: list[dict]) -> tuple[str, ...]:
    """The prompt under decontam's normalization (lower case, punctuation split away)."""
    return tuple(decontam.exact_key(decontam.tokenize(t)) for t in user_turns(messages))


def dedupe_pool(pools: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict]:
    """Drop exact duplicate examples (example_key) within and across the sets, walking them in
    ORDER; a set's later copy is dropped. Returns (pools, duplicates per set)."""
    seen: set = set()
    out, dups = {}, {}
    for name in ORDER:
        keep = []
        for rec in pools.get(name, []):
            k = example_key(rec["messages"])
            if k not in seen:
                seen.add(k)
                keep.append(rec)
        out[name] = keep
        dups[name] = len(pools.get(name, [])) - len(keep)
    return out, dups


def judge_input(rec: dict, model: str) -> dict:
    """One example in the judge's generations format."""
    p, resp = last_turns(rec["messages"])
    return {"model": model, "prompt_id": rec["id"], "sample": 0, "source": rec["source"],
            "prompt": p, "response": resp}


def turn_inputs(rec: dict, model: str) -> list[dict]:
    """The earlier assistant turns of one example in the judge's generations format."""
    return [{"model": model, "prompt_id": turn_id(rec["id"], k), "sample": 0,
             "source": rec["source"], "prompt": p, "response": resp}
            for k, p, resp in earlier_turns(rec["messages"])]


def fold_earlier(labels: dict[str, dict], recs: Iterable[dict]) -> dict:
    """Set labels[id]["earlier_clean"] on every multi-turn record: each earlier assistant turn
    non-empty, labelled, uncut and response_harmful False. Returns counts per source."""
    out: dict[str, dict] = defaultdict(lambda: {"rows": 0, "turns": 0, "turns_empty": 0,
                                                "turns_truncated": 0, "rows_failing": 0})
    for rec in recs:
        turns = earlier_turns(rec["messages"])
        if not turns:
            continue
        ids = [turn_id(rec["id"], k) for k, _, _ in turns]
        missing = [t for t in ids if t not in labels]
        if missing:
            raise ValueError(f"{len(missing)} earlier turns have no WildGuard label, "
                             f"e.g. {missing[0]}")
        labs = [labels[t] for t in ids]
        empty = sum(not content.strip() for _, _, content in turns)
        cut = sum(bool(x.get("truncated")) for x in labs)
        clean = not empty and not cut and all(x["response_harmful"] is False for x in labs)
        labels[rec["id"]]["earlier_clean"] = clean
        c = out[rec["source"]]
        c["rows"] += 1
        c["turns"] += len(turns)
        c["turns_empty"] += empty
        c["turns_truncated"] += cut
        c["rows_failing"] += not clean
    return {s: out[s] for s in sorted(out)}


def harmful_refusal(lab: dict) -> bool:
    """Pass 1: WildGuard calls the prompt harmful, the target a refusal, not a harmful response."""
    return lab["prompt_harmful"] is True and lab["refusal"] is True \
        and lab["response_harmful"] is False


def benign_compliance(lab: dict) -> bool:
    """Pass 2: a benign prompt, a target that is neither a refusal nor a harmful response."""
    return lab["prompt_harmful"] is False and lab["refusal"] is False \
        and lab["response_harmful"] is False


PASSES = (("pass1_harmful_refusal", harmful_refusal), ("pass2_benign_compliance", benign_compliance))


def usable(lab: dict, eligible) -> bool:
    """The pass's predicate on a label computed on the whole response, with every earlier
    assistant turn clean (fold_earlier)."""
    return not lab.get("truncated") and lab.get("earlier_clean", True) and eligible(lab)


def _removed(r: dict, rec: dict, why: dict) -> None:
    r["removed"] += 1
    for part in ("panel_in_candidate", "candidate_in_panel", "exact"):
        r[f"removed_{part}"] += part in why
    r["removed_ids"].append({"id": rec["id"], "panel_pids": sorted(
        {int(p) for v in why.values() for p in v})[:5]})


def _row(rec: dict, name: str) -> dict:
    return {"id": f"castor_swap/{rec['id']}", "messages": rec["messages"],
            "source": SWAP_SOURCE[name]}


def select(pools: dict[str, list[dict]], labels: dict[str, dict], cap: int, n: int,
           index: decontam.PanelIndex, kept: frozenset = frozenset()
           ) -> tuple[list[dict], dict]:
    """Two passes over the pools in ORDER (each already shuffled and deduplicated): pass 1 takes
    up to `cap` harmful-prompt refusals, pass 2 fills to `n` with benign-prompt compliance.

    Tier 1 (D127): one row per normalized prompt; a candidate is skipped if its prompt was taken
    already (in either pass) or is in `kept` (the kept Tulu rows' norm_prompts), or if it fails
    the panel decontamination. Tier 2 (D128), only when tier 1 falls short of the pass's target:
    further eligible responses to prompts tier 1 took in the same pass, by rounds (copy 2
    of every such prompt, then copy 3, ...; within a round in ORDER and pool order; a copy's
    number counts the tier-1 row as copy 1 and the prompt's other eligible rows in ORDER
    and pool order). labels: id -> WildGuard row. The pass-1 target is min(cap, n). Returns
    (rows, report); report["short"] is set when the pools cannot fill n."""
    taken: set = set()
    seen_rows: set = set()
    why_cache: dict[tuple, dict] = {}
    rows: list[dict] = []
    report: dict = {}

    def why_of(rec, pk):
        if pk not in why_cache:
            why_cache[pk] = contamination(index, user_turns(rec["messages"]))
        return why_cache[pk]

    for pname, eligible in PASSES:
        target = min(cap, n) if pname.startswith("pass1") else n - len(rows)
        got = 0
        owned: dict[tuple, str] = {}  # prompts tier 1 took in this pass -> source
        taken_ids: set = set()
        report[pname] = rep = {"target": target}
        for name in ORDER:
            r = {"scanned": 0, "eligible": 0, "duplicate_prompt": 0, "duplicate_of_kept": 0,
                 "removed_panel_in_candidate": 0, "removed_candidate_in_panel": 0,
                 "removed_exact": 0, "removed": 0, "removed_ids": [], "taken_tier1": 0}
            rep[name] = r
            for rec in pools.get(name, []):
                if got == target:
                    break
                r["scanned"] += 1
                if not usable(labels[rec["id"]], eligible):
                    continue
                r["eligible"] += 1
                pk = norm_prompt(rec["messages"])
                if pk in taken or pk in kept:
                    r["duplicate_prompt"] += 1
                    r["duplicate_of_kept"] += pk in kept
                    continue
                why = why_of(rec, pk)
                if why:
                    _removed(r, rec, why)
                    continue
                taken.add(pk)
                owned[pk] = name
                taken_ids.add(rec["id"])
                seen_rows.add(example_key(rec["messages"]))
                rows.append(_row(rec, name))
                r["taken_tier1"] += 1
                got += 1
        rep["taken_tier1"] = got

        t2: dict = {"candidates_by_round": {}, "taken_by_round": {}, "cross_source": 0}
        rep["tier2"] = t2
        for name in ORDER:
            rep[name].update({"tier2_candidates": 0, "taken_tier2": 0})
        if got < target:
            copies = dict.fromkeys(owned, 1)
            cands = []
            for si, name in enumerate(ORDER):
                for pos, rec in enumerate(pools.get(name, [])):
                    if rec["id"] in taken_ids or not usable(labels[rec["id"]], eligible):
                        continue
                    pk = norm_prompt(rec["messages"])
                    if pk not in owned or example_key(rec["messages"]) in seen_rows:
                        continue
                    copies[pk] += 1
                    cands.append((copies[pk], si, pos, name, rec, pk))
                    rep[name]["tier2_candidates"] += 1
            cands.sort(key=lambda c: c[:3])
            for rnd, _, _, name, rec, pk in cands:
                t2["candidates_by_round"][rnd] = t2["candidates_by_round"].get(rnd, 0) + 1
            for rnd, _, _, name, rec, pk in cands:
                if got == target:
                    break
                ek = example_key(rec["messages"])
                if ek in seen_rows:
                    continue
                seen_rows.add(ek)
                rows.append(_row(rec, name))
                rep[name]["taken_tier2"] += 1
                t2["taken_by_round"][rnd] = t2["taken_by_round"].get(rnd, 0) + 1
                t2["cross_source"] += name != owned[pk]
                got += 1
        for name in ORDER:
            rep[name]["taken"] = rep[name]["taken_tier1"] + rep[name]["taken_tier2"]
        rep["taken_tier2"] = got - rep["taken_tier1"]
        rep["taken"] = got
    report["short"] = n - len(rows)
    return rows, report


def prompt_multiplicity(rows: list[dict]) -> dict:
    """Rows per normalized prompt (norm_prompt, the selection's key): unique prompts, repeat
    rows (second-and-later copies) and the histogram {copies: prompts}."""
    counts: dict[tuple, int] = defaultdict(int)
    for r in rows:
        counts[norm_prompt(r["messages"])] += 1
    hist: dict[int, int] = defaultdict(int)
    for c in counts.values():
        hist[c] += 1
    return {"rows": len(rows), "unique_norm_prompts": len(counts),
            "repeat_rows": len(rows) - len(counts),
            "histogram": {str(k): hist[k] for k in sorted(hist)}}


def unique_prompts(rows: list[dict]) -> dict:
    """Distinct (stripped) user-turn sequences, per source and in total."""
    by: dict[str, set] = defaultdict(set)
    for r in rows:
        by[r["source"]].add(prompt_key(r["messages"]))
    return {"unique_prompts": {s: len(v) for s, v in sorted(by.items())},
            "unique_prompts_total": len(set().union(*by.values()))}


def lengths(rows: list[dict]) -> dict:
    n = [len(decontam.tokenize(r["messages"][-1]["content"])) for r in rows]
    return {"response_tokens_median": statistics.median(n), "response_tokens_mean":
            round(statistics.fmean(n), 1)}


DPO_COUNTS = ("dpo_rows", "exact", "ngram", "either")


def dpo_overlap(index: decontam.PanelIndex, dpo_rows: Iterable[dict]) -> dict:
    """Counts of DPO rows whose prompt matches one removed-slice prompt. index = the slice's
    user turns. A row matches exactly if a user turn's normalized string equals a slice user
    turn, by 8-gram if more than half of a user turn's tokens lie in 8-grams of one slice
    user turn."""
    n = exact = ngram = either = 0
    for row in dpo_rows:
        n += 1
        turns = dict.fromkeys(m["content"] for m in row["chosen"] if m["role"] == "user")
        e = g = False
        for t in turns:
            toks = decontam.tokenize(t)
            e = e or bool(index.exact.get(decontam.exact_key(toks)))
            g = g or any(s > decontam.THRESHOLD for s in covered_by_one(index, toks).values())
        exact += e
        ngram += g
        either += e or g
    return {"dpo_rows": n, "exact": exact, "ngram": ngram, "either": either}


def dpo_shares(parts: Iterable[dict]) -> dict:
    """Sum per-chunk dpo_overlap counts and add the shares."""
    tot = dict.fromkeys(DPO_COUNTS, 0)
    for p in parts:
        for k in DPO_COUNTS:
            tot[k] += p[k]
    if not tot["dpo_rows"]:
        raise ValueError("no DPO rows")
    return {**tot, **{f"share_{k}": tot[k] / tot["dpo_rows"] for k in DPO_COUNTS[1:]}}


def _share(xs: list) -> float | None:
    ok = [x for x in xs if x is not None]
    return sum(ok) / len(ok) if ok else None


def composition(rows: list[dict], labels: dict[str, dict], label_id=lambda r: r["id"]) -> dict:
    """The D127 composition of one set of rows under WildGuard, overall and per source: share of
    harmful prompts, refusal rate overall and on harmful prompts, benign-prompt refusals,
    harmful responses, unparsed labels, unique prompts and response lengths."""
    def one(sub: list[dict]) -> dict:
        lab = [labels[label_id(r)] for r in sub]
        harm = [x for x in lab if x["prompt_harmful"] is True]
        return {"rows": len(sub),
                "prompt_harmful_share": _share([x["prompt_harmful"] for x in lab]),
                "refusal_rate": _share([x["refusal"] for x in lab]),
                "refusal_rate_harmful_prompts": _share([x["refusal"] for x in harm]),
                "harmful_prompt_refusals": sum(harmful_refusal(x) for x in lab),
                "benign_prompt_refusals": sum(x["prompt_harmful"] is False and x["refusal"] is True
                                              for x in lab),
                "harmful_responses": sum(x["response_harmful"] is True for x in lab),
                "unparsed": {k: sum(x[k] is None for x in lab) for k in LABELS},
                **unique_prompts(sub), **lengths(sub)}
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    out = one(rows)
    out["by_source"] = {s: {k: v for k, v in one(sub).items() if k != "unique_prompts"}
                        for s, sub in sorted(by.items())}
    return out


def agreement(recs: list[dict], labels: dict[str, dict]) -> dict:
    """Cross-tabs of a set's own labels (Nemotron-SFT-Safety-v2: prompt_label, response_policy)
    against WildGuard's prompt_harmful and refusal, as {own: {wildguard: count}}."""
    tabs: dict = {"prompt_label_x_prompt_harmful": defaultdict(lambda: defaultdict(int)),
                  "response_policy_x_refusal": defaultdict(lambda: defaultdict(int))}
    for rec in recs:
        lab = labels[rec["id"]]
        tabs["prompt_label_x_prompt_harmful"][str(rec["meta"]["prompt_label"])][
            str(lab["prompt_harmful"])] += 1
        tabs["response_policy_x_refusal"][str(rec["meta"]["response_policy"])][
            str(lab["refusal"])] += 1
    return {k: {a: dict(sorted(b.items())) for a, b in sorted(v.items())} for k, v in tabs.items()}


def stop_rule(targets: float | None, slice_: float | None) -> bool:
    """D125/D126: stop if the replacements refuse at less than half the slice's rate."""
    if targets is None or slice_ is None:
        raise ValueError("a refusal rate is undefined: no WildGuard label parsed")
    return targets < 0.5 * slice_
