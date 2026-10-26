"""8-gram decontamination of the panel against training mixes.

Follows the Tulu 3 paper's eval decontamination (Lambert et al. 2024, Sec. 3.2, p.14) as
implemented in open-instruct/decontamination (commit d19de7d27):
- training side: every user turn of every training instance is one document
  (index.py read_dataset, `--query_filter role:user`);
- tokens: open-instruct's `tulu_analyzer` (index.py create_text_index): split on its regex
  `[ ,.?!:;()"-]|\n|\\`, lowercase;
- a test token matches a training document if it lies inside an 8-gram of the test prompt
  that also occurs in that document; the test prompt's score is the largest fraction of
  its tokens matched by any single document (search.py ngram_match);
- contaminated if score > 0.5 (paper: "more than 50% of the test tokens").

Deviations from the original tooling, all making the check stricter or equal:
- exhaustive in-memory matching instead of Elasticsearch top-`search_size` hits;
- the same analyzer tokenizes both sides (original: spaCy splits the query into 8-grams, ES
  re-analyzes each 8-gram as a phrase);
- every Unicode whitespace character splits (tab, CR, NBSP, ...), not only space and
  newline, so text that differs only in whitespace style (CRLF vs LF, tabs) still matches;
- ties between equally-covering documents go to the smallest doc id, so the report does not
  depend on the order training shards are scanned.
We also flag exact matches (normalized full-string equality) with a training user turn, which
catches prompts shorter than 8 tokens that the n-gram rule cannot score.
"""

import re
from collections import defaultdict
from collections.abc import Iterable, Iterator

TULU_SPLIT = re.compile(r"[\s,.?!:;()\"-]|\\")
N = 8
THRESHOLD = 0.5


def tokenize(text: str) -> list[str]:
    return [t for t in TULU_SPLIT.split(text.lower()) if t]


def ngrams(tokens: list[str], n: int = N) -> Iterator[tuple[int, str]]:
    for i in range(len(tokens) - n + 1):
        yield i, " ".join(tokens[i:i + n])


def exact_key(tokens: list[str]) -> str:
    """Normalized full-string key for the exact-match check."""
    return " ".join(tokens)


class PanelIndex:
    """All 8-grams of the panel prompts, so training documents can be streamed past it."""

    def __init__(self, prompts: list[str], n: int = N):
        self.n = n
        self.tokens = [tokenize(p) for p in prompts]
        self.grams: dict[str, list[tuple[int, int]]] = defaultdict(list)  # gram -> (pid, pos)
        for pid, toks in enumerate(self.tokens):
            for pos, g in ngrams(toks, n):
                self.grams[g].append((pid, pos))
        self.exact = defaultdict(list)
        for pid, toks in enumerate(self.tokens):
            self.exact[exact_key(toks)].append(pid)

    def hits(self, doc: str) -> tuple[set[str], list[int]]:
        """Panel 8-grams present in one training document, and exact-match panel ids."""
        toks = tokenize(doc)
        found = {g for _, g in ngrams(toks, self.n) if g in self.grams}
        return found, self.exact.get(exact_key(toks), [])


def score(index: PanelIndex, doc_grams: Iterable[tuple[str, set[str]]]) -> list[dict]:
    """doc_grams: (doc_id, panel 8-grams found in that doc). Returns per panel prompt the
    best single-document token coverage and that document's id."""
    cover: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for doc_id, grams in doc_grams:
        for g in grams:
            for pid, pos in index.grams[g]:
                cover[pid][doc_id].update(range(pos, pos + index.n))
    out = []
    for pid, toks in enumerate(index.tokens):
        best, best_doc = 0.0, None
        for doc_id, covered in cover.get(pid, {}).items():
            s = len(covered) / len(toks)
            if s > best or (s == best and best_doc is not None and doc_id < best_doc):
                best, best_doc = s, doc_id
        out.append({"score": best, "doc": best_doc, "n_tokens": len(toks)})
    return out


def removal_reasons(per_set: dict[str, list[dict]],
                    exact_by_set: dict[str, dict[int, list[str]]],
                    pid: int) -> dict[str, dict]:
    """Why panel prompt `pid` is contaminated, per training set ({} if clean).
    per_set: set -> `score` output; exact_by_set: set -> pid -> matching doc ids."""
    why: dict[str, dict] = {}
    for name, scores in per_set.items():
        s = scores[pid]
        if s["score"] > THRESHOLD:
            why[name] = {"ngram_score": round(s["score"], 4), "doc": s["doc"]}
    for name, by_pid in exact_by_set.items():
        docs = by_pid.get(pid)
        if docs:
            why.setdefault(name, {})["exact"] = sorted(docs)[:5]
            why[name]["n_exact"] = len(docs)
    return why
