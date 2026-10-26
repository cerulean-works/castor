"""Bibliography audit: parse refs.bib, read fetched records, classify discrepancies.

Every field in a proposed entry or a "fetched" column comes from a saved API response under
runs/068/records/<key>/; nothing is filled from memory. A record that failed to fetch is None here.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"

_ACCENTS = {
    '\\"': "\u0308", "\\'": "\u0301", "\\`": "\u0300", "\\^": "\u0302", "\\~": "\u0303",
    "\\c": "\u0327", "\\v": "\u030c", "\\u": "\u0306", "\\=": "\u0304", "\\.": "\u0307",
}


@dataclass
class BibEntry:
    type: str
    key: str
    fields: dict[str, str]
    line: int
    raw: str = ""

    @property
    def arxiv_id(self) -> str | None:
        for f in ("eprint", "journal", "note", "url", "howpublished"):
            m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", self.fields.get(f, ""))
            if m and (f == "eprint" or "arxiv" in self.fields.get(f, "").lower()):
                return m.group(1)
        return None

    @property
    def doi(self) -> str | None:
        d = self.fields.get("doi")
        return d.strip() if d else None

    @property
    def venue(self) -> str:
        return self.fields.get("booktitle") or self.fields.get("journal") or \
            self.fields.get("publisher") or self.fields.get("howpublished") or ""

    @property
    def authors(self) -> list[str]:
        return split_authors(self.fields.get("author", ""))


def _field_value(s: str, i: int) -> tuple[str, int]:
    """Read one field value starting at s[i] (a brace group, a quoted string or a bare word)."""
    if s[i] == "{":
        depth, j = 0, i
        while True:
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    return s[i + 1:j], j + 1
            j += 1
    if s[i] == '"':
        j = s.index('"', i + 1)
        return s[i + 1:j], j + 1
    m = re.match(r"[^,}\s]+", s[i:])
    return m.group(0), i + m.end()


def parse_bib(text: str) -> list[BibEntry]:
    """Minimal BibTeX parser for refs.bib's flat entries (no @string, no concatenation)."""
    entries = []
    for m in re.finditer(r"^@(\w+)\s*\{\s*([^,\s]+)\s*,", text, re.MULTILINE):
        typ, key, i = m.group(1).lower(), m.group(2), m.end()
        fields: dict[str, str] = {}
        while True:
            fm = re.compile(r"\s*(\w+)\s*=\s*").match(text, i)
            if not fm:
                break
            val, i = _field_value(text, fm.end())
            fields[fm.group(1).lower()] = " ".join(val.split())
            cm = re.compile(r"\s*,?").match(text, i)
            i = cm.end()
        end = re.compile(r"\s*\}").match(text, i)
        if not end:
            raise ValueError(f"malformed entry {key} near offset {i}")
        entries.append(BibEntry(typ, key, fields, text.count("\n", 0, m.start()) + 1,
                                text[m.start():end.end()]))
    return entries


def delatex(s: str) -> str:
    for cmd, mark in _ACCENTS.items():
        s = re.sub(re.escape(cmd) + r"\{?\s*([A-Za-z])\}?", lambda m, mark=mark: m.group(1) + mark, s)
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    return unicodedata.normalize("NFC", s.replace("{", "").replace("}", "").replace("~", " "))


def fold(s: str) -> str:
    """Lower-case, strip accents and LaTeX, keep letters, digits and spaces."""
    s = delatex(s)  # before the Sk strip: ` and ^ are Sk and start LaTeX accents
    s = "".join(c for c in s if unicodedata.category(c) != "Sk")  # PDF text's detached ¨ ´ ˆ
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("ı", "i").replace("ł", "l").replace("ø", "o").replace("ß", "ss")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def title_tokens(s: str) -> list[str]:
    return fold(s).split()


def token_overlap(a: str, b: str) -> float:
    """|A ∩ B| / max(|A|, |B|) over folded title token sets (1.0 = same words)."""
    ta, tb = set(title_tokens(a)), set(title_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def contained(title: str, text: str) -> float:
    """Share of the title's tokens that occur in a page's text (for the PDF-in-hand check)."""
    tt, pt = set(title_tokens(title)), set(title_tokens(text))
    return len(tt & pt) / len(tt) if tt else 0.0


def split_authors(s: str) -> list[str]:
    return [a.strip() for a in re.split(r"\s+and\s+", s) if a.strip()] if s else []


def surname(name: str) -> str:
    """Surname of a BibTeX ('Last, First') or display ('First Last') name, folded."""
    name = name.strip()
    if name.lower() == "others":
        return "others"
    if name.startswith("{") and name.endswith("}"):
        return fold(name)
    if "," in name:
        return fold(name.split(",")[0])
    parts = fold(name).split()
    return parts[-1] if parts else ""


def key_pdf(key: str, pdf_dir: Path) -> Path | None:
    """refs/pdf naming: <author><year>_<keyword>.pdf for key <author><year><keyword>."""
    m = re.match(r"^([a-z]+?)(\d{4})(.+)$", key)
    if not m:
        return None
    p = pdf_dir / f"{m.group(1)}{m.group(2)}_{m.group(3)}.pdf"
    return p if p.exists() else None


# ---------- fetched-record readers (inputs are the saved raw responses) ----------

def read_arxiv(xml_text: str) -> dict | None:
    root = ET.fromstring(xml_text)
    for e in root.findall(f"{ATOM}entry"):
        idurl = e.findtext(f"{ATOM}id") or ""
        if "api/errors" in idurl or not e.findtext(f"{ATOM}title"):
            return None
        m = re.search(r"abs/(.+?)(v(\d+))?$", idurl)
        doi = e.findtext(f"{ARXIV}doi")
        return {
            "id": m.group(1) if m else None,
            "version": int(m.group(3)) if m and m.group(3) else None,
            "title": " ".join((e.findtext(f"{ATOM}title") or "").split()),
            "authors": [a.findtext(f"{ATOM}name") for a in e.findall(f"{ATOM}author")],
            "published": e.findtext(f"{ATOM}published"),
            "updated": e.findtext(f"{ATOM}updated"),
            "journal_ref": e.findtext(f"{ARXIV}journal_ref"),
            "doi": doi,
            "comment": e.findtext(f"{ARXIV}comment"),
        }
    return None


def read_crossref_work(obj: dict) -> dict:
    """One Crossref work item → the fields we compare and build from."""
    def date(o):
        for k in ("published-print", "published-online", "issued", "published"):
            dp = (o.get(k) or {}).get("date-parts") or [[None]]
            if dp[0] and dp[0][0]:
                return dp[0][0]
        return None
    return {
        "title": " ".join((obj.get("title") or [""])[0].split()),
        "authors": [" ".join(x for x in (a.get("given"), a.get("family")) if x) or a.get("name", "")
                    for a in obj.get("author", [])],
        "venue": (obj.get("container-title") or [""])[0],
        "year": date(obj),
        "doi": obj.get("DOI"),
        "type": obj.get("type"),
        "publisher": obj.get("publisher"),
        "volume": obj.get("volume"),
        "issue": obj.get("issue"),
        "pages": obj.get("page"),
        "score": obj.get("score"),
    }


def read_crossref(obj: dict) -> list[dict]:
    msg = obj.get("message", {})
    items = msg.get("items") if "items" in msg else [msg]
    return [read_crossref_work(i) for i in items]


def read_dblp_sparql(obj: dict) -> list[dict]:
    """DBLP SPARQL JSON results (one row per author signature) → one record per publication."""
    recs: dict[str, dict] = {}
    for b in (obj.get("results") or {}).get("bindings", []):
        v = lambda k, b=b: b[k]["value"] if k in b else None
        r = recs.setdefault(v("p"), {
            "title": (v("t") or "").rstrip("."), "authors": {}, "venue": v("v"),
            "year": int(v("y")) if v("y") else None,
            "doi": re.sub(r"^https?://doi.org/", "", v("doi")) if v("doi") else None,
            "pages": v("pages"), "volume": v("vol"), "url": v("ee"),
            "type": (v("btype") or "").rsplit("#", 1)[-1] or None, "dblp": v("p"),
        })
        if v("ord") and v("name"):
            r["authors"][int(v("ord"))] = re.sub(r"\s\d{4}$", "", v("name"))
    out = []
    for r in recs.values():
        r["authors"] = [r["authors"][k] for k in sorted(r["authors"])]
        out.append(r)
    return out


def read_openreview(obj: dict) -> list[dict]:
    out = []
    for n in obj.get("notes", []):
        c = n.get("content", {})
        v = lambda k, c=c: (c.get(k) or {}).get("value") if isinstance(c.get(k), dict) else c.get(k)
        out.append({
            "title": v("title") or "", "venue": v("venue"),
            "authors": [a.get("fullname", "") if isinstance(a, dict) else a for a in v("authors") or []],
            "venueid": v("venueid"), "forum": n.get("forum"), "id": n.get("id"),
            "year": _year_in(v("venue")), "pdf": v("pdf"),
        })
    return out


def read_s2(obj: dict) -> dict | None:
    if not obj or "paperId" not in obj:
        return None
    pv = obj.get("publicationVenue") or {}
    j = obj.get("journal") or {}
    return {
        "title": obj.get("title") or "", "authors": [a.get("name") for a in obj.get("authors", [])],
        "venue": obj.get("venue") or pv.get("name"), "year": obj.get("year"),
        "doi": (obj.get("externalIds") or {}).get("DOI"),
        "dblp": (obj.get("externalIds") or {}).get("DBLP"),
        "acl": (obj.get("externalIds") or {}).get("ACL"),
        "journal": j.get("name"), "volume": j.get("volume"), "pages": j.get("pages"),
        "date": obj.get("publicationDate"),
    }


def _year_in(s) -> int | None:
    m = re.search(r"(19|20)\d{2}", s or "")
    return int(m.group(0)) if m else None


# ---------- venue detection and the proposed entry ----------

_PREPRINT = re.compile(r"arxiv|corr|biorxiv|ssrn|preprint|submitted|withdrawn|rejected", re.IGNORECASE)


def is_venue(v: str | None) -> bool:
    return bool(v) and not _PREPRINT.search(v)


@dataclass
class VenueHit:
    source: str
    venue: str
    year: int | None
    record: dict = field(default_factory=dict)


def venue_hits(title: str, rec: dict, min_overlap: float = 0.9,
               first_author: str | None = None) -> list[VenueHit]:
    """Non-preprint venues named by a fetched record whose title matches the entry's (and, when
    the entry names a person as first author, whose author list contains that surname)."""
    def same_people(r: dict) -> bool:
        return not first_author or first_author in {surname(a) for a in r.get("authors") or []}
    hits = []
    ax = rec.get("arxiv")
    if ax and ax.get("journal_ref") and is_venue(ax["journal_ref"]):
        hits.append(VenueHit("arxiv journal-ref", ax["journal_ref"], _year_in(ax["journal_ref"]), ax))
    for src in ("crossref", "dblp", "openreview"):
        for r in rec.get(src) or []:
            if token_overlap(title, r["title"]) >= min_overlap and is_venue(r.get("venue")) \
                    and r.get("type") not in ("posted-content",) and same_people(r):
                if src == "openreview" and not r.get("venueid", "").count("/"):
                    continue
                hits.append(VenueHit(src, r["venue"], r.get("year"), r))
    s2 = rec.get("s2")
    if s2 and token_overlap(title, s2["title"]) >= min_overlap and is_venue(s2.get("venue")) \
            and same_people(s2):
        hits.append(VenueHit("semantic scholar", s2["venue"], s2.get("year"), s2))
    return hits


def bibtex_name(display: str) -> str:
    """'First Middle Last' (as fetched) → 'Last, First Middle'. Already 'Last, First' passes."""
    if "," in display:
        return display
    p = display.split()
    return display if len(p) < 2 else f"{p[-1]}, {' '.join(p[:-1])}"


def build_entry(key: str, hit: VenueHit, arxiv: dict | None) -> str:
    """A proposed @inproceedings/@article from one fetched record; only fetched fields."""
    r = hit.record
    authors = r.get("authors") or []
    journal = r.get("type") in ("journal-article", "Article")
    typ = "article" if journal else "inproceedings"
    f: list[tuple[str, str]] = [("title", "{" + r.get("title", "") + "}" if r.get("title") else "")]
    f.append(("author", " and ".join(bibtex_name(a) for a in authors)))
    f.append(("journal" if typ == "article" else "booktitle", hit.venue))
    for k in ("volume", "pages"):
        if r.get(k):
            f.append((k, str(r[k]).replace("-", "--") if k == "pages" else str(r[k])))
    if r.get("publisher"):
        f.append(("publisher", r["publisher"]))
    f.append(("year", str(hit.year) if hit.year else ""))
    if r.get("doi"):
        f.append(("doi", r["doi"]))
    if arxiv and arxiv.get("id"):
        f += [("eprint", arxiv["id"]), ("archivePrefix", "arXiv")]
    f = [(k, v) for k, v in f if v]
    w = max(len(k) for k, _ in f)
    body = ",\n".join(f"  {k.ljust(w)} = {{{v}}}" for k, v in f)
    src = f"% proposed by 068 from {hit.source}; every field fetched, none recalled\n"
    return f"{src}@{typ}{{{key},\n{body}\n}}\n"


def build_arxiv_entry(key: str, arxiv: dict) -> str:
    """Preprint form from the arXiv record (the 'still a preprint' case)."""
    f = [("title", "{" + arxiv["title"] + "}"),
         ("author", " and ".join(bibtex_name(a) for a in arxiv["authors"])),
         ("year", arxiv["published"][:4]),
         ("eprint", arxiv["id"]), ("archivePrefix", "arXiv")]
    if arxiv.get("doi"):
        f.append(("doi", arxiv["doi"]))
    w = max(len(k) for k, _ in f)
    body = ",\n".join(f"  {k.ljust(w)} = {{{v}}}" for k, v in f)
    return f"% proposed by 068 from the arXiv API record (v{arxiv.get('version')}); no venue found\n" \
           f"@misc{{{key},\n{body}\n}}\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ---- install pasted venue entries under the cited keys -------------------------------------

# Booktitle unification: the ML conferences get one year-free name each; ACL / ACM / IEEE short
# DBLP names are replaced by the Crossref container title fetched for the entry's DOI. Workshop
# titles are left as pasted.
VENUES = [
    (r"neural information processing systems|^neurips$", "Advances in Neural Information Processing Systems"),
    (r"international conference on machine learning|^icml$", "International Conference on Machine Learning"),
    (r"international conference on learning representations|^iclr$",
     "International Conference on Learning Representations"),
    (r"conference on language modeling|^colm$", "Conference on Language Modeling"),
]
KEEP = {
    "inproceedings": ("author", "title", "booktitle", "year", "pages", "doi", "url"),
    "article": ("author", "title", "journal", "volume", "number", "pages", "year", "doi", "url"),
    "misc": ("author", "title", "howpublished", "year", "eprint", "archiveprefix", "primaryclass"),
}
CASED = ("archivePrefix", "primaryClass")
PROPER = {"Tulu", "Olmo", "Paloma", "Pythia"}  # capitalised-only names sentence case would lower


def unify_booktitle(bt: str, container: str | None = None) -> str:
    if "workshop" in bt.lower():
        return bt
    for pat, name in VENUES:
        if re.search(pat, bt.strip(), re.IGNORECASE):
            return name
    return container or bt


def protect_case(title: str) -> str:
    """Brace words the bst's sentence-casing would damage (inner capitals, PROPER); keep existing braces."""
    # a run mixing a pasted brace group with letters ({OR}-Bench, Multi{BERT}s) is braced whole
    def whole(m):
        run, tail = re.match(r"^(.*?)([:;,.?!]*)$", m.group(0)).groups()
        bare = run.replace("{", "").replace("}", "")
        if run.startswith("{") and run.endswith("}") and run.count("{") == 1:
            return m.group(0)
        return ("{" + run + "}" if re.search(r"[A-Z]", bare[1:]) else run) + tail
    title = re.sub(r"[^\s{}]*\{[^{}\s]*\}[^\s{}]*", whole, title)
    out, depth, word = [], 0, ""

    def flush():
        nonlocal word
        if word:
            prev = "".join(out).rstrip()
            if prev.endswith(("?", "!")) and word[0].isupper():  # sentence case keeps only ': X'
                out.append("{" + word + "}")
                word = ""
                return
            parts = re.split(r"[-–]", word)
            inner = any(re.search(r"[A-Z]", q[1:]) for q in parts) or \
                any(len(q) == 1 and q.isupper() for q in parts[1:])  # Best-of-N
            out.append("{" + word + "}" if inner or parts[0] in PROPER else word)
            word = ""
    for c in title:
        if depth == 0 and (c.isalnum() or c in "-'") and c != "{":
            word += c
            continue
        flush()
        depth += (c == "{") - (c == "}")
        out.append(c)
    flush()
    return "".join(out)


def parse_ris(text: str) -> dict[str, list[str]]:
    rec: dict[str, list[str]] = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Z][A-Z0-9])  - ?(.*)$", line)
        if m:
            rec.setdefault(m.group(1), []).append(m.group(2).strip())
    return rec


def ris_to_fields(rec: dict[str, list[str]]) -> tuple[str, dict[str, str]]:
    typ = "article" if rec.get("TY", [""])[0] == "JOUR" else "misc"
    f = {"author": " and ".join(rec.get("AU", [])), "title": rec.get("TI", [""])[0],
         "journal": rec.get("JO", [""])[0], "volume": rec.get("VL", [""])[0],
         "number": rec.get("IS", [""])[0], "pages": rec.get("SP", [""])[0],
         "year": rec.get("PY", [""])[0], "doi": rec.get("DO", [""])[0]}
    return typ, {k: v for k, v in f.items() if v}


def normalise(key: str, typ: str, fields: dict[str, str], container: str | None = None
              ) -> tuple[str, list[str]]:
    """A pasted entry → the installed entry under `key`; returns (bibtex, changes made)."""
    typ = typ.lower()
    typ = typ if typ in KEEP else "misc"
    f = {k.lower(): v for k, v in fields.items()}
    changes = []
    if typ == "misc" and f.get("eprint") and "howpublished" not in f:
        f["howpublished"] = f"arXiv preprint arXiv:{f['eprint']}"
        changes.append("howpublished ← arXiv preprint arXiv:<eprint> (bst prints no eprint)")
    if f.get("url") and (f.get("doi") or f["url"].lower().endswith(".pdf")
                         or re.search(r"/[0-9a-f]{32}", f["url"])):
        why = "a DOI is given" if f.get("doi") else "a PDF-file or hashed proceedings link; overfull"
        changes.append(f"url dropped ({why})")
        del f["url"]
    if "booktitle" in f:
        new = unify_booktitle(f["booktitle"], container)
        if new != f["booktitle"]:
            changes.append(f"booktitle {f['booktitle']!r} → {new!r}")
            f["booktitle"] = new
    if "pages" in f:
        new = re.sub(r"\s*(--|–|-)\s*", "--", f["pages"])
        if new != f["pages"]:
            changes.append(f"pages {f['pages']} → {new}")
            f["pages"] = new
    if "author" in f:
        names = [a for a in split_authors(f["author"]) if a.strip() not in (":",)]
        names = ["{" + a + "}" if re.match(r"^Team \w+$", a) else a for a in names]
        new = " and ".join(names)
        if new != f["author"]:
            changes.append("author: corporate 'Team …' braced / stray ':' author dropped")
            f["author"] = new
    if "title" in f:
        new = protect_case(f["title"])
        if new != f["title"]:
            changes.append(f"title case-protected: {new}")
            f["title"] = new
    dropped = sorted(k for k in f if k not in KEEP[typ])
    if dropped:
        changes.append("dropped " + ", ".join(dropped))
    keep = [(k, f[k]) for k in KEEP[typ] if f.get(k)]
    names = {k.lower(): k for k in CASED}
    w = max(len(k) for k, _ in keep)
    body = ",\n".join(f"  {names.get(k, k).ljust(w)} = {{{v}}}" for k, v in keep)
    return f"@{typ}{{{key},\n{body}\n}}", changes
