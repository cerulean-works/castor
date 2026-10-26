"""Terminal labeller for the blinded audit sheets written by scripts/audit_sample.py.

    uv run --frozen python scripts/audit_label.py AUDIT_DIR/sheet_harry.csv

One item per screen: prompt, response, current label. Every change is saved to the CSV at
once (atomic replace; a .bak copy of the sheet is kept on first open), so quitting any time
is safe and the next start resumes at the first unlabelled item. The CSV stays the format
scripts/audit_score.py reads (label R/H/N/I, partial 1 or empty, notes).

Keys
  r h n i      label REFUSAL / HARMFUL-COMPLIANCE / NON-HARMFUL-COMPLIANCE / INCOHERENT-OR-OFF-TOPIC,
               then go to the next item
  p            toggle partial (mixes refusal and compliance); stays on the item
  c            edit the note
  x            clear label and partial
  → / space / l      next item          ← / b / backspace   previous item
  ↓ / j, ↑ / k       scroll one line    PgDn / d, PgUp / e  scroll one page
  g            jump to the next unlabelled item
  ?            help (the labelling instructions)
  q            save and quit
"""

import argparse
import csv
import curses
import os
import shutil
import sys
import textwrap
from pathlib import Path

LABELS = {"r": ("R", "REFUSAL"), "h": ("H", "HARMFUL-COMPLIANCE"),
          "n": ("N", "NON-HARMFUL-COMPLIANCE"), "i": ("I", "INCOHERENT / OFF-TOPIC")}
NAMES = {code: name for code, name in LABELS.values()}
KEYS = ("r/h/n/i label  p partial  c note  x clear  ←/→ prev/next  j/k scroll  "
        "g next unlabelled  ? help  q quit")


class Sheet:
    """The CSV: a header, optional '#' instruction rows, then one row per item."""

    def __init__(self, path: Path):
        self.path = path
        with open(path, newline="") as f:
            self.rows = list(csv.reader(f))
        self.header = self.rows[0]
        self.col = {c: i for i, c in enumerate(self.header)}
        for c in ("item_id", "prompt", "response", "label", "partial", "notes"):
            if c not in self.col:
                raise ValueError(f"{path}: missing column {c!r}")
        self.items = [r for r in self.rows[1:] if not r[self.col["item_id"]].startswith("#")]
        self.instructions = " ".join(r[self.col["prompt"]] for r in self.rows[1:]
                                     if r[self.col["item_id"]].startswith("#"))
        for r in self.items:  # pad short rows so every column index exists
            r.extend([""] * (len(self.header) - len(r)))

    def get(self, i: int, c: str) -> str:
        return self.items[i][self.col[c]]

    def set(self, i: int, c: str, v: str) -> None:
        self.items[i][self.col[c]] = v
        self.save()

    def save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", newline="") as f:
            csv.writer(f).writerows(self.rows)
        os.replace(tmp, self.path)

    def n_done(self) -> int:
        return sum(bool(self.get(i, "label").strip()) for i in range(len(self.items)))

    def next_unlabelled(self, start: int = 0) -> int | None:
        n = len(self.items)
        for k in range(n):
            i = (start + k) % n
            if not self.get(i, "label").strip():
                return i
        return None


def wrap(text: str, width: int) -> list[str]:
    out = []
    for para in text.split("\n"):
        out.extend(textwrap.wrap(para, width, replace_whitespace=False,
                                 drop_whitespace=True) or [""])
    return out


def body_lines(sheet: Sheet, i: int, width: int) -> list[tuple[str, int]]:
    """(text, attr) lines: the prompt, then the response."""
    head = curses.A_BOLD | curses.A_REVERSE
    lines = [(" PROMPT ", head)]
    lines += [(s, curses.A_DIM) for s in wrap(sheet.get(i, "prompt"), width)]
    lines += [("", 0), (" RESPONSE ", head)]
    resp = sheet.get(i, "response")
    lines += [(s, 0) for s in wrap(resp, width)] if resp.strip() else [("(empty response)",
                                                                        curses.A_DIM)]
    return lines


def ask(scr, prompt: str, initial: str) -> str:
    """One-line text input on the bottom row (Enter accepts, Esc cancels)."""
    h, w = scr.getmaxyx()
    buf = list(initial)
    curses.curs_set(1)
    try:
        while True:
            line = prompt + "".join(buf)
            scr.move(h - 1, 0)
            scr.clrtoeol()
            scr.addnstr(h - 1, 0, line[-(w - 1):], w - 1)
            ch = scr.get_wch()
            if ch in ("\n", "\r", curses.KEY_ENTER):
                return "".join(buf)
            if ch == "\x1b":
                return initial
            if ch in (curses.KEY_BACKSPACE, "\x7f", "\b"):
                if buf:
                    buf.pop()
            elif isinstance(ch, str) and ch.isprintable():
                buf.append(ch)
    finally:
        curses.curs_set(0)


def show_help(scr, sheet: Sheet) -> None:
    h, w = scr.getmaxyx()
    scr.erase()
    text = (__doc__.split("Keys", 1)[1] + "\n\nInstructions from the sheet:\n"
            + sheet.instructions)
    for y, s in enumerate(["Keys"] + wrap(text, w - 2)):
        if y >= h - 1:
            break
        scr.addnstr(y, 1, s, w - 2)
    scr.addnstr(h - 1, 1, "any key to return", w - 2, curses.A_REVERSE)
    scr.getch()


ESCAPES = {"[A": curses.KEY_UP, "[B": curses.KEY_DOWN, "[C": curses.KEY_RIGHT,
           "[D": curses.KEY_LEFT, "OA": curses.KEY_UP, "OB": curses.KEY_DOWN,
           "OC": curses.KEY_RIGHT, "OD": curses.KEY_LEFT, "[5~": curses.KEY_PPAGE,
           "[6~": curses.KEY_NPAGE}


def read_key(scr) -> int:
    """getch, decoding arrow / page escape sequences curses left undecoded (terminals differ in
    which form they send)."""
    ch = scr.getch()
    if ch != 27:
        return ch
    scr.nodelay(True)
    seq = ""
    try:
        while len(seq) < 3:
            c = scr.getch()
            if c == -1:
                break
            seq += chr(c)
            if seq in ESCAPES:
                return ESCAPES[seq]
    finally:
        scr.nodelay(False)
    return 27


def run(scr, sheet: Sheet, start: int) -> None:
    curses.curs_set(0)
    curses.use_default_colors()
    i, top, msg = start, 0, ""
    n = len(sheet.items)
    while True:
        h, w = scr.getmaxyx()
        scr.erase()
        width = max(20, w - 2)
        lines = body_lines(sheet, i, width)
        view = max(1, h - 4)
        top = max(0, min(top, len(lines) - view))
        lab, part = sheet.get(i, "label").strip().upper(), sheet.get(i, "partial").strip()
        status = (f" item {i + 1}/{n}  id {sheet.get(i, 'item_id')}   done {sheet.n_done()}/{n}"
                  f"   label: {NAMES.get(lab, '\u2014') if lab else '\u2014'}"
                  f"   partial: {'yes' if part in ('1', 'y', 'Y', 'yes') else 'no'}")
        scr.addnstr(0, 0, status.ljust(w), w - 1, curses.A_REVERSE | curses.A_BOLD)
        for y, (s, attr) in enumerate(lines[top:top + view]):
            scr.addnstr(1 + y, 1, s, width, attr)
        more = f" lines {top + 1}-{min(top + view, len(lines))} of {len(lines)} "
        note = sheet.get(i, "notes")
        scr.addnstr(h - 3, 0, (more + ("  (more ↓)" if top + view < len(lines) else "")
                               ).ljust(w), w - 1, curses.A_DIM)
        scr.addnstr(h - 2, 0, (f" note: {note}" if note else msg or " ").ljust(w), w - 1)
        scr.addnstr(h - 1, 0, (" " + KEYS).ljust(w), w - 1, curses.A_REVERSE)
        msg = ""
        ch = read_key(scr)
        key = chr(ch).lower() if 0 <= ch < 256 else ""
        if key in LABELS:
            sheet.set(i, "label", LABELS[key][0])
            if i < n - 1:
                i, top = i + 1, 0
            else:
                msg = " last item \u2014 press g for any unlabelled one, q to quit"
        elif key == "p":
            sheet.set(i, "partial", "" if part in ("1", "y", "Y", "yes") else "1")
        elif key == "c":
            sheet.set(i, "notes", ask(scr, " note: ", note))
        elif key == "x":
            sheet.set(i, "label", "")
            sheet.set(i, "partial", "")
        elif ch in (curses.KEY_RIGHT,) or key in (" ", "l"):
            i, top = min(n - 1, i + 1), 0
        elif ch in (curses.KEY_LEFT, curses.KEY_BACKSPACE) or key in ("b", "\x7f", "\b"):
            i, top = max(0, i - 1), 0
        elif ch == curses.KEY_DOWN or key == "j":
            top += 1
        elif ch == curses.KEY_UP or key == "k":
            top -= 1
        elif ch == curses.KEY_NPAGE or key == "d":
            top += view - 1
        elif ch == curses.KEY_PPAGE or key == "e":
            top -= view - 1
        elif key == "g":
            j = sheet.next_unlabelled(i + 1)
            if j is None:
                msg = " all items labelled \u2014 q to quit"
            else:
                i, top = j, 0
        elif key == "?":
            show_help(scr, sheet)
        elif key == "q":
            return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sheet", type=Path, help="sheet_<annotator>.csv from audit_sample.py")
    ap.add_argument("--start", type=int, help="1-based item to open (default: first unlabelled)")
    args = ap.parse_args()
    if not sys.stdin.isatty():
        sys.exit("audit_label.py needs an interactive terminal")
    sheet = Sheet(args.sheet)
    bak = args.sheet.with_suffix(args.sheet.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(args.sheet, bak)
    start = (args.start - 1) if args.start else (sheet.next_unlabelled() or 0)
    curses.wrapper(run, sheet, max(0, min(start, len(sheet.items) - 1)))
    print(f"{sheet.n_done()}/{len(sheet.items)} labelled; saved to {args.sheet}")


if __name__ == "__main__":
    main()
