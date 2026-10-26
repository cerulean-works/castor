"""The size ladder: the main recipe's twins at 150M and 300M, the 1B point, and the
pre-registered trend rule.

Models: DCLM-Baseline at 150M@step37500 and 300M@step45000 (step-matched), pretraining seeds
{default, small-aux-2, small-aux-3} (released; each sets init and data order jointly) ×
post-training seeds k ∈ {1, 2, 3, 4} (k → SFT seed k, DPO seed 110 + k). Recipe, configs and
instrument are castor.sanityb's: only the base differs.

`Cell` has the interface of `sanityb.Cell`, so the launcher functions
(`scripts/launch_006.py` status / train_cmd) and `sanityb.check_base` take it as is.

Pre-registered:
  - per size: the gate (`sanityb.gate`) on its 12 twins; a size that fails is reported and
    leaves the trend;
  - primary = ceiling-corrected r_MZ − r_U on adv_harmful × wg_vulnerable;
  - the prediction holds iff gap(150M) > gap(300M) > gap(1B) as point estimates AND the
    prompt-bootstrap 95 % interval of gap(150M) − gap(1B) lies above 0 (the same prompt
    resample for every size, so the difference is paired on prompts).
"""

import json
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from math import comb
from pathlib import Path
from types import MappingProxyType

import numpy as np

from castor import convert, grid, reliability, sanityb

PROMPT_ID = "016"
RECIPE = sanityb.RECIPE
SIZES = {  # size -> (HF repo, step)
    "150M": ("allenai/DataDecide-dclm-baseline-150M", "step37500"),
    "300M": ("allenai/DataDecide-dclm-baseline-300M", "step45000"),
}
PRETRAIN_SEEDS = ("default", "small-aux-2", "small-aux-3")  # small-aux-* below 1B
POSTTRAIN_SEEDS = sanityb.POSTTRAIN_SEEDS
MODELS_ROOT = sanityb.MODELS_ROOT
TREND = ("150M", "300M", "1B")  # predicted order of the gap, largest first
# pairs the trend's gap averages at every size: 3 bases x C(4, 2) MZ, C(3, 2) x 4 x 4 U
# (the 1B twins are also 3 bases x 4 post-training seeds)
TREND_PAIRS = {"MZ": len(PRETRAIN_SEEDS) * comb(len(POSTTRAIN_SEEDS), 2),
               "U": comb(len(PRETRAIN_SEEDS), 2) * len(POSTTRAIN_SEEDS) ** 2}

# ---- the launcher configuration of record ---------------------------------------------------
# Every post-training cell trains its SFT on this DeepSpeed config (ZeRO-1), DPO on the stage
# default; path and sha256 are recorded in each run's provenance by posttrain_run.py render. The
# launcher refuses any other configuration and the analysis refuses a twin trained on another
# one (no mixed-configuration comparison).
SFT_DS = "configs/posttrain/ds/sft_zero1.json"
SFT_DS_SHA256 = "b615b6daa5ea9aa8e67d5b9055093b8581287439f84f73038730af2d6d70ab37"
LAUNCHER = {"ds_config": {"sft": SFT_DS}, "launch_env": None}  # castor.grid's control shape

# ---- the 1B point of the trend: the main-recipe rerun ----------------------------------------
# The 1B anchor is the 12 A0 cells of the grid_008 manifest (3 dclm-baseline bases x
# post-training seeds 1-4, SFT on SFT_DS), not the ZeRO-3 twins. Their labels live under
# runs/<model-id>/008/, their signal file of record (scripts/signal_006.py --twins a0) at
# ANCHOR_SIGNAL.
GRID_008 = grid.ROOT / "data/manifests/grid_008.json"  # repo-anchored: any cwd, any snapshot
ANCHOR_PROMPT_ID = "008"
ANCHOR_SIGNAL = grid.ROOT / "runs" / "008" / "signal_a0.json"


@dataclass(frozen=True)
class Cell:
    size: str
    pretrain_seed: str
    k: int

    @property
    def repo(self) -> str:
        return SIZES[self.size][0]

    @property
    def revision(self) -> str:
        return f"{SIZES[self.size][1]}-seed-{self.pretrain_seed}"

    @property
    def base_ref(self) -> str:
        return f"{self.repo}@{self.revision}"

    @property
    def base_dir(self) -> Path:
        return convert.converted_dir(self.repo, self.revision)

    @property
    def group(self) -> str:
        return f"dd-dclm-{self.size}-seed-{self.pretrain_seed}"

    @property
    def model_id(self) -> str:
        return f"dd-dclm-{self.size.lower()}-{self.pretrain_seed}-dpo-seed{self.k}"

    @property
    def tag(self) -> str:
        return f"{self.size}-{self.pretrain_seed}-s{self.k}"

    def out(self, stage: str, root: Path = MODELS_ROOT) -> Path:
        return root / self.group / stage / f"seed{self.k}"

    @property
    def info(self) -> reliability.ModelInfo:
        return reliability.ModelInfo(RECIPE, self.pretrain_seed, str(self.k))


def cells(sizes: tuple[str, ...] = tuple(SIZES)) -> list[Cell]:
    return [Cell(z, s, k) for z in sizes for s in PRETRAIN_SEEDS for k in POSTTRAIN_SEEDS]


@dataclass(frozen=True)
class AnchorCell:
    """One of the 12 main-recipe rerun cells (A0) with `Cell`'s / `sanityb.Cell`'s interface,
    so the launcher functions, `signal_016` and the configuration check take it as they take a
    ladder cell. Ids `dd-dclm-baseline-1b-<s>-dpo-seed<k>`, generation directory
    `runs/<id>/008/` (castor.grid naming; never the ZeRO-3 `dd-dclm-1b-…`)."""
    g: grid.Cell

    size = "1B"

    @property
    def k(self) -> int:
        return self.g.k

    @property
    def pretrain_seed(self) -> str:
        return self.g.base.pretrain_seed

    @property
    def repo(self) -> str:
        return self.g.base.repo

    @property
    def revision(self) -> str:
        return self.g.base.revision

    @property
    def base_ref(self) -> str:
        return self.g.base.ref

    @property
    def base_dir(self) -> Path:
        return self.g.base.base_dir

    @property
    def group(self) -> str:
        return self.g.base.group

    @property
    def model_id(self) -> str:
        return self.g.model_id

    @property
    def tag(self) -> str:
        return self.g.tag

    def out(self, stage: str, root: Path = MODELS_ROOT) -> Path:
        return self.g.out(stage, root)

    @property
    def info(self) -> reliability.ModelInfo:
        return reliability.ModelInfo(RECIPE, self.pretrain_seed, str(self.k))


def anchor_cells(manifest: Path = GRID_008) -> list[AnchorCell]:
    """The 12 A0 cells of the grid_008 manifest: the main recipe's bases (sanityb.REPO,
    sanityb's three pretraining seeds and step) x post-training seeds 1-4. Raises (ValueError,
    not assert) unless the manifest is the reference instrument and holds exactly those cells."""
    g = grid.load(manifest)
    if g.prompt_id != ANCHOR_PROMPT_ID:
        raise ValueError(f"{manifest}: prompt_id {g.prompt_id}, not {ANCHOR_PROMPT_ID}")
    if (g.sft_config, g.dpo_config) != (sanityb.SFT_CONFIG, sanityb.DPO_CONFIG):
        raise ValueError(f"{manifest}: not 006's post-training configs")
    out = [AnchorCell(c) for c in g.cells() if c.base.repo == sanityb.REPO]
    want = {c.base_ref: set(POSTTRAIN_SEEDS) for c in sanityb.cells()}
    got: dict[str, set[int]] = {}
    for c in out:
        got.setdefault(c.base_ref, set()).add(c.k)
    if got != want:
        raise ValueError(f"{manifest}: main-recipe cells {got} are not 006's bases x "
                         f"seeds {sorted(POSTTRAIN_SEEDS)}: {want}")
    return out


# ---- one launcher configuration across every size ---------------------------------------------


def _pin(stages: Mapping[str, Mapping], what: str = SFT_DS) -> None:
    """The pin, spelt once: the resolved SFT sha must be SFT_DS_SHA256. No hashing: it reads
    the resolved value, so `_launcher` can repeat it on every `want` it is handed; `what` names
    what is refused (the file, or a stale Launcher). Raises, not asserts: it must hold under
    python -O too."""
    sha = stages["sft"]["deepspeed_config_sha256"]
    if sha != SFT_DS_SHA256:
        raise ValueError(f"{what} is {sha}, not the ratified configuration {SFT_DS_SHA256} (D067a)")


def _freeze(stages: dict[str, dict]) -> Mapping[str, Mapping]:
    """A deep read-only view of the resolved record, {stage: {deepspeed_config_sha256: str,
    env: {KEY: VALUE}}} (the shape castor.grid.expected_launcher makes, and the only one frozen
    here): MappingProxyType at every level, so item assignment anywhere is a TypeError, and
    equal to the plain dict it wraps."""
    return MappingProxyType({st: MappingProxyType(v | {"env": MappingProxyType(v["env"])})
                             for st, v in stages.items()})


@dataclass(frozen=True, eq=False)
class Launcher:
    """The configuration of record (`LAUNCHER`) resolved per stage: `stages` = {stage:
    {DeepSpeed sha256, exported environment}} (castor.grid.resolve_launcher), as a deep
    read-only view (`_freeze`); `as_dict()` returns a copy. Constructing one runs the pin
    (`_pin`), so an edited `sft_zero1.json` is refused by every entry point.
    `check_launcher_request` and `check_cell_launcher` accept only a `Launcher` (`_launcher`,
    which repeats the pin). Resolve once per invocation and pass it on (`want=`). Identity, not
    value (eq=False): hashable, never compared field-wise."""
    stages: Mapping[str, Mapping] = field(init=False)

    def __post_init__(self) -> None:
        stages = grid.resolve_launcher(LAUNCHER)
        _pin(stages)
        object.__setattr__(self, "stages", _freeze(stages))

    def as_dict(self) -> dict[str, dict]:
        """`stages` as a plain copy, for a JSON record; changing it changes nothing here."""
        return {st: {**v, "env": dict(v["env"])} for st, v in self.stages.items()}


def _launcher(want: Launcher | None) -> Launcher:
    """`want` as the entry points pass it (one `Launcher()` per invocation), or the record
    resolved here when None; anything else is refused, and the pin is repeated on `want`
    (no hashing) so a `Launcher` built before SFT_DS_SHA256 changed cannot pass as current."""
    if want is None:
        return Launcher()
    if not isinstance(want, Launcher):
        raise TypeError(f"{type(want).__name__} is not a ladder.Launcher")
    _pin(want.stages, "the Launcher handed as want (resolved before SFT_DS_SHA256 changed)")
    return want


def check_launcher_request(ds_config: str | None, launch_env: list[str] | None,
                           want: Launcher | None = None) -> dict:
    """What a launcher invocation asks for ({ds_config: {sft: path} | None, launch_env}) must
    resolve, stage by stage, to the configuration of record (same DeepSpeed bytes, same exported
    environment); anything else is refused.
    `want` is the record (`Launcher`, pinned by construction; resolved here if None). A request
    for the record itself is `want`: nothing to resolve or compare. Returns the control dict
    to pass on."""
    launcher = {"ds_config": {"sft": ds_config} if ds_config else None,
                "launch_env": list(launch_env) if launch_env else None}
    want = _launcher(want)
    if launcher != LAUNCHER:
        got = grid.resolve_launcher(launcher)
        assert got == want.stages, f"launcher {launcher} resolves to {got}, not the " \
                                   f"configuration of record {LAUNCHER} -> {want.as_dict()} " \
                                   "(D067a, D073b)"
    return launcher


def check_cell_launcher(cell, root: Path = MODELS_ROOT,
                        stages: tuple[str, ...] = ("sft", "dpo"),
                        want: Launcher | None = None) -> dict[str, dict]:
    """Every present provenance of `cell` was trained on the configuration of record
    (`LAUNCHER`; castor.grid.check_launcher: DeepSpeed sha256 and exported environment, CPU
    count aside). Returns stage -> launcher_of. A ZeRO-3 twin is refused here. Stages without a
    provenance are skipped (the caller decides whether that is ok). `want` = the `Launcher`
    resolved once by a caller that checks many cells; there is no other configuration to check
    against, so no `launcher` argument that `want` could silently disagree with."""
    plain = _launcher(want).as_dict()  # the record's values as grid prints them in a refusal
    out = {}
    for st in stages:
        p = cell.out(st, root) / "provenance.json"
        if not p.exists():
            continue
        prov = json.loads(p.read_text())
        grid.check_launcher(prov, st, plain[st], cell.out(st, root))
        out[st] = grid.launcher_of(prov, st)
    return out


_NO_CELLS = "no A0 cells to check the anchor against (ladder.anchor_cells())"


def check_anchor_twins(signal: dict, cells: list, where: Path = ANCHOR_SIGNAL) -> list[str]:
    """The signal file of record for the A0 twins must name exactly their 12 model ids as its
    twins (so runs/006/signal.json, whose models are the ZeRO-3 `dd-dclm-1b-…` ids, is refused
    by name). Independent of the labels at hand: it runs whenever the file exists. Empty `cells`
    would make it vacuous, so they are refused. Returns the sorted ids."""
    assert cells, _NO_CELLS
    want = sorted(c.model_id for c in cells)
    got = sorted(m for m, v in signal["models"].items() if not v.get("reference"))
    assert got == want, f"{where}: its twins {got} are not the A0 rerun cells {want} " \
                        "(runs/006/signal.json is 006's ZeRO-3 twins, not the anchor on the " \
                        "new configuration)"
    return want


@dataclass(frozen=True, eq=False)
class MissingAnchor:
    """`read_anchor`'s result when the file does not exist (allowed only under
    `allow_incomplete`): it carries the path it looked at, so `anchor_block` records one source
    from one object on either branch and takes no second `where`. Identity, not value
    (eq=False), as `Anchor` is."""
    where: Path


@dataclass(frozen=True, eq=False)
class Anchor:
    """A parsed anchor signal file whose twins are exactly the A0 cells. Constructing one runs
    `check_anchor_twins` (`cells` is an init-only argument, not state), so a wrong file cannot
    become an Anchor and nothing that takes one (`check_anchor_signal`, `anchor_block`) checks
    the twins again: the by-name refusal runs once, in `read_anchor`, signal_016's fail-fast
    point. Identity, not value (eq=False): hashable, never compared field-wise."""
    signal: dict
    where: Path
    cells: InitVar[list]
    twins: list[str] = field(init=False)

    def __post_init__(self, cells: list) -> None:
        object.__setattr__(self, "twins", check_anchor_twins(self.signal, cells, self.where))


def read_anchor(cells: list, where: Path = ANCHOR_SIGNAL,
                allow_incomplete: bool = False) -> Anchor | MissingAnchor:
    """The anchor signal file as an `Anchor` (parsed, twins checked against `cells`: a wrong
    file is wrong whatever the labels, so it is refused whenever the file exists); a
    `MissingAnchor` if it does not exist, which is allowed only under `allow_incomplete` (the
    ladder before the A0 twins are judged). `cells` must be the A0 cells (never empty): refused
    first, before any file is read, so an empty list is named as such and not left to a parse
    error. The policy on a missing file lives here and in `anchor_block`'s record of it,
    nowhere else: signal_016 calls this first so an absent or wrong file fails before the
    per-size statistics, not after."""
    assert cells, _NO_CELLS
    if where.exists():
        return Anchor(json.loads(where.read_text()), where, cells)
    assert allow_incomplete, _anchor_missing(where)
    return MissingAnchor(where)


def _anchor_missing(where: Path) -> str:
    return f"{where} does not exist (write it with signal_006.py --twins a0)"


def check_anchor_signal(anchor: Anchor, gap_recomputed: float) -> dict:
    """The 1B anchor: the file's corrected gap must equal the gap recomputed
    here from the same (complete) labels with the same functions, to 1e-9. Its twins were
    checked when the `Anchor` was constructed."""
    assert isinstance(anchor, Anchor), f"{type(anchor).__name__} is not an Anchor (read_anchor)"
    lv = anchor.signal[sanityb.PRIMARY_SUBSET]["primary"]["levels"]
    g = gap(lv)
    assert abs(g - gap_recomputed) < 1e-9, \
        f"{anchor.where}: gap {g} != recomputed {gap_recomputed}"
    return {"checked": True, "signal": str(anchor.where), "gap_signal": g,
            "gap_recomputed": gap_recomputed, "signal_git_commit": anchor.signal.get("git_commit"),
            "signal_script": anchor.signal.get("script"), "twins": anchor.twins}


def anchor_block(anchor: Anchor | MissingAnchor, gap_recomputed: float,
                 labels_complete: bool, allow_incomplete: bool) -> dict:
    """signal_016's anchor entry; `anchor` is `read_anchor`'s result: an `Anchor` (so its twins
    are the A0 cells by construction, whatever the labels) or a `MissingAnchor` for a file that
    does not exist; either carries the file it came from (`where`), so the entry names one
    source whichever branch wrote it and nothing else is passed in (a raw dict or None is
    refused). The gap cross-check (`check_anchor_signal`) runs when the file exists and the 1B
    labels are complete; it is skipped, recorded as `checked` False with the reason, only under
    `allow_incomplete` (the ladder before the A0 twins are judged: `runs/008/signal_a0.json`
    does not exist yet, or a partially labelled A0 set cannot equal a file computed on complete
    labels). Without `allow_incomplete` a missing file or incomplete 1B labels raise, as every
    other missing input does. A `checked` False entry makes `trend_rule` INCOMPLETE
    (`anchor_checked`): never a verdict with the 1B anchor unverified."""
    assert isinstance(anchor, (Anchor, MissingAnchor)), \
        f"{type(anchor).__name__} is not an Anchor or MissingAnchor (read_anchor)"
    where = anchor.where
    reasons = []
    if isinstance(anchor, MissingAnchor):
        reasons.append(_anchor_missing(where))
    if not labels_complete:
        reasons.append("1B (A0) twins not all labelled on every cell")
    if not reasons:
        return check_anchor_signal(anchor, gap_recomputed)
    assert allow_incomplete, "; ".join(reasons)
    return {"checked": False, "signal": str(where), "gap_recomputed": gap_recomputed,
            "reason": "; ".join(reasons)}


# ---- the trend --------------------------------------------------------------------------------


def gap(levels: dict) -> float:
    """Ceiling-corrected r_MZ − r_U from twin_correlations(...)["levels"]."""
    return levels["MZ"]["r_corrected"] - levels["U"]["r_corrected"]


def gap_every_pair(tw: dict, n_pairs: dict[str, int] = TREND_PAIRS) -> float:
    """`gap` for the trend: NaN unless every MZ and U pair of the size is present (`n_pairs`,
    18 and 48) and its correction is defined (`levels[lvl]["n_pairs_defined"]`).
    `twin_correlations` averages r_corrected per level with nanmean, so a sample on which one
    twin's split-half reliability is ≤ 0 (`disattenuate` NaN for its pairs) still gives a finite
    level mean over the remaining pairs, and a size with a twin missing gives a finite mean over
    fewer pairs. Neither is the pre-registered statistic, so the gap is undefined instead: on a
    bootstrap draw `reliability._percentile_ci` then drops the draw (`n_draws` < `n_boot`), on
    the full sample `trend_rule` sees a NaN point estimate; either makes the rule undecided, not
    a verdict."""
    lv = tw["levels"]
    every = all(lv[lvl]["n_pairs"] == lv[lvl]["n_pairs_defined"] == n for lvl, n in n_pairs.items())
    return gap(lv) if every else float("nan")


def paired_size_bootstrap(ys: dict[str, dict[str, np.ndarray]], ids: dict[str, list[str]],
                          info: dict[str, dict[str, reliability.ModelInfo]],
                          n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Prompt bootstrap of each size's corrected gap and of every pairwise difference, with
    one row resample shared by all sizes (`reliability.bootstrap_prompts` over the models of
    every size at once). `ids[z]` names the rows of size z's matrices; they must be the same
    prompts in the same order for every size, or the differences are not paired.

    Keys are canonical, independent of the order of `ys`: `gap_<z>` per size and
    `diff_<a>_<b>` = gap(a) − gap(b) for a before b in TREND (e.g. `diff_150M_1B`). Each value
    is reliability's `{"estimate", "lo", "hi", "n_draws"}`. A size's gap is `gap_every_pair`
    (NaN if any MZ or U correction is undefined, on the full sample or on a draw), so
    `n_draws` < `n_boot` names exactly the draws on which some pair was undefined."""
    assert set(ys) == set(TREND) == set(ids) == set(info), \
        f"sizes {sorted(ys)} / ids {sorted(ids)} / info {sorted(info)} != {list(TREND)}"
    ref = list(ids[TREND[0]])
    for z in TREND:
        assert list(ids[z]) == ref, f"{z}: prompt ids differ from {TREND[0]}'s (order or set)"
        for m, y in ys[z].items():
            assert y.shape[0] == len(ref), f"{z}/{m}: {y.shape[0]} rows for {len(ref)} prompt ids"
    flat = {f"{z}/{m}": y for z in TREND for m, y in ys[z].items()}
    flat_info = {f"{z}/{m}": info[z][m] for z in TREND for m in ys[z]}

    def analyze(ys_, info_, method):
        return {z: gap_every_pair(reliability.twin_correlations(
            {m.split("/", 1)[1]: y for m, y in ys_.items() if m.startswith(f"{z}/")},
            {m.split("/", 1)[1]: i for m, i in info_.items() if m.startswith(f"{z}/")},
            method)) for z in TREND}

    def stat(g):
        out = {f"gap_{z}": g[z] for z in TREND}
        for i, a in enumerate(TREND):
            for b in TREND[i + 1:]:
                out[f"diff_{a}_{b}"] = g[a] - g[b]
        return out

    return reliability.bootstrap_prompts(flat, flat_info, n_boot, seed, alpha, stat=stat,
                                         analyze=analyze) | {"n_boot": n_boot, "seed": seed}


VERDICTS = ("prediction holds", "prediction fails", "undecided", "INCOMPLETE")


def trend_rule(boot: dict, passed: dict[str, bool], *, labels_complete: bool,
               anchor_checked: bool) -> dict:
    """The pre-registered rule: gap(150M) > gap(300M) > gap(1B) as point estimates AND
    the paired prompt-bootstrap 95 % interval of gap(150M) − gap(1B) above 0 (`lo > 0`: the
    prediction is directional, the gap shrinks with size, so an interval entirely below 0 is
    evidence against it). Only sizes whose gate passed enter the trend; if any size is out, the
    prediction cannot hold as stated.

    `verdict` is one of VERDICTS, `reason` says why. "prediction fails" is given without the
    bootstrap when a clause is decided against the prediction on its own: a size out of the
    trend, or finite point estimates out of order. The interval decides ("holds" / "fails")
    only when it is the pre-registered one: every point estimate finite and every one of the
    `n_boot` draws finite at every size (`gap_every_pair`; `reliability._percentile_ci` drops
    the others, so every `gap_<z>` and `diff_150M_1B` must have `n_draws` == `n_boot` > 0 and
    `lo` must be finite). Anything else is "undecided" (`holds` None): never a verdict either
    way from an undefined correction or a partial bootstrap. `labels_complete` False
    (signal_016 --allow-incomplete: a twin or a label cell missing) gives "INCOMPLETE",
    `holds` None, in the same shape; so does `anchor_checked` False (the 1B anchor's signal
    file absent or unverified, `anchor_block`): a decisive verdict needs the recomputed 1B
    gap confirmed against the file of record. Both are keyword-only without a default: a
    caller states what it verified, nothing vouches for an unchecked anchor silently. A
    missing key raises (KeyError).

    `ordered` reports the order test on its own (finite point estimates in the predicted
    order), whatever the gates said; `complete` and `sizes_in_trend` carry the gates."""
    in_trend = [z for z in TREND if passed.get(z)]
    complete = in_trend == list(TREND)
    g = {z: float(boot[f"gap_{z}"]["estimate"]) for z in TREND}
    undefined = [z for z in TREND if not np.isfinite(g[z])]
    ordered = not undefined and g["150M"] > g["300M"] > g["1B"]
    d = boot["diff_150M_1B"]
    n_draws, n_boot = int(d["n_draws"]), int(boot["n_boot"])
    n_draws_size = {z: int(boot[f"gap_{z}"]["n_draws"]) for z in TREND}
    partial = [z for z in TREND if n_draws_size[z] != n_boot]
    lo = float(d["lo"])
    full = n_boot > 0 and not partial and n_draws == n_boot and bool(np.isfinite(lo))
    above = bool(np.isfinite(lo) and lo > 0)
    if not labels_complete:
        holds, verdict = None, "INCOMPLETE"
        reason = "not every twin labelled on every (prompt, sample) cell"
    elif not anchor_checked:
        holds, verdict = None, "INCOMPLETE"
        reason = "the 1B anchor signal file is absent or its cross-check did not run " \
                 "(anchor_checked false; the anchor entry's reason says why)"
    elif not complete:
        holds, verdict = False, "prediction fails"
        reason = f"gate failed at {[z for z in TREND if z not in in_trend]}: out of the trend"
    elif undefined:
        holds, verdict = None, "undecided"
        reason = f"gap point estimate undefined at {undefined} (an MZ or U pair missing or " \
                 "its correction undefined)"
    elif not ordered:
        holds, verdict = False, "prediction fails"
        reason = "point estimates not ordered gap(150M) > gap(300M) > gap(1B)"
    elif not full:
        holds, verdict = None, "undecided"
        reason = f"{n_draws}/{n_boot} bootstrap draws finite for gap(150M) - gap(1B), " \
                 f"{n_draws_size} per size (a correction undefined on the rest): not the " \
                 "pre-registered interval"
    else:
        holds, verdict = above, "prediction holds" if above else "prediction fails"
        reason = "ordered and the CI of gap(150M) - gap(1B) lies above 0" if above else \
            "ordered but the CI of gap(150M) - gap(1B) does not lie above 0"
    return {"holds": holds, "verdict": verdict, "reason": reason, "ordered": bool(ordered),
            "undefined_point_estimates": undefined, "ci_150M_1B_above_0": above,
            "n_draws": n_draws, "n_draws_per_size": n_draws_size, "n_boot": n_boot,
            "full_bootstrap": full,
            "sizes_in_trend": in_trend, "complete": complete, "labels_complete": labels_complete,
            "anchor_checked": anchor_checked,
            "rule": "gap(150M) > gap(300M) > gap(1B) (point estimates) and the paired "
                    "prompt-bootstrap 95% CI of gap(150M) - gap(1B) excludes 0 on the "
                    "predicted side (lo > 0), every MZ and U correction defined on the full "
                    "sample and on every bootstrap draw; gap = corrected r_MZ - r_U, "
                    "adv_harmful x wg_vulnerable"}
