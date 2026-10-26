"""Projection transplant (089): a dose test of the refusal direction at one site, then a
per-prompt transplant of one run's projection scalar into another run.

Quantities (pre-registered in prompts/*/089-projection-transplant.md, D125):
- **Site** (layer, position): resid_pre at block `layer` (castor.directions), position counted
  from the end of the chat-formatted prompt (−1 = last prompt token). Main site (13, −1),
  comparison site (11, −2).
- **Direction** d̂_M = unit(mean_diff[layer, pos]) of model M (021's candidates.safetensors).
- **Projection** proj_M(p) = 021's stored panel activation of prompt p at the site · d̂_M;
  μ_M, σ_M over the adversarial-harmful prompts (population SD); z_M = (proj_M − μ_M) / σ_M.
- **Edit**: add c_p · d̂_M at the site position, every later prompt position, and every
  generated position; nothing before the site position. Dose arm: c_p = k σ_M for every p.
  Transplant arm (A edited, B donor): c_p = σ_A (z_B(p) − z_A(p)), so A′'s projection at the
  site becomes μ_A + σ_A z_B(p).
- **Permuted control**: z_B read through a derangement of the prompts (castor.transfer's,
  default_rng(2028)), which keeps B's marginal and breaks the per-prompt assignment.
- **Corrected correlation** r_c(a, b) = corr(P_a, P_b) / √(rel_a rel_b) (castor.reliability).
- **Closure** = (r_c(A′, B) − r_c(A, B)) / (r_c(A, A_MZ) − r_c(A, B)): the share of the gap
  between the pair and A's same-base ceiling that the edit closes.
- **Uncertainty**: percentile prompt bootstrap (rows resampled, reliabilities recomputed,
  the same draw for every model, arm and pair), seed 0, as castor.reliability.

Pure numpy; torch is imported lazily by the hook.
"""

from collections.abc import Callable, Sequence
from itertools import pairwise

import numpy as np

from castor import reliability as R
from castor import transfer

SITE = (13, -1)
COMPARE_SITE = (11, -2)
DOSES = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
GATE_DOSES = (-0.5, 0.5)
GATE_MIN = 0.05
IDENTITY_PREDICTED = 0.95
IDENTITY_REVIEW = 0.90
PERMUTE_SEED = transfer.SEED
N_BOOT = 1000
BOOT_SEED = 0
ALPHA = 0.05


# ---- site quantities ------------------------------------------------------------------------


def site_stats(proj: np.ndarray) -> tuple[float, float]:
    """(n,) projections -> (μ, σ), population SD."""
    p = np.asarray(proj, np.float64)
    return float(p.mean()), float(p.std())


def zscore(proj: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (np.asarray(proj, np.float64) - mu) / sigma


def dose_coeffs(n: int, k: float, sigma: float) -> np.ndarray:
    return np.full(n, k * sigma)


def transplant_coeffs(z_a: np.ndarray, z_b: np.ndarray, sigma_a: float) -> np.ndarray:
    """Per-prompt scalar added along d̂_A so that A's z-scored projection becomes B's."""
    return sigma_a * (np.asarray(z_b, np.float64) - np.asarray(z_a, np.float64))


def permuted(z: np.ndarray, seed: int = PERMUTE_SEED) -> np.ndarray:
    """z read through 018's derangement: entry i is z of another prompt."""
    z = np.asarray(z, np.float64)
    return z[transfer.derangement(len(z), seed)]


# ---- outcome statistics ---------------------------------------------------------------------


def corrected(ya: np.ndarray, yb: np.ndarray) -> dict:
    """Corrected correlation of two (n, K) outcome matrices on the same prompts."""
    pa, rel_a = R.outcome_measure(ya)
    pb, rel_b = R.outcome_measure(yb)
    r = R.corr(pa, pb)
    return {"r": r, "rel_a": rel_a, "rel_b": rel_b, "r_corrected": R.disattenuate(r, rel_a, rel_b)}


def mean_p(y: np.ndarray) -> float:
    """Mean over prompts of per-prompt P (sanityb's 'mean vulnerability')."""
    return float(np.nanmean(R.per_prompt_p(y)))


def closure_value(r_edit: float, r_pair: float, r_ceiling: float) -> float:
    return (r_edit - r_pair) / (r_ceiling - r_pair)


def _percentile(draws: list[dict], point: dict, alpha: float = ALPHA) -> dict:
    out = {}
    for k, v in point.items():
        if isinstance(v, dict):
            out[k] = _percentile([d[k] for d in draws], v, alpha)
            continue
        x = np.array([d[k] for d in draws], dtype=float)
        x = x[np.isfinite(x)]
        lo, hi = (np.quantile(x, [alpha / 2, 1 - alpha / 2]) if len(x)
                  else (np.nan, np.nan))
        out[k] = {"point": v, "lo": float(lo), "hi": float(hi), "n_finite_draws": len(x)}
    return out


def bootstrap(ys: dict[str, np.ndarray], stat: Callable[[dict], dict], n_boot: int = N_BOOT,
              seed: int = BOOT_SEED, alpha: float = ALPHA) -> dict:
    """Percentile CIs of every scalar in stat(ys) (nested dicts allowed), prompts resampled
    with the same rows for every matrix."""
    rng = np.random.default_rng(seed)
    n = next(iter(ys.values())).shape[0]
    assert all(y.shape[0] == n for y in ys.values()), "matrices on different prompts"
    point = stat(ys)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        draws.append(stat({m: y[idx] for m, y in ys.items()}))
    return _percentile(draws, point, alpha)


# ---- step 1: dose gate ----------------------------------------------------------------------


def dose_stat(doses: Sequence[float] = DOSES, lo: float = GATE_DOSES[0],
              hi: float = GATE_DOSES[1]) -> Callable[[dict], dict]:
    """ys = {"dose_<k>": Y} -> mean P per dose and P(lo) − P(hi)."""
    def stat(ys: dict) -> dict:
        m = {dose_key(k): mean_p(ys[dose_key(k)]) for k in doses}
        return {"mean_p": m, "gate_diff": m[dose_key(lo)] - m[dose_key(hi)]}
    return stat


def dose_key(k: float) -> str:
    return f"dose_{k:+g}"


def gate_verdict(mean_by_dose: dict[float, float], diff_ci: dict,
                 minimum: float = GATE_MIN) -> dict:
    """Pass iff the mean P strictly decreases across the doses in order and P(lo) − P(hi)
    ≥ minimum with its interval excluding 0."""
    ks = sorted(mean_by_dose)
    steps = [mean_by_dose[b] - mean_by_dose[a] for a, b in pairwise(ks)]
    monotone = all(s < 0 for s in steps)
    size = diff_ci["point"] >= minimum
    excl = diff_ci["lo"] > 0
    return {"monotone": monotone, "steps": steps, "diff_ge_min": size, "ci_excludes_0": excl,
            "pass": bool(monotone and size and excl)}


def identity_reading(r_c: float) -> str:
    if r_c >= IDENTITY_PREDICTED:
        return "as predicted (>= 0.95)"
    if r_c >= IDENTITY_REVIEW:
        return "below prediction, above the review line (0.90-0.95)"
    return "below 0.90: the hooked path differs from the sampler of record; review resolves"


# ---- step 2: closure ------------------------------------------------------------------------


def closure_stat(pairs: Sequence[dict], arms: Sequence[str]) -> Callable[[dict], dict]:
    """pairs: [{"name", "a", "b", "mz_a", "arms": {arm: key of A′ under that arm}}], where
    a, b, mz_a and the arm values are keys of ys. Returns per-pair and mean closure per arm,
    and the paired differences the predictions name."""
    def stat(ys: dict) -> dict:
        per = {}
        for p in pairs:
            r_ab = corrected(ys[p["a"]], ys[p["b"]])["r_corrected"]
            r_mz = corrected(ys[p["a"]], ys[p["mz_a"]])["r_corrected"]
            row = {"r_ab": r_ab, "r_mz": r_mz}
            for arm in arms:
                r_e = corrected(ys[p["arms"][arm]], ys[p["b"]])["r_corrected"]
                row[f"r_edit_{arm}"] = r_e
                row[f"closure_{arm}"] = closure_value(r_e, r_ab, r_mz)
            per[p["name"]] = row
        mean = {arm: float(np.mean([per[p["name"]][f"closure_{arm}"] for p in pairs]))
                for arm in arms}
        out = {"per_pair": per, "mean_closure": mean}
        diffs = {}
        if "transplant" in arms and "site11" in arms:
            diffs["transplant_minus_site11"] = mean["transplant"] - mean["site11"]
        if "transplant" in arms and "mzdonor" in arms:
            diffs["mzdonor_minus_transplant"] = mean["mzdonor"] - mean["transplant"]
        out["differences"] = diffs
        return out
    return stat


def predictions(ci: dict) -> dict:
    """The four fixed predictions of 089 on the bootstrap output of closure_stat."""
    m, d = ci["mean_closure"], ci["differences"]
    return {
        "i_transplant_gt_0": bool(m["transplant"]["point"] > 0 and m["transplant"]["lo"] > 0),
        "ii_permuted_le_0": bool(m["permuted"]["point"] <= 0 or m["permuted"]["lo"] <= 0),
        "iii_mzdonor_within_0.10": bool(abs(d["mzdonor_minus_transplant"]["point"]) <= 0.10),
        "iv_site11_below": bool(d["transplant_minus_site11"]["point"] > 0
                                and d["transplant_minus_site11"]["lo"] > 0),
    }


# ---- the edit (torch) -----------------------------------------------------------------------


class Addition:
    """Forward pre-hook on block `layer`: adds coeff[i] · direction to row i of the residual
    stream at the last `n_site` prompt positions (prefill) and at every generated position
    (each decode step's single position). Set `coeff` (batch,) before each generate call.
    Prefill vs decode is read from the layer's `cache_position` (prefill starts at 0)."""

    def __init__(self, direction, position: int):
        import torch

        assert position < 0, "site position counts from the end of the prompt"
        self.n_site = -position
        self.direction = torch.as_tensor(np.asarray(direction, np.float64), dtype=torch.float32)
        self.coeff = None
        self.calls = {"prefill": 0, "decode": 0}
        self._v = (None, None)

    def __call__(self, module, args, kwargs):
        from castor.directions import _edit_first

        cp = kwargs.get("cache_position")
        assert cp is not None, "the block received no cache_position"
        prefill = int(cp[0]) == 0
        return _edit_first(args, kwargs, lambda x: self._add(x, prefill))

    def _offset(self, device):
        import torch

        if self._v[0] is not self.coeff or self._v[1].device != device:
            c = self.coeff.to(device, torch.float32)
            v = (c[:, None] * self.direction.to(device)[None, :])[:, None, :]  # (B, 1, d)
            self._v = (self.coeff, v)
        return self._v[1]

    def _add(self, x, prefill: bool):
        v = self._offset(x.device)
        if prefill:
            assert x.shape[1] >= self.n_site
            self.calls["prefill"] += 1
            x = x.clone()
            x[:, -self.n_site:, :] = (x[:, -self.n_site:, :].float() + v).to(x.dtype)
            return x
        assert x.shape[1] == 1, f"decode step with {x.shape[1]} positions"
        self.calls["decode"] += 1
        return (x.float() + v).to(x.dtype)
