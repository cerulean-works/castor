"""Cross-twin ablation transfer and its pre-registered rule.

Twin A's refusal direction is removed from twin B by weight orthogonalization (Arditi et al.
2024, 2406.11717 Eq. 5), either as is or transported by an orthogonal Procrustes map fitted on
paired activations of the two twins (Schoenemann 1966; the transport of Bosco et al. 2026,
2609.04721). Pure numpy; the torch parts live in scripts/transfer_018.py.

- **Donors** (deterministic): for target (base b, post-training seed k), the same-base donor is
  (b, k mod 4 + 1) and the different-seed donor is (next base in default -> large-aux-2 ->
  large-aux-3 -> default, k). Every twin is a donor exactly once per arm.
- **Procrustes**: R = argmin_{R^T R = I} ||X_A R - X_B||_F = U V^T with U S V^T = X_A^T X_B, on
  mean-centred rows (one row per paired token); a row vector moves as d -> d R.
- **Mismatched control**: the same fit with prompts paired to other prompts (a derangement
  from default_rng(2028)); within a pair, the tokens are paired by independent random
  position permutations over the shorter length (Bosco et al. App. A.7: "source and target
  activations from different prompts"). Aligning the two prompts from the end instead pairs
  the shared chat-template tail id for id in every pair (34 % of the rows in the fit set),
  which leaves the control with a real component of the map.
- **Standardized fit** (Bosco et al. App. A.9, the `diff-rot-std` arm): the same
  Procrustes on rows divided by their per-dimension std; a difference of means standardizes
  as d / sigma_A, moves as a row vector, and de-standardizes as * sigma_B.
- **Transfer ratio** T = (P[arm] - P[base]) / (P[own] - P[base]), P = mean per-prompt
  vulnerability; bootstrap over prompts resampled jointly across arms.
"""

from collections.abc import Sequence

import numpy as np

from castor.directions import unit

BASES = ("default", "large-aux-2", "large-aux-3")
N_SEEDS = 4
ARMS = ("same-unrot", "diff-unrot", "diff-rot", "diff-mismatch", "diff-rot-std")
RULE_ARMS = ARMS[:4]  # diff-rot-std is a reported secondary with no rule
DONOR_OF_ARM = {"same-unrot": "same", "diff-unrot": "diff", "diff-rot": "diff",
                "diff-mismatch": "diff", "diff-rot-std": "diff"}
MAP_OF_ARM = {"same-unrot": None, "diff-unrot": None, "diff-rot": "matched",
              "diff-mismatch": "mismatched", "diff-rot-std": "standardized"}
SEED = 2028

# pre-registered rule
SAME_MIN, DIFF_UNROT_MAX, ROT_MIN, ROT_MARGIN = 0.8, 0.2, 0.5, 0.3
N_SANITY, N_GO = 10, 8


def model_id(base: str, k: int) -> str:
    return f"dd-dclm-1b-{base}-dpo-seed{k}"


def parse_id(mid: str) -> tuple[str, int]:
    head, k = mid.removeprefix("dd-dclm-1b-").rsplit("-dpo-seed", 1)
    assert head in BASES and 1 <= int(k) <= N_SEEDS, mid
    return head, int(k)


def donors(target: str) -> dict[str, str]:
    """{"same": same-base donor, "diff": different-seed donor} of a target twin."""
    b, k = parse_id(target)
    return {"same": model_id(b, k % N_SEEDS + 1),
            "diff": model_id(BASES[(BASES.index(b) + 1) % len(BASES)], k)}


def targets() -> list[str]:
    return [model_id(b, k) for b in BASES for k in range(1, N_SEEDS + 1)]


# ---- Procrustes -----------------------------------------------------------------------------


def centred(xa: np.ndarray, xb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both matrices in float64 with their column means removed (per model)."""
    a = np.asarray(xa, np.float64)
    b = np.asarray(xb, np.float64)
    return a - a.mean(0), b - b.mean(0)


def procrustes(xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    """(n, d), (n, d) paired rows -> orthogonal (d, d) R minimising ||(xa - mean) R -
    (xb - mean)||_F. Rows are centred per model; the cross-covariance and SVD in float64."""
    a, b = centred(xa, xb)
    u, _, vt = np.linalg.svd(a.T @ b)
    return u @ vt


def fit_residual(xa: np.ndarray, xb: np.ndarray, r: np.ndarray) -> float:
    """||(xa - mean) R - (xb - mean)||_F / ||xb - mean||_F."""
    a, b = centred(xa, xb)
    return float(np.linalg.norm(a @ r - b) / np.linalg.norm(b))


def standardized_procrustes(xa: np.ndarray, xb: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R fitted on per-dimension standardized rows (Bosco et al. App. A.9), with the two
    std vectors (sigma_A, sigma_B) that transport_standardized needs."""
    a, b = centred(xa, xb)
    sa, sb = a.std(0), b.std(0)
    assert sa.min() > 0 and sb.min() > 0, "a zero-variance dimension cannot be standardized"
    return procrustes(a / sa, b / sb), sa, sb


def transport_standardized(d: np.ndarray, r: np.ndarray, sa: np.ndarray, sb: np.ndarray
                           ) -> np.ndarray:
    """unit(((unit(d) / sigma_A) R) * sigma_B): the direction moved through the
    standardization on both sides."""
    return unit(((unit(d) / sa) @ r) * sb)


def derangement(n: int, seed: int = SEED) -> np.ndarray:
    """partner[i] != i for every i: a random cycle through a default_rng(seed) permutation."""
    perm = np.random.default_rng(seed).permutation(n)
    partner = np.empty(n, dtype=int)
    partner[perm] = np.roll(perm, -1)
    return partner


def token_rows(lengths: Sequence[int], skip_first: int = 1) -> list[np.ndarray]:
    """Row indices of each prompt's tokens in the concatenated activation matrix, without its
    first `skip_first` tokens (BOS)."""
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return [np.arange(s + skip_first, s + n) for s, n in zip(starts, lengths)]


def matched_rows(lengths: Sequence[int], skip_first: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Row indices pairing every token of a prompt in A with the same token in B (one tokenizer,
    so positions pair exactly)."""
    r = np.concatenate(token_rows(lengths, skip_first))
    return r, r


def mismatched_rows(lengths: Sequence[int], partner: np.ndarray, skip_first: int = 1,
                    seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Prompt i of A paired with prompt partner[i] of B: m = min(len_i, len_j) tokens of each,
    drawn by independent random position permutations from default_rng(seed), so no
    position (and no shared template token) pairs with itself by construction. BOS excluded
    on both sides."""
    rows = token_rows(lengths, skip_first)
    rng = np.random.default_rng(seed)
    ia, ib = [], []
    for i, j in enumerate(partner):
        m = min(len(rows[i]), len(rows[j]))
        if m:
            ia.append(rng.permutation(rows[i])[:m])
            ib.append(rng.permutation(rows[j])[:m])
    return np.concatenate(ia), np.concatenate(ib)


def template_span(ids: Sequence[Sequence[int]]) -> tuple[int, int]:
    """(n_prefix, n_suffix): the token prefix (BOS included) and suffix shared by every
    prompt, i.e. the chat template around the content."""
    first = list(ids[0])
    shortest = min(len(x) for x in ids)
    pre = next((k for k in range(shortest) if any(x[k] != first[k] for x in ids)), shortest)
    suf = next((k for k in range(shortest - pre)
                if any(x[-1 - k] != first[-1 - k] for x in ids)), shortest - pre)
    return pre, suf


def content_rows(lengths: Sequence[int], n_prefix: int, n_suffix: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Matched rows of the content tokens only: every token outside the shared template
    prefix (BOS included) and suffix."""
    rows = token_rows(lengths, n_prefix)
    r = np.concatenate([x[:len(x) - n_suffix] for x in rows])
    return r, r


def transport(d: np.ndarray, r: np.ndarray | None) -> np.ndarray:
    """Unit direction moved into the target's space: unit(unit(d) R), or unit(d) unrotated."""
    d = unit(d)
    return d if r is None else unit(d @ r)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(unit(a) @ unit(b))


# ---- transfer ratio and the rule ------------------------------------------------------------


def ratio(p_arm: float, p_base: float, p_own: float) -> float:
    den = p_own - p_base
    return float((p_arm - p_base) / den) if den != 0 else float("nan")


def bootstrap_ratio(pp: dict[str, np.ndarray], arms: Sequence[str], n_boot: int = 1000,
                    seed: int = SEED) -> dict[str, dict]:
    """pp: per-prompt P (same prompt order) for "base", "own" and each arm. Prompts are
    resampled jointly across arms; NaN prompts are skipped inside each mean (sanityb.model_stats
    takes np.nanmean). Returns point estimate and 95 % percentile interval per arm."""
    n = len(pp["base"])
    assert all(len(v) == n for v in pp.values())
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    m = {k: np.nanmean(np.asarray(v, float)[idx], axis=1) for k, v in pp.items()}
    out = {}
    for a in arms:
        den = m["own"] - m["base"]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (m[a] - m["base"]) / den
        point = ratio(np.nanmean(pp[a]), np.nanmean(pp["base"]), np.nanmean(pp["own"]))
        lo, hi = np.nanpercentile(t, [2.5, 97.5])
        out[a] = {"T": point, "ci": [float(lo), float(hi)], "n_boot": n_boot}
    return out


def verdict(t: dict[str, dict[str, float]]) -> dict:
    """t[target][arm] = point T. The pre-registered rule: sanity arms same-unrot >= 0.8 and
    diff-unrot <= 0.2 in >= 10 of 12 each; GO if diff-rot >= 0.5 and diff-rot - diff-mismatch
    >= 0.3 in >= 8 of 12."""
    tg = sorted(t)
    same = [m for m in tg if t[m]["same-unrot"] >= SAME_MIN]
    unrot = [m for m in tg if t[m]["diff-unrot"] <= DIFF_UNROT_MAX]
    rot = [m for m in tg if t[m]["diff-rot"] >= ROT_MIN
           and t[m]["diff-rot"] - t[m]["diff-mismatch"] >= ROT_MARGIN]
    rot_only = [m for m in tg if t[m]["diff-rot"] >= ROT_MIN]
    sanity = len(same) >= N_SANITY and len(unrot) >= N_SANITY
    if not sanity:
        reading = "inconclusive: a sanity arm failed"
    elif len(rot) >= N_GO:
        reading = "GO: the same object across pretraining seeds, up to a rotation"
    else:
        reading = "not transportable by a rotation at this size"
    return {"n_targets": len(tg), "same_unrot_pass": same, "diff_unrot_pass": unrot,
            "diff_rot_pass": rot, "diff_rot_ge_half": rot_only,
            "sanity_same": len(same) >= N_SANITY, "sanity_diff_unrot": len(unrot) >= N_SANITY,
            "go": bool(sanity and len(rot) >= N_GO), "reading": reading,
            "predictions": {"same_unrot_ge_0.8_in_10": len(same) >= N_SANITY,
                            "diff_unrot_le_0.2_in_10": len(unrot) >= N_SANITY,
                            "diff_rot_ge_0.5_majority": len(rot_only) > len(tg) / 2}}
