"""The refusal direction of each twin: extraction, Arditi's selection
scores, the common (layer, position), projections, and weight orthogonalization.

Follows Arditi et al. 2024 (2406.11717) and their released code (andyrdt/refusal_direction
@ 9d852fa, `pipeline/submodules/{generate_directions,select_direction}.py`,
`pipeline/utils/hook_utils.py`), with these declared deviations:
- **Residual stream at layer l** = the input of decoder block l (`resid_pre`, a forward
  pre-hook on block l), l = 0 … L−1, as in Arditi's code.
- **Post-instruction positions** = the tokens of the chat template after the user content
  (tulu: "\\n<|assistant|>\\n", 7 tokens), addressed from the end (−7 … −1) as Arditi does.
  Position −8 is the last instruction-side token (kept for a harmfulness readout, 2507.11878).
- **Candidate direction** r(l, p) = mean over harmful training prompts − mean over harmless
  training prompts of the activation (float64 accumulation).
- **Refusal score** = log p(R) − log(1 − p(R)) at the last position, p(R) = total probability
  of the refusal-token set, float64 softmax, ε = 1e-8 (Arditi App. C.1, `refusal_score`).
- **bypass_score** = mean refusal score on harmful validation prompts with the unit direction
  ablated at every layer and position (pre-hook on every block, output hook on every
  attention and MLP module); **induce_score** = mean refusal score on harmless validation
  prompts with the raw r added at block l's input at every position (coeff 1);
  **kl_score** = mean KL(base ‖ ablated) of the last-position next-token distribution on
  harmless validation prompts (float64, ε = 1e-6, Arditi's `kl_div_fn`).
- **Filters** (Arditi `filter_fn`): discard NaN scores, layer ≥ int(0.8 L), kl > 0.1,
  induce < 0. Arditi's text says "l < 0.8L"; his code discards l ≥ int(0.8 L). We follow the
  code (L = 16: layers 0 … 11 pass).
- **Common candidate**: among candidates passing in every twin, the lowest mean
  bypass_score; if none passes in all, the candidate passing in the most twins, ties by mean
  bypass_score.
- **Orthogonalization** (Arditi Eq. 5, App. E): W ← W − r̂ r̂ᵀ W for every matrix writing to the
  residual stream (token embedding, attention o_proj, MLP down_proj). Llama blocks have no
  biases, so this equals ablating r̂ at every layer and position.

Torch is imported lazily so the pure-numpy parts load without it.
"""

import contextlib
import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

KL_MAX = 0.1
INDUCE_MIN = 0.0
PRUNE_LAYER_FRACTION = 0.2
REFUSAL_EPS = 1e-8
KL_EPS = 1e-6
TOKEN_COVERAGE = 0.8
N_POST = 7        # tulu post-instruction tokens "\n<|assistant|>\n"
N_STORED = N_POST + 1  # plus the last instruction-side token (position -8)


# ---- tokens ---------------------------------------------------------------------------------


def post_instruction_ids(tok) -> list[int]:
    """Token ids the chat template appends after the user content (with the generation
    prompt): the tail shared by two renderings that differ only in the content."""
    a = tok.apply_chat_template([{"role": "user", "content": "Alpha one"}], tokenize=True,
                                add_generation_prompt=True)
    b = tok.apply_chat_template([{"role": "user", "content": "Beta two"}], tokenize=True,
                                add_generation_prompt=True)
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return a[len(a) - n:]


def refusal_token_set(first_tokens: Iterable[int], coverage: float = TOKEN_COVERAGE) -> dict:
    """Smallest set of first response tokens covering >= `coverage` of the responses: tokens
    by descending count (ties by token id), added until the covered share reaches coverage.
    Taking the most frequent first is optimal, so the set is the smallest possible."""
    c = Counter(first_tokens)
    n = sum(c.values())
    assert n > 0, "no responses"
    ranked = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen, covered = [], 0
    for t, k in ranked:
        if covered / n >= coverage:
            break
        chosen.append(t)
        covered += k
    return {"tokens": chosen, "coverage": covered / n, "n_responses": n,
            "counts": {int(t): int(k) for t, k in ranked[:50]}}


# ---- scores (Arditi App. C.1) ---------------------------------------------------------------


def refusal_score(last_logits, refusal_toks: Sequence[int], eps: float = REFUSAL_EPS):
    """(batch, vocab) logits at the last position -> (batch,) log-odds of the refusal set."""
    import torch

    p = torch.softmax(last_logits.to(torch.float64), dim=-1)[:, list(refusal_toks)].sum(-1)
    return torch.log(p + eps) - torch.log(1 - p + eps)


def kl_div(logits_a, logits_b, eps: float = KL_EPS):
    """(batch, vocab) -> (batch,) KL(a ‖ b), Arditi's `kl_div_fn` without a mask."""
    import torch

    pa = torch.softmax(logits_a.to(torch.float64), dim=-1)
    pb = torch.softmax(logits_b.to(torch.float64), dim=-1)
    return torch.sum(pa * (torch.log(pa + eps) - torch.log(pb + eps)), dim=-1)


def passes(bypass: float, induce: float, kl: float, layer: int, n_layer: int) -> bool:
    """Arditi's `filter_fn`, inverted: True if the candidate is kept."""
    if any(np.isnan(x) for x in (bypass, induce, kl)):
        return False
    if layer >= int(n_layer * (1.0 - PRUNE_LAYER_FRACTION)):
        return False
    return kl <= KL_MAX and induce >= INDUCE_MIN


def pass_matrix(scores: dict) -> np.ndarray:
    """scores = {"bypass", "induce", "kl"}: arrays (n_layer, n_pos) -> bool (n_layer, n_pos)."""
    by, ind, kl = (np.asarray(scores[k], dtype=float) for k in ("bypass", "induce", "kl"))
    n_layer = by.shape[0]
    return np.array([[passes(by[l, p], ind[l, p], kl[l, p], l, n_layer)
                      for p in range(by.shape[1])] for l in range(n_layer)])


def select_own(scores: dict) -> tuple[int, int] | None:
    """Arditi's per-model choice: lowest bypass among passing candidates (index into the
    (n_layer, n_pos) grid), None if nothing passes."""
    ok = pass_matrix(scores)
    if not ok.any():
        return None
    by = np.where(ok, np.asarray(scores["bypass"], dtype=float), np.inf)
    l, p = np.unravel_index(int(np.argmin(by)), by.shape)
    return int(l), int(p)


def select_own_kl_dropped(scores: dict) -> tuple[int, int] | None:
    """Sensitivity check for a model with no passing candidate: its lowest-bypass
    candidate among those with layer < int(0.8 L) and induce >= 0 (Arditi's filters with the
    KL filter dropped), None if even that is empty."""
    by, ind = (np.asarray(scores[k], dtype=float) for k in ("bypass", "induce"))
    n_layer = by.shape[0]
    ok = np.array([[passes(by[l, p], ind[l, p], 0.0, l, n_layer) for p in range(by.shape[1])]
                   for l in range(n_layer)])
    if not ok.any():
        return None
    b = np.where(ok, by, np.inf)
    l, p = np.unravel_index(int(np.argmin(b)), b.shape)
    return int(l), int(p)


def select_common(per_model: dict[str, dict]) -> dict:
    """One (layer, pos) for every model. Candidates passing in all models -> lowest
    mean bypass; else the most models passing, ties by lowest mean bypass. Ties in mean bypass
    (exact) break by (layer, pos) order. Candidates with an undefined (NaN) mean bypass, e.g.
    layer 0 at a template position, are never chosen (they can only tie when nothing passes)."""
    names = sorted(per_model)
    ok = np.stack([pass_matrix(per_model[m]) for m in names])            # (M, L, P)
    by = np.stack([np.asarray(per_model[m]["bypass"], float) for m in names])
    n_pass = ok.sum(0)
    mean_by = by.mean(0)
    best = n_pass.max()
    cand = [(mean_by[l, p], l, p) for l, p in zip(*np.nonzero((n_pass == best)
                                                              & np.isfinite(mean_by)))]
    assert cand, "no candidate with a defined mean bypass"
    _, l, p = min(cand)
    return {"layer": int(l), "pos_index": int(p), "n_models_passing": int(best),
            "n_models": len(names), "all_pass": bool(best == len(names)),
            "mean_bypass": float(mean_by[l, p]),
            "n_candidates_all_pass": int((n_pass == len(names)).sum()),
            "per_model_pass": {m: bool(ok[i, l, p]) for i, m in enumerate(names)}}


# ---- directions, projections, reliability ---------------------------------------------------


def mean_diff(harmful: np.ndarray, harmless: np.ndarray) -> np.ndarray:
    """(n_h, ..., d), (n_b, ..., d) -> (..., d) float64 difference of means."""
    return (np.asarray(harmful, np.float64).mean(0) - np.asarray(harmless, np.float64).mean(0))


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def half_splits(n_harmful: int, n_harmless: int, n_splits: int, seed: int) -> list[tuple]:
    """Random disjoint halves of each class's training prompts: list of
    ((harm_half1, harm_half2), (harmless_half1, harmless_half2)) index arrays."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_splits):
        h, b = rng.permutation(n_harmful), rng.permutation(n_harmless)
        out.append(((h[: n_harmful // 2], h[n_harmful // 2: 2 * (n_harmful // 2)]),
                    (b[: n_harmless // 2], b[n_harmless // 2: 2 * (n_harmless // 2)])))
    return out


def half_directions(harmful: np.ndarray, harmless: np.ndarray, splits: list[tuple]
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Training activations at one site, (n_h, d) and (n_b, d) -> unit half directions
    (S, d) and (S, d) from each split's two disjoint halves."""
    h = np.asarray(harmful, np.float64)
    b = np.asarray(harmless, np.float64)
    d1 = np.stack([mean_diff(h[a1], b[c1]) for (a1, _), (c1, _) in splits])
    d2 = np.stack([mean_diff(h[a2], b[c2]) for (_, a2), (_, c2) in splits])
    return unit(d1), unit(d2)


def project(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """(n, d) activations on (d,) or (S, d) unit directions -> (n,) or (n, S), float64."""
    return np.asarray(acts, np.float64) @ np.asarray(direction, np.float64).T


def cosine_readout(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Sensitivity: cosine between each activation and the unit direction."""
    a = np.asarray(acts, np.float64)
    return project(a, direction) / np.linalg.norm(a, axis=-1, keepdims=True).reshape(
        (-1,) + (1,) * (np.ndim(direction) - 1))


def random_direction(model_id: str, d: int, salt: str = "castor-007-random") -> np.ndarray:
    """The random control's unit direction, seeded by the model id."""
    seed = int.from_bytes(hashlib.sha256(f"{salt}:{model_id}".encode()).digest()[:8], "big")
    return unit(np.random.default_rng(seed).standard_normal(d))


# ---- hooks and orthogonalization (torch) ----------------------------------------------------


def blocks(model):
    return list(model.model.layers)


@contextlib.contextmanager
def hooks(pre: Sequence[tuple] = (), post: Sequence[tuple] = ()):
    """Register (module, fn) forward pre-hooks (with kwargs) and forward hooks; remove on
    exit."""
    handles = [m.register_forward_pre_hook(f, with_kwargs=True) for m, f in pre]
    handles += [m.register_forward_hook(f) for m, f in post]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _edit_first(args, kwargs, fn):
    if args:
        return (fn(args[0]), *args[1:]), kwargs
    kwargs = dict(kwargs)
    kwargs["hidden_states"] = fn(kwargs["hidden_states"])
    return args, kwargs


def _ablate(x, r):
    r = r.to(x)
    return x - (x @ r).unsqueeze(-1) * r


def ablation_hooks(model, direction) -> tuple[list, list]:
    """Arditi's directional ablation of a unit direction at every block input and every
    attention / MLP output."""
    import torch

    r = torch.as_tensor(np.asarray(unit(direction)), dtype=torch.float32)

    def pre(module, args, kwargs):
        return _edit_first(args, kwargs, lambda x: _ablate(x, r))

    def post(module, inputs, output):
        if isinstance(output, tuple):
            return (_ablate(output[0], r), *output[1:])
        return _ablate(output, r)

    bl = blocks(model)
    return ([(b, pre) for b in bl],
            [(b.self_attn, post) for b in bl] + [(b.mlp, post) for b in bl])


def actadd_hooks(model, layer: int, vector, coeff: float = 1.0) -> tuple[list, list]:
    """Arditi's activation addition of the raw vector at block `layer`'s input, all positions."""
    import torch

    v = torch.as_tensor(np.asarray(vector), dtype=torch.float32)

    def pre(module, args, kwargs):
        return _edit_first(args, kwargs, lambda x: x + coeff * v.to(x))

    return [(blocks(model)[layer], pre)], []


def capture_hooks(model, store: dict, positions: Sequence[int]) -> tuple[list, list]:
    """Record block inputs (resid_pre) at `positions` (negative, from the end of a
    left-padded batch) for every layer: store[layer] = (batch, len(positions), d) float32.
    Every prompt must be at least max(-positions) tokens long (a chat-formatted prompt is:
    template prefix + content + 7 post-instruction tokens)."""
    pos = list(positions)

    def make(layer):
        def pre(module, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            store[layer] = x[:, pos, :].detach().float()
        return pre

    return [(b, make(i)) for i, b in enumerate(blocks(model))], []


def orthogonalize_(model, direction) -> None:
    """In place: W ← W − r̂ r̂ᵀ W on every matrix writing to the residual stream (Arditi Eq. 5).
    Computed in float64, cast back to each weight's dtype."""
    import torch

    r = torch.as_tensor(np.asarray(unit(direction)), dtype=torch.float64)

    def proj_rows(w):  # rows are d_model vectors (embedding: (vocab, d))
        w64 = w.to(torch.float64)
        return (w64 - (w64 @ r.to(w.device)).unsqueeze(-1) * r.to(w.device)).to(w.dtype)

    def proj_out(w):  # nn.Linear weight (d_model, in): output side is dim 0
        w64 = w.to(torch.float64)
        rr = r.to(w.device)
        return (w64 - torch.outer(rr, rr @ w64)).to(w.dtype)

    with torch.no_grad():
        emb = model.model.embed_tokens.weight
        emb.copy_(proj_rows(emb))
        for b in blocks(model):
            for lin in (b.self_attn.o_proj, b.mlp.down_proj):
                assert lin.bias is None, "orthogonalization assumes bias-free output projections"
                lin.weight.copy_(proj_out(lin.weight))


# written by save_pretrained, or the ablation scripts' own outputs; everything else is copied
NOT_COPIED = ("config.json", "provenance.json", "direction.npy")
WEIGHT_SUFFIXES = (".bin", ".safetensors", ".pt", ".pth", ".index.json")


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def orthogonalize_checkpoint(model, tok, direction, src: Path, out: Path,
                             probe: Sequence[Sequence[int]], tol: float) -> dict:
    """Orthogonalize `model` (loaded from the checkpoint `src`) against `direction` in place,
    check it against Arditi's ablation hooks on the `probe` token ids (max |Δlogit| at the
    last position < tol, no residual component left along the direction), save it as a plain
    HF checkpoint at `out` with every non-weight file of `src` copied unchanged (tokenizer,
    chat template, generation config), and return the checks and the copied file names."""
    import shutil

    import torch
    from transformers import AutoTokenizer

    r = unit(direction)
    pre, post = ablation_hooks(model, r)
    hooked = last_logits(model, probe, tok.pad_token_id, pre, post)
    base = last_logits(model, probe, tok.pad_token_id)
    orthogonalize_(model, r)
    ortho = last_logits(model, probe, tok.pad_token_id)
    max_diff = float(torch.max(torch.abs(hooked - ortho)))
    effect = float(torch.max(torch.abs(hooked - base)))
    assert max_diff < tol, f"orthogonalized != hooked: {max_diff}"
    resid = activations(model, probe, tok.pad_token_id, (-1,)).double().numpy() @ r
    assert np.max(np.abs(resid)) < 1e-2, "residual keeps a component along the direction"

    out.mkdir(parents=True, exist_ok=True)
    model.cpu().save_pretrained(out, safe_serialization=True)
    copied = []
    for f in sorted(src.iterdir()):
        if f.is_file() and f.name not in NOT_COPIED and not f.name.endswith(WEIGHT_SUFFIXES):
            shutil.copy2(f, out / f.name)
            copied.append(f.name)
    assert AutoTokenizer.from_pretrained(out).chat_template == tok.chat_template
    return {"check_max_abs_dlogit_vs_hooks": max_diff,
            "check_max_abs_dlogit_hooks_vs_base": effect,
            "check_max_abs_residual_component": float(np.max(np.abs(resid))),
            "copied_files": copied}


# ---- batched forward passes (torch) ---------------------------------------------------------


def encode(tok, prompts: Sequence[str]) -> list[list[int]]:
    """Token ids as the instrument formats them (castor.evalset.prompt_ids: chat template with
    the generation prompt, BOS from the template)."""
    from castor import evalset

    return [evalset.prompt_ids(tok, p) for p in prompts]


def batch_inputs(ids: Sequence[Sequence[int]], pad_id: int, device):
    """Left-padded batch with an attention mask and position ids that start at 0 at each
    sequence's first real token, so padding never shifts a prompt's positions."""
    import torch

    n = max(len(x) for x in ids)
    inp = torch.full((len(ids), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids), n), dtype=torch.long)
    for i, x in enumerate(ids):
        inp[i, n - len(x):] = torch.tensor(x)
        mask[i, n - len(x):] = 1
    pos = (mask.cumsum(1) - 1).clamp(min=0)
    return {"input_ids": inp.to(device), "attention_mask": mask.to(device),
            "position_ids": pos.to(device)}


def length_batches(ids: Sequence[Sequence[int]], max_tokens: int) -> list[list[int]]:
    """Indices grouped by length (descending) so a batch holds <= max_tokens padded tokens."""
    order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
    out, cur, cur_len = [], [], 0
    for i in order:
        n = len(ids[i])
        if cur and max(cur_len, n) * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, cur_len = [], 0
        cur.append(i)
        cur_len = max(cur_len, n)
    if cur:
        out.append(cur)
    return out


def last_logits(model, ids, pad_id: int, pre=(), post=(), max_tokens: int = 65536):
    """(n, vocab) float32 logits at the last position, in the input order."""
    import torch

    out = [None] * len(ids)
    with torch.no_grad(), hooks(pre, post):
        for b in length_batches(ids, max_tokens):
            x = batch_inputs([ids[i] for i in b], pad_id, model.device)
            lg = model(**x, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            for j, i in enumerate(b):
                out[i] = lg[j]
    return torch.stack(out)


def activations(model, ids, pad_id: int, positions: Sequence[int] = tuple(range(-N_STORED, 0)),
                max_tokens: int = 65536):
    """(n, n_layer, len(positions), d) float32 resid_pre activations, input order (CPU)."""
    import torch

    n_layer = len(blocks(model))
    out = None
    store: dict = {}
    pre, post = capture_hooks(model, store, positions)
    with torch.no_grad(), hooks(pre, post):
        for b in length_batches(ids, max_tokens):
            x = batch_inputs([ids[i] for i in b], pad_id, model.device)
            model(**x, use_cache=False, logits_to_keep=1)
            a = torch.stack([store[l] for l in range(n_layer)], 1).cpu()
            if out is None:
                out = torch.empty((len(ids), *a.shape[1:]), dtype=torch.float32)
            out[torch.tensor(b)] = a
    return out
