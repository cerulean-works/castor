"""Castor-side helpers for the Tulu 3 SFT + DPO pipeline.

Training itself runs in open-instruct's own locked env (shell/posttrain/train.sh); nothing
here imports open-instruct. This module:
- exports the pinned Tulu 3 datasets to local jsonl, because open-instruct @ fc7748ea4
  hardcodes `dataset_revision="main"` for hub datasets (dataset_transformation.py:1213) but
  loads a local `.jsonl` verbatim (:955-961), which is how the revision is pinned;
- reproduces open-instruct's SFT tokenize-and-truncate label mask
  (`sft_tulu_tokenize_and_truncate_v1`, dataset_transformation.py:682-747, and the filter
  `sft_tulu_filter_v1`, :750-751) to count how many rows a given max length truncates or drops;
- holds the chat template: open-instruct's "tulu" template, taken from the released
  `allenai/OLMo-2-0425-1B-SFT` tokenizer (which is "{{ bos_token }}" + tulu, as open-instruct
  builds it with --add_bos, :520-528).
"""

import hashlib
import json
from pathlib import Path

TULU3_DIR = Path("/work/harrychang08/castor/data/tulu3")
SFT_REPO, SFT_SHA = "allenai/tulu-3-sft-olmo-2-mixture-0225", "d91a0785ade02942520280fb484866fce41e448f"
DPO_REPO, DPO_SHA = "allenai/olmo-2-0425-1b-preference-mix", "c6454f3d364622718a8ecd3d307e524febbba569"
TEMPLATE_REPO, TEMPLATE_SHA = "allenai/OLMo-2-0425-1B-SFT", "0d85a3d037876ce6ac7d4311d994400fc66ac27f"
BOS_PREFIX = "{{ bos_token }}"


def snapshot_dir(repo: str, sha: str) -> Path:
    return TULU3_DIR / f"{repo.split('/')[1]}@{sha[:12]}"


def jsonl_path(repo: str, sha: str) -> Path:
    return TULU3_DIR / "jsonl" / f"{repo.split('/')[1]}@{sha[:12]}.jsonl"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ordered_hash(rows) -> str:
    """sha256 over the row order and content (json with sorted keys, one per line)."""
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True, ensure_ascii=False).encode())
        h.update(b"\n")
    return h.hexdigest()


def tulu_template(bos: bool) -> str:
    """open-instruct's tulu chat template, from the released OLMo-2-1B-SFT tokenizer."""
    from huggingface_hub import hf_hub_download

    cfg = json.loads(Path(hf_hub_download(TEMPLATE_REPO, "tokenizer_config.json",
                                          revision=TEMPLATE_SHA)).read_text())
    t = cfg["chat_template"]
    assert t.startswith(BOS_PREFIX), "released SFT template no longer starts with bos_token"
    return t if bos else t[len(BOS_PREFIX):]


def _ids(tok, messages, gen_prompt: bool) -> list[int]:
    return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=gen_prompt)


def sft_labels(tok, messages: list[dict], max_len: int | None) -> tuple[int, list[bool]]:
    """(untruncated length, trainable mask of the truncated sequence), as open-instruct does.

    open-instruct masks each non-assistant message from the end of the preceding prefix to the
    end of that message (plus the "<|assistant|>\\n" generation prompt when an assistant turn
    follows), with every prefix tokenized under the same right truncation; truncating each
    prefix at max_len is the same as clipping its length, so the mask is computed untruncated
    and clipped.
    """
    full = _ids(tok, messages, False)
    n = len(full)
    keep = n if max_len is None else min(n, max_len)
    train = [True] * keep
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            continue
        start = 0 if i == 0 else len(_ids(tok, messages[:i], False))
        nxt_asst = i < len(messages) - 1 and messages[i + 1]["role"] == "assistant"
        end = len(_ids(tok, messages[: i + 1], nxt_asst))
        for j in range(min(start, keep), min(end, keep)):
            train[j] = False
        if max_len and end >= max_len:
            break
    return n, train


def length_stats(tok, rows, max_len: int) -> dict:
    """Rows truncated at max_len, and rows open-instruct drops (no trainable token left)."""
    n = trunc = dropped = 0
    lengths = []
    for r in rows:
        length, train = sft_labels(tok, r["messages"], max_len)
        n += 1
        lengths.append(length)
        trunc += length > max_len
        dropped += not any(train)
    lengths.sort()
    return {"max_len": max_len, "rows": n, "truncated": trunc, "dropped": dropped,
            "frac_truncated": trunc / n, "frac_dropped": dropped / n,
            "len_median": lengths[n // 2], "len_p99": lengths[int(0.99 * (n - 1))],
            "len_max": lengths[-1]}


# Post-training seed k -> open-instruct --seed. k = 1 reproduces Ai2's runs (SFT seed 1, DPO
# seed 111; open-instruct docs/olmo2.md @ 89b4f62a5); k = 2, 3, ... continue both series. The
# seed drives the dataset shuffle and the per-epoch sampler order only.
OI_SEED_BASE = {"sft": 0, "dpo": 110}
OI_ENTRY = {"sft": "open_instruct/finetune.py", "dpo": "open_instruct/dpo_tune_cache.py"}
OI_DS_CONFIG = {"sft": "configs/ds_configs/stage3_no_offloading_accelerate.conf",
                "dpo": "configs/ds_configs/stage2_accelerate.conf"}
OI_DIR = Path("/work/harrychang08/castor/third_party/open-instruct")
OI_COMMIT = "fc7748ea4e111d0ae3865e176cdf73227cca0c39"
REPO_ROOT = Path(__file__).resolve().parent.parent

# Launcher environment of every open-instruct run. The defaults are what
# shell/posttrain/train.sh has always exported; BEAKER_ASSIGNED_CPU_COUNT sets open-instruct's
# tokenization workers only (dataset_transformation.py, cached after the first run).
LAUNCH_ENV_DEFAULTS = {"TOKENIZERS_PARALLELISM": "false"}
# Settings a launcher change may touch; anything else is refused, so a
# recipe-relevant variable (e.g. a seed or an HF flag) cannot ride in as "environment".
LAUNCH_ENV_ALLOWED = {"BEAKER_ASSIGNED_CPU_COUNT", "TOKENIZERS_PARALLELISM", "NCCL_CUMEM_ENABLE",
                      "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_NVLS_ENABLE", "NCCL_P2P_LEVEL",
                      "OMP_NUM_THREADS"}


def ds_config_path(stage: str, ds_config: str | None = None) -> Path:
    """The DeepSpeed config a run uses: the stage default under open-instruct's tree
    (OI_DS_CONFIG, the current configuration) unless --ds-config names one explicitly. An
    explicit relative path is resolved against the castor repo (configs/posttrain/ds/...); an
    open-instruct-shipped file is passed by its absolute path."""
    if ds_config is None:
        return OI_DIR / OI_DS_CONFIG[stage]
    p = Path(ds_config)
    return p if p.is_absolute() else REPO_ROOT / p


def ds_config_record(p: Path) -> tuple[str, str | None]:
    """(path, tree) recorded in provenance for a DeepSpeed config: relative to the tree it lives
    in, so the same file is always recorded the same way (open-instruct's shipped configs as
    configs/ds_configs/...), and a job's snapshot worktree (its absolute REPO_ROOT,
    deleted when the job ends) never appears. Anything else is recorded absolute."""
    for tree, root in (("open-instruct", OI_DIR), ("castor", REPO_ROOT)):
        if p.is_relative_to(root):
            return str(p.relative_to(root)), tree
    return str(p), None


def launch_env(cpus: int, overrides: list[str] | None = None) -> dict[str, str]:
    """Environment exported to the open-instruct launch: defaults, the CPU count, then the
    explicit KEY=VALUE overrides (each must be in LAUNCH_ENV_ALLOWED)."""
    env = LAUNCH_ENV_DEFAULTS | {"BEAKER_ASSIGNED_CPU_COUNT": str(cpus)}
    for kv in overrides or []:
        # raise, not assert: the allow-list must hold under python -O too
        k, sep, v = kv.partition("=")
        if not sep or not k:
            raise ValueError(f"--launch-env {kv!r}: expected KEY=VALUE")
        if k not in LAUNCH_ENV_ALLOWED:
            raise ValueError(f"{k} is not a launcher setting (LAUNCH_ENV_ALLOWED)")
        env[k] = v
    return env


def oi_seed(stage: str, k: int) -> int:
    assert k >= 1
    return OI_SEED_BASE[stage] + k
