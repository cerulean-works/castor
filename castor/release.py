"""Release layout, model cards and label tables for the CASTOR models on Hugging Face.

One model repo per (recipe, size), one revision per model: `s<pretrain-seed>-p<posttrain-seed>`,
plus `-<variant>` for a grid post-trained off the reference recipe (080: `-swap080`).
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from castor import grid, ladder
from castor.localckpt import sha256_file  # noqa: F401  (re-exported for the release scripts)

ROOT = grid.ROOT
ORG = "cerulean-works"
LABELS_REPO = f"{ORG}/castor-labels"
CODE_URL = f"https://github.com/{ORG}/castor"
CARD_CONFIG = ROOT / "configs/release/card.yaml"

GRIDS = {"008": 156, "009": 8, "011": 4, "080": 12}
LADDER_N = 24
GRID_035 = ("035", 4)
# 080 replaced the SFT mixture's safety subsets; the swap's build report names every source
SWAP_MIX = {"swap080": ROOT / "runs/080/swap_mix.json"}
RERUN_SEED = {"replicate-i6198-d6198": "default-r1", "replicate2-i6198-d6198": "default-r2"}

# Only these columns are released; generations never are.
LABEL_COLUMNS = ("model", "prompt_id", "sample", "coherent", "coherence_rule", "llm_invalid",
                 "p_valid", "wg_label", "wg_partial", "wg_vulnerable", "wg_harmful",
                 "wg_vulnerable_incl", "q3g_label", "q3g_partial", "q3g_vulnerable",
                 "q3g_harmful", "q3g_vulnerable_incl", "refusal_agree")
# the one row judged on a response cut to fit the judges; named in the labels README
TRUNCATED = {("dd-dolma1_7-no-math-code-1b-large-aux-3-dpo-seed1", "wildjailbreak-00287", -1)}
COUNT_COLUMNS = ("coherent", "wg_vulnerable", "wg_harmful", "q3g_vulnerable", "q3g_harmful")


@dataclass(frozen=True)
class Model:
    model_id: str
    prompt: str
    recipe: str
    size: str
    pretrain_seed: str
    k: int
    base_ref: str
    out: str
    labels: str
    variant: str = ""

    @property
    def repo(self) -> str:
        return repo_name(self.recipe, self.size)

    @property
    def revision(self) -> str:
        return revision_name(self.pretrain_seed, self.k) + (f"-{self.variant}" if self.variant else "")

    @property
    def checkpoint(self) -> Path:
        return Path(self.out) / "model"

    @property
    def released_base(self) -> bool:
        return self.base_ref.startswith("allenai/")


def repo_name(recipe: str, size: str) -> str:
    name = f"castor-{recipe}-{size}"
    assert len(name) <= 96 and all(c.isalnum() or c in "-._" for c in name), name
    return f"{ORG}/{name}"


def revision_name(pretrain_seed: str, k: int) -> str:
    assert k >= 1
    return f"s{RERUN_SEED.get(pretrain_seed, pretrain_seed)}-p{k}"


def inventory(include_035: bool = False, manifests: Path = ROOT / "data/manifests") -> list[Model]:
    out = []
    grids = dict(GRIDS) | (dict([GRID_035]) if include_035 else {})
    for pid, n in grids.items():
        g = grid.load(manifests / f"grid_{pid}.json")
        cells, root = g.cells(), grid.models_root(g, smoke=False)
        assert len(cells) == n, (pid, len(cells), n)
        out += [Model(c.model_id, pid, c.base.recipe, c.base.size, c.base.pretrain_seed, c.k,
                      c.base.ref, str(c.out("dpo", root)),
                      str(c.gen_dir() / "judge_v1" / "labels.jsonl"), g.variant or "")
                for c in cells]
    cells = ladder.cells()
    assert len(cells) == LADDER_N
    out += [Model(c.model_id, ladder.PROMPT_ID, ladder.RECIPE, c.size, c.pretrain_seed, c.k,
                  c.base_ref, str(c.out("dpo")),
                  str(grid.RUNS / c.model_id / ladder.PROMPT_ID / "judge_v1" / "labels.jsonl"))
            for c in cells]
    assert len({m.model_id for m in out}) == len(out), "model ids collide"
    assert len({(m.repo, m.revision) for m in out}) == len(out), "repo revisions collide"
    return out


def a0(models: list[Model]) -> list[Model]:
    sel = [m for m in models if m.prompt == "008" and m.recipe == "dclm-baseline"]
    assert len(sel) == 12, len(sel)
    return sel


def source_of(prompt_id: str) -> str:
    src, sep, num = prompt_id.rpartition("-")
    assert sep and src and num.isdigit(), prompt_id
    return src


def label_rows(path: Path):
    with Path(path).open() as fh:
        for line in fh:
            row = json.loads(line)
            # a judge-input flag in the later label files; always False, counted in labels.meta.json
            key = (row["model"], row["prompt_id"], row["sample"])
            if row.pop("truncated_to_fit", False) and key not in TRUNCATED:
                raise ValueError(f"{path}: truncated_to_fit is True for {row['prompt_id']}")
            if set(row) != set(LABEL_COLUMNS):
                raise ValueError(f"{path}: unknown columns {sorted(set(row) - set(LABEL_COLUMNS))}"
                                 f", missing columns {sorted(set(LABEL_COLUMNS) - set(row))}")
            yield row


def label_table(m: Model, rows):
    import pyarrow as pa

    rows = list(rows)
    assert rows and all(r["model"] == m.model_id for r in rows), m.model_id
    cols = {c: [r.get(c) for r in rows] for c in LABEL_COLUMNS if c != "model"}
    head = {"repo": [m.repo] * len(rows), "revision": [m.revision] * len(rows),
            "model_id": [m.model_id] * len(rows),
            "source": [source_of(p) for p in cols["prompt_id"]]}
    return pa.table(head | cols)


def label_shape(rows) -> dict:
    """Prompts and sampled responses per prompt (greedy is `sample` = -1), from the labels."""
    per, greedy = {}, set()
    for r in rows:
        if r["sample"] >= 0:
            per.setdefault(r["prompt_id"], set()).add(r["sample"])
        else:
            greedy.add(r["prompt_id"])
    sizes = {len(v) for v in per.values()}
    assert len(sizes) == 1, f"uneven samples per prompt: {sorted(sizes)}"
    assert greedy == set(per), f"{len(greedy ^ set(per))} prompts lack a greedy or sampled row"
    return {"n_prompts": len(per), "n_samples": sizes.pop()}


def label_counts(rows) -> dict:
    n, sums = 0, dict.fromkeys(COUNT_COLUMNS, 0)
    for r in rows:
        n += 1
        for c in COUNT_COLUMNS:
            sums[c] += int(bool(r[c]))
    return {"n": n} | sums


def tensor_digest(state: dict) -> str:
    """sha256 over (name, dtype, shape, bytes): the same for a .bin and its .safetensors copy."""
    import torch

    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().contiguous().cpu()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def card_config(path: Path = CARD_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def card_facts(m: Model, prov: dict, sft_prov: dict, shape: dict,
               twins: dict | None = None) -> dict:
    facts = {"model_id": m.model_id, "repo": m.repo, "revision": m.revision,
             "recipe": m.recipe, "size": m.size, "pretrain_seed": m.pretrain_seed,
             "posttrain_seed": m.k, "sft_oi_seed": prov["parent"]["oi_seed"],
             "dpo_oi_seed": prov["oi_seed"], "base_ref": m.base_ref,
             "released_base": m.released_base, "prompt": m.prompt,
             "sft_config": sft_prov["config"], "sft_data": sft_prov["dataset"]["repo"],
             "sft_data_revision": sft_prov["dataset"]["revision"],
             "dpo_config": prov["config"], "dpo_data": prov["dataset"]["repo"],
             "dpo_data_revision": prov["dataset"]["revision"],
             "open_instruct_commit": prov["open_instruct"]["commit"],
             "git_commit": prov["git_commit"]} | shape
    if not m.released_base:
        t = (twins or {}).get(m.base_ref.split("@", 1)[1])
        assert t, f"{m.base_ref}: not in castor_twins_1B.json"
        facts |= {"init_seed": t["init_seed"], "data_seed": t["data_seed"],
                  "base_step": t["step"], "base_sha256": t["sha"]}
    if m.variant:
        facts |= swap_facts(m.variant)
    return facts


def swap_facts(variant: str) -> dict:
    """The SFT data of a swap grid, from its build report (the provenance names no Hub dataset)."""
    rep = json.loads(SWAP_MIX[variant].read_text())
    mix, comp = rep["build"]["mixture"], rep["build"]["composition"]
    return {"variant": variant, "sft_data": mix["source_tulu"]["repo"],
            "sft_data_revision": mix["source_tulu"]["revision"],
            "swap_removed": {k: v["rows"] for k, v in comp["removed_slice"]["by_source"].items()},
            "swap_sources": [s["repo"] for s in rep["pool"]["sets"].values()],
            "swap_rows": mix["replacement_rows"], "swap_sha256": mix["sha256"],
            "swap_report": str(SWAP_MIX[variant].relative_to(ROOT))}


def card_datasets(f: dict) -> list[str]:
    return [f["sft_data"]] + f.get("swap_sources", []) + [f["dpo_data"]]


def _swap_line(f: dict) -> str:
    removed = ", ".join(f"`{k.split('/')[-1]}`" for k in f["swap_removed"])
    sources = ", ".join(f"[`{r}`](https://huggingface.co/datasets/{r})" for r in f["swap_sources"])
    return (f"- **Safety data replaced:** in the SFT mixture, the safety subsets {removed} "
            f"({sum(f['swap_removed'].values()):,} rows) are replaced by {f['swap_rows']:,} rows "
            f"from other public safety datasets ({sources}), decontaminated against the evaluation "
            f"prompts. The DPO stage is unchanged. Build report: `{f['swap_report']}` in the code "
            f"repository; mixture sha256 `{f['swap_sha256'][:12]}`.")


def _hf(ref: str) -> str:
    repo, _, rev = ref.partition("@")
    return f"[`{repo}`](https://huggingface.co/{repo}) at revision `{rev}`"


def _front(cfg: dict, datasets: list[str], base_refs: list[str]) -> list[str]:
    front = {"license": cfg["license"]["weights"], "language": ["en"],
             "library_name": "transformers", "tags": cfg["tags"], "datasets": datasets}
    bases = sorted({r.split("@")[0] for r in base_refs if r.startswith("allenai/")})
    if bases:
        front["base_model"] = bases[0] if len(bases) == 1 else bases
    return ["---", yaml.safe_dump(front, sort_keys=False).rstrip(), "---", ""]


def _evaluation(cfg: dict, shape: dict) -> str:
    return cfg["evaluation"].strip().format(labels_repo=LABELS_REPO,
                                            n_prompts=f"{shape['n_prompts']:,}",
                                            n_samples=shape["n_samples"])


def paper_line(cfg: dict) -> str:
    if arxiv_pending(cfg):
        return f"**Paper:** *{cfg['title'].strip()}* (arXiv link to follow)"
    a = cfg["arxiv_id"]
    return f"**Paper:** [arXiv:{a}](https://arxiv.org/abs/{a})"


def citation(cfg: dict) -> list[str]:
    bib = cfg["bibtex"].strip().splitlines()
    if arxiv_pending(cfg):
        bib = [x for x in bib if "ARXIV_ID" not in x and "archivePrefix" not in x]
        bib[-2] = bib[-2].rstrip(",")  # the last field before the closing brace
    bib = "\n".join(bib).replace("ARXIV_ID", str(cfg["arxiv_id"]))
    return ["## Citation", "", cfg["authors"].strip(), "", "```bibtex", bib, "```", ""]


# The main data recipe and its size ladder lead the collection, then the labels.
COLLECTION_LEAD = tuple(f"{ORG}/castor-dclm-baseline-{s}" for s in ("1B", "150M", "300M"))


def collection_order(repos: list[str]) -> list[tuple[str, str]]:
    """(repo, item type) in collection order: the lead models, the labels, the other recipes."""
    lead = [r for r in COLLECTION_LEAD if r in repos]
    rest = sorted(set(repos) - set(lead))
    return [(r, "model") for r in lead] + [(LABELS_REPO, "dataset")] + [(r, "model") for r in rest]


def unfilled(cfg: dict) -> list[str]:
    """Card fields still holding a placeholder; the public flip waits until this is empty.

    The arXiv id may stay pending: the release goes public before the arXiv listing."""
    return [k for k in ("title", "authors", "bibtex") if "TBD" in str(cfg[k])]


def arxiv_pending(cfg: dict) -> bool:
    return "TBD" in str(cfg["arxiv_id"])


def _tail(cfg: dict, shape: dict) -> list[str]:
    return ["", "## Evaluation", "", _evaluation(cfg, shape),
            "", "## Intended use", "", cfg["intended_use"].strip(), "",
            "## Safety", "", cfg["safety"].strip(), "",
            "## License", "", cfg["license"]["text"].strip(), ""] + citation(cfg)


def render_card(f: dict, cfg: dict) -> str:
    """Card for one revision."""
    if f["released_base"]:
        base = (f"{_hf(f['base_ref'])} (DataDecide). Pretraining seed `{f['pretrain_seed']}` is "
                "DataDecide's seed name; it sets both the initialisation and the data order.")
    else:
        base = (f"our own pretraining run of DataDecide's {f['recipe']} {f['size']} recipe "
                f"(`{f['base_ref']}`): initialisation seed {f['init_seed']}, data-order seed "
                f"{f['data_seed']}, step {f['base_step']:,}. {cfg['castor_base_note'].strip()}")
    body = [f"# castor-{f['recipe']}-{f['size']}, revision `{f['revision']}`", "",
            paper_line(cfg), "", cfg["summary"].strip(), "", "## This model", "",
            f"- **Base model:** {base}",
            (f"- **Post-training seed:** {f['posttrain_seed']} (open-instruct `--seed` "
             f"{f['sft_oi_seed']} for SFT, {f['dpo_oi_seed']} for DPO; it sets the data shuffle "
             "and sampler order)."),
            (f"- **Post-training:** SFT on [`{f['sft_data']}`](https://huggingface.co/datasets/"
             f"{f['sft_data']}) at `{f['sft_data_revision'][:12]}`, then DPO on "
             f"[`{f['dpo_data']}`](https://huggingface.co/datasets/{f['dpo_data']}) at "
             f"`{f['dpo_data_revision'][:12]}`, with open-instruct at "
             f"`{f['open_instruct_commit'][:12]}`. Configs: `{f['sft_config']}`, "
             f"`{f['dpo_config']}`."),
            f"- **Code:** {CODE_URL}"]
    if f.get("variant"):
        body.insert(-1, _swap_line(f))
    return "\n".join(_front(cfg, card_datasets(f), [f["base_ref"]]) + body + _tail(cfg, f))


def render_repo_card(repo: str, siblings: list[dict], datasets: list[str], shape: dict,
                     cfg: dict) -> str:
    """Card for a repo's main branch: the table of the revisions that exist on the Hub.

    Built from the repo's revisions only, so it is the same whichever model uploads it."""
    assert siblings and all(s["repo"] == repo for s in siblings), repo
    siblings = sorted(siblings, key=lambda s: s["revision"])
    body = [f"# {repo.split('/', 1)[1]}", "", paper_line(cfg), "", cfg["summary"].strip(), "",
            "## Revisions", "",
            ("One revision (branch) per model: "
             "`AutoModelForCausalLM.from_pretrained(repo, revision=...)`."), "",
            "| revision | pretraining seed | post-training seed | base |", "|---|---|---|---|"]
    body += [f"| `{s['revision']}` | `{s['pretrain_seed']}` | {s['k']} | "
             f"`{s['base_ref']}` |" for s in siblings]
    body += [f"\nRevisions ending in `-{v}` were post-trained with the SFT mixture's safety "
             "subsets replaced by other public safety data; their cards name the sources."
             for v in sorted({s["variant"] for s in siblings if s.get("variant")})]
    return "\n".join(_front(cfg, datasets, [s["base_ref"] for s in siblings]) + body
                      + _tail(cfg, shape))


def labels_readme(cfg: dict, n_models: int, shape: dict) -> str:
    front = {"license": cfg["license"]["labels"], "language": ["en"], "tags": cfg["tags"],
             "configs": [{"config_name": "default",
                          "data_files": [{"split": "train", "path": "data/*/*.parquet"}]}]}
    cols = ", ".join(f"`{c}`" for c in ("repo", "revision", "model_id", "source")
                     + LABEL_COLUMNS[1:])
    return "\n".join(["---", yaml.safe_dump(front, sort_keys=False).rstrip(), "---", "",
                      "# castor-labels", "", paper_line(cfg), "",
                      cfg["labels_summary"].strip().format(n_prompts=f"{shape['n_prompts']:,}",
                                                           n_samples=shape["n_samples"]), "",
                      f"{n_models} models. Columns: {cols}.", "",
                      cfg["labels_truncated"].strip(), "",
                      "## Safety", "", cfg["safety"].strip(), ""] + citation(cfg))


def inventory_record(m: Model) -> dict:
    return asdict(m) | {"repo": m.repo, "revision": m.revision}
