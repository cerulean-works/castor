![Cerulean Labs Castor banner](assets/banner.png)

---

# CASTOR: Common Ancestry, Seed-Twinned Outcomes of Refusal

**Paper:** [arXiv:2610.34074](https://arxiv.org/abs/2610.34074)

Train the same language model twice, on the same data with the same recipe, changing only the
random seed. Do the two copies fail on the same jailbreaks?

CASTOR answers this with seed twins. The base models come from DataDecide: each data recipe and
size was pretrained under several seeds, so two bases that share a recipe differ only in their
pretraining seed. Every base receives the same safety post-training (SFT then DPO, the OLMo 2 1B
Instruct recipe) under several post-training seeds, and every resulting model answers the same
panel of public jailbreak and over-refusal prompts, judged per response. Comparing models that
share a pretraining run, share only a data recipe, or share nothing measures how much of a
model's per-prompt jailbreak vulnerability its pretraining seed fixes.

## What is released

- **Models** (Hugging Face, [`cerulean-works`](https://huggingface.co/cerulean-works)): 208 post-trained
  models in 27 repos, one repo per data recipe and size (`cerulean-works/castor-<recipe>-<size>`)
  and one revision per model (`s<pretraining seed>-p<post-training seed>`): 156 models at
  1B across 25 DataDecide data recipes, 24 twins at 150M and 300M, 16
  models built on our own pretraining reruns of one recipe, and 12 1B models whose SFT
  safety data were replaced by other public safety data (revisions ending in `-swap080`).
- **Labels** ([`cerulean-works/castor-labels`](https://huggingface.co/datasets/cerulean-works/castor-labels)): per-response WildGuard and
  Qwen3Guard labels for every model, keyed by benchmark prompt id. No prompt text, no generations.
- **This repository:** the code and the aggregate results behind every number in the paper. The
  paper's LaTeX source is on arXiv.

## Layout

| path | contents |
|---|---|
| `castor/` | library: panel, generation, judging, twin statistics, release |
| `scripts/` | one entry point per experiment step |
| `configs/` | post-training and evaluation configs; `third_party/` pins open-instruct, OLMo and OLMES |
| `shell/` | Slurm submission scripts |
| `runs/` | aggregate artifacts; `runs/numbers.json` names the source of each paper number |
| `tests/` | tests for the code that computes reported numbers |
| `refs/refs.bib` | BibTeX for the prior work the code and the paper cite |

## Setup

```
uv sync
uv run pytest tests    # skips tests whose inputs are not released
```

Evaluation prompts are not redistributed. The panel is rebuilt from the public benchmarks with
`scripts/build_panel.py` (see `data/panel/MANIFEST.json`). Training uses open-instruct and OLMo
unmodified at the commits pinned in `configs/third_party/`.

## Safety

The models were safety-tuned on public data, and their bases are public. They can still produce
harmful text when prompted, as the released labels show. Model outputs on harmful prompts are not
released.

## License

Code: Apache 2.0 (`LICENSE`). Model weights: Apache 2.0, as the DataDecide base models. Labels:
ODC-BY 1.0.

## Citation

Chi-Wei Chang, Mu-Tien Kuo, Kuan-Wei Lee, Meng-Hsun Tsai

```bibtex
@misc{chang2026identicallytrainedlanguagemodels,
  title         = {Identically Trained Language Models Break on Different Jailbreaks},
  author        = {Chang, Chi-Wei and Kuo, Mu-Tien and Lee, Kuan-Wei and Tsai, Meng-Hsun},
  year          = {2026},
  eprint        = {2610.34074},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.34074}
}
```
