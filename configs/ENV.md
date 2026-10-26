# Environment (2026-09-26)

One uv project at the repo root (`pyproject.toml`, `uv.lock` committed). Jobs run
`uv run --frozen`. Python 3.12.14 (uv-managed, `UV_PYTHON_INSTALL_DIR=/work/$USER/.local/uv-python`).

| package | version | note |
|---|---|---|
| torch | 2.11.0+cu130 | CUDA 13.0 wheels; NCHC driver 580.65.06 |
| transformers | 4.57.6 | ceiling set by vllm 0.23.0 |
| vllm | 0.23.0 | same pin astra runs on this cluster |
| datasets | 5.0.1 | 2.14.x (first resolve) breaks on pyarrow 25 |
| huggingface-hub | 0.36.2 | |
| accelerate | 1.15.0 | |
| safetensors / tokenizers | 0.8.0 / 0.22.2 | |
| triton / flashinfer-python | 3.6.0 / 0.6.12 | vllm deps |
| pandas / pyarrow / numpy | 3.0.6 / 25.0.1 / 2.3.5 | |
| dev: pytest, ruff, pypdf | 9.1.1, 0.16.9, 6.19.0 | |
| dev: omegaconf, boto3, cached-path, importlib-resources | | imports of the reference OLMo code (validation only) |

Hardware: compute nodes 8x H200 141 GB (`dev` / `8gpus` partitions); login node 1x H100 NVL.
**Login node limits (cgroup): 20 GiB RAM, 4 CPUs per user.** Anything heavier runs as a Slurm
job (`shell/nchc/submit.sh --dev`); exceeding the cap kills every process of the session.

## External code (not installed; pinned by commit)

| code | where | commit | use |
|---|---|---|---|
| allenai/OLMo, branch `DataDecide` | `/work/harrychang08/castor/third_party/OLMo` | `7094aab0448096f4994cae881edbd629d6c2f3de` (2025-03-13) | DataDecide training code; reference forward pass in `scripts/validate_conversion.py` (via `PYTHONPATH`) |
| allenai/open-instruct | `/work/harrychang08/castor/third_party/open-instruct`, own uv project, Python 3.11.16 | `fc7748ea4e111d0ae3865e176cdf73227cca0c39` (2025-04-23), used for SFT and DPO: vs `d19de7d27` the SFT/DPO scripts differ only by a guarded deepspeed import + `NCCL_CUMEM_ENABLE=0` | Tulu 3 SFT + DPO. Installed with Ai2's own lock: `uv sync --frozen --extra compile` → torch 2.6.0+cu124, transformers 4.50.0, deepspeed 0.15.4, accelerate 1.2.1, datasets 3.2.0, vllm 0.8.2, flash-attn 2.7.2.post1. Lock copy: `configs/third_party/open-instruct.lock`. Entry: `shell/posttrain/train.sh` |

## Model format

DataDecide checkpoints are released in Ai2's legacy `hf_olmo` format, loadable only by the
old `ai2-olmo` package (not by current transformers or vLLM). `scripts/convert_datadecide.py`
(`castor/convert.py`) re-indexes them into `LlamaForCausalLM` (exact; architecture is a Llama
block). Converted copies: `/work/harrychang08/castor/models/datadecide-llama/<repo>/<revision>/`
with `provenance.json`. Validation on `allenai/DataDecide-dclm-baseline-1B@step69369-seed-default`:
max |logit diff| vs the reference OLMo code 4.96e-5 (fp32, 2048 tokens), top-1 agreement 1.0;
HF and vLLM both load and generate (notes/002).

Tokenizer: `allenai_gpt-neox-olmo-dolma-v1_5` (vocab 50280, embedding rows 50304), **no BOS
token**, EOS `<|endoftext|>` (50279), pad `<|padding|>` (1). Context length 2048.
