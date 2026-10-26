# NCHC (Nano 4, H200) environment for CASTOR. Source it; do not execute it:
#   source shell/nchc/env.sh
# Every job from shell/nchc/submit.sh sources this. Python env is uv-managed (.venv from
# uv.lock); run code with `uv run --frozen ...` or after this file activates .venv.

export CASTOR_ROOT="${CASTOR_ROOT:-$HOME/castor}"
# cron promises LOGNAME, HOME and SHELL only (man 5 crontab); every /work path below needs USER
export USER="${USER:-$(id -un)}"
export NCHC_ACCOUNT="${NCHC_ACCOUNT:-acd115198}"
export CASTOR_WORK="/work/$USER/castor"          # checkpoints, generations, activations

# Model weights and tool caches -> /work, never /home. HF_HOME is shared with astra.
export HF_HOME="/work/$USER/hf"
export TRITON_CACHE_DIR="/work/$USER/.cache/triton"
export TORCH_HOME="/work/$USER/.cache/torch"
export VLLM_CACHE_ROOT="/work/$USER/.cache/vllm"
export UV_CACHE_DIR="/work/$USER/.cache/uv"
export UV_PYTHON_INSTALL_DIR="/work/$USER/.local/uv-python"
export UV_LINK_MODE=copy

case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) export PATH="$HOME/.local/bin:$PATH" ;; esac
mkdir -p "$CASTOR_WORK"

if [ -f "$CASTOR_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$CASTOR_ROOT/.venv/bin/activate"
else
    echo "shell/nchc/env.sh: no venv at $CASTOR_ROOT/.venv — run: cd $CASTOR_ROOT && uv sync --frozen" >&2
fi
