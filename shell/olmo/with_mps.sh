#!/bin/bash
# Run a command under a private CUDA MPS daemon limited to PCT% of the GPU's SMs.
# Used to emulate other GPUs' SM counts (CUDA RNG kernels size their grid by SM count),
# only if the step-0 check fails on an H200 for every candidate seed.
#   bash shell/olmo/with_mps.sh <PCT> <command ...>
set -euo pipefail
PCT=$1; shift
ID=${SLURM_JOB_ID:-$$}   # also works outside Slurm (login node)
BASE=${TMPDIR:-/tmp}/mps-$USER-$ID
export CUDA_MPS_PIPE_DIRECTORY=$BASE/pipe CUDA_MPS_LOG_DIRECTORY=$BASE/log
rm -rf "$BASE"; mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
cleanup() { echo quit | nvidia-cuda-mps-control || true; rm -rf "$BASE"; }
trap cleanup EXIT
nvidia-cuda-mps-control -d
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$PCT "$@"
