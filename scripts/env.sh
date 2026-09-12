#!/usr/bin/env bash
# Shared environment for llama.cpp binaries on DGX Spark (GB10 / Blackwell).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

BIN_DIR="${REPO_ROOT}/vendor/llama.cpp/build/bin"
CUDA_COMPAT="/usr/local/cuda-13/compat"
# Also try generic cuda compat path
if [[ ! -d "${CUDA_COMPAT}" ]]; then
  CUDA_COMPAT="/usr/local/cuda/compat"
fi

export PATH="${BIN_DIR}:/usr/local/cuda/bin:${HOME}/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${BIN_DIR}:${CUDA_COMPAT}:${LD_LIBRARY_PATH:-}"

# Prefer the uv-tool cmake/ninja over the system cmake 3.28
if [[ -x "${HOME}/.local/bin/cmake" ]]; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
