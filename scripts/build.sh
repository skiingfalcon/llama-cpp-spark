#!/usr/bin/env bash
# Build llama.cpp with CUDA for NVIDIA DGX Spark GB10 (sm_121a).
#
# Primary target: CMAKE_CUDA_ARCHITECTURES=121a-real
#   Required for native MXFP4 block-scale kernels used by gpt-oss models.
# Fallback: 121 + GGML_NATIVE=OFF if ptxas rejects block_scale on plain 121.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

VERSION_FILE="${REPO_ROOT}/LLAMA_CPP_VERSION"
VENDOR="${REPO_ROOT}/vendor/llama.cpp"
REPO_URL="https://github.com/ggml-org/llama.cpp.git"

if [[ ! -f "${VERSION_FILE}" ]]; then
  echo "error: missing ${VERSION_FILE}" >&2
  exit 1
fi
PINNED="$(tr -d '[:space:]' < "${VERSION_FILE}")"
if [[ -z "${PINNED}" ]]; then
  echo "error: LLAMA_CPP_VERSION is empty" >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "error: cmake not found (install with: uv tool install 'cmake>=3.30')" >&2
  exit 1
fi
if ! command -v ninja >/dev/null 2>&1; then
  echo "error: ninja not found (install with: uv tool install ninja)" >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc not found; ensure CUDA toolkit is on PATH" >&2
  exit 1
fi

echo "==> cmake $(cmake --version | head -1)"
echo "==> ninja $(ninja --version)"
echo "==> nvcc  $(nvcc --version | tail -1)"
echo "==> pin   ${PINNED}"

mkdir -p "${REPO_ROOT}/vendor"
if [[ ! -d "${VENDOR}/.git" ]]; then
  echo "==> cloning llama.cpp"
  git clone --filter=blob:none "${REPO_URL}" "${VENDOR}"
fi

echo "==> checking out ${PINNED}"
git -C "${VENDOR}" fetch --depth 1 origin "${PINNED}" 2>/dev/null \
  || git -C "${VENDOR}" fetch --depth 1 origin "${PINNED}" \
  || git -C "${VENDOR}" fetch --unshallow 2>/dev/null \
  || true
git -C "${VENDOR}" fetch origin "${PINNED}" --depth 1 2>/dev/null || git -C "${VENDOR}" fetch --depth 50 origin
git -C "${VENDOR}" checkout --force "${PINNED}"

configure_and_build() {
  local arch="$1"
  shift
  local extra=("$@")
  echo "==> configuring (CUDA arch=${arch}${extra[*]:+ ${extra[*]}})"
  rm -rf "${VENDOR}/build"
  cmake -S "${VENDOR}" -B "${VENDOR}/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DLLAMA_CURL=ON \
    -DLLAMA_OPENSSL=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${arch}" \
    "${extra[@]}"
  echo "==> building"
  cmake --build "${VENDOR}/build" --config Release -j"$(nproc)"
}

LOG="$(mktemp)"
trap 'rm -f "${LOG}"' EXIT

set +e
configure_and_build "121a-real" 2>&1 | tee "${LOG}"
STATUS=${PIPESTATUS[0]}
set -e

if [[ ${STATUS} -ne 0 ]]; then
  if grep -qiE 'block_scale|not supported on .target .sm_121|ptxas.*error|CUDA_ARCHITECTURES' "${LOG}"; then
    echo "==> primary build failed (likely MXFP4 / arch); falling back to 121 + GGML_NATIVE=OFF"
    configure_and_build "121" -DGGML_NATIVE=OFF
  else
    echo "error: build failed; see log above" >&2
    exit "${STATUS}"
  fi
fi

SERVER="${VENDOR}/build/bin/llama-server"
if [[ ! -x "${SERVER}" ]]; then
  echo "error: expected binary missing: ${SERVER}" >&2
  exit 1
fi

echo "==> verifying binary"
"${SERVER}" --version 2>&1 | head -20 || true
echo "==> build complete: ${SERVER}"
