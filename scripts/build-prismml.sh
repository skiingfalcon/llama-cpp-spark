#!/usr/bin/env bash
# Build PrismML's llama.cpp fork with CUDA for NVIDIA DGX Spark GB10 (sm_121a).
#
# Only needed to serve PrismML ternary models (e.g. bonsai-2-27b): their PQ2_0/PTQ1_0 tensor
# types are rejected by stock llama.cpp, so they need this fork's custom kernels instead.
# This is a SEPARATE checkout/build from vendor/llama.cpp (scripts/build.sh) — it must not
# replace the pinned upstream binary every other Spark model relies on. Point
# LOCAL_LLM_LLAMA_BIN_DIR at this script's output when serving a PrismML model; leave it unset
# for everything else.
#
# Same CMake flags and sm_121a-real / 121+GGML_NATIVE=OFF fallback as scripts/build.sh: the
# fork's build.md documents no extra flags for the ternary kernels.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VERSION_FILE="${REPO_ROOT}/PRISMML_LLAMA_CPP_VERSION"
VENDOR="${REPO_ROOT}/vendor/prismml-llama.cpp"
REPO_URL="https://github.com/PrismML-Eng/llama.cpp.git"

if [[ ! -f "${VERSION_FILE}" ]]; then
  echo "error: missing ${VERSION_FILE}" >&2
  exit 1
fi
PINNED="$(tr -d '[:space:]' < "${VERSION_FILE}")"
if [[ -z "${PINNED}" ]]; then
  echo "error: PRISMML_LLAMA_CPP_VERSION is empty" >&2
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
echo "==> pin   ${PINNED} (PrismML fork)"

mkdir -p "${REPO_ROOT}/vendor"
if [[ ! -d "${VENDOR}/.git" ]]; then
  echo "==> cloning PrismML-Eng/llama.cpp"
  git clone --filter=blob:none "${REPO_URL}" "${VENDOR}"
fi

echo "==> checking out ${PINNED}"
# Release tags live on PrismML's `prism` branch (their README: never build prism-v6). Fetch the
# tag by ref so an existing checkout picks it up even when it is not near the branch head.
git -C "${VENDOR}" fetch origin "refs/tags/${PINNED}:refs/tags/${PINNED}" 2>/dev/null \
  || git -C "${VENDOR}" fetch --tags origin
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

BUILT_ARCH="121a-real"
set +e
configure_and_build "121a-real" 2>&1 | tee "${LOG}"
STATUS=${PIPESTATUS[0]}
set -e

if [[ ${STATUS} -ne 0 ]]; then
  if grep -qiE 'block_scale|not supported on .target .sm_121|ptxas.*error|CUDA_ARCHITECTURES' "${LOG}"; then
    echo "==> primary build failed (likely arch); falling back to 121 + GGML_NATIVE=OFF"
    configure_and_build "121" -DGGML_NATIVE=OFF
    BUILT_ARCH="121+GGML_NATIVE=OFF"
  else
    echo "error: build failed; see log above" >&2
    exit "${STATUS}"
  fi
fi

echo "${BUILT_ARCH}" > "${VENDOR}/build/spark-arch.txt"
echo "==> recorded CUDA arch ${BUILT_ARCH} in build/spark-arch.txt"

SERVER="${VENDOR}/build/bin/llama-server"
if [[ ! -x "${SERVER}" ]]; then
  echo "error: expected binary missing: ${SERVER}" >&2
  exit 1
fi

echo "==> verifying binary"
"${SERVER}" --version 2>&1 | head -20 || true
echo "==> build complete: ${SERVER}"
echo "==> serve a PrismML model with:"
echo "    LOCAL_LLM_LLAMA_BIN_DIR=${VENDOR}/build/bin uv run local-llm serve bonsai-2-27b"
