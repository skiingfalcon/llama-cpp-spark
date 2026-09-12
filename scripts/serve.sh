#!/usr/bin/env bash
# Thin wrapper: prefer the Python CLI for serving; this exists for Makefile/docs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

MODEL="${1:-gpt-oss-20b}"
shift || true
cd "${REPO_ROOT}"
exec uv run spark-llm serve "${MODEL}" "$@"
