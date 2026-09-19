#!/usr/bin/env bash
# Register-and-measure pass for newly added models on the DGX Spark: Gemma 4 31B, Gemma 4 26B-A4B,
# DeepSeek V4 Flash, Laguna S 2.1 (models.toml). Runs on the Spark; everything it learns lands under
# state/evals/ (the only committed part of state/) so the numbers can be read back on a machine
# without a GPU.
#
#   FULL=1 PUSH=1 ./scripts/spark-new-models-smoke.sh
#
# Per model and its thinking-toggle twin: download, serve, record /props + server-log excerpt + GPU
# memory, a chat smoke (raw JSON keeps reasoning_content visible), `eval swe check` on the twin, a
# 22-item extract-full gate run, then a gate check (no errors, answers present, warm-row prompt-cache
# ratio >= 0.8, reasoning present/absent as the twin implies). Models that pass get the full
# 121-question run when FULL=1. Re-runnable: steps whose output exists are skipped; RESUME=<stamp>
# reuses a dir.
#
# Twin naming: every model here defaults to thinking ON and has a "-nothink" twin that turns it off,
# except Laguna, whose vendor default is thinking OFF, so its twin is "-thinking" (turns it on) and
# the bare name is already the primary, thinking-off row.
#
# Env: MODELS (default below) FULL=1 PUSH=1 DRY_RUN=1 RESUME=<stamp> MIN_FREE_GB (default 135)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODELS="${MODELS:-gemma-4-26b-a4b gemma-4-31b deepseek-v4-flash laguna-s-2.1}"
FULL="${FULL:-0}"
PUSH="${PUSH:-0}"
DRY_RUN="${DRY_RUN:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-175}"
MODELS_DIR="${LOCAL_LLM_MODELS_DIR:-/opt/models}"
export LOCAL_LLM_HEALTH_TIMEOUT_S="${LOCAL_LLM_HEALTH_TIMEOUT_S:-1500}"   # a 97 GB load beats the 300 s default

STAMP="${RESUME:-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE="state/evals/smoke/${STAMP}"
STEPS="${SMOKE}/steps.log"

run() {
  # run <step-id> <output-path-that-marks-done> <command...>
  local id="$1" done_mark="$2"; shift 2
  if [[ -e "${done_mark}" ]]; then
    echo "==> skip ${id} (have ${done_mark})"; return 0
  fi
  echo "==> ${id}: $*"
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local t0; t0="$(date -u +%FT%TZ)"
  set +e; "$@"; local rc=$?; set -e
  printf '%s\t%s\t%s\t%s\n' "${t0}" "$(date -u +%FT%TZ)" "${rc}" "${id}" >> "${STEPS}"
  return "${rc}"
}

mk() { [[ "${DRY_RUN}" == "1" ]] || mkdir -p "$@"; }
stop_all() { echo "==> stop"; [[ "${DRY_RUN}" == "1" ]] || uv run local-llm stop || true; }
port_of() { uv run python -c "from spark_llm.registry import load_registry; print(load_registry().get('$1').port)"; }

gate() {
  # gate <model> <smoke-dir>: writes gate.json from the newest 22-item run of <model>; exit 1 on fail
  uv run python - "$1" "$2" <<'PY'
import glob, json, os, statistics, sys
model, out = sys.argv[1], sys.argv[2]
runs = sorted(glob.glob(f"state/evals/sec/{model}-spark-cuda/*-extract-full"))
runs = [r for r in runs if os.path.isfile(f"{r}/run.json")
        and json.load(open(f"{r}/run.json")).get("task_config", {}).get("limit") == 22]
if not runs:
    sys.exit(f"gate: no 22-item run for {model}")
run = runs[-1]
rj = json.load(open(f"{run}/run.json"))
rows = [json.loads(l) for l in open(f"{run}/results.jsonl")]
warm = [r for r in rows if r.get("warm") and r.get("prompt_tokens") and not r.get("skipped")]
ratios = [(r.get("cached_prompt_tokens") or 0) / r["prompt_tokens"] for r in warm]
gaps = [r["prompt_tokens"] - (r.get("cached_prompt_tokens") or 0) for r in warm]
thinking_off_expected = model.endswith("-nothink") or (
    model.startswith("laguna-") and not model.endswith("-thinking")
)  # Laguna defaults to thinking off; its twin ("-thinking") turns it on
reasoning = [r.get("reasoning_tokens") or 0 for r in rows if not r.get("skipped")]
checks = {
    "finished": bool(rj.get("finished")),
    "no_errors": (rj.get("summary") or {}).get("errors", 0) == 0,
    "no_skips": all(not r.get("skipped") for r in rows),
    "answers_present": all((r.get("answer") or "").strip() for r in rows if not r.get("skipped")),
    "no_think_tags_in_answers": all("<think>" not in (r.get("answer") or "") for r in rows),
    "warm_cache_ratio_ok": bool(ratios) and statistics.median(ratios) >= 0.8,
    "reasoning_as_expected": (max(reasoning, default=0) == 0) if thinking_off_expected else (max(reasoning, default=0) > 0),
}
report = {
    "model": model, "run": run, "n": len(rows), "checks": checks, "passed": all(checks.values()),
    "warm_cache_ratio_median": round(statistics.median(ratios), 3) if ratios else None,
    "warm_gap_median_tokens": int(statistics.median(gaps)) if gaps else None,
    "score": (rj.get("summary") or {}).get("score"),
    "truncated": (rj.get("summary") or {}).get("truncated"),
    "decode_tps_p50": (rj.get("summary") or {}).get("decode_tps_p50"),
    "reasoning_tokens": (rj.get("summary") or {}).get("reasoning_tokens"),
    "chat_template_sha": (rj.get("server") or {}).get("chat_template_sha"),
    "build_info": (rj.get("server") or {}).get("build_info"),
}
os.makedirs(out, exist_ok=True)
json.dump(report, open(f"{out}/gate.json", "w"), indent=2)
print(json.dumps(report, indent=2))
sys.exit(0 if report["passed"] else 1)
PY
}

mk "${SMOKE}/preflight"
echo "==> smoke dir ${SMOKE}"

# 0. Preflight
run preflight.pull "${SMOKE}/preflight/git.txt" bash -c "git pull --ff-only | tee '${SMOKE}/preflight/git.txt'"
run preflight.sync "${SMOKE}/preflight/uv-sync.txt" bash -c "uv sync 2>&1 | tee '${SMOKE}/preflight/uv-sync.txt'"
run preflight.doctor "${SMOKE}/preflight/doctor.txt" bash -c "uv run local-llm doctor 2>&1 | tee '${SMOKE}/preflight/doctor.txt' || true"
run preflight.disk "${SMOKE}/preflight/df.txt" bash -c "df -h '${MODELS_DIR}' | tee '${SMOKE}/preflight/df.txt'; uname -r >> '${SMOKE}/preflight/df.txt'"
run preflight.gpu "${SMOKE}/preflight/nvidia-smi.txt" bash -c "nvidia-smi -q -d MEMORY > '${SMOKE}/preflight/nvidia-smi.txt' 2>&1 || true"
stop_all

if [[ "${DRY_RUN}" != "1" ]]; then
  free_gb="$(df -BG --output=avail "${MODELS_DIR}" | tail -1 | tr -dc '0-9')"
  if [[ " ${MODELS} " == *" deepseek-v4-flash "* || " ${MODELS} " == *" laguna-s-2.1 "* ]] \
     && [[ "${free_gb}" -lt "${MIN_FREE_GB}" ]]; then
    echo "error: ${free_gb} GB free under ${MODELS_DIR}; need ${MIN_FREE_GB} GB for the four downloads" \
         "(17.65 + 14.44 + 96.8 + ~40 GB). Free space or run with MODELS='gemma-4-26b-a4b gemma-4-31b'." >&2
    exit 1
  fi
fi

# 1. Per model: download, then for base and twin: serve, smoke, gate
for M in ${MODELS}; do
  T="${M}-nothink"
  [[ "${M}" == laguna-* ]] && T="${M}-thinking"   # Laguna defaults to thinking off; its twin turns it on
  run "download.${M}" "${SMOKE}/${M}/download.txt" bash -c "mkdir -p '${SMOKE}/${M}'; uv run local-llm download '${M}' 2>&1 | tee '${SMOKE}/${M}/download.txt'"

  order=("${M}" "${T}")
  [[ "${M}" == deepseek-* ]] && order=("${T}" "${M}")   # thinking twin last: it may spend the whole budget per question
  for X in "${order[@]}"; do
    D="${SMOKE}/${X}"; mk "${D}"
    PORT="$(port_of "${X}")"
    if [[ "${X}" == deepseek-* ]]; then
      run "dropcaches.${X}" "${D}/dropcaches.txt" bash -c "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' > '${D}/dropcaches.txt' 2>&1 || echo 'no passwordless sudo; skipped' > '${D}/dropcaches.txt'"
    fi
    # serve is keyed on the smoke run's output, so a resume after a crash re-serves instead of skipping
    run "serve.${X}" "${D}/smoke-run.txt" bash -c "uv run local-llm serve '${X}' 2>&1 | tee '${D}/serve.txt'"
    run "props.${X}" "${D}/props.json" bash -c "curl -sf 'http://127.0.0.1:${PORT}/props' > '${D}/props.json'"
    run "log.${X}" "${D}/server-log-excerpt.txt" bash -c "grep -E 'KV self size|SWA|swa|chat template|Chat format|chat_format|model size|compute buffer|load time|checkpoint|reasoning' 'state/${X}.log' > '${D}/server-log-excerpt.txt' || true"
    run "gpumem.${X}" "${D}/gpu-mem.txt" bash -c "nvidia-smi --query-gpu=memory.used,memory.total --format=csv > '${D}/gpu-mem.txt' 2>&1 || free -g > '${D}/gpu-mem.txt'"
    run "chat.${X}" "${D}/chat.txt" bash -c "uv run local-llm chat '${X}' --no-stream -m 'Reply with the single word: ready.' 2>&1 | tee '${D}/chat.txt'"
    run "chatraw.${X}" "${D}/chat-raw.json" bash -c "curl -s 'http://127.0.0.1:${PORT}/v1/chat/completions' -H 'Content-Type: application/json' -d '{\"model\":\"${X}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 times 23? Answer with the number only.\"}],\"max_tokens\":512,\"temperature\":0}' > '${D}/chat-raw.json'"
    if [[ "${X}" == "${T}" ]]; then
      run "swecheck.${X}" "${D}/swe-check.exit" bash -c "uv run local-llm eval swe check '${X}' > '${D}/swe-check.txt' 2>&1; echo \$? > '${D}/swe-check.exit'; cat '${D}/swe-check.txt'"
    fi
    run "smoke22.${X}" "${D}/smoke-run.txt" bash -c "uv run local-llm eval sec run '${X}' --task extract-full --forms 10-K --tickers AAPL,XOM --limit 22 2>&1 | tee '${D}/smoke-run.txt'"
    stop_all
    set +e; run "gate.${X}" "${D}/gate.json" gate "${X}" "${D}"; gate_rc=$?; set -e
    if [[ "${DRY_RUN}" != "1" ]]; then
      passed="$(uv run python -c "import json,sys; print(json.load(open('${D}/gate.json'))['passed'])" 2>/dev/null || echo False)"
    else
      passed="False"
    fi
    echo "==> gate ${X}: passed=${passed} (rc ${gate_rc})"

    # 2. Full run only behind the gate
    if [[ "${FULL}" == "1" && "${passed}" == "True" ]]; then
      run "serve.full.${X}" "${D}/full-run.txt" bash -c "uv run local-llm serve '${X}' 2>&1 | tee '${D}/serve-full.txt'"
      run "full.${X}" "${D}/full-run.txt" bash -c "uv run local-llm eval sec run '${X}' --task extract-full --forms 10-K 2>&1 | tee '${D}/full-run.txt'"
      stop_all
    fi
  done
done

# 3. Report, commit, push
run report "${SMOKE}/report.txt" bash -c "uv run local-llm eval report --suite sec 2>&1 | tail -5 | tee '${SMOKE}/report.txt'"
if [[ "${PUSH}" == "1" && "${DRY_RUN}" != "1" ]]; then
  git add state/evals
  git commit -m "Add Gemma 4 / DeepSeek V4 Flash Spark smoke and 10-K eval results (${STAMP})" || echo "nothing to commit"
  git push
fi
echo "==> done; steps in ${STEPS}"
[[ -f "${STEPS}" ]] && column -t -s $'\t' "${STEPS}" || true
