# Convenience targets for the DGX Spark / Linux box (GNU make + bash). On the Windows Halo box
# run the `uv run local-llm …` commands directly; see README "Windows / AMD Strix Halo".
.PHONY: build serve bench test lint doctor stop models download eval-fetch eval-sec eval-swe eval-report

MODEL ?= gpt-oss-20b

build:
	uv run local-llm build

serve:
	uv run local-llm serve $(MODEL)

bench:
	uv run local-llm bench $(MODEL)

# Evals need a served model: `make serve` first.
eval-fetch:
	uv run local-llm eval sec fetch

eval-sec:
	uv run local-llm eval sec run $(MODEL) --task extract-full
	uv run local-llm eval sec run $(MODEL) --task extract-chunked
	uv run local-llm eval sec perf $(MODEL)

eval-swe:
	uv run local-llm eval swe check $(MODEL)
	uv run local-llm eval swe run $(MODEL) --tier 1

eval-report:
	uv run local-llm eval report --suite sec
	uv run local-llm eval report --suite swe

download:
	uv run local-llm download $(MODEL)

models:
	uv run local-llm models

doctor:
	uv run local-llm doctor

stop:
	uv run local-llm stop

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
