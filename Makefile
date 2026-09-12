.PHONY: build serve bench test lint doctor stop models download eval-fetch eval-sec eval-swe eval-report

build:
	./scripts/build.sh

serve:
	uv run local-llm serve gpt-oss-20b

bench:
	uv run local-llm bench gpt-oss-20b

# Evals need a served model: `uv run local-llm serve $(MODEL)` first.
MODEL ?= gpt-oss-20b
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
	uv run local-llm download gpt-oss-20b

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
