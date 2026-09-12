.PHONY: build serve bench test lint doctor stop models download eval-fetch eval-sec eval-swe eval-report

build:
	./scripts/build.sh

serve:
	uv run spark-llm serve gpt-oss-20b

bench:
	uv run spark-llm bench gpt-oss-20b

# Evals need a served model: `uv run spark-llm serve $(MODEL)` first.
MODEL ?= gpt-oss-20b
eval-fetch:
	uv run spark-llm eval sec fetch

eval-sec:
	uv run spark-llm eval sec run $(MODEL) --task extract-full
	uv run spark-llm eval sec run $(MODEL) --task extract-chunked
	uv run spark-llm eval sec perf $(MODEL)

eval-swe:
	uv run spark-llm eval swe check $(MODEL)
	uv run spark-llm eval swe run $(MODEL) --tier 1

eval-report:
	uv run spark-llm eval report --suite sec
	uv run spark-llm eval report --suite swe

download:
	uv run spark-llm download gpt-oss-20b

models:
	uv run spark-llm models

doctor:
	uv run spark-llm doctor

stop:
	uv run spark-llm stop

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
