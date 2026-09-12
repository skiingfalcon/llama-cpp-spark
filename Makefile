.PHONY: build serve bench test lint doctor stop models download

build:
	./scripts/build.sh

serve:
	uv run spark-llm serve gpt-oss-20b

bench:
	uv run spark-llm bench gpt-oss-20b

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
