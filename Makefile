.PHONY: install lint fmt type test check data

install:
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

type:
	uv run mypy

test:
	uv run pytest -q

check: lint type test

data:
	uv run modelrouter data prepare
