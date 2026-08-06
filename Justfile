ruff_version := "0.16.1"
basedpyright_version := "1.39.9"

default: fmt-check lint types

install:
    uv sync --frozen

lock:
    uv lock

fmt:
    uvx ruff@{{ruff_version}} format

fmt-check:
    uvx ruff@{{ruff_version}} format --check

lint:
    uvx ruff@{{ruff_version}} check

lint-fix:
    uvx ruff@{{ruff_version}} check --fix

types:
    npx basedpyright@{{basedpyright_version}}

run-example:
    PYTHONPATH=. uv run python -m experiments.basic.simple_optimization

run-r:
    PYTHONPATH=. uv run python -m experiments.cross_validation.des_vs_r
