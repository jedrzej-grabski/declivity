ruff_version := "0.16.1"
basedpyright_version := "1.39.9"

default: fmt-check lint types

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
