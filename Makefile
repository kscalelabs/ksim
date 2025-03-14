# Makefile

format:
	@black ksim tests examples
	@ruff format ksim tests examples
	@ruff check --fix ksim tests examples
.PHONY: format

static-checks:
	@black --diff --check ksim tests examples
	@ruff check ksim tests examples
	@mypy --install-types --non-interactive ksim tests examples
.PHONY: lint

test:
	python -m pytest
.PHONY: test
