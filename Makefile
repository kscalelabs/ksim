# Makefile

format:
	@echo "=== Running Ruff Formatting ==="
	@ruff format ksim tests examples
	@echo ""
	@echo "=== Running Ruff Checks ==="
	@ruff check --fix ksim tests examples
.PHONY: format

static-checks:
	@echo "=== Running Ruff Checks ==="
	@ruff check ksim tests examples
	@echo ""
	@echo "=== Running MyPy ==="
	@mypy --install-types --non-interactive ksim tests examples
.PHONY: lint

test:
	python -m pytest
.PHONY: test
