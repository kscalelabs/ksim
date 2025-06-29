# Makefile

format:
	@echo "=== Running Ruff Formatting ==="
	@/Users/claywarren/ksim/venv/bin/ruff format ksim tests examples
	@echo ""
	@echo "=== Running Ruff Checks ==="
	@/Users/claywarren/ksim/venv/bin/ruff check --fix ksim tests examples
.PHONY: format

static-checks:
	@echo "=== Running Ruff Checks ==="
	@/Users/claywarren/ksim/venv/bin/ruff check ksim tests examples
	@echo ""
	@echo "=== Running MyPy ==="
	@/Users/claywarren/ksim/venv/bin/mypy ksim tests examples
.PHONY: lint

test:
	python -m pytest
.PHONY: test
