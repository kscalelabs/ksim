# Makefile
.DEFAULT_GOAL := help # Sets default action to be help

define PRINT_HELP_PYSCRIPT # start of Python section
import re, sys

help_lines = []
max_len = 0

# Collect matching lines and calculate max target length
for line in sys.stdin:
    match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
    if match:
        target, help = match.groups()
        help_lines.append((target, help))
        max_len = max(max_len, len(target))

# Format and print help
for target, help in sorted(help_lines):
    print(f"{target.ljust(max_len + 2)}{help}")
endef
export PRINT_HELP_PYSCRIPT # End of python section

help: ## Show help
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

format: ## Format code with Ruff
	@echo "=== Running Ruff Formatting ==="
	@ruff format ksim tests examples
	@echo ""
	@echo "=== Running Ruff Checks ==="
	@ruff check --fix ksim tests examples

static-checks: ## Perform static checks with Ruff and MyPy
	@echo "=== Running Ruff Checks ==="
	@ruff check ksim tests examples
	@echo ""
	@echo "=== Running MyPy ==="
	@mypy --install-types --non-interactive ksim tests examples

test: ## Run tests
	python -m pytest

fast-check: ## Run Ruff and MyPy on staged Python files
	@echo "=== Running Ruff, MyPy and Pytest on staged files ==="
	@STAGED=$$(git diff --name-only --cached --diff-filter=ACMR | grep '\.py$$' || true); \
	if [ -z "$$STAGED" ]; then echo "No staged Python files."; exit 0; fi; \
	echo "Files: $$STAGED"; \
	ruff check $$STAGED && \
	mypy --install-types --non-interactive $$STAGED && \
	python -m pytest

.PHONY: help format static-checks test
