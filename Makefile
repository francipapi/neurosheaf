# Makefile for Neurosheaf development
# Use this file to run common development tasks

# Variables
PYTHON = python
PIP = pip
PYTEST = pytest
BLACK = black
ISORT = isort
FLAKE8 = flake8
MYPY = mypy
PACKAGE = neurosheaf
TESTS = tests
DOCS = docs

# Colors for output
GREEN = \033[0;32m
RED = \033[0;31m
BLUE = \033[0;34m
YELLOW = \033[0;33m
NC = \033[0m # No Color

.PHONY: help install install-dev install-all clean test test-unit test-integration test-performance test-coverage lint format type-check docs docs-serve benchmark profile clean-cache clean-build clean-all git-hooks

# Default target
help:
	@echo "$(BLUE)Neurosheaf Development Makefile$(NC)"
	@echo ""
	@echo "Available targets:"
	@echo "  $(GREEN)install$(NC)         Install package for development"
	@echo "  $(GREEN)install-dev$(NC)     Install with development dependencies"
	@echo "  $(GREEN)install-all$(NC)     Install with all optional dependencies"
	@echo ""
	@echo "  $(GREEN)test$(NC)            Run all tests"
	@echo "  $(GREEN)test-unit$(NC)       Run unit tests only"
	@echo "  $(GREEN)test-integration$(NC) Run integration tests only"
	@echo "  $(GREEN)test-performance$(NC) Run performance tests"
	@echo "  $(GREEN)test-coverage$(NC)   Run tests with coverage report"
	@echo ""
	@echo "  $(GREEN)lint$(NC)            Run all linting checks"
	@echo "  $(GREEN)format$(NC)          Format code with black and isort"
	@echo "  $(GREEN)type-check$(NC)      Run type checking with mypy"
	@echo ""
	@echo "  $(GREEN)docs$(NC)            Build documentation"
	@echo "  $(GREEN)docs-serve$(NC)      Serve documentation locally"
	@echo ""
	@echo "  $(GREEN)benchmark$(NC)       Run performance benchmarks"
	@echo "  $(GREEN)profile$(NC)         Run memory profiling"
	@echo ""
	@echo "  $(GREEN)clean$(NC)           Clean cache and build files"
	@echo "  $(GREEN)clean-all$(NC)       Clean everything including venv"
	@echo "  $(GREEN)git-hooks$(NC)       Install pre-commit hooks"

# Installation targets
install:
	@echo "$(BLUE)Installing Neurosheaf for development...$(NC)"
	$(PIP) install -e .

install-dev:
	@echo "$(BLUE)Installing Neurosheaf with development dependencies...$(NC)"
	$(PIP) install -e ".[dev]"

install-all:
	@echo "$(BLUE)Installing Neurosheaf with all dependencies...$(NC)"
	$(PIP) install -e ".[all]"

# Testing targets
test:
	@echo "$(BLUE)Running all tests...$(NC)"
	$(PYTEST) $(TESTS) -v

test-unit:
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTEST) $(TESTS)/unit -v

test-integration:
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTEST) $(TESTS)/integration -v

test-performance:
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(PYTEST) $(TESTS)/performance -v -m benchmark

test-coverage:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) $(TESTS) --cov=$(PACKAGE) --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

test-fast:
	@echo "$(BLUE)Running fast tests (excluding slow tests)...$(NC)"
	$(PYTEST) $(TESTS) -v -m "not slow"

test-gpu:
	@echo "$(BLUE)Running GPU tests...$(NC)"
	$(PYTEST) $(TESTS) -v -m gpu

test-phase1:
	@echo "$(BLUE)Running Phase 1 tests...$(NC)"
	$(PYTEST) $(TESTS) -v -m phase1

# Code quality targets
lint:
	@echo "$(BLUE)Running linting checks...$(NC)"
	$(FLAKE8) $(PACKAGE) $(TESTS)
	$(BLACK) --check $(PACKAGE) $(TESTS)
	$(ISORT) --check-only $(PACKAGE) $(TESTS)

format:
	@echo "$(BLUE)Formatting code...$(NC)"
	$(BLACK) $(PACKAGE) $(TESTS)
	$(ISORT) $(PACKAGE) $(TESTS)
	@echo "$(GREEN)Code formatted successfully$(NC)"

type-check:
	@echo "$(BLUE)Running type checking...$(NC)"
	$(MYPY) $(PACKAGE)

check-all: lint type-check
	@echo "$(GREEN)All checks passed!$(NC)"

# Documentation targets
docs:
	@echo "$(BLUE)Building documentation...$(NC)"
	cd $(DOCS) && make html
	@echo "$(GREEN)Documentation built in docs/_build/html/$(NC)"

docs-serve:
	@echo "$(BLUE)Serving documentation locally...$(NC)"
	cd $(DOCS)/_build/html && $(PYTHON) -m http.server 8000

docs-clean:
	@echo "$(BLUE)Cleaning documentation...$(NC)"
	cd $(DOCS) && make clean

# Performance targets
benchmark:
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	$(PYTHON) -m $(PACKAGE).utils.benchmarking

profile:
	@echo "$(BLUE)Running memory profiling...$(NC)"
	$(PYTHON) -m $(PACKAGE).utils.profiling

profile-baseline:
	@echo "$(BLUE)Running baseline profiling...$(NC)"
	$(PYTHON) benchmarks/profile_baseline.py

# Git and development setup
git-hooks:
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	pre-commit install
	@echo "$(GREEN)Pre-commit hooks installed$(NC)"

git-hooks-update:
	@echo "$(BLUE)Updating pre-commit hooks...$(NC)"
	pre-commit autoupdate

# Cleaning targets
clean-cache:
	@echo "$(BLUE)Cleaning cache files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

clean-build:
	@echo "$(BLUE)Cleaning build files...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .coverage.*

clean-docs:
	@echo "$(BLUE)Cleaning documentation...$(NC)"
	rm -rf $(DOCS)/_build/

clean: clean-cache clean-build clean-docs
	@echo "$(GREEN)Cleaned cache and build files$(NC)"

clean-all: clean
	@echo "$(BLUE)Cleaning everything...$(NC)"
	rm -rf venv/
	rm -rf .venv/
	@echo "$(GREEN)All files cleaned$(NC)"

# Development workflow targets
dev-setup: install-dev git-hooks
	@echo "$(GREEN)Development environment set up successfully!$(NC)"
	@echo "$(YELLOW)Run 'make test' to verify installation$(NC)"

quick-check: format lint test-fast
	@echo "$(GREEN)Quick development check complete!$(NC)"

full-check: format lint type-check test-coverage
	@echo "$(GREEN)Full development check complete!$(NC)"

# Release targets
check-release: clean format lint type-check test-coverage
	@echo "$(BLUE)Checking release readiness...$(NC)"
	$(PYTHON) -m build --check
	@echo "$(GREEN)Release checks passed!$(NC)"

build-release:
	@echo "$(BLUE)Building release packages...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)Release packages built in dist/$(NC)"

# CI/CD simulation
ci-test:
	@echo "$(BLUE)Running CI/CD simulation...$(NC)"
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test-coverage
	@echo "$(GREEN)CI/CD simulation complete!$(NC)"

# Monitoring and profiling
memory-profile:
	@echo "$(BLUE)Running memory profiling...$(NC)"
	$(PYTHON) -m memory_profiler $(PACKAGE)/utils/profiling.py

line-profile:
	@echo "$(BLUE)Running line profiling...$(NC)"
	kernprof -l -v $(PACKAGE)/utils/profiling.py

# Utilities
show-deps:
	@echo "$(BLUE)Showing dependency tree...$(NC)"
	pipdeptree

show-outdated:
	@echo "$(BLUE)Showing outdated packages...$(NC)"
	$(PIP) list --outdated

update-deps:
	@echo "$(BLUE)Updating development dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e ".[dev]"

# Help for specific phases
help-phase1:
	@echo "$(BLUE)Phase 1 (Foundation) Development Commands:$(NC)"
	@echo "  make install-dev     - Install development dependencies"
	@echo "  make git-hooks       - Set up pre-commit hooks"
	@echo "  make test-phase1     - Run Phase 1 tests"
	@echo "  make benchmark       - Run baseline benchmarks"
	@echo "  make profile         - Run memory profiling"

# Validation targets for Phase 1
validate-phase1: install-dev git-hooks test-phase1 benchmark
	@echo "$(GREEN)Phase 1 validation complete!$(NC)"

# Show project status
status:
	@echo "$(BLUE)Neurosheaf Project Status$(NC)"
	@echo "=========================="
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Package: $(shell $(PYTHON) -c 'import $(PACKAGE); print($(PACKAGE).__version__)')"
	@echo "Tests: $(shell find $(TESTS) -name '*.py' | wc -l) test files"
	@echo "Coverage: $(shell coverage report --show-missing 2>/dev/null | tail -1 | awk '{print $$4}' || echo 'Not measured')"
	@echo "Dependencies: $(shell $(PIP) list | wc -l) packages installed"