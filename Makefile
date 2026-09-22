# APEX -- development entry points.
#
# Targets are added by the phase that makes them meaningful. There are deliberately no stub
# targets that print "not implemented yet": if `make <target>` exists, it does the thing.

UV ?= uv
.DEFAULT_GOAL := help

.PHONY: help setup lint format types test check clean ingest ingest-all laps calibrate race replay validate

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## Create the Python 3.11 environment and install git hooks
	@command -v $(UV) >/dev/null 2>&1 || { \
		echo "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; }
	$(UV) sync --group dev
	$(UV) run pre-commit install

lint: ## Check style and import order
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Apply autofixes and formatting
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

types: ## Type-check under mypy strict
	$(UV) run mypy

test: ## Run the test suite, excluding anything that needs the network
	$(UV) run pytest -m "not network"

check: lint types test ## Everything CI runs

ingest: ## Download the pilot races (Bahrain and Monaco, all configured seasons)
	$(UV) run --extra data apex ingest --pilot

ingest-all: ## Download every race of every configured season -- slow, and hits the timing API
	$(UV) run --extra data apex ingest

laps: ## Report what the exclusion rules keep, and what they cost
	$(UV) run --extra data apex laps

calibrate: ## Fit every simulator parameter and write params/ and the calibration report
	$(UV) run --extra data --extra fit apex calibrate

race: ## Simulate one race from a grid and print the result
	$(UV) run --extra data --extra fit apex race --circuit Monza --laps 53

replay: ## Simulate a race and open a watchable top-down replay in the browser
	$(UV) run --extra data --extra fit apex replay --circuit Monza --laps 53

validate: ## Measure the simulator against the held-out races and print the error
	$(UV) run --extra data --extra fit apex validate --per-race

clean: ## Remove caches and build artefacts, but never downloaded race data
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
