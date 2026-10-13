PY      ?= .venv/bin/python
BUNDLE  ?= build/bundle
KEYS    ?= runtime/keys
IMAGE   ?= mlseal:local
export PYTHONPATH := src

.DEFAULT_GOAL := help
.PHONY: help install lock data-provenance validate train gate scan keys sign verify pipeline \
	promote serve docker docker-run audit sbom test lint format clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv from uv.lock (hash-verified) with dev and audit tools
	uv sync --locked --extra dev --extra audit

lock: ## Re-resolve uv.lock and re-export requirements/serve.txt (review the diff)
	uv lock
	uv export --frozen --no-dev --no-emit-project --format requirements-txt -o requirements/serve.txt

data-provenance: ## Re-download the UCI archive and compare it with the pinned digest
	$(PY) -m mlseal data-provenance

validate: ## Data gate only -> build/data_validation.json
	$(PY) -m mlseal validate

train: ## Data gate, de-duplicated split, CV selection, evaluation, export -> build/bundle/
	$(PY) -m mlseal train --bundle $(BUNDLE)

gate: ## Quality gates against the committed baseline (reports/metrics.json)
	$(PY) -m mlseal gate --bundle $(BUNDLE)

scan: ## Pickle scan of the bundle (a release bundle may contain no pickle at all)
	$(PY) -m mlseal scan $(BUNDLE) --forbid-pickle

keys: ## Create a local signing key in runtime/keys/ (git-ignored) unless one exists
	@test -f $(KEYS)/signing.key || $(PY) -m mlseal keygen --out-dir $(KEYS)

sign: keys ## Sign the bundle manifest (refused unless every gate passed)
	$(PY) -m mlseal sign --bundle $(BUNDLE) --key $(KEYS)/signing.key

verify: ## Verify signature, every digest and the strict model load
	$(PY) -m mlseal verify --bundle $(BUNDLE) --public-key $(KEYS)/signing.pub

pipeline: train gate scan sign verify ## Everything CI does to produce a signed release

promote: ## Make the last run the new baseline in reports/ (then review and commit)
	$(PY) -m mlseal promote --bundle $(BUNDLE)

serve: ## Serve the verified bundle on http://127.0.0.1:8080
	MLSEAL_PUBLIC_KEY=$(KEYS)/signing.pub $(PY) -m mlseal serve --bundle $(BUNDLE)

docker: ## Build the serving image
	docker build -t $(IMAGE) .

docker-run: ## Run the image hardened: read-only, no capabilities, bundle and key mounted read-only
	docker run --rm -p 127.0.0.1:8080:8080 --read-only --cap-drop ALL \
		--security-opt no-new-privileges --pids-limit 64 --memory 256m \
		-v "$(CURDIR)/$(BUNDLE):/models:ro" -v "$(CURDIR)/$(KEYS)/signing.pub:/keys/signing.pub:ro" \
		$(IMAGE)

audit: ## Vulnerability audit of every locked package, then lint the CI workflows
	@mkdir -p build
	uv export --frozen --all-extras --no-emit-project --format requirements-txt -o build/requirements-all.txt
	$(PY) -m pip_audit --requirement build/requirements-all.txt --require-hashes --disable-pip --strict
	.venv/bin/zizmor --offline .github/workflows

sbom: ## CycloneDX SBOM of the serving dependencies -> build/sbom.cdx.json
	@mkdir -p build
	$(PY) -m cyclonedx_py requirements requirements/serve.txt --pyproject pyproject.toml \
		--output-reproducible -o build/sbom.cdx.json

test: ## Run the test suite (trains small models on the real data, ~1 min)
	$(PY) -m pytest --cov --cov-report=term-missing

lint: ## Lint (including the Bandit security rules) and check formatting
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format: ## Auto-format the code
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

clean: ## Remove caches and pipeline output (keeps runtime/keys)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
